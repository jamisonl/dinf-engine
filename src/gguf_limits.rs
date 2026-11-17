//! Application-level limits for untrusted GGUF headers and metadata.
//!
//! The layer limit bounds every model and pipeline layer range the engine
//! accepts; the larger entry limits allow realistic models to have many
//! tensors and tokenizer values per layer without allowing a forged count to
//! request an unbounded allocation.

#[cfg(test)]
use crate::verified_file::SparseSetLen;
use std::collections::{HashMap, hash_map::DefaultHasher};
use std::fs::File;
use std::hash::Hasher;
use std::io::{Read, Seek, SeekFrom};

use anyhow::{Context, Result};

/// Maximum executable transformer layers accepted by the engine.
pub(crate) const MAX_MODEL_LAYERS: usize = 4_096;
/// Maximum tensor records in one GGUF header.
pub(crate) const MAX_GGUF_TENSOR_COUNT: u64 = (MAX_MODEL_LAYERS as u64) * 256;
/// Maximum metadata key/value records in one GGUF header.
pub(crate) const MAX_GGUF_METADATA_COUNT: u64 = (MAX_MODEL_LAYERS as u64) * 64;
/// Maximum values in one metadata array.  This accommodates large token
/// vocabularies while bounding `Vec<Value>` allocations in the raw reader.
pub(crate) const MAX_GGUF_ARRAY_ELEMENTS: u64 = (MAX_MODEL_LAYERS as u64) * 256;
/// Maximum parsed metadata values (array elements included) in one header.
/// Every value becomes a `gguf_file::Value` of `size_of::<Value>()` bytes in
/// memory, so the on-disk byte budget alone admits ~32x amplification from
/// one-byte array elements. Real tokenizers use a few million at most.
pub(crate) const MAX_GGUF_METADATA_VALUES: u64 = 8 * 1024 * 1024;
/// Maximum header budget considered by application-level preflight checks.
pub(crate) const MAX_GGUF_HEADER_BYTES: u64 = 256 * 1024 * 1024;
/// Maximum number of split shards accepted by the loader.
pub(crate) const MAX_GGUF_SHARDS: usize = 128;
/// Maximum total parsed-header storage retained across a split model. Shards
/// normally repeat only a tiny split descriptor; this bound prevents a model
/// from multiplying the generous per-file header limit by the shard count.
pub(crate) const MAX_GGUF_AGGREGATE_HEADER_BYTES: u64 = 512 * 1024 * 1024;
/// Maximum logical elements in a single tensor. This still permits very large
/// packed expert tensors while rejecting forged shapes before multiplication
/// reaches Candle or an accelerator allocator.
pub(crate) const MAX_GGUF_TENSOR_ELEMENTS: u64 = 1 << 36;
/// Maximum encoded bytes in one tensor. Real packed expert tensors are well
/// below this; the bound prevents a sparse file from advertising a single
/// allocation hundreds of gigabytes in size.
pub(crate) const MAX_GGUF_TENSOR_BYTES: u64 = 64 * 1024 * 1024 * 1024;

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_SWAPPED_MAGIC: u32 = 0x4747_5546;
const MAX_GGUF_TENSOR_DIMS: u32 = 4;
const MAX_GGUF_VALUE_DEPTH: usize = 64;

#[derive(Clone, Copy)]
enum GgufVersion {
    V1,
    V2,
    V3,
}

impl GgufVersion {
    fn read_len(self, file: &mut File, file_size: u64) -> Result<u64> {
        match self {
            Self::V1 => Ok(read_u32(file, file_size)? as u64),
            Self::V2 | Self::V3 => read_u64(file, file_size),
        }
    }

    fn length_prefix_bytes(self) -> u64 {
        match self {
            Self::V1 => 4,
            Self::V2 | Self::V3 => 8,
        }
    }

    fn read_dimension(self, file: &mut File, file_size: u64) -> Result<u64> {
        match self {
            Self::V1 => Ok(read_u32(file, file_size)? as u64),
            Self::V2 | Self::V3 => read_u64(file, file_size),
        }
    }
}

/// Validate a GGUF header before handing it to a downstream GGUF reader.
///
/// Candle's reader has count limits, but a file can still stay below those
/// limits while putting arbitrarily large strings/arrays in each record.  A
/// sparse model file makes this especially easy to miss: seeking across a
/// forged header does not consume disk space.  Scan the records with checked
/// seeks first so the *complete* metadata + tensor table, including padding,
/// is bounded before any downstream parser allocates from it.
///
/// The duplicate check keeps only a bounded fingerprint index of record names
/// (one file offset and normalized length per record); it never allocates from
/// an attacker-controlled string length or retains the strings themselves.
/// The existing record-count and header-size limits bound that index.  Hash
/// matches are verified byte-for-byte so a fingerprint collision cannot turn
/// two distinct names into a false duplicate.  The file cursor is restored on
/// both success and failure.
/// What a validated header will cost once parsed, beyond its disk bytes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GgufHeaderStats {
    pub(crate) metadata_values: u64,
}

impl GgufHeaderStats {
    /// Disk header bytes plus the parsed metadata values' in-memory size.
    pub(crate) fn parsed_bytes(&self, header_bytes: u64) -> u64 {
        let value_bytes = std::mem::size_of::<candle_core::quantized::gguf_file::Value>() as u64;
        header_bytes.saturating_add(self.metadata_values.saturating_mul(value_bytes))
    }
}

struct ValueBudget(u64);

impl ValueBudget {
    fn charge(&mut self, values: u64) -> Result<()> {
        self.0 = self
            .0
            .checked_add(values)
            .context("GGUF metadata value count overflow")?;
        anyhow::ensure!(
            self.0 <= MAX_GGUF_METADATA_VALUES,
            "GGUF metadata values ({}) exceed application limit ({MAX_GGUF_METADATA_VALUES})",
            self.0
        );
        Ok(())
    }
}

pub(crate) fn validate_gguf_header_budget(file: &mut File) -> Result<GgufHeaderStats> {
    let original_position = file.stream_position()?;
    let mut values = ValueBudget(0);
    let result = (|| {
        file.seek(SeekFrom::Start(0))?;
        let file_size = file.metadata()?.len();
        let version = match read_u32(file, file_size)? {
            GGUF_MAGIC | GGUF_SWAPPED_MAGIC => match read_u32(file, file_size)? {
                1 => GgufVersion::V1,
                2 => GgufVersion::V2,
                3 => GgufVersion::V3,
                version => anyhow::bail!("unsupported GGUF version {version}"),
            },
            magic => anyhow::bail!("invalid GGUF magic 0x{magic:08x}"),
        };
        let tensor_count = version.read_len(file, file_size)?;
        let metadata_count = version.read_len(file, file_size)?;
        anyhow::ensure!(
            tensor_count <= MAX_GGUF_TENSOR_COUNT,
            "GGUF tensor count ({tensor_count}) exceeds application limit ({MAX_GGUF_TENSOR_COUNT})"
        );
        anyhow::ensure!(
            metadata_count <= MAX_GGUF_METADATA_COUNT,
            "GGUF metadata count ({metadata_count}) exceeds application limit ({MAX_GGUF_METADATA_COUNT})"
        );

        // Reject impossible counts before scanning any attacker-controlled
        // record lengths.  These are lower bounds, not the final budget check.
        let metadata_min_bytes = version
            .length_prefix_bytes()
            .checked_add(5)
            .context("GGUF metadata minimum size overflow")?;
        let tensor_min_bytes = version
            .length_prefix_bytes()
            .checked_add(16)
            .context("GGUF tensor minimum size overflow")?;
        let minimum_records = metadata_count
            .checked_mul(metadata_min_bytes)
            .and_then(|bytes| {
                tensor_count
                    .checked_mul(tensor_min_bytes)
                    .and_then(|tensor_bytes| bytes.checked_add(tensor_bytes))
            })
            .context("GGUF header minimum size overflow")?;
        let position = file.stream_position()?;
        let header_remaining = MAX_GGUF_HEADER_BYTES
            .checked_sub(position)
            .context("GGUF fixed header exceeds application limit")?;
        anyhow::ensure!(
            minimum_records <= header_remaining,
            "GGUF header minimum size ({minimum_records} bytes) exceeds application limit"
        );
        anyhow::ensure!(
            minimum_records <= file_size.saturating_sub(position),
            "GGUF header declares more entries than the file can contain"
        );

        let mut alignment = 32u64;
        let mut deepseek41 = false;
        let mut engram_encoding = false;
        let (mut split_no, mut split_count, mut split_tensors) = (None, None, None);
        let mut metadata_keys = DuplicateStringIndex::default();
        for _ in 0..metadata_count {
            values.charge(1)?;
            let key = read_string_scan(file, version, file_size)?;
            metadata_keys.record(file, file_size, key, "metadata key")?;
            let value_type = read_u32(file, file_size)?;
            if key.is_general_alignment() {
                // Candle only honors 8-, 16-, and 32-bit integer alignment
                // metadata. Accepting U64/I64 here would validate one data
                // offset while Candle reads another and treats padding as
                // weights. Keep both readers on the same accepted types.
                anyhow::ensure!(
                    matches!(value_type, 0..=5),
                    "GGUF general.alignment must use an 8-, 16-, or 32-bit integer"
                );
                alignment = skip_scalar_u64(file, version, file_size, value_type, &mut values)?
                    .context("GGUF general.alignment must be a non-negative integer")?;
                anyhow::ensure!(
                    alignment.is_power_of_two() && alignment <= 1024 * 1024,
                    "GGUF alignment must be a power of two no greater than 1048576"
                );
            } else if key.matches(b"split.no")
                || key.matches(b"split.count")
                || key.matches(b"split.tensors.count")
            {
                let value = skip_scalar_u64(file, version, file_size, value_type, &mut values)?;
                if key.matches(b"split.no") {
                    split_no = value;
                } else if key.matches(b"split.count") {
                    split_count = value;
                } else {
                    split_tensors = value;
                }
            } else if value_type == 8
                && (key.matches(b"general.architecture")
                    || key.matches(b"deepseek41.engram.encoding"))
            {
                let value = read_string_scan(file, version, file_size)?;
                if key.matches(b"general.architecture") {
                    deepseek41 = value.matches(b"deepseek41");
                } else {
                    engram_encoding = value.matches(b"e4m3_e8m0_32_row264");
                }
            } else {
                skip_value(file, version, file_size, value_type, 0, &mut values)?;
            }
        }
        let mut maximum_tensor_end = 0u64;
        let mut tensor_names = DuplicateStringIndex::default();
        let mut spans = Vec::new();
        for _ in 0..tensor_count {
            let name = read_string_scan(file, version, file_size)?;
            tensor_names.record(file, file_size, name, "tensor name")?;
            let dimensions = read_u32(file, file_size)?;
            anyhow::ensure!(
                (1..=MAX_GGUF_TENSOR_DIMS).contains(&dimensions),
                "GGUF tensor dimensions ({dimensions}) are outside 1..={MAX_GGUF_TENSOR_DIMS}"
            );
            let mut element_count = 1u64;
            let mut first_dimension = None;
            let mut shape = [0u64; 4];
            for dimension_index in 0..dimensions {
                let dimension = version.read_dimension(file, file_size)?;
                anyhow::ensure!(dimension > 0, "GGUF tensor dimensions must be positive");
                shape[dimension_index as usize] = dimension;
                if dimension_index == 0 {
                    first_dimension = Some(dimension);
                }
                element_count = element_count
                    .checked_mul(dimension)
                    .context("GGUF tensor element count overflow")?;
            }
            let tensor_type = read_u32(file, file_size)?;
            let tensor_offset = read_u64(file, file_size)?;
            let engram_name = name.matches(b"blk.1.engram_embd.weight")
                || name.matches(b"blk.14.engram_embd.weight");
            let maya_engram = deepseek41
                && engram_encoding
                && engram_name
                && tensor_type == 24
                && dimensions == 2
                && shape[0] == 264
                && shape[1]
                    == if name.matches(b"blk.1.engram_embd.weight") {
                        384006168
                    } else {
                        384016682
                    };
            // Header permission only; the native loader still verifies the
            // complete pinned artifact before it can read rows. Never lift
            // the resident tensor cap for other shapes, parts or quant types.
            let coder_engram = split_no == Some(5)
                && split_count == Some(6)
                && split_tensors == Some(1043)
                && tensor_count == 1
                && file_size == crate::gsq_coder::PARTS[5].0
                && name.matches(b"per_layer_token_embd.weight")
                && tensor_type == 20
                && dimensions == 2
                && shape[..2] == [256, 768022850];
            let disk_engram = maya_engram || coder_engram;
            if deepseek41 && name.display_name().contains("engram_embd") {
                anyhow::ensure!(
                    disk_engram,
                    "invalid disk-only DeepSeek V4.1 Engram descriptor"
                );
            }
            anyhow::ensure!(
                disk_engram || element_count <= MAX_GGUF_TENSOR_ELEMENTS,
                "GGUF tensor element count ({element_count}) exceeds application limit ({MAX_GGUF_TENSOR_ELEMENTS})"
            );
            anyhow::ensure!(
                tensor_type != 24 || disk_engram,
                "I8 is supported only for exact disk-only DeepSeek V4.1 Engram tables"
            );
            let (block_elements, block_bytes) = ggml_block_layout(tensor_type)?;
            anyhow::ensure!(
                first_dimension.is_some_and(|dimension| dimension % block_elements == 0),
                "GGUF tensor innermost dimension is not divisible by type {tensor_type} block size {block_elements}"
            );
            let tensor_bytes = element_count
                .checked_div(block_elements)
                .and_then(|blocks| blocks.checked_mul(block_bytes))
                .context("GGUF tensor byte length overflow")?;
            anyhow::ensure!(
                disk_engram || tensor_bytes <= MAX_GGUF_TENSOR_BYTES,
                "GGUF tensor byte length ({tensor_bytes}) exceeds application limit ({MAX_GGUF_TENSOR_BYTES})"
            );
            let tensor_end = tensor_offset
                .checked_add(tensor_bytes)
                .context("GGUF tensor data offset overflow")?;
            maximum_tensor_end = maximum_tensor_end.max(tensor_end);
            if deepseek41 || coder_engram {
                anyhow::ensure!(
                    tensor_offset.is_multiple_of(alignment),
                    "unaligned DeepSeek V4.1 tensor offset"
                );
            }
            // Every file: tensors must not share bytes, so the total declared
            // tensor data cannot exceed the file size.
            spans.push((tensor_offset, tensor_end));
        }
        spans.sort_unstable();
        for adjacent in spans.windows(2) {
            anyhow::ensure!(
                adjacent[0].1 <= adjacent[1].0,
                "overlapping GGUF tensor spans"
            );
        }

        let position = file.stream_position()?;
        ensure_header_position(position)?;
        let aligned = position
            .checked_add(alignment.saturating_sub(1))
            .and_then(|value| value.checked_div(alignment))
            .and_then(|value| value.checked_mul(alignment))
            .context("GGUF header alignment overflow")?;
        ensure_header_position(aligned)?;
        let data_end = aligned
            .checked_add(maximum_tensor_end)
            .context("GGUF tensor data extent overflow")?;
        anyhow::ensure!(
            data_end <= file_size,
            "GGUF tensor data extends beyond the end of the file"
        );
        Ok(())
    })();
    let restore = file.seek(SeekFrom::Start(original_position));
    match (result, restore) {
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error.into()),
        (Ok(()), Ok(_)) => Ok(GgufHeaderStats {
            metadata_values: values.0,
        }),
    }
}

fn ensure_header_position(position: u64) -> Result<()> {
    anyhow::ensure!(
        position <= MAX_GGUF_HEADER_BYTES,
        "GGUF parsed header/table position ({position} bytes) exceeds application limit ({} bytes)",
        MAX_GGUF_HEADER_BYTES
    );
    Ok(())
}

fn read_exact(file: &mut File, file_size: u64, bytes: &mut [u8]) -> Result<()> {
    let position = file.stream_position()?;
    let end = position
        .checked_add(bytes.len() as u64)
        .context("GGUF header offset overflow")?;
    ensure_header_position(end)?;
    anyhow::ensure!(
        end <= file_size,
        "GGUF header extends beyond the end of the file"
    );
    file.read_exact(bytes).context("truncated GGUF header")
}

fn read_u32(file: &mut File, file_size: u64) -> Result<u32> {
    let mut bytes = [0u8; 4];
    read_exact(file, file_size, &mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u8(file: &mut File, file_size: u64) -> Result<u8> {
    let mut bytes = [0u8; 1];
    read_exact(file, file_size, &mut bytes)?;
    Ok(bytes[0])
}

fn read_u64(file: &mut File, file_size: u64) -> Result<u64> {
    let mut bytes = [0u8; 8];
    read_exact(file, file_size, &mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_u16(file: &mut File, file_size: u64) -> Result<u16> {
    let mut bytes = [0u8; 2];
    read_exact(file, file_size, &mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn skip_bytes(file: &mut File, file_size: u64, length: u64) -> Result<()> {
    let position = file.stream_position()?;
    let end = position
        .checked_add(length)
        .context("GGUF header offset overflow")?;
    ensure_header_position(end)?;
    anyhow::ensure!(
        end <= file_size,
        "GGUF header extends beyond the end of the file"
    );
    file.seek(SeekFrom::Start(end))?;
    Ok(())
}

#[derive(Clone, Copy)]
struct StringScan {
    /// First byte of the string payload in the file.
    start: u64,
    /// Length after the same trailing-NUL normalization used by GGUF readers.
    len: u64,
    /// Hash of the normalized payload.  Equal hashes are always byte-checked.
    hash: u64,
    /// Prefix used for the small structural metadata key lookup.
    small: [u8; 64],
    small_len: usize,
}

impl StringScan {
    fn matches(self, expected: &[u8]) -> bool {
        self.len == expected.len() as u64
            && expected.len() <= self.small_len
            && self.small[..expected.len()] == *expected
    }

    fn is_general_alignment(self) -> bool {
        let mut len = self.small_len;
        while len > 0 && self.small[len - 1] == 0 {
            len -= 1;
        }
        self.len == b"general.alignment".len() as u64
            && len == b"general.alignment".len()
            && self.small[..len] == b"general.alignment"[..]
    }

    fn display_name(self) -> String {
        let mut len = self.small_len;
        while len > 0 && self.small[len - 1] == 0 {
            len -= 1;
        }
        if self.small_len == 64 && self.len > len as u64 {
            format!("<{} bytes>", self.len)
        } else {
            String::from_utf8_lossy(&self.small[..len]).into_owned()
        }
    }
}

/// Exact duplicate detection without retaining untrusted GGUF strings.
///
/// The index is bounded by the already-validated record count and header
/// budget.  It stores only offsets and normalized lengths; a hash match is
/// checked against the file before it is reported as a duplicate.
#[derive(Default)]
struct DuplicateStringIndex {
    by_hash: HashMap<u64, Vec<(u64, u64)>>,
}

impl DuplicateStringIndex {
    fn record(
        &mut self,
        file: &mut File,
        file_size: u64,
        scan: StringScan,
        label: &str,
    ) -> Result<()> {
        if let Some(previous) = self.by_hash.get(&scan.hash) {
            for &(previous_start, previous_len) in previous {
                if previous_len == scan.len
                    && equal_file_ranges(file, file_size, previous_start, scan.start, scan.len)?
                {
                    anyhow::bail!("duplicate GGUF {label} {:?}", scan.display_name());
                }
            }
        }

        // Make collection growth fallible.  The total number of entries is
        // bounded by the validated GGUF record count, rather than by a string
        // length supplied by the file.
        if !self.by_hash.contains_key(&scan.hash) {
            self.by_hash.try_reserve(1).map_err(|error| {
                anyhow::anyhow!("GGUF duplicate index allocation failed: {error}")
            })?;
        }
        let bucket = self.by_hash.entry(scan.hash).or_default();
        bucket
            .try_reserve(1)
            .map_err(|error| anyhow::anyhow!("GGUF duplicate index allocation failed: {error}"))?;
        bucket.push((scan.start, scan.len));
        Ok(())
    }
}

fn read_string_scan(file: &mut File, version: GgufVersion, file_size: u64) -> Result<StringScan> {
    let raw_len = version.read_len(file, file_size)?;
    let start = file.stream_position()?;
    let end = start
        .checked_add(raw_len)
        .context("GGUF string offset overflow")?;
    ensure_header_position(end)?;
    anyhow::ensure!(
        end <= file_size,
        "GGUF string extends beyond the end of the file"
    );

    let small_len = usize::try_from(raw_len.min(64)).expect("bounded GGUF key prefix");
    let mut small = [0u8; 64];
    let mut hasher = DefaultHasher::new();
    // Keep at most three bytes from a UTF-8 character split across reads.
    // Both downstream readers decode invalid UTF-8 lossily, which can turn
    // distinct byte names into the same HashMap key. Reject those names
    // before the duplicate check instead of letting one record replace another.
    let mut buffer = [0u8; 4099];
    let mut utf8_prefix = 0usize;
    let mut remaining = raw_len;
    let mut offset = 0u64;
    let mut logical_len = 0u64;
    let mut pending_zeroes = 0u64;
    while remaining > 0 {
        let chunk_len = remaining.min(4096) as usize;
        let read_end = utf8_prefix + chunk_len;
        read_exact(file, file_size, &mut buffer[utf8_prefix..read_end])?;
        let utf8_tail = match std::str::from_utf8(&buffer[..read_end]) {
            Ok(_) => 0,
            Err(error) if error.error_len().is_none() => read_end - error.valid_up_to(),
            Err(_) => anyhow::bail!("GGUF record name or structural string is not valid UTF-8"),
        };
        anyhow::ensure!(
            remaining > chunk_len as u64 || utf8_tail == 0,
            "GGUF record name or structural string ends with incomplete UTF-8"
        );
        let chunk = &buffer[utf8_prefix..read_end];
        if offset < small_len as u64 {
            let copy_len = (small_len as u64 - offset).min(chunk_len as u64) as usize;
            let copy_start = offset as usize;
            small[copy_start..copy_start + copy_len].copy_from_slice(&chunk[..copy_len]);
        }
        for (index, &byte) in chunk.iter().enumerate() {
            if byte == 0 {
                pending_zeroes += 1;
            } else {
                for _ in 0..pending_zeroes {
                    hasher.write_u8(0);
                }
                pending_zeroes = 0;
                hasher.write_u8(byte);
                logical_len = offset + index as u64 + 1;
            }
        }
        offset += chunk_len as u64;
        remaining -= chunk_len as u64;
        buffer.copy_within(read_end - utf8_tail..read_end, 0);
        utf8_prefix = utf8_tail;
    }

    Ok(StringScan {
        start,
        len: logical_len,
        hash: hasher.finish(),
        small,
        small_len,
    })
}

fn equal_file_ranges(
    file: &mut File,
    file_size: u64,
    first_start: u64,
    second_start: u64,
    length: u64,
) -> Result<bool> {
    let original_position = file.stream_position()?;
    let result = (|| {
        let mut first = [0u8; 4096];
        let mut second = [0u8; 4096];
        let mut offset = 0u64;
        while offset < length {
            let chunk_len = (length - offset).min(first.len() as u64) as usize;
            let first_position = first_start
                .checked_add(offset)
                .context("GGUF duplicate comparison offset overflow")?;
            file.seek(SeekFrom::Start(first_position))?;
            read_exact(file, file_size, &mut first[..chunk_len])?;
            let second_position = second_start
                .checked_add(offset)
                .context("GGUF duplicate comparison offset overflow")?;
            file.seek(SeekFrom::Start(second_position))?;
            read_exact(file, file_size, &mut second[..chunk_len])?;
            if first[..chunk_len] != second[..chunk_len] {
                return Ok(false);
            }
            offset += chunk_len as u64;
        }
        Ok(true)
    })();
    let restore = file.seek(SeekFrom::Start(original_position));
    match (result, restore) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error.into()),
        (Ok(equal), Ok(_)) => Ok(equal),
    }
}

fn skip_string(file: &mut File, version: GgufVersion, file_size: u64) -> Result<()> {
    let length = version.read_len(file, file_size)?;
    skip_bytes(file, file_size, length)
}

fn skip_scalar_u64(
    file: &mut File,
    version: GgufVersion,
    file_size: u64,
    value_type: u32,
    values: &mut ValueBudget,
) -> Result<Option<u64>> {
    Ok(match value_type {
        0 => Some(read_u8(file, file_size)? as u64),
        1 => {
            let value = read_u8(file, file_size)? as i8;
            (value >= 0).then_some(value as u64)
        }
        2 => Some(read_u16(file, file_size)? as u64),
        3 => {
            let value = read_u16(file, file_size)? as i16;
            (value >= 0).then_some(value as u64)
        }
        4 => Some(read_u32(file, file_size)? as u64),
        5 => {
            let value = read_u32(file, file_size)? as i32;
            (value >= 0).then_some(value as u64)
        }
        10 => Some(read_u64(file, file_size)?),
        11 => {
            let value = read_u64(file, file_size)? as i64;
            (value >= 0).then_some(value as u64)
        }
        _ => {
            skip_value(file, version, file_size, value_type, 0, values)?;
            None
        }
    })
}

fn skip_value(
    file: &mut File,
    version: GgufVersion,
    file_size: u64,
    value_type: u32,
    depth: usize,
    values: &mut ValueBudget,
) -> Result<()> {
    anyhow::ensure!(
        depth <= MAX_GGUF_VALUE_DEPTH,
        "GGUF metadata nesting is too deep"
    );
    match value_type {
        0 | 1 | 7 => skip_bytes(file, file_size, 1)?,
        2 | 3 => skip_bytes(file, file_size, 2)?,
        4..=6 => skip_bytes(file, file_size, 4)?,
        8 => skip_string(file, version, file_size)?,
        9 => {
            let element_type = read_u32(file, file_size)?;
            let count = version.read_len(file, file_size)?;
            anyhow::ensure!(
                count <= MAX_GGUF_ARRAY_ELEMENTS,
                "GGUF array element count ({count}) exceeds application limit ({MAX_GGUF_ARRAY_ELEMENTS})"
            );
            values.charge(count)?;
            let minimum = count
                .checked_mul(min_value_disk_bytes(version, element_type)?)
                .context("GGUF array minimum size overflow")?;
            let position = file.stream_position()?;
            let minimum_end = position
                .checked_add(minimum)
                .context("GGUF array header offset overflow")?;
            ensure_header_position(minimum_end)?;
            anyhow::ensure!(
                minimum_end <= file_size,
                "GGUF array declares more elements than the file can contain"
            );
            match element_type {
                0 | 1 | 7 => skip_bytes(file, file_size, count)?,
                2 | 3 => skip_bytes(
                    file,
                    file_size,
                    count.checked_mul(2).context("GGUF array size overflow")?,
                )?,
                4..=6 => skip_bytes(
                    file,
                    file_size,
                    count.checked_mul(4).context("GGUF array size overflow")?,
                )?,
                10..=12 => skip_bytes(
                    file,
                    file_size,
                    count.checked_mul(8).context("GGUF array size overflow")?,
                )?,
                8 => {
                    for _ in 0..count {
                        skip_string(file, version, file_size)?;
                    }
                }
                9 => {
                    for _ in 0..count {
                        skip_value(file, version, file_size, 9, depth + 1, values)?;
                    }
                }
                _ => unreachable!("array element type validated by min_value_disk_bytes"),
            }
        }
        10..=12 => skip_bytes(file, file_size, 8)?,
        value_type => anyhow::bail!("unsupported GGUF value type {value_type}"),
    }
    Ok(())
}

fn ggml_block_layout(tensor_type: u32) -> Result<(u64, u64)> {
    Ok(match tensor_type {
        0 => (1, 4),   // F32
        1 => (1, 2),   // F16
        2 => (32, 18), // Q4_0
        3 => (32, 20), // Q4_1
        6 => (32, 22), // Q5_0
        7 => (32, 24), // Q5_1
        8 => (32, 34), // Q8_0
        9 => (32, 36), // Q8_1
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8 (disk-only Engram on V4.1)
        26 => (1, 4),     // I32
        29 => (256, 56),  // IQ1_M
        30 => (1, 2),     // BF16
        39 => (32, 17),   // MXFP4
        value => anyhow::bail!("unsupported GGUF tensor type {value}"),
    })
}

fn min_value_disk_bytes(version: GgufVersion, value_type: u32) -> Result<u64> {
    Ok(match value_type {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        8 => version.length_prefix_bytes(),
        9 => 4 + version.length_prefix_bytes(),
        value_type => anyhow::bail!("unsupported GGUF array type {value_type}"),
    })
}

pub(crate) fn validate_layer_count(count: usize, label: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        (1..=MAX_MODEL_LAYERS).contains(&count),
        "{label} ({count}) is outside 1..={MAX_MODEL_LAYERS}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};
    use tempfile::NamedTempFile;

    fn v3_fixture(metadata_keys: &[&[u8]], tensor_names: &[&[u8]]) -> Vec<u8> {
        v3_fixture_with_offset_step(metadata_keys, tensor_names, 32)
    }

    /// One-element F32 tensors placed `offset_step` bytes apart.
    fn v3_fixture_with_offset_step(
        metadata_keys: &[&[u8]],
        tensor_names: &[&[u8]],
        offset_step: u64,
    ) -> Vec<u8> {
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
        for (index, name) in tensor_names.iter().enumerate() {
            bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
            bytes.extend_from_slice(name);
            bytes.extend_from_slice(&1u32.to_le_bytes()); // one dimension
            bytes.extend_from_slice(&1u64.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes()); // F32
            bytes.extend_from_slice(&(index as u64 * offset_step).to_le_bytes());
        }
        let aligned_header = (bytes.len() + 31) & !31;
        let last_offset = tensor_names.len().saturating_sub(1) * offset_step as usize;
        bytes.resize(aligned_header + last_offset + 4, 0);
        bytes
    }

    fn open_fixture(metadata_keys: &[&[u8]], tensor_names: &[&[u8]]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&v3_fixture(metadata_keys, tensor_names))
            .unwrap();
        file
    }

    #[test]
    fn rejects_duplicate_metadata_keys_before_downstream_hashmap() {
        let mut file = open_fixture(&[b"general.architecture", b"general.architecture\0"], &[]);
        file.seek(SeekFrom::Start(7)).unwrap();
        let error = validate_gguf_header_budget(file.as_file_mut())
            .unwrap_err()
            .to_string();
        assert!(error.contains("duplicate GGUF metadata key"), "{error}");
        assert_eq!(file.stream_position().unwrap(), 7);
    }

    #[test]
    fn rejects_duplicate_tensor_names_before_downstream_hashmap() {
        let mut file = open_fixture(&[], &[b"blk.0.attn", b"blk.0.attn"]);
        let error = validate_gguf_header_budget(file.as_file_mut())
            .unwrap_err()
            .to_string();
        assert!(error.contains("duplicate GGUF tensor name"), "{error}");
    }

    #[test]
    fn rejects_tensors_that_share_file_bytes() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&v3_fixture_with_offset_step(&[], &[b"first", b"second"], 0))
            .unwrap();
        let error = validate_gguf_header_budget(file.as_file_mut())
            .unwrap_err()
            .to_string();
        assert!(error.contains("overlapping GGUF tensor spans"), "{error}");
    }

    #[test]
    fn accepts_distinct_names_and_restores_cursor() {
        let mut file = open_fixture(&[b"one", b"two"], &[b"first", b"second"]);
        file.seek(SeekFrom::Start(7)).unwrap();
        validate_gguf_header_budget(file.as_file_mut()).unwrap();
        assert_eq!(file.stream_position().unwrap(), 7);
    }

    fn alignment_fixture(value_type: u32, value: &[u8]) -> NamedTempFile {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata count
        bytes.extend_from_slice(&17u64.to_le_bytes());
        bytes.extend_from_slice(b"general.alignment");
        bytes.extend_from_slice(&value_type.to_le_bytes());
        bytes.extend_from_slice(value);
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.push(b'x');
        bytes.extend_from_slice(&1u32.to_le_bytes()); // one dimension
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // F32
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.resize(bytes.len().div_ceil(64) * 64, 0);
        bytes.extend_from_slice(&123.0f32.to_le_bytes());
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        file
    }

    #[test]
    fn accepted_alignment_types_read_weights_at_the_same_offset() -> Result<()> {
        use candle_core::{Device, quantized::gguf_file};
        for (value_type, value) in [
            (0, vec![64]),
            (1, vec![64]),
            (2, 64u16.to_le_bytes().to_vec()),
            (3, 64i16.to_le_bytes().to_vec()),
            (4, 64u32.to_le_bytes().to_vec()),
            (5, 64i32.to_le_bytes().to_vec()),
        ] {
            let mut file = alignment_fixture(value_type, &value);
            validate_gguf_header_budget(file.as_file_mut())?;
            file.seek(SeekFrom::Start(0))?;
            let candle = gguf_file::Content::read(file.as_file_mut())?;
            let raw = crate::deepseek4_gguf::RawGgufContent::read(file.as_file_mut())?;
            assert_eq!(candle.tensor_data_offset, raw.tensor_data_offset);
            let values = candle
                .tensor(file.as_file_mut(), "x", &Device::Cpu)?
                .dequantize(&Device::Cpu)?
                .to_vec1::<f32>()?;
            assert_eq!(values, [123.0], "alignment type {value_type}");
        }
        Ok(())
    }

    #[test]
    fn rejects_alignment_types_ignored_by_candle_before_reading_weights() {
        for value_type in [10, 11] {
            // U64 and I64
            let mut file = alignment_fixture(value_type, &64u64.to_le_bytes());
            file.seek(SeekFrom::Start(7)).unwrap();
            let error = validate_gguf_header_budget(file.as_file_mut())
                .unwrap_err()
                .to_string();
            assert!(error.contains("general.alignment"), "{error}");
            assert_eq!(file.stream_position().unwrap(), 7);
        }
    }

    #[test]
    fn rejects_names_that_lossy_utf8_decoding_would_merge() {
        for tensor_names in [false, true] {
            let names: &[&[u8]] = &[b"name\xff", b"name\xfe"];
            let mut file = if tensor_names {
                open_fixture(&[], names)
            } else {
                open_fixture(names, &[])
            };
            file.seek(SeekFrom::Start(7)).unwrap();
            let error = validate_gguf_header_budget(file.as_file_mut())
                .unwrap_err()
                .to_string();
            assert!(error.contains("UTF-8"), "{error}");
            assert_eq!(file.stream_position().unwrap(), 7);
        }
    }

    #[test]
    fn validates_utf8_across_read_chunks_without_changing_duplicate_detection() {
        for prefix_len in 4093..=4096 {
            let mut name = vec![b'k'; prefix_len];
            name.extend_from_slice("🦀".as_bytes());
            let mut file = open_fixture(&[&name], &["valid-�".as_bytes()]);
            validate_gguf_header_budget(file.as_file_mut()).unwrap();

            let mut duplicate = name.clone();
            duplicate.push(0);
            let mut file = open_fixture(&[&name, &duplicate], &[]);
            let error = validate_gguf_header_budget(file.as_file_mut())
                .unwrap_err()
                .to_string();
            assert!(error.contains("duplicate GGUF metadata key"), "{error}");

            name.pop();
            let mut file = open_fixture(&[&name], &[]);
            let error = validate_gguf_header_budget(file.as_file_mut())
                .unwrap_err()
                .to_string();
            assert!(error.contains("UTF-8"), "{error}");
        }
    }

    #[test]
    fn rejects_long_duplicate_keys_across_read_chunks_and_trailing_nuls() {
        let long = vec![b'k'; 4097];
        let mut same_after_normalization = long.clone();
        same_after_normalization.extend_from_slice(&[0, 0]);
        let keys = [long.as_slice(), same_after_normalization.as_slice()];
        let mut file = open_fixture(&keys, &[]);
        let error = validate_gguf_header_budget(file.as_file_mut())
            .unwrap_err()
            .to_string();
        assert!(error.contains("duplicate GGUF metadata key"), "{error}");
    }

    #[test]
    fn hash_collisions_are_verified_against_file_bytes() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"firstother").unwrap();
        let mut index = DuplicateStringIndex::default();
        let mut first_small = [0u8; 64];
        first_small[..5].copy_from_slice(b"first");
        let first = StringScan {
            start: 0,
            len: 5,
            hash: 17,
            small: first_small,
            small_len: 5,
        };
        let mut other_small = [0u8; 64];
        other_small[..5].copy_from_slice(b"other");
        let other = StringScan {
            start: 5,
            len: 5,
            hash: 17,
            small: other_small,
            small_len: 5,
        };
        file.seek(SeekFrom::Start(3)).unwrap();
        let file_size = file.as_file().metadata().unwrap().len();
        index
            .record(file.as_file_mut(), file_size, first, "tensor name")
            .unwrap();
        index
            .record(file.as_file_mut(), file_size, other, "tensor name")
            .unwrap();
        assert_eq!(file.stream_position().unwrap(), 3);

        let file_size = file.as_file().metadata().unwrap().len();
        let error = index
            .record(file.as_file_mut(), file_size, first, "tensor name")
            .unwrap_err()
            .to_string();
        assert!(error.contains("duplicate GGUF tensor name"), "{error}");
        assert_eq!(file.stream_position().unwrap(), 3);
    }
    #[test]
    fn exact_disk_only_engram_exception_accepts_both_pinned_tables() {
        use crate::deepseek4_gguf::fixtures::{maya_metadata, sparse};
        let md = maya_metadata(false);
        for (name, rows) in [
            ("blk.1.engram_embd.weight", 384006168),
            ("blk.14.engram_embd.weight", 384016682),
        ] {
            let (mut file, _) = sparse(&md, &[(name, &[rows, 264], 24, 0)]);
            file.seek(SeekFrom::Start(7)).unwrap();
            validate_gguf_header_budget(file.as_file_mut()).unwrap();
            assert_eq!(file.stream_position().unwrap(), 7);
        }
    }

    #[test]
    fn coder_disk_only_iq4_exception_is_exact_and_never_allows_whole_load() {
        use crate::deepseek4_gguf::{RawGgufContent, fixtures::sparse};
        use candle_core::quantized::gguf_file::Value;
        let md = std::collections::HashMap::from([
            ("split.no".into(), Value::U16(5)),
            ("split.count".into(), Value::U16(6)),
            ("split.tensors.count".into(), Value::U32(1043)),
        ]);
        let name = "per_layer_token_embd.weight";
        let (mut file, _) = sparse(&md, &[(name, &[768022850, 256], 20, 0)]);
        file.as_file()
            .set_len_sparse(crate::gsq_coder::PARTS[5].0)
            .unwrap();
        validate_gguf_header_budget(file.as_file_mut()).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        let content = RawGgufContent::read(file.as_file_mut()).unwrap();
        assert!(
            content
                .load_tensor(file.as_file_mut(), name, &candle_core::Device::Cpu)
                .is_err()
        );

        for (key, value) in [
            ("split.no", 4),
            ("split.count", 7),
            ("split.tensors.count", 1044),
        ] {
            let mut changed = md.clone();
            changed.insert(key.into(), Value::U32(value));
            let (mut file, _) = sparse(&changed, &[(name, &[768022850, 256], 20, 0)]);
            file.as_file()
                .set_len_sparse(crate::gsq_coder::PARTS[5].0)
                .unwrap();
            assert!(
                validate_gguf_header_budget(file.as_file_mut()).is_err(),
                "accepted wrong {key}"
            );
        }
        for (name, shape, dtype) in [
            ("other.weight", [768022850, 256], 20),
            (name, [768022851, 256], 20),
            (name, [768022850, 256], 0),
        ] {
            let (mut file, _) = sparse(&md, &[(name, &shape, dtype, 0)]);
            file.as_file()
                .set_len_sparse(crate::gsq_coder::PARTS[5].0)
                .unwrap();
            assert!(validate_gguf_header_budget(file.as_file_mut()).is_err());
        }
        let (mut file, _) = sparse(&md, &[(name, &[768022850, 256], 20, 0)]);
        file.as_file()
            .set_len_sparse(crate::gsq_coder::PARTS[5].0 + 1)
            .unwrap();
        assert!(validate_gguf_header_budget(file.as_file_mut()).is_err());
    }

    #[test]
    fn oversized_generic_tensors_keep_element_and_byte_caps() {
        use crate::deepseek4_gguf::fixtures::{maya_metadata, sparse};
        let md = maya_metadata(false);
        for (shape, dtype, message) in [
            (vec![1usize << 37], 24, "element count"),
            (vec![1usize << 35], 0, "byte length"),
            (vec![384006168, 264], 24, "element count"),
        ] {
            let (mut file, _) = sparse(&md, &[("not_engram.weight", &shape, dtype, 0)]);
            let error = validate_gguf_header_budget(file.as_file_mut())
                .unwrap_err()
                .to_string();
            assert!(error.contains(message), "{error}");
        }
    }

    #[test]
    fn disk_only_exception_rejects_wrong_architecture_encoding_dtype_shape_name() {
        use crate::deepseek4_gguf::fixtures::{maya_metadata, sparse};
        use candle_core::quantized::gguf_file::Value;
        for (key, value) in [
            ("general.architecture", Value::String("deepseek4".into())),
            ("deepseek41.engram.encoding", Value::String("other".into())),
            ("deepseek41.engram.encoding", Value::U32(1)),
        ] {
            let mut md = maya_metadata(false);
            md.insert(key.into(), value);
            let (mut file, _) = sparse(
                &md,
                &[("blk.1.engram_embd.weight", &[384006168, 264], 24, 0)],
            );
            assert!(validate_gguf_header_budget(file.as_file_mut()).is_err());
        }
        for (name, shape, dtype) in [
            ("blk.1.engram_embd.weight", vec![384006168, 264], 0),
            ("blk.1.engram_embd.weight", vec![384006168, 263], 24),
            ("blk.1.engram_embd.weight", vec![384006169, 264], 24),
            ("blk.1.engram_embd.weight", vec![384006168, 1, 264], 24),
            ("blk.14.engram_embd.weight", vec![384006168, 264], 24),
            ("blk.2.engram_embd.weight", vec![384006168, 264], 24),
        ] {
            let (mut file, _) = sparse(&maya_metadata(false), &[(name, &shape, dtype, 0)]);
            assert!(
                validate_gguf_header_budget(file.as_file_mut()).is_err(),
                "{name} {shape:?} {dtype}"
            );
        }
    }

    #[test]
    fn disk_only_exception_preserves_offsets_overlap_duplicates_and_file_bounds() {
        use crate::deepseek4_gguf::fixtures::{maya_metadata, sparse};
        let md = maya_metadata(false);
        let name = "blk.1.engram_embd.weight";
        for offset in [1, u64::MAX] {
            let (mut file, _) = sparse(&md, &[(name, &[384006168, 264], 24, offset)]);
            assert!(validate_gguf_header_budget(file.as_file_mut()).is_err());
        }
        let (mut overlap, _) = sparse(
            &md,
            &[
                (name, &[384006168, 264], 24, 0),
                ("blk.14.engram_embd.weight", &[384016682, 264], 24, 32),
            ],
        );
        assert!(
            validate_gguf_header_budget(overlap.as_file_mut())
                .unwrap_err()
                .to_string()
                .contains("overlapping")
        );
        let (mut duplicate, _) = sparse(
            &md,
            &[
                (name, &[384006168, 264], 24, 0),
                (name, &[384006168, 264], 24, 0),
            ],
        );
        assert!(
            validate_gguf_header_budget(duplicate.as_file_mut())
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
        let (mut truncated, data) = sparse(&md, &[(name, &[384006168, 264], 24, 0)]);
        truncated
            .as_file()
            .set_len_sparse(data + 384006168 * 264 - 1)
            .unwrap();
        assert!(
            validate_gguf_header_budget(truncated.as_file_mut())
                .unwrap_err()
                .to_string()
                .contains("end of the file")
        );
        truncated.as_file().set_len(data - 64).unwrap();
        assert!(validate_gguf_header_budget(truncated.as_file_mut()).is_err());
    }
}
