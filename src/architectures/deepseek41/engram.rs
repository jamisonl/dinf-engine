//! Native Maya-XS Engram hashing and bounded, uncached GGUF row reads.
//!
//! Ported from ds4.c at 0dff39f91721bcf3472813c6d2bebf9353df2a1a;
//! see NOTICE and LICENSE-MIT. This is not a model runner. The integrator supplies
//! parsed metadata and trusted artifact paths, suppresses Engram at masked image
//! positions, and commits `HashBatch::next_history` only after the whole model
//! step succeeds. Chunk long prefixes at `MAX_BATCH_TOKENS` boundaries.
//!
//! Integration order: validate `LayoutMetadata` once with `Layout::new`; resolve
//! GGUF tensor offsets to absolute artifact offsets and validate each payload
//! with `Layout::table_span`; construct one `Table::from_file` per layer.
//! The verified loader must expose `clone_file() -> Result<File>` (try_clone on
//! its owned descriptor, never reopen a path) and
//! `file_identity() -> FileIdentity`, captured from that descriptor BEFORE
//! full SHA verification or authorized local receipt validation and checked
//! again afterward. Pass that same identity with the cloned descriptor.
//! Timestamp/sample checks detect ordinary mutation, not a hostile writer
//! able to restore metadata. Artifacts must remain immutable during inference.
//! Per request, start with `History::default`, hash a bounded token chunk, select
//! that layer's 24 IDs from each token, and call `Table::read_rows`. Consume rows in
//! token/column order. Only then, after all other model work succeeds, replace
//! the request history with the candidate. Disk/decode errors discard the
//! candidate and all temporary output. No fallback inference backend is used.

#[cfg(test)]
use crate::verified_file::SparseSetLen;
use std::fs::File;
use std::io;
use std::path::Path;

use anyhow::{Context, Result, ensure};

/// Supported GGUF architecture.
pub const ARCHITECTURE: &str = "deepseek41";
/// The only supported table encoding.
pub const ENCODING: &str = "e4m3_e8m0_32_row264";
/// Transformer layers containing Engram, in hash-output order.
pub const LAYER_IDS: [u32; 2] = [1, 14];
/// Pinned Maya-XS table row counts.
pub const TABLE_ROWS: [u32; 2] = [384_006_168, 384_016_682];
/// Uncompressed tokenizer vocabulary size.
pub const VOCAB_SIZE: u32 = 129_280;
/// Compressed Engram vocabulary size.
pub const COMPRESSED_VOCAB_SIZE: u32 = 99_092;
/// Heads for each of the 2-, 3-, and 4-grams, per layer.
pub const COLS: usize = 24;
/// Decoded elements per row.
pub const DIM: usize = 256;
/// Encoded row bytes: 256 E4M3 values, then eight E8M0 scales.
pub const ROW_BYTES: usize = 264;
/// History sentinel; a masked image position is a dead barrier.
pub const DEAD: i32 = -1;
/// Maximum tokens per hash operation; callers chunk larger prefixes.
pub const MAX_BATCH_TOKENS: usize = 256;
/// Maximum rows per disk operation (6 MiB decoded), independent of table size.
pub const MAX_READ_ROWS: usize = MAX_BATCH_TOKENS * COLS;

/// Internal candidate only: at most 512 tokens of 24 flattened rows. The scalar
/// reference retains its independent `MAX_READ_ROWS` limit.
pub const MAX_BATCHED_READ_ROWS: usize = 512 * COLS;
/// One reusable encoded I/O buffer; spans are not rounded to page boundaries.
pub const BATCHED_READ_BUFFER_BYTES: usize = 16 * 1024;
/// Merge across at most one unrequested row, never a large random gap.
pub const BATCHED_READ_MAX_GAP_BYTES: usize = ROW_BYTES;
/// Requested allocation payload at the row cap: decoded output, (row, caller
/// index) sort plan, and encoded buffer. Unstable sorting uses no heap scratch.
/// Excludes allocator bookkeeping, Vec headers, and fixed-size stack locals.
pub const MAX_BATCHED_READ_HEAP_BYTES: usize = MAX_BATCHED_READ_ROWS
    * (std::mem::size_of::<[f32; DIM]>() + std::mem::size_of::<(u32, u32)>())
    + BATCHED_READ_BUFFER_BYTES;

/// A parsed GGUF table descriptor; offsets are supplied separately as spans.
#[derive(Clone, Debug)]
pub struct TableGeometry {
    /// Exact GGUF tensor name, not a filesystem path.
    pub name: String,
    /// GGUF dtype ID; Engram requires I8 (24).
    pub dtype: u32,
    /// GGUF dimensions, in file order: [264, rows].
    pub dimensions: [u64; 2],
}

/// Untrusted parsed metadata, consumed by `Layout::new`.
#[derive(Clone, Debug)]
pub struct LayoutMetadata {
    /// `general.architecture`.
    pub architecture: String,
    /// `deepseek41.engram.encoding`.
    pub encoding: String,
    /// Uncompressed vocabulary size.
    pub vocab_size: u32,
    /// `deepseek41.engram.compressed_vocab_size`.
    pub compressed_vocab_size: u32,
    /// `deepseek41.engram.pad_id`, already in compressed space.
    pub pad_id: u32,
    /// `deepseek41.engram.layer_ids`.
    pub layer_ids: [u32; 2],
    /// `deepseek41.engram.rows`.
    pub rows: [u64; 2],
    /// `deepseek41.engram.token_map`.
    pub token_map: Vec<u32>,
    /// `deepseek41.engram.primes`, reshaped [layer][column].
    pub primes: [[u32; COLS]; 2],
    /// `deepseek41.engram.multipliers`, reshaped [layer][ngram position].
    pub multipliers: [[u64; 4]; 2],
    /// Descriptors for blk.1 and blk.14, in that order.
    pub tables: [TableGeometry; 2],
}

/// Validated immutable layout, shareable across requests without request state.
#[derive(Debug)]
pub struct Layout {
    token_map: Vec<u32>,
    pad_id: u32,
    rows: [u32; 2],
    primes: [[u32; COLS]; 2],
    multipliers: [[u64; 4]; 2],
    row_bytes: usize,
}

impl Layout {
    /// Validate the exact pinned geometry, every map entry, and hash arithmetic.
    /// Raw GGUF array lengths must be checked before reshaping into fixed arrays.
    pub fn new(metadata: LayoutMetadata) -> Result<Self> {
        Self::validate(metadata, TABLE_ROWS)
    }

    // Tests can reduce table rows, never vocabulary, encoding, or tensor width.
    fn validate(m: LayoutMetadata, expected_rows: [u32; 2]) -> Result<Self> {
        ensure!(
            m.architecture == ARCHITECTURE,
            "unsupported Engram architecture"
        );
        let iq4 = m.encoding == "iq4_nl";
        ensure!(m.encoding == ENCODING || iq4, "unsupported Engram encoding");
        ensure!(
            m.vocab_size == VOCAB_SIZE && m.compressed_vocab_size == COMPRESSED_VOCAB_SIZE,
            "unsupported Engram vocabulary geometry"
        );
        ensure!(
            m.layer_ids == LAYER_IDS,
            "unsupported Engram layer IDs/order"
        );
        ensure!(
            m.token_map.len() == VOCAB_SIZE as usize,
            "invalid Engram token map length"
        );
        ensure!(
            m.pad_id < COMPRESSED_VOCAB_SIZE,
            "invalid compressed pad ID"
        );
        ensure!(
            m.token_map.iter().all(|&id| id < COMPRESSED_VOCAB_SIZE),
            "token map entry outside compressed vocabulary"
        );
        for layer in 0..2 {
            ensure!(
                m.rows[layer] == u64::from(expected_rows[layer]),
                "unsupported Engram row count"
            );
            let table = &m.tables[layer];
            if iq4 {
                ensure!(
                    table.name == "per_layer_token_embd.weight"
                        && table.dtype == 20
                        && table.dimensions == [DIM as u64, m.rows[layer]],
                    "unsupported IQ4_NL Engram geometry"
                );
            } else {
                ensure!(
                    table.name == format!("blk.{}.engram_embd.weight", LAYER_IDS[layer]),
                    "unexpected Engram table name/order"
                );
                ensure!(
                    table.dtype == 24 && table.dimensions == [ROW_BYTES as u64, m.rows[layer]],
                    "unsupported Engram table geometry/dtype"
                );
            }
            for &multiplier in &m.multipliers[layer] {
                ensure!(
                    multiplier & 1 == 1
                        && multiplier <= i64::MAX as u64 / u64::from(COMPRESSED_VOCAB_SIZE),
                    "Engram multiplier must be odd and nonoverflowing"
                );
            }
            let mut sum = 0u32;
            for &prime in &m.primes[layer] {
                ensure!(is_prime(prime), "invalid Engram prime modulus");
                sum = sum
                    .checked_add(prime)
                    .context("Engram prime sum overflow")?;
            }
            ensure!(
                sum == expected_rows[layer],
                "Engram primes do not sum to table rows"
            );
        }
        Ok(Self {
            token_map: m.token_map,
            pad_id: m.pad_id,
            rows: expected_rows,
            primes: m.primes,
            multipliers: m.multipliers,
            row_bytes: if iq4 { 144 } else { ROW_BYTES },
        })
    }

    /// Hash without mutating request state. Nonzero mask entries mean text;
    /// zero means a DEAD image barrier. Even masked token IDs must be valid.
    /// Masked positions still have row IDs; the model must suppress their output.
    pub fn hash(
        &self,
        history: &History,
        tokens: &[i32],
        mask: Option<&[u8]>,
    ) -> Result<HashBatch> {
        ensure!(
            tokens.len() <= MAX_BATCH_TOKENS,
            "Engram hash batch exceeds token limit"
        );
        ensure!(
            mask.is_none_or(|m| m.len() == tokens.len()),
            "Engram mask length mismatch"
        );
        ensure!(
            history
                .tail
                .iter()
                .all(|&id| id == DEAD || (0..COMPRESSED_VOCAB_SIZE as i32).contains(&id)),
            "invalid compressed Engram history"
        );
        ensure!(
            tokens
                .iter()
                .all(|&id| (0..VOCAB_SIZE as i32).contains(&id)),
            "invalid Engram token ID"
        );
        let mut rows = Vec::new();
        rows.try_reserve_exact(tokens.len())?;
        let mut next = history.clone();
        for (pos, &token) in tokens.iter().enumerate() {
            let current = if mask.is_some_and(|m| m[pos] == 0) {
                DEAD
            } else {
                self.token_map[token as usize] as i32
            };
            let mut ids = [0u32; 4];
            let mut blocked = false;
            for (j, id) in std::iter::once(current).chain(next.tail).enumerate() {
                blocked |= id == DEAD;
                ids[j] = if blocked { self.pad_id } else { id as u32 };
            }
            let mut token_rows = [[0u32; COLS]; 2];
            for (layer, output) in token_rows.iter_mut().enumerate() {
                let mut hash = u64::from(ids[0]) * self.multipliers[layer][0];
                let mut offset = 0u32;
                for (j, &id) in ids.iter().enumerate().skip(1) {
                    hash ^= u64::from(id) * self.multipliers[layer][j];
                    for head in 0..8 {
                        let col = (j - 1) * 8 + head;
                        let prime = self.primes[layer][col];
                        output[col] = (hash % u64::from(prime)) as u32 + offset;
                        offset += prime; // Sum and every product validated at load.
                    }
                }
            }
            rows.push(token_rows);
            next.tail = [current, next.tail[0], next.tail[1]];
        }
        Ok(HashBatch {
            rows,
            next_history: next,
        })
    }

    /// Validate a caller-supplied absolute artifact span for one supported layer.
    /// `byte_len` is the tensor payload length, excluding alignment padding.
    pub fn table_span(&self, layer_id: u32, offset: u64, byte_len: u64) -> Result<TableSpan> {
        let layer = LAYER_IDS
            .iter()
            .position(|&id| id == layer_id)
            .context("unsupported Engram layer")?;
        let rows = self.rows[layer];
        let expected = u64::from(rows)
            .checked_mul(self.row_bytes as u64)
            .context("Engram span overflow")?;
        ensure!(byte_len == expected, "Engram span length mismatch");
        let end = offset
            .checked_add(byte_len)
            .context("Engram span end overflow")?;
        ensure!(
            end <= i64::MAX as u64,
            "Engram span exceeds signed file offsets"
        );
        Ok(TableSpan {
            offset,
            end,
            rows,
            row_bytes: self.row_bytes,
        })
    }
}

fn is_prime(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    let mut d = 3;
    while d <= n / d {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// Request-owned rolling compressed IDs, newest first. Clone to snapshot/branch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct History {
    tail: [i32; 3],
}

impl Default for History {
    fn default() -> Self {
        Self { tail: [DEAD; 3] }
    }
}

impl History {
    /// Restore a checkpoint; rejects invalid compressed IDs atomically.
    pub fn from_compressed(tail: [i32; 3]) -> Result<Self> {
        ensure!(
            tail.iter()
                .all(|&id| id == DEAD || (0..COMPRESSED_VOCAB_SIZE as i32).contains(&id)),
            "invalid compressed Engram history"
        );
        Ok(Self { tail })
    }
    /// Snapshot the last three compressed IDs, newest first.
    pub fn compressed_tail(&self) -> [i32; 3] {
        self.tail
    }
    /// Start a new request; no process-global history exists.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Successful speculative hash output. Discard this value to roll back.
#[derive(Debug, PartialEq, Eq)]
pub struct HashBatch {
    /// Row IDs in `[token][layer][column]` order.
    pub rows: Vec<[[u32; COLS]; 2]>,
    /// Commit by replacing the request history only after the model step succeeds.
    pub next_history: History,
}

/// Validated payload bounds; construct through `Layout::table_span`.
#[derive(Clone, Copy, Debug)]
pub struct TableSpan {
    offset: u64,
    end: u64,
    rows: u32,
    row_bytes: usize,
}

/// File identity captured from the owned descriptor, never from metadata paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileIdentity {
    /// Full artifact byte length at open.
    pub length: u64,
    /// Device containing the artifact.
    #[cfg(unix)]
    pub device: u64,
    /// Inode of the open artifact (path replacement cannot redirect reads).
    #[cfg(unix)]
    pub inode: u64,
    /// Modification time, seconds and nanoseconds since the Unix epoch.
    #[cfg(unix)]
    pub modified: (i64, i64),
    /// Status-change time, seconds and nanoseconds since the Unix epoch.
    #[cfg(unix)]
    pub changed: (i64, i64),
    /// Descriptor modification timestamp (including Windows precision).
    #[cfg(not(unix))]
    pub modified: std::time::SystemTime,
}

impl FileIdentity {
    /// Capture identity from descriptor metadata, never path metadata. The
    /// verified loader captures this before hashing and compares it afterward.
    pub fn from_metadata(metadata: &std::fs::Metadata) -> Result<Self> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        ensure!(metadata.is_file(), "Engram artifact must be a regular file");
        Ok(Self {
            length: metadata.len(),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            modified: (metadata.mtime(), metadata.mtime_nsec()),
            #[cfg(unix)]
            changed: (metadata.ctime(), metadata.ctime_nsec()),
            #[cfg(not(unix))]
            modified: metadata.modified()?,
        })
    }
}

/// Owned read-only descriptor and validated span; never maps or caches table data.
#[derive(Debug)]
pub struct Table {
    file: File,
    span: TableSpan,
    identity: FileIdentity,
}

impl Table {
    /// Open a caller-selected artifact, not a path taken from model metadata.
    /// Standalone convenience only; production uses the verified descriptor.
    pub fn open(artifact_path: impl AsRef<Path>, span: TableSpan) -> Result<Self> {
        let file = File::open(artifact_path.as_ref()).context("opening Engram artifact")?;
        let expected = FileIdentity::from_metadata(&file.metadata()?)?;
        Self::from_file(file, expected, span)
    }

    /// Adopt an already-owned descriptor and the identity captured by the
    /// verified loader, rejecting substitution or changes since verification.
    /// No path is opened. Each bounded batch rechecks identity before/after I/O.
    pub fn from_file(file: File, expected: FileIdentity, span: TableSpan) -> Result<Self> {
        ensure!(
            FileIdentity::from_metadata(&file.metadata()?)? == expected,
            "Engram artifact identity changed since verification"
        );
        ensure!(
            span.end <= expected.length,
            "Engram span exceeds artifact length"
        );
        disable_cache(&file)?;
        let table = Self {
            file,
            span,
            identity: expected,
        };
        table.check_identity()?;
        Ok(table)
    }

    fn check_identity(&self) -> Result<()> {
        ensure!(
            FileIdentity::from_metadata(&self.file.metadata()?)? == self.identity,
            "Engram artifact identity/length/timestamps changed"
        );
        Ok(())
    }

    /// Identity and full byte length captured at open.
    pub fn file_identity(&self) -> FileIdentity {
        self.identity
    }

    /// Positional reads of exactly 264 bytes per requested row. Duplicates are
    /// valid and retain caller order. The only allocation is bounded decoded
    /// output, returned only on complete success; no partial output escapes.
    pub fn read_rows(&self, rows: &[u32]) -> Result<Vec<[f32; DIM]>> {
        self.read_rows_with(rows, read_exact_at)
    }

    /// Bounded, opt-in candidate; production callers stay on `read_rows` until
    /// native quality/performance qualification authorizes integration.
    ///
    /// Preserves flat caller order, duplicates, and decoded bits. A sorted
    /// (row, caller-index) plan avoids a retained row cache. Each unique row is
    /// read/decoded once, then scattered directly into the sole output Vec.
    /// Exact row endpoints, not page alignment, handle 264-byte rows crossing
    /// 4096-byte pages and non-aligned table endpoints without overreading.
    ///
    /// For N caller rows and U unique rows: N <= MAX_BATCHED_READ_ROWS, at most
    /// U logical exact-read calls, and at most (2*U - 1)*ROW_BYTES requested
    /// bytes when U > 0 (zero for empty input). Each call is at most 16 KiB.
    /// These are logical I/O counts, not syscall/device-byte bounds: short
    /// reads and EINTR may require retries inside `read_exact_at`.
    ///
    /// Requested heap payload is N*(1024 + 8) + 16384 bytes on supported
    /// targets. Decoded locals/results are fixed 1024-byte arrays; stack/ABI
    /// and allocator overhead are separate. Empty input allocates nothing but still checks
    /// identity pre/post. All allocation requests are fallible. No partial
    /// output is returned after an I/O, decoding, or identity error.
    ///
    /// Existing host diagnostics expose EngramRead calls/requested/completed
    /// bytes, EngramDecode unique-row calls, and one metadata pre/post pair.
    pub fn read_rows_batched(&self, rows: &[u32]) -> Result<Vec<[f32; DIM]>> {
        if self.span.row_bytes == 144 {
            return self.read_rows(rows);
        }
        self.read_rows_batched_with(rows, read_exact_at)
    }

    fn read_rows_batched_with(
        &self,
        rows: &[u32],
        mut read: impl FnMut(&File, &mut [u8], u64) -> io::Result<()>,
    ) -> Result<Vec<[f32; DIM]>> {
        use super::profile::{self, Detail};
        ensure!(
            rows.len() <= MAX_BATCHED_READ_ROWS,
            "Engram batched read exceeds row limit"
        );
        ensure!(
            rows.iter().all(|&row| row < self.span.rows),
            "Engram row ID out of bounds"
        );
        let metadata = profile::detail(Detail::EngramMetadataPre);
        self.check_identity()?;
        drop(metadata);

        let mut output = Vec::new();
        let mut plan = Vec::new();
        let mut raw = Vec::new();
        if !rows.is_empty() {
            let output_bytes = rows
                .len()
                .checked_mul(std::mem::size_of::<[f32; DIM]>())
                .context("Engram batched output size overflow")?;
            let plan_bytes = rows
                .len()
                .checked_mul(std::mem::size_of::<(u32, u32)>())
                .context("Engram batched plan size overflow")?;
            let heap_bytes = output_bytes
                .checked_add(plan_bytes)
                .and_then(|n| n.checked_add(BATCHED_READ_BUFFER_BYTES))
                .context("Engram batched allocation size overflow")?;
            ensure!(
                heap_bytes <= MAX_BATCHED_READ_HEAP_BYTES,
                "Engram batched allocation exceeds limit"
            );
            output.try_reserve_exact(rows.len())?;
            output.resize(rows.len(), [0.0; DIM]);
            plan.try_reserve_exact(rows.len())?;
            for (index, &row) in rows.iter().enumerate() {
                plan.push((
                    row,
                    u32::try_from(index).context("Engram caller index overflow")?,
                ));
            }
            plan.sort_unstable();
            raw.try_reserve_exact(BATCHED_READ_BUFFER_BYTES)?;
            raw.resize(BATCHED_READ_BUFFER_BYTES, 0);
        }

        let mut cursor = 0;
        while cursor < plan.len() {
            let first_row = plan[cursor].0;
            let offset = self.batched_row_offset(first_row)?;
            let mut end = offset
                .checked_add(ROW_BYTES as u64)
                .context("Engram row end overflow")?;
            let mut last_row = first_row;
            let mut stop = cursor + 1;
            while stop < plan.len() {
                let row = plan[stop].0;
                if row != last_row {
                    let next = self.batched_row_offset(row)?;
                    let next_end = next
                        .checked_add(ROW_BYTES as u64)
                        .context("Engram row end overflow")?;
                    let gap = next
                        .checked_sub(end)
                        .context("Engram batched row order overflow")?;
                    let bytes = next_end
                        .checked_sub(offset)
                        .context("Engram batched span overflow")?;
                    if gap > BATCHED_READ_MAX_GAP_BYTES as u64
                        || bytes > BATCHED_READ_BUFFER_BYTES as u64
                    {
                        break;
                    }
                    end = next_end;
                    last_row = row;
                }
                stop += 1;
            }
            let bytes = usize::try_from(
                end.checked_sub(offset)
                    .context("Engram batched span overflow")?,
            )
            .context("Engram batched span exceeds host size")?;
            let reading = profile::read_detail(Detail::EngramRead, bytes);
            read(&self.file, &mut raw[..bytes], offset)?;
            profile::read_completed(&reading, bytes);
            drop(reading);

            let mut index = cursor;
            while index < stop {
                let row = plan[index].0;
                let relative = usize::try_from(
                    row.checked_sub(first_row)
                        .context("Engram batched relative row overflow")?,
                )
                .context("Engram batched row exceeds host size")?
                .checked_mul(ROW_BYTES)
                .context("Engram batched relative offset overflow")?;
                let row_end = relative
                    .checked_add(ROW_BYTES)
                    .context("Engram batched row end overflow")?;
                let encoded: &[u8; ROW_BYTES] = raw[..bytes]
                    .get(relative..row_end)
                    .context("Engram batched row exceeds read span")?
                    .try_into()?;
                let decoding = profile::detail(Detail::EngramDecode);
                let decoded = decode_row(encoded)?;
                drop(decoding);
                while index < stop && plan[index].0 == row {
                    output[plan[index].1 as usize] = decoded;
                    index += 1;
                }
            }
            cursor = stop;
        }

        let metadata = profile::detail(Detail::EngramMetadataPost);
        self.check_identity()?;
        drop(metadata);
        Ok(output)
    }

    fn batched_row_offset(&self, row: u32) -> Result<u64> {
        ensure!(row < self.span.rows, "Engram row ID out of bounds");
        let offset = u64::from(row)
            .checked_mul(ROW_BYTES as u64)
            .and_then(|n| self.span.offset.checked_add(n))
            .context("Engram row offset overflow")?;
        let end = offset
            .checked_add(ROW_BYTES as u64)
            .context("Engram row end overflow")?;
        ensure!(
            end <= self.span.end && end <= self.identity.length && end <= i64::MAX as u64,
            "Engram row exceeds file/span bounds"
        );
        Ok(offset)
    }

    fn read_rows_with(
        &self,
        rows: &[u32],
        mut read: impl FnMut(&File, &mut [u8], u64) -> io::Result<()>,
    ) -> Result<Vec<[f32; DIM]>> {
        use super::profile::{self, Detail};
        ensure!(rows.len() <= MAX_READ_ROWS, "Engram read exceeds row limit");
        ensure!(
            rows.iter().all(|&row| row < self.span.rows),
            "Engram row ID out of bounds"
        );
        let metadata = profile::detail(Detail::EngramMetadataPre);
        self.check_identity()?;
        drop(metadata);
        let mut output = Vec::new();
        output.try_reserve_exact(rows.len())?;
        let mut raw = [0u8; ROW_BYTES];
        for &row in rows {
            let offset = u64::from(row)
                .checked_mul(self.span.row_bytes as u64)
                .and_then(|n| self.span.offset.checked_add(n))
                .context("Engram row offset overflow")?;
            let end = offset
                .checked_add(self.span.row_bytes as u64)
                .context("Engram row end overflow")?;
            ensure!(
                end <= self.span.end && end <= self.identity.length && end <= i64::MAX as u64,
                "Engram row exceeds file/span bounds"
            );
            let reading = profile::read_detail(Detail::EngramRead, self.span.row_bytes);
            read(&self.file, &mut raw[..self.span.row_bytes], offset)?;
            profile::read_completed(&reading, self.span.row_bytes);
            drop(reading);
            let decoding = profile::detail(Detail::EngramDecode);
            output.push(if self.span.row_bytes == 144 {
                decode_iq4_row(&raw[..144])?
            } else {
                decode_row(&raw)?
            });
            drop(decoding);
        }
        let metadata = profile::detail(Detail::EngramMetadataPost);
        self.check_identity()?;
        drop(metadata);
        Ok(output)
    }
}

/// Standard GGML IQ4_NL: eight 18-byte blocks, low nibbles then high nibbles.
pub fn decode_iq4_row(raw: &[u8]) -> Result<[f32; DIM]> {
    ensure!(raw.len() == 144, "IQ4_NL Engram row length mismatch");
    const VALUES: [i8; 16] = [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ];
    let mut out = [0.0; DIM];
    for (block, bytes) in raw.chunks_exact(18).enumerate() {
        let scale = half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
        ensure!(scale.is_finite(), "nonfinite IQ4_NL Engram scale");
        for i in 0..16 {
            out[block * 32 + i] = scale * f32::from(VALUES[(bytes[2 + i] & 15) as usize]);
            out[block * 32 + i + 16] = scale * f32::from(VALUES[(bytes[2 + i] >> 4) as usize]);
        }
    }
    Ok(out)
}

#[cfg(test)]
#[test]
fn coder_iq4_nl_nonzero_row_golden_and_invalid_scale() -> Result<()> {
    let table = [
        -127., -104., -83., -65., -49., -35., -22., -10., 1., 13., 25., 38., 53., 69., 89., 113.,
    ];
    let mut bytes = [0u8; 144];
    let scales = [0.25f32, -0.5, 0., 0.03125, 2., -1., 0.125, 0.0625];
    for block in 0..8 {
        bytes[18 * block..18 * block + 2]
            .copy_from_slice(&half::f16::from_f32(scales[block]).to_le_bytes());
        for i in 0..16 {
            bytes[18 * block + 2 + i] = ((15 - i) << 4 | i) as u8;
        }
    }
    let actual = decode_iq4_row(&bytes)?;
    for i in 0..256 {
        let lane = i % 32;
        let code = if lane < 16 { lane } else { 31 - lane };
        assert_eq!(
            actual[i].to_bits(),
            (scales[i / 32] * table[code]).to_bits()
        );
    }
    assert!(decode_iq4_row(&bytes[..143]).is_err());
    bytes[..2].copy_from_slice(&half::f16::INFINITY.to_le_bytes());
    assert!(decode_iq4_row(&bytes).is_err());
    Ok(())
}

#[cfg(target_os = "macos")]
fn disable_cache(file: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    // Darwin fcntl's variadic ABI, restricted here to documented integer flags.
    // F_NOCACHE=48 disables page caching; F_RDAHEAD=45 disables read-ahead.
    unsafe extern "C" {
        fn fcntl(fd: std::ffi::c_int, cmd: std::ffi::c_int, ...) -> std::ffi::c_int;
    }
    // SAFETY: `file` owns a live descriptor for both calls; these commands take
    // one c_int, no pointers, and neither close nor transfer descriptor ownership.
    let result = unsafe {
        if fcntl(file.as_raw_fd(), 48, 1 as std::ffi::c_int) == -1 {
            -1
        } else {
            fcntl(file.as_raw_fd(), 45, 0 as std::ffi::c_int)
        }
    };
    if result == -1 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
fn disable_cache(_: &File) -> io::Result<()> {
    Ok(())
}

fn read_exact_at(file: &File, mut bytes: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !bytes.is_empty() {
        #[cfg(unix)]
        let result = std::os::unix::fs::FileExt::read_at(file, bytes, offset);
        #[cfg(windows)]
        let result = std::os::windows::fs::FileExt::seek_read(file, bytes, offset);
        #[cfg(not(any(unix, windows)))]
        let result = {
            use std::io::{Read, Seek, SeekFrom};
            let mut reader = file;
            reader
                .seek(SeekFrom::Start(offset))
                .and_then(|_| reader.read(bytes))
        };
        match result {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "truncated Engram row",
                ));
            }
            Ok(n) => {
                offset += n as u64;
                bytes = &mut bytes[n..];
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Decode one row to BF16-rounded values in F32 storage, preserving signed zero.
/// Scaling is an F32 multiplication by an exactly representable power of two,
/// including scale 0 (2^-127). No F16 conversion or table-sized lookup is used.
/// Invalid E4M3 NaNs, E8M0 255, and F32/BF16 overflow reject the entire row.
pub fn decode_row(raw: &[u8; ROW_BYTES]) -> Result<[f32; DIM]> {
    let mut output = [0.0; DIM];
    for group in 0..8 {
        let scale = raw[DIM + group];
        ensure!(scale != 255, "invalid Engram E8M0 scale");
        let factor = f32::from_bits(if scale == 0 {
            0x0040_0000
        } else {
            u32::from(scale) << 23
        });
        for j in group * 32..(group + 1) * 32 {
            let code = raw[j];
            ensure!(code & 127 != 127, "invalid Engram E4M3 code");
            let exponent = (code >> 3) & 15;
            let mantissa = u32::from(code & 7);
            let magnitude = if exponent == 0 {
                mantissa as f32 * (1.0 / 512.0)
            } else {
                f32::from_bits((u32::from(exponent) + 120) << 23 | mantissa << 20)
            };
            let signed = if code & 128 != 0 {
                -magnitude
            } else {
                magnitude
            };
            let value = signed * factor;
            ensure!(value.is_finite(), "Engram F32 scaling overflow");
            let bits = value.to_bits();
            let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
            let value = f32::from_bits(rounded);
            ensure!(value.is_finite(), "Engram BF16 rounding overflow");
            output[j] = value;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};

    const PINNED_PRIMES: [[u32; COLS]; 2] = [
        [
            16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183,
            16000189, 16000207, 16000211, 16000253, 16000277, 16000289, 16000307, 16000321,
            16000339, 16000381, 16000393, 16000399, 16000403, 16000409, 16000447, 16000463,
        ],
        [
            16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627,
            16000667, 16000669, 16000693, 16000697, 16000711, 16000729, 16000759, 16000769,
            16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889,
        ],
    ];

    fn metadata(primes: [[u32; COLS]; 2]) -> LayoutMetadata {
        let rows = primes.map(|p| p.iter().map(|&x| u64::from(x)).sum());
        LayoutMetadata {
            architecture: ARCHITECTURE.into(),
            encoding: ENCODING.into(),
            vocab_size: VOCAB_SIZE,
            compressed_vocab_size: COMPRESSED_VOCAB_SIZE,
            pad_id: 2,
            layer_ids: LAYER_IDS,
            rows,
            token_map: (0..VOCAB_SIZE)
                .map(|id| id % COMPRESSED_VOCAB_SIZE)
                .collect(),
            primes,
            multipliers: [
                [
                    76632096046245,
                    4839876093313,
                    35959672319349,
                    73987337458391,
                ],
                [
                    67716810739261,
                    51510806800915,
                    30921347202721,
                    82619226485591,
                ],
            ],
            tables: std::array::from_fn(|i| TableGeometry {
                name: format!("blk.{}.engram_embd.weight", LAYER_IDS[i]),
                dtype: 24,
                dimensions: [ROW_BYTES as u64, rows[i]],
            }),
        }
    }

    fn small_metadata() -> LayoutMetadata {
        metadata([[17; COLS]; 2])
    }
    fn small_layout() -> Layout {
        Layout::validate(small_metadata(), [408; 2]).unwrap()
    }

    // Port of upstream tests/test_engram.c's independent full-history oracle:
    // recompute each column separately in u128, without rolling state/hash reuse.
    fn upstream_reference(
        m: &LayoutMetadata,
        tokens: &[i32],
        mask: Option<&[u8]>,
    ) -> Vec<[[u32; COLS]; 2]> {
        (0..tokens.len())
            .map(|pos| {
                std::array::from_fn(|layer| {
                    std::array::from_fn(|col| {
                        let mut hash = 0u128;
                        let mut blocked = false;
                        for shift in 0..col / 8 + 2 {
                            let p = pos.checked_sub(shift);
                            blocked |= p.is_none_or(|p| mask.is_some_and(|m| m[p] == 0));
                            let id = if blocked {
                                m.pad_id
                            } else {
                                m.token_map[tokens[p.unwrap()] as usize]
                            };
                            hash ^= u128::from(id) * u128::from(m.multipliers[layer][shift]);
                        }
                        (hash % u128::from(m.primes[layer][col])) as u32
                            + m.primes[layer][..col].iter().sum::<u32>()
                    })
                })
            })
            .collect()
    }

    #[test]
    fn deepseek41_pinned_layout_and_spans() {
        let layout = Layout::new(metadata(PINNED_PRIMES)).unwrap();
        let a = layout
            .table_span(1, 162_949_808_128 + 5_832_704, 101_377_628_352)
            .unwrap();
        let b = layout
            .table_span(14, 264_327_446_528 + 5_832_704, 101_380_404_048)
            .unwrap();
        assert_eq!(a.end - a.offset + b.end - b.offset, 202_758_032_400);
        assert!(layout.table_span(0, 0, 1).is_err());
        assert!(layout.table_span(1, 0, 1).is_err());
        assert!(layout.table_span(1, u64::MAX, 101_377_628_352).is_err());
        assert!(
            layout
                .table_span(1, i64::MAX as u64, 101_377_628_352)
                .is_err()
        );
        assert!(Layout::new(small_metadata()).is_err());
    }

    #[test]
    fn deepseek41_rejects_malformed_layout() {
        let mutations: &[fn(&mut LayoutMetadata)] = &[
            |m| m.architecture = "deepseek4".into(),
            |m| m.encoding = "f16".into(),
            |m| m.vocab_size -= 1,
            |m| m.compressed_vocab_size -= 1,
            |m| m.pad_id = COMPRESSED_VOCAB_SIZE,
            |m| {
                m.token_map.pop();
            },
            |m| m.token_map[17] = COMPRESSED_VOCAB_SIZE,
            |m| m.layer_ids.swap(0, 1),
            |m| m.rows[0] -= 1,
            |m| m.tables[0].dtype = 1,
            |m| m.tables[0].dimensions[0] = 256,
            |m| m.tables[1].dimensions[1] += 1,
            |m| m.tables[0].name = "../untrusted".into(),
            |m| m.multipliers[1][3] = 0,
            |m| m.multipliers[1][3] = 2,
            |m| m.multipliers[1][3] = u64::MAX,
            |m| m.primes[0][0] = 0,
            |m| m.primes[0][0] = 1,
            |m| m.primes[0][0] = 15,
            |m| m.primes[0][0] = 19,
            |m| m.primes[0] = [4_294_967_291; COLS],
        ];
        for change in mutations {
            let mut m = small_metadata();
            change(&mut m);
            assert!(Layout::validate(m, [408; 2]).is_err());
        }
    }

    #[test]
    fn deepseek41_hash_literal_xor_golden() {
        let mut m = small_metadata();
        m.multipliers = [[1, 3, 5, 7], [11, 13, 17, 19]];
        let layout = Layout::validate(m, [408; 2]).unwrap();
        // Hand-derived ds4 XOR residues for 2/3/4 grams; offsets are 17*column.
        let residues = [
            [[2, 8, 6], [3, 3, 16]],
            [[10, 0, 14], [16, 16, 12]],
            [[9, 14, 0], [5, 14, 14]],
            [[1, 12, 16], [6, 11, 10]],
        ];
        let actual = layout
            .hash(&History::default(), &[4, 6, 8, 10], None)
            .unwrap();
        for (token, layers) in residues.iter().enumerate() {
            for (layer, grams) in layers.iter().enumerate() {
                for col in 0..COLS {
                    assert_eq!(
                        actual.rows[token][layer][col],
                        grams[col / 8] + 17 * col as u32
                    );
                }
            }
        }
        assert_eq!(actual.next_history.compressed_tail(), [10, 8, 6]);
    }

    #[test]
    fn deepseek41_upstream_hash_vectors_chunk_clone_reset_rollback() {
        let mut m = metadata([[16_000_057; COLS]; 2]);
        m.pad_id = 1;
        for (i, entry) in m.token_map.iter_mut().take(256).enumerate() {
            *entry = (i / 2) as u32;
        }
        for layer in 0..2 {
            for j in 0..4 {
                m.multipliers[layer][j] = 35_184_372_088_831 - 2 * (j + 4 * layer) as u64;
            }
        }
        let tokens: Vec<i32> = (0..513).map(|i| (i * 97 + i / 3) % 256).collect();
        let mask: Vec<u8> = (0..513)
            .map(|i| u8::from(i % 17 != 0 && !(125..=131).contains(&i)))
            .collect();
        let layout = Layout::validate(m.clone(), [384_001_368; 2]).unwrap();
        for mask in [None, Some(mask.as_slice())] {
            let expected = upstream_reference(&m, &tokens, mask);
            for chunk in [1, 2, 3, 7, 31, 128, 255, 256] {
                let mut history = History::default();
                let mut actual = Vec::new();
                for start in (0..tokens.len()).step_by(chunk) {
                    let end = (start + chunk).min(tokens.len());
                    let snapshot = history.clone();
                    let batch = layout
                        .hash(&history, &tokens[start..end], mask.map(|m| &m[start..end]))
                        .unwrap();
                    assert_eq!(history, snapshot); // Candidate is not an implicit commit.
                    assert_eq!(
                        batch,
                        layout
                            .hash(&snapshot, &tokens[start..end], mask.map(|m| &m[start..end]))
                            .unwrap()
                    );
                    actual.extend(batch.rows);
                    history = batch.next_history;
                }
                assert_eq!(actual, expected);
                assert_eq!(
                    history.compressed_tail(),
                    [
                        m.token_map[tokens[512] as usize] as i32,
                        m.token_map[tokens[511] as usize] as i32,
                        if mask.is_some() {
                            DEAD
                        } else {
                            m.token_map[tokens[510] as usize] as i32
                        }
                    ]
                );
                history.reset();
                assert_eq!(history, History::default());
            }
        }
    }

    #[test]
    fn deepseek41_masked_dead_barriers_and_transaction_errors() {
        let layout = small_layout();
        let history = layout
            .hash(&History::default(), &[4, 5, 6], None)
            .unwrap()
            .next_history;
        let snapshot = history.clone();
        for tokens in [&[7, -1][..], &[7, VOCAB_SIZE as i32][..]] {
            assert!(layout.hash(&history, tokens, Some(&[1, 0])).is_err());
            assert_eq!(history, snapshot);
        }
        assert!(layout.hash(&history, &[7], Some(&[])).is_err());
        assert!(
            layout
                .hash(&history, &vec![0; MAX_BATCH_TOKENS + 1], None)
                .is_err()
        );
        assert!(History::from_compressed([-2, 0, 0]).is_err());
        assert!(History::from_compressed([COMPRESSED_VOCAB_SIZE as i32, 0, 0]).is_err());
        let invalid = History { tail: [0, -2, 0] };
        assert!(layout.hash(&invalid, &[], None).is_err());
        assert_eq!(invalid.tail, [0, -2, 0]);
        assert_eq!(
            layout.hash(&history, &[], None).unwrap().next_history,
            history
        );
        let batch = layout
            .hash(&history, &[8, 9, 10], Some(&[0, 1, 1]))
            .unwrap();
        assert_eq!(batch.next_history.tail, [10, 9, DEAD]);
        let from_empty = layout
            .hash(&History::default(), &[8, 9, 10], Some(&[0, 1, 1]))
            .unwrap();
        assert_eq!(batch, from_empty); // Nothing before the barrier contributes.
        let resumed = layout.hash(&batch.next_history, &[11], None).unwrap();
        let reset = layout
            .hash(&History::default(), &[9, 10, 11], None)
            .unwrap();
        assert_eq!(resumed.rows[0], reset.rows[2]);
        assert_eq!(history, snapshot);
    }

    fn uniform_row(code: u8, scale: u8) -> [u8; ROW_BYTES] {
        let mut raw = [code; ROW_BYTES];
        raw[DIM..].fill(scale);
        raw
    }

    #[test]
    fn host_diag_engram_bytes_errors_and_post_identity() -> Result<()> {
        if crate::test_in_subprocess(concat!(
            module_path!(),
            "::host_diag_engram_bytes_errors_and_post_identity"
        )) {
            return Ok(());
        }
        use super::super::profile::{self, Detail};
        let span = small_layout().table_span(1, 32, 408 * ROW_BYTES as u64)?;
        let mut artifact = tempfile::NamedTempFile::new()?;
        artifact.as_file().set_len_sparse(span.end)?;
        artifact.seek(SeekFrom::Start(span.offset))?;
        artifact.write_all(&uniform_row(0x38, 127))?;
        let table = Table::open(artifact.path(), span)?;
        let _guard = profile::HostGuard::enable();
        assert_eq!(table.read_rows(&[0, 0])?, vec![[1.; DIM]; 2]);
        let snapshot = profile::host_snapshot();
        let read = snapshot.details[Detail::EngramRead as usize];
        assert_eq!(read.calls, 2);
        assert_eq!(read.requested_bytes, 2 * ROW_BYTES as u64);
        assert_eq!(read.completed_bytes, read.requested_bytes);
        assert_eq!(snapshot.details[Detail::EngramDecode as usize].calls, 2);
        assert_eq!(
            snapshot.details[Detail::EngramMetadataPre as usize].calls,
            1
        );
        assert_eq!(
            snapshot.details[Detail::EngramMetadataPost as usize].calls,
            1
        );
        profile::reset();
        assert!(
            table
                .read_rows_with(&[0], |_, _, _| {
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "CPU read failure",
                    ))
                })
                .is_err()
        );
        let snapshot = profile::host_snapshot();
        let read = snapshot.details[Detail::EngramRead as usize];
        assert_eq!(read.calls, 1);
        assert_eq!(read.requested_bytes, ROW_BYTES as u64);
        assert_eq!(read.completed_bytes, 0);
        assert_eq!(snapshot.details[Detail::EngramDecode as usize].calls, 0);
        profile::reset();
        let changed = table.read_rows_with(&[0], |file, bytes, offset| {
            read_exact_at(file, bytes, offset)?;
            advance_modified(artifact.as_file());
            Ok(())
        });
        assert!(changed.unwrap_err().to_string().contains("identity"));
        let snapshot = profile::host_snapshot();
        assert_eq!(
            snapshot.details[Detail::EngramRead as usize].completed_bytes,
            ROW_BYTES as u64
        );
        assert_eq!(
            snapshot.details[Detail::EngramMetadataPost as usize].calls,
            1
        );
        profile::reset();
        assert!(table.read_rows(&[0]).is_err());
        assert_eq!(
            profile::host_snapshot().details[Detail::EngramRead as usize].calls,
            0
        );
        Ok(())
    }

    #[test]
    fn deepseek41_decode_scalar_fixtures_all_scale_groups() {
        let mut raw = uniform_row(0x38, 127); // +1
        raw[DIM..].copy_from_slice(&[127, 128, 126, 120, 134, 118, 136, 0]);
        let expected = [
            1.0,
            2.0,
            0.5,
            1.0 / 128.0,
            128.0,
            1.0 / 512.0,
            512.0,
            f32::from_bits(0x0040_0000),
        ];
        let decoded = decode_row(&raw).unwrap();
        for group in 0..8 {
            assert_eq!(
                &decoded[group * 32..(group + 1) * 32],
                &[expected[group]; 32]
            );
        }
        // Independent literal E4M3 sign/subnormal/max fixtures, scale 1.
        for (code, bits) in [
            (0, 0),
            (128, 0x8000_0000),
            (1, 0x3b00_0000),
            (129, 0xbb00_0000),
            (7, 0x3c60_0000),
            (8, 0x3c80_0000),
            (0x38, 0x3f80_0000),
            (0xb8, 0xbf80_0000),
            (0x7e, 0x43e0_0000),
            (0xfe, 0xc3e0_0000),
        ] {
            assert_eq!(
                decode_row(&uniform_row(code, 127)).unwrap()[0].to_bits(),
                bits
            );
        }
        // BF16 subnormal halfway ties: 1*2^-134 -> even 0;
        // 3*2^-134 -> even 2*2^-133; 5*2^-134 -> same even 2.
        for (code, bits) in [
            (1, 0),
            (3, 0x0002_0000),
            (5, 0x0002_0000),
            (0x81, 0x8000_0000),
            (0x83, 0x8002_0000),
        ] {
            assert_eq!(
                decode_row(&uniform_row(code, 2)).unwrap()[0].to_bits(),
                bits
            );
        }
        for raw in [
            uniform_row(127, 127),
            uniform_row(255, 127),
            uniform_row(0, 255),
            uniform_row(0x7e, 254),
            uniform_row(0xfe, 254),
        ] {
            assert!(decode_row(&raw).is_err());
        }
        assert_eq!(
            decode_row(&uniform_row(0x38, 254)).unwrap()[0].to_bits(),
            0x7f00_0000
        );
    }

    #[test]
    fn deepseek41_decode_exhaustive_independent_scalar_oracle() {
        // Binary64 scalar values are exact here. BF16 is rounded by distances
        // between adjacent representable values, not the production bit bias.
        for code in 0u16..256 {
            for scale in 0u16..256 {
                let e = (code >> 3) & 15;
                let magnitude = if e == 0 {
                    f64::from(code & 7) / 512.0
                } else {
                    (1.0 + f64::from(code & 7) / 8.0) * 2.0f64.powi(i32::from(e) - 7)
                };
                let sign = if code & 128 != 0 { -1.0 } else { 1.0 };
                let scaled = (sign * magnitude * 2.0f64.powi(i32::from(scale) - 127)) as f32;
                let abs = scaled.abs();
                let lower_bits = abs.to_bits() & 0xffff_0000;
                let lower = f32::from_bits(lower_bits);
                let upper = f32::from_bits(lower_bits.saturating_add(0x1_0000));
                let dl = f64::from(abs) - f64::from(lower);
                let du = f64::from(upper) - f64::from(abs);
                let expected = if dl < du || (dl == du && (lower_bits >> 16).is_multiple_of(2)) {
                    lower
                } else {
                    upper
                } * sign as f32;
                let valid =
                    code & 127 != 127 && scale != 255 && scaled.is_finite() && expected.is_finite();
                let decoded = decode_row(&uniform_row(code as u8, scale as u8));
                assert_eq!(decoded.is_ok(), valid, "code={code} scale={scale}");
                if valid {
                    assert_eq!(
                        decoded.unwrap().map(f32::to_bits),
                        [expected.to_bits(); DIM],
                        "code={code} scale={scale}"
                    );
                }
            }
        }
    }

    #[test]
    fn deepseek41_disk_sparse_huge_table_duplicate_rows_and_owned_identity() {
        let layout = Layout::new(metadata(PINNED_PRIMES)).unwrap();
        let offset = (1u64 << 33) + 32;
        let span = layout
            .table_span(1, offset, u64::from(TABLE_ROWS[0]) * ROW_BYTES as u64)
            .unwrap();
        let mut artifact = tempfile::NamedTempFile::new().unwrap();
        artifact.as_file().set_len_sparse(span.end).unwrap(); // Sparse 100GB table, not resident weights.
        artifact.seek(SeekFrom::Start(offset)).unwrap();
        artifact.write_all(&uniform_row(0x38, 127)).unwrap();
        artifact
            .seek(SeekFrom::Start(span.end - ROW_BYTES as u64))
            .unwrap();
        artifact.write_all(&uniform_row(0xb8, 128)).unwrap();
        let file = File::open(artifact.path()).unwrap();
        // Unlink before verification: ctime changes are intentionally rejected
        // after verification, even when the owned descriptor remains usable.
        artifact.close().unwrap();
        let original_identity = FileIdentity::from_metadata(&file.metadata().unwrap()).unwrap();
        let table = Table::from_file(file, original_identity, span).unwrap();
        assert_eq!(original_identity.length, span.end);
        let output = table
            .read_rows(&[TABLE_ROWS[0] - 1, 0, TABLE_ROWS[0] - 1])
            .unwrap();
        assert_eq!(output, vec![[-2.0; DIM], [1.0; DIM], [-2.0; DIM]]);
        assert!(table.read_rows(&[TABLE_ROWS[0]]).is_err());
        assert!(table.read_rows(&[u32::MAX]).is_err());
        assert!(table.read_rows(&vec![0; MAX_READ_ROWS + 1]).is_err());
        assert_eq!(table.file_identity(), original_identity);
        assert!(table.read_rows(&[]).unwrap().is_empty());
    }

    #[test]
    fn deepseek41_disk_mutation_and_truncation_reject_without_committing_history() {
        let layout = small_layout();
        let span = layout.table_span(1, 32, 408 * ROW_BYTES as u64).unwrap();
        let mut artifact = tempfile::NamedTempFile::new().unwrap();
        assert!(Table::open(artifact.path(), span).is_err());
        artifact.as_file().set_len_sparse(span.end).unwrap();
        artifact.seek(SeekFrom::Start(32)).unwrap();
        artifact.write_all(&uniform_row(0x38, 127)).unwrap();
        let table = Table::open(artifact.path(), span).unwrap();
        let history = History::default();
        let pending = layout.hash(&history, &[0], None).unwrap();
        assert_eq!(table.read_rows(&[0]).unwrap(), vec![[1.0; DIM]]);
        // A same-size, valid decoded row mutation must not return any output.
        artifact.seek(SeekFrom::Start(32)).unwrap();
        artifact.write_all(&uniform_row(0xb8, 127)).unwrap();
        advance_modified(artifact.as_file());
        assert!(table.read_rows(&[0]).is_err());
        assert!(table.read_rows(&[]).is_err());
        let file = File::open(artifact.path()).unwrap();
        assert!(Table::from_file(file, table.file_identity(), span).is_err());
        let table = Table::open(artifact.path(), span).unwrap();
        assert_eq!(table.read_rows(&[0]).unwrap(), vec![[-1.0; DIM]]);
        artifact.as_file().set_len_sparse(span.end - 1).unwrap();
        assert!(table.read_rows(&[0]).is_err());
        assert!(Table::open(artifact.path(), span).is_err());
        drop(pending); // Whole-model failure: candidate hash state is discarded.
        assert_eq!(history.compressed_tail(), [DEAD; 3]);
        let directory = tempfile::tempdir().unwrap();
        assert!(Table::open(directory.path(), span).is_err());
    }

    fn advance_modified(file: &File) {
        // Force a distinct timestamp even on filesystems with coarse clocks.
        let modified =
            file.metadata().unwrap().modified().unwrap() + std::time::Duration::from_secs(2);
        file.set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
    }

    #[test]
    fn deepseek41_verified_descriptor_rejects_substitution_and_metadata_changes() {
        let span = small_layout()
            .table_span(1, 32, 408 * ROW_BYTES as u64)
            .unwrap();
        let mut artifact = tempfile::NamedTempFile::new().unwrap();
        artifact.as_file().set_len_sparse(span.end).unwrap();
        artifact.seek(SeekFrom::Start(span.offset)).unwrap();
        artifact.write_all(&uniform_row(0x38, 127)).unwrap();
        let file = File::open(artifact.path()).unwrap();
        let expected = FileIdentity::from_metadata(&file.metadata().unwrap()).unwrap();
        let table = Table::from_file(file.try_clone().unwrap(), expected, span).unwrap();
        assert_eq!(table.read_rows(&[0]).unwrap(), vec![[1.0; DIM]]);
        let other = tempfile::NamedTempFile::new().unwrap();
        other.as_file().set_len_sparse(span.end).unwrap();
        advance_modified(other.as_file());
        assert!(Table::from_file(File::open(other.path()).unwrap(), expected, span).is_err());
        // Metadata-only changes also invalidate the verified descriptor.
        advance_modified(artifact.as_file());
        assert!(table.read_rows(&[0]).is_err());
        assert!(Table::from_file(file, expected, span).is_err());
        let current = File::open(artifact.path()).unwrap();
        let actual = FileIdentity::from_metadata(&current.metadata().unwrap()).unwrap();
        assert_ne!(actual, expected);
        #[cfg(unix)]
        {
            // Verify nanoseconds are part of both timestamp comparisons.
            let mut changed = actual;
            changed.changed.1 ^= 1;
            assert!(Table::from_file(current.try_clone().unwrap(), changed, span).is_err());
            let mut modified = actual;
            modified.modified.1 ^= 1;
            assert!(Table::from_file(current.try_clone().unwrap(), modified, span).is_err());
        }
        let table = Table::from_file(current.try_clone().unwrap(), actual, span).unwrap();
        artifact.as_file().set_len_sparse(span.end - 1).unwrap();
        assert!(table.read_rows(&[0]).is_err());
        assert!(Table::from_file(current, actual, span).is_err());
    }

    #[test]
    fn deepseek41_identity_checked_after_bounded_batch_before_output() {
        let span = small_layout()
            .table_span(1, 32, 408 * ROW_BYTES as u64)
            .unwrap();
        let mut artifact = tempfile::NamedTempFile::new().unwrap();
        artifact.as_file().set_len_sparse(span.end).unwrap();
        artifact.seek(SeekFrom::Start(span.offset)).unwrap();
        artifact.write_all(&uniform_row(0x38, 127)).unwrap();
        let table = Table::open(artifact.path(), span).unwrap();
        let mut calls = 0;
        let result = table.read_rows_with(&[0, 0], |file, bytes, offset| {
            read_exact_at(file, bytes, offset)?;
            calls += 1;
            if calls == 2 {
                // Both reads and decodes are valid. A before-only identity
                // check would return output even though verification is stale.
                artifact.seek(SeekFrom::Start(span.offset))?;
                artifact.write_all(&uniform_row(0xb8, 127))?;
                advance_modified(artifact.as_file());
            }
            Ok(())
        });
        assert_eq!(calls, 2);
        assert!(result.unwrap_err().to_string().contains("identity"));
        assert!(table.read_rows(&[0]).is_err());
    }

    // Literal decoded values are independent of decode_row: exercise signed
    // zero, E4M3 subnormals, both signs, and every scale group.
    fn batched_expected_row(row: u32) -> [f32; DIM] {
        let values = [
            0.0f32,
            -0.0,
            1.0 / 512.0,
            -1.0 / 512.0,
            1.0,
            -1.0,
            448.0,
            -448.0,
        ];
        let factors = [1.0, 2.0, 0.5, 0.125, 8.0, 0.25, 4.0, 16.0];
        std::array::from_fn(|j| values[(row as usize + j) % values.len()] * factors[j / 32])
    }

    fn batched_fixture(offset: u64) -> Result<(tempfile::NamedTempFile, Table)> {
        let span = small_layout().table_span(1, offset, 408 * ROW_BYTES as u64)?;
        let mut artifact = tempfile::NamedTempFile::new()?;
        artifact.as_file().set_len_sparse(span.end)?;
        artifact.seek(SeekFrom::Start(span.offset))?;
        let codes = [0, 128, 1, 129, 0x38, 0xb8, 0x7e, 0xfe];
        for row in 0..span.rows {
            let mut raw = uniform_row(0, 127);
            for (j, byte) in raw[..DIM].iter_mut().enumerate() {
                *byte = codes[(row as usize + j) % codes.len()];
            }
            raw[DIM..].copy_from_slice(&[127, 128, 126, 124, 130, 125, 129, 131]);
            artifact.write_all(&raw)?;
        }
        let table = Table::open(artifact.path(), span)?;
        Ok((artifact, table))
    }

    fn assert_batched_bits(rows: &[u32], output: &[[f32; DIM]]) {
        assert_eq!(output.len(), rows.len());
        for (&row, actual) in rows.iter().zip(output) {
            assert_eq!(
                actual.map(f32::to_bits),
                batched_expected_row(row).map(f32::to_bits),
                "decoded row {row}"
            );
        }
    }

    #[test]
    fn deepseek41_batched_exact_order_duplicates_and_unaligned_endpoints() -> Result<()> {
        // Includes first/last rows, row 15 crossing a 4096-byte boundary, and
        // offsets above 32 bits. Duplicate rows occupy nonadjacent caller slots.
        for offset in [37, (1u64 << 33) + 37] {
            let (_artifact, table) = batched_fixture(offset)?;
            let rows = [407, 0, 15, 16, 30, 31, 61, 62, 1, 15, 407, 0];
            let mut reads = Vec::new();
            let output = table.read_rows_batched_with(&rows, |file, bytes, at| {
                assert!(at >= table.span.offset);
                assert!(at + bytes.len() as u64 <= table.span.end);
                reads.push((at, bytes.len()));
                read_exact_at(file, bytes, at)
            })?;
            assert_batched_bits(&rows, &output);
            let scalar = table.read_rows(&rows)?;
            for (actual, reference) in output.iter().zip(&scalar) {
                assert_eq!(actual.map(f32::to_bits), reference.map(f32::to_bits));
            }
            assert_eq!(reads.first().unwrap().0, table.span.offset);
            let &(last, bytes) = reads.last().unwrap();
            assert_eq!(last + bytes as u64, table.span.end);
            assert_ne!(table.span.offset % 4096, 0);
            assert_ne!(table.span.end % 4096, 0);
            assert!(offset + 15 * (ROW_BYTES as u64) < (offset / 4096 + 1) * 4096);
            assert!(offset + 16 * ROW_BYTES as u64 > (offset / 4096 + 1) * 4096);
        }
        Ok(())
    }

    #[test]
    fn deepseek41_batched_coalescing_bounds_dense_nearby_and_random() -> Result<()> {
        let (_artifact, table) = batched_fixture(37)?;
        let dense: Vec<u32> = (0..408).rev().collect();
        let nearby: Vec<u32> = (0..408).step_by(2).rev().collect();
        let random = vec![407, 100, 6, 0, 3];
        for rows in [&dense, &nearby, &random] {
            let mut reads = Vec::new();
            let output = table.read_rows_batched_with(rows, |file, bytes, offset| {
                assert!(bytes.len() <= BATCHED_READ_BUFFER_BYTES);
                reads.push((offset, bytes.len()));
                read_exact_at(file, bytes, offset)
            })?;
            assert_batched_bits(rows, &output);
            let total: usize = reads.iter().map(|&(_, bytes)| bytes).sum();
            assert!(reads.len() <= rows.len());
            assert!(total <= (2 * rows.len() - 1) * ROW_BYTES);
            assert!(
                reads
                    .windows(2)
                    .all(|pair| pair[0].0 + pair[0].1 as u64 <= pair[1].0)
            );
            if std::ptr::eq(rows, &random) {
                // No claim of fewer calls for separated/random rows.
                assert_eq!(reads.len(), rows.len());
                assert_eq!(total, rows.len() * ROW_BYTES);
            } else {
                assert!(reads.len() <= 8, "dense/nearby rows must actually coalesce");
            }
            if std::ptr::eq(rows, &dense) {
                assert_eq!(total, rows.len() * ROW_BYTES);
            }
        }
        // A one-row gap is permitted, but the unrequested bytes are not decoded.
        let output = table.read_rows_batched_with(&[4, 0, 2, 2], |file, bytes, offset| {
            assert_eq!(offset, table.span.offset);
            assert_eq!(bytes.len(), 5 * ROW_BYTES);
            read_exact_at(file, bytes, offset)?;
            bytes[ROW_BYTES..2 * ROW_BYTES].fill(255);
            bytes[3 * ROW_BYTES..4 * ROW_BYTES].fill(255);
            Ok(())
        })?;
        assert_batched_bits(&[4, 0, 2, 2], &output);
        Ok(())
    }

    #[test]
    fn deepseek41_batched_cap_invalid_rows_and_empty() -> Result<()> {
        let (_artifact, table) = batched_fixture(37)?;
        for rows in [
            vec![408],
            vec![u32::MAX],
            vec![0; MAX_BATCHED_READ_ROWS + 1],
        ] {
            let error = table
                .read_rows_batched_with(&rows, |_, _, _| {
                    panic!("invalid requests must fail before I/O")
                })
                .unwrap_err();
            assert!(error.to_string().contains("row"));
        }
        let output =
            table.read_rows_batched_with(&[], |_, _, _| panic!("empty requests must not read"))?;
        assert!(output.is_empty());
        let rows = vec![407; MAX_BATCHED_READ_ROWS];
        let mut reads = 0;
        let output = table.read_rows_batched_with(&rows, |file, bytes, offset| {
            reads += 1;
            assert_eq!(bytes.len(), ROW_BYTES);
            read_exact_at(file, bytes, offset)
        })?;
        assert_eq!(reads, 1);
        assert_batched_bits(&rows, &output);
        Ok(())
    }

    #[test]
    fn deepseek41_batched_short_read_and_decode_errors_are_atomic() -> Result<()> {
        let (_artifact, table) = batched_fixture(37)?;
        let rows = [0, 407]; // Two separate spans; the first one succeeds.
        let mut reads = 0;
        let error = table
            .read_rows_batched_with(&rows, |file, bytes, offset| {
                reads += 1;
                if reads == 2 {
                    let half = bytes.len() / 2;
                    read_exact_at(file, &mut bytes[..half], offset)?;
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "injected short span",
                    ));
                }
                read_exact_at(file, bytes, offset)
            })
            .unwrap_err();
        assert_eq!(reads, 2);
        assert_eq!(
            error.downcast_ref::<io::Error>().unwrap().kind(),
            io::ErrorKind::UnexpectedEof
        );
        for corrupt_scale in [false, true] {
            let error = table
                .read_rows_batched_with(&rows, |file, bytes, offset| {
                    read_exact_at(file, bytes, offset)?;
                    if offset > table.span.offset {
                        bytes[if corrupt_scale { DIM } else { 0 }] = 255;
                    }
                    Ok(())
                })
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains(if corrupt_scale { "scale" } else { "code" })
            );
        }
        // A failed candidate must not retain poisoned output or encoded bytes.
        assert_batched_bits(&rows, &table.read_rows_batched(&rows)?);
        Ok(())
    }

    #[test]
    fn deepseek41_batched_source_mutation_pre_post_and_empty() -> Result<()> {
        let (mut artifact, table) = batched_fixture(37)?;
        let rows = [0, 407];
        let mut reads = 0;
        let error = table
            .read_rows_batched_with(&rows, |file, bytes, offset| {
                read_exact_at(file, bytes, offset)?;
                reads += 1;
                if reads == 2 {
                    // Mutate valid bytes after every requested byte has been read.
                    artifact.seek(SeekFrom::Start(table.span.offset))?;
                    artifact.write_all(&uniform_row(0xb8, 127))?;
                    advance_modified(artifact.as_file());
                }
                Ok(())
            })
            .unwrap_err();
        assert_eq!(reads, 2);
        assert!(error.to_string().contains("identity"));
        for rows in [&[][..], &[0][..]] {
            let error = table
                .read_rows_batched_with(rows, |_, _, _| {
                    panic!("stale identity must fail before I/O")
                })
                .unwrap_err();
            assert!(error.to_string().contains("identity"));
        }
        let (artifact, truncated) = batched_fixture(37)?;
        artifact.as_file().set_len_sparse(truncated.span.end - 1)?;
        let error = truncated
            .read_rows_batched_with(&[407], |_, _, _| {
                panic!("truncation must fail identity before I/O")
            })
            .unwrap_err();
        assert!(error.to_string().contains("identity"));
        Ok(())
    }

    #[test]
    fn host_diag_engram_batched_counts_bytes_and_unique_decodes() -> Result<()> {
        if crate::test_in_subprocess(concat!(
            module_path!(),
            "::host_diag_engram_batched_counts_bytes_and_unique_decodes"
        )) {
            return Ok(());
        }
        use super::super::profile::{self, Detail};
        let (_artifact, table) = batched_fixture(37)?;
        let _guard = profile::HostGuard::enable();
        profile::reset();
        assert_batched_bits(
            &[4, 0, 2, 2, 407],
            &table.read_rows_batched(&[4, 0, 2, 2, 407])?,
        );
        let snapshot = profile::host_snapshot();
        let read = snapshot.details[Detail::EngramRead as usize];
        assert_eq!(read.calls, 2);
        assert_eq!(read.requested_bytes, 6 * ROW_BYTES as u64);
        assert_eq!(read.completed_bytes, read.requested_bytes);
        assert_eq!(snapshot.details[Detail::EngramDecode as usize].calls, 4);
        assert_eq!(
            snapshot.details[Detail::EngramMetadataPre as usize].calls,
            1
        );
        assert_eq!(
            snapshot.details[Detail::EngramMetadataPost as usize].calls,
            1
        );
        profile::reset();
        assert!(table.read_rows_batched(&[])?.is_empty());
        let snapshot = profile::host_snapshot();
        assert_eq!(snapshot.details[Detail::EngramRead as usize].calls, 0);
        assert_eq!(snapshot.details[Detail::EngramDecode as usize].calls, 0);
        assert_eq!(
            snapshot.details[Detail::EngramMetadataPre as usize].calls,
            1
        );
        assert_eq!(
            snapshot.details[Detail::EngramMetadataPost as usize].calls,
            1
        );
        profile::reset();
        let error = table
            .read_rows_batched_with(&[0, 407], |file, bytes, offset| {
                if offset > table.span.offset {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "injected EOF"));
                }
                read_exact_at(file, bytes, offset)
            })
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<io::Error>().unwrap().kind(),
            io::ErrorKind::UnexpectedEof
        );
        let snapshot = profile::host_snapshot();
        let read = snapshot.details[Detail::EngramRead as usize];
        assert_eq!(read.calls, 2);
        assert_eq!(read.requested_bytes, 2 * ROW_BYTES as u64);
        assert_eq!(read.completed_bytes, ROW_BYTES as u64);
        assert_eq!(snapshot.details[Detail::EngramDecode as usize].calls, 1);
        assert_eq!(
            snapshot.details[Detail::EngramMetadataPre as usize].calls,
            1
        );
        assert_eq!(
            snapshot.details[Detail::EngramMetadataPost as usize].calls,
            0
        );
        Ok(())
    }

    #[test]
    fn deepseek41_batched_owned_descriptor_without_path() -> Result<()> {
        let (artifact, original) = batched_fixture(37)?;
        let file = original.file.try_clone()?;
        let span = original.span;
        drop(original);
        // Unlink before capturing identity: removing a name changes ctime.
        artifact.close()?;
        let expected = FileIdentity::from_metadata(&file.metadata()?)?;
        let table = Table::from_file(file, expected, span)?;
        assert_batched_bits(
            &[407, 15, 0, 15],
            &table.read_rows_batched(&[407, 15, 0, 15])?,
        );
        assert_eq!(table.file_identity(), expected);
        Ok(())
    }

    #[test]
    fn deepseek41_batched_checked_offset_and_span_failures() -> Result<()> {
        let (_artifact, mut table) = batched_fixture(37)?;
        let original = table.span;
        table.span.offset = u64::MAX;
        let error = table
            .read_rows_batched_with(&[1], |_, _, _| panic!("overflow must fail before I/O"))
            .unwrap_err();
        assert!(error.to_string().contains("overflow"));
        table.span = original;
        table.span.end = table.span.offset + ROW_BYTES as u64 - 1;
        let error = table
            .read_rows_batched_with(&[0], |_, _, _| panic!("invalid span must fail before I/O"))
            .unwrap_err();
        assert!(error.to_string().contains("bounds"));
        table.span = original;
        assert_batched_bits(&[0, 407], &table.read_rows_batched(&[0, 407])?);
        Ok(())
    }
}
