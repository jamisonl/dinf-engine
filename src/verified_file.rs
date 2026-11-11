//! Pinned GGUF artifact verification.
//!
//! A full SHA-256 pass establishes trust in a pinned artifact. On Unix, a
//! small owner-only verification receipt stored beside the artifact records
//! the pin, file identity and a sampled fingerprint so later opens of the same
//! unchanged file can skip the full pass. The sampled fingerprint format is
//! stable and documented below, so the value can be recomputed independently.

use crate::architectures::deepseek41::engram::FileIdentity;
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::Path;
use std::time::Instant;

const SAMPLE_CHUNK: usize = 8 * 1024 * 1024;
const MAX_RECEIPT_BYTES: u64 = 16 * 1024;

pub(crate) fn check_identity(file: &File, expected: FileIdentity) -> Result<()> {
    ensure!(
        FileIdentity::from_metadata(&file.metadata()?)? == expected,
        "GGUF source identity/length/timestamps changed"
    );
    Ok(())
}

pub(crate) fn read_exact_at(
    file: &File,
    mut bytes: &mut [u8],
    mut offset: u64,
) -> std::io::Result<()> {
    while !bytes.is_empty() {
        #[cfg(unix)]
        let result = std::os::unix::fs::FileExt::read_at(file, bytes, offset);
        #[cfg(windows)]
        let result = std::os::windows::fs::FileExt::seek_read(file, bytes, offset);
        #[cfg(not(any(unix, windows)))]
        let result: std::io::Result<usize> = Err(std::io::ErrorKind::Unsupported.into());
        match result {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(count) => {
                offset += count as u64;
                bytes = &mut bytes[count..];
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// SHA256(sizeLE ++ head8MiB ++ tail8MiB). Tail is included only above 8MiB,
/// including overlap. Exact bounded positional reads never change the seek cursor.
/// Identity is checked before and after; this sample is not full SHA verification.
pub fn fingerprint_descriptor(file: &File) -> Result<[u8; 32]> {
    let identity = FileIdentity::from_metadata(&file.metadata()?)?;
    fingerprint(file, identity).map(|(digest, _)| digest)
}

fn fingerprint(file: &File, identity: FileIdentity) -> Result<([u8; 32], u64)> {
    check_identity(file, identity)?;
    let head = identity.length.min(SAMPLE_CHUNK as u64) as usize;
    let mut scratch = vec![0; head];
    let mut hash = Sha256::new();
    hash.update(identity.length.to_le_bytes());
    read_exact_at(file, &mut scratch, 0)?;
    hash.update(&scratch);
    let mut bytes = head as u64;
    if identity.length > SAMPLE_CHUNK as u64 {
        read_exact_at(file, &mut scratch, identity.length - SAMPLE_CHUNK as u64)?;
        hash.update(&scratch);
        bytes += SAMPLE_CHUNK as u64;
    }
    check_identity(file, identity)?;
    Ok((hash.finalize().into(), bytes))
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Pin {
    pub model_id: String,
    pub source_revision: String,
    pub native_version: u32,
    pub expected_bytes: u64,
    pub expected_sha256: [u8; 32],
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    version: u32,
    pin: Pin,
    identity: FileIdentity,
    sample_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct Verification {
    pub sample_hit: bool,
    pub full_passes: u64,
    pub full_read_bytes: u64,
    pub sample_bytes: u64,
    pub elapsed_ns: u128,
}

// Test-only guard for ignored timing tests. It never overrides the pin or adds
// an artifact open/sample; it rejects an unexpected file identity before any
// verification I/O, so a warm (cached-only) run cannot silently fall back to a
// full SHA pass on a cache miss.
#[cfg(test)]
std::thread_local! {
    static OPEN_GUARD: std::cell::Cell<Option<(FileIdentity, bool)>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_open_guard<T>(
    identity: FileIdentity,
    allow_fullcheck: bool,
    open: impl FnOnce() -> Result<T>,
) -> Result<T> {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            OPEN_GUARD.with(|guard| guard.set(None));
        }
    }
    ensure!(
        OPEN_GUARD.with(|guard| guard.get().is_none()),
        "nested timing guard"
    );
    OPEN_GUARD.with(|guard| guard.set(Some((identity, allow_fullcheck))));
    let _reset = Reset;
    open()
}

// In production only the compiled-in Maya pin reaches this function. A test
// fixture pin cannot make a `RawGgufSource::new` source count as verified.
pub(crate) fn verify_cached(
    file: &File,
    path: &Path,
    identity: FileIdentity,
    pin: &Pin,
    fullcheck: impl FnOnce() -> Result<()>,
) -> Result<Verification> {
    let started = Instant::now();
    #[cfg(test)]
    if let Some((expected, _)) = OPEN_GUARD.with(std::cell::Cell::get) {
        ensure!(
            identity == expected,
            "retained source differs from timing control identity"
        );
    }
    check_identity(file, identity)?;
    ensure!(
        identity.length == pin.expected_bytes,
        "Maya artifact length mismatch: expected {}, got {}",
        pin.expected_bytes,
        identity.length
    );
    let mut stats = Verification::default();
    #[cfg(unix)]
    let cache = unix_cache::Cache::open(path).ok();
    #[cfg(not(unix))]
    let _ = path;
    #[cfg(unix)]
    let receipt = cache.as_ref().and_then(|cache| cache.read().ok());
    #[cfg(not(unix))]
    let receipt: Option<Receipt> = None;
    if let Some(receipt) = receipt
        && receipt.version == 1
        && receipt.pin == *pin
        && receipt.identity == identity
    {
        let (sample, bytes) = fingerprint(file, identity)?;
        stats.sample_bytes += bytes;
        stats.sample_hit = sample == receipt.sample_sha256;
    }
    if !stats.sample_hit {
        #[cfg(test)]
        if let Some((_, allow_fullcheck)) = OPEN_GUARD.with(std::cell::Cell::get) {
            ensure!(
                allow_fullcheck,
                "cached-only timing refused a full SHA cache miss"
            );
        }
        fullcheck()?;
        stats.full_passes = 1;
        stats.full_read_bytes = identity.length;
        check_identity(file, identity)?;
        #[cfg(unix)]
        if let Some(cache) = cache {
            let (sample_sha256, bytes) = fingerprint(file, identity)?;
            stats.sample_bytes += bytes;
            let receipt = Receipt {
                version: 1,
                pin: pin.clone(),
                identity,
                sample_sha256,
            };
            // A valid model remains usable in a read-only/unavailable cache.
            let _ = cache.write(&receipt);
        }
    }
    check_identity(file, identity)?;
    stats.elapsed_ns = started.elapsed().as_nanos();
    tracing::info!(
        kind = if stats.sample_hit {
            "local_receipt"
        } else {
            "full_sha256"
        },
        sample_hit = stats.sample_hit,
        full_passes = stats.full_passes,
        full_read_bytes = stats.full_read_bytes,
        sample_bytes = stats.sample_bytes,
        elapsed_ns = stats.elapsed_ns as u64,
        "Maya artifact verification"
    );
    Ok(stats)
}

#[cfg(unix)]
mod unix_cache {
    use super::*;
    use std::ffi::CString;
    use std::io::{Read, Write};
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::{
        ffi::OsStrExt,
        fs::{MetadataExt, OpenOptionsExt},
    };
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(super) struct Cache {
        directory: File,
        name: CString,
    }
    impl Cache {
        pub(super) fn open(path: &Path) -> Result<Self> {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            let directory = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(parent)?;
            let metadata = directory.metadata()?;
            ensure!(
                metadata.uid() == unsafe { libc::geteuid() } && metadata.mode() & 0o022 == 0,
                "unsafe receipt directory"
            );
            let mut name = path
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("missing artifact name"))?
                .as_bytes()
                .to_vec();
            name.extend_from_slice(b".dinf-verified.json");
            Ok(Self {
                directory,
                name: CString::new(name)?,
            })
        }
        fn open_at(&self, name: &CString, flags: i32) -> Result<File> {
            let fd = unsafe {
                libc::openat(
                    self.directory.as_raw_fd(),
                    name.as_ptr(),
                    flags | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK,
                    0o600,
                )
            };
            if fd < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            Ok(unsafe { File::from_raw_fd(fd) })
        }
        pub(super) fn read(&self) -> Result<Receipt> {
            let file = self.open_at(&self.name, libc::O_RDONLY)?;
            let metadata = file.metadata()?;
            ensure!(
                metadata.is_file()
                    && metadata.uid() == unsafe { libc::geteuid() }
                    && metadata.mode() & 0o777 == 0o600
                    && metadata.nlink() == 1
                    && metadata.len() <= MAX_RECEIPT_BYTES,
                "unsafe/oversized receipt"
            );
            let mut bytes = Vec::with_capacity(metadata.len() as usize);
            file.take(MAX_RECEIPT_BYTES + 1).read_to_end(&mut bytes)?;
            ensure!(bytes.len() as u64 <= MAX_RECEIPT_BYTES, "oversized receipt");
            Ok(serde_json::from_slice(&bytes)?)
        }
        pub(super) fn write(&self, receipt: &Receipt) -> Result<()> {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let temporary = CString::new(format!(
                ".dinf-verified-{}-{}.tmp",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ))?;
            let mut file =
                self.open_at(&temporary, libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL)?;
            let result = (|| -> Result<()> {
                // umask may remove permission bits, never add them.
                let bytes = serde_json::to_vec(receipt)?;
                ensure!(bytes.len() as u64 <= MAX_RECEIPT_BYTES, "oversized receipt");
                file.write_all(&bytes)?;
                file.sync_all()?;
                ensure!(
                    unsafe {
                        libc::renameat(
                            self.directory.as_raw_fd(),
                            temporary.as_ptr(),
                            self.directory.as_raw_fd(),
                            self.name.as_ptr(),
                        )
                    } == 0,
                    "receipt atomic rename failed: {}",
                    std::io::Error::last_os_error()
                );
                Ok(())
            })();
            if result.is_err() {
                unsafe {
                    libc::unlinkat(self.directory.as_raw_fd(), temporary.as_ptr(), 0);
                }
            }
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom, Write};

    #[test]
    fn maya_fingerprint_parity_cpu() -> Result<()> {
        let mut fixture = tempfile::NamedTempFile::new()?;
        for size in [0, 3, SAMPLE_CHUNK, SAMPLE_CHUNK + 1, 2 * SAMPLE_CHUNK + 1] {
            let bytes: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
            fixture.as_file_mut().set_len(0)?;
            fixture.as_file_mut().seek(SeekFrom::Start(0))?;
            fixture.write_all(&bytes)?;
            // Reference: a plain sequential seek/read over the same byte ranges.
            let mut legacy = File::open(fixture.path())?;
            let mut hash = Sha256::new();
            hash.update((size as u64).to_le_bytes());
            let mut scratch = vec![0; SAMPLE_CHUNK];
            let n = legacy.read(&mut scratch)?;
            hash.update(&scratch[..n]);
            if size > SAMPLE_CHUNK {
                legacy.seek(SeekFrom::Start((size - SAMPLE_CHUNK) as u64))?;
                let n = legacy.read(&mut scratch)?;
                hash.update(&scratch[..n]);
            }
            let reference: [u8; 32] = hash.finalize().into();
            legacy.seek(SeekFrom::Start(7))?;
            let actual = fingerprint_descriptor(&legacy)?;
            assert_eq!(actual, reference, "size={size}");
            assert_eq!(legacy.stream_position()?, 7);
            let identity = FileIdentity::from_metadata(&legacy.metadata()?)?;
            let (_, count) = fingerprint(&legacy, identity)?;
            assert_eq!(
                count,
                if size > SAMPLE_CHUNK {
                    2 * SAMPLE_CHUNK
                } else {
                    size
                } as u64
            );
            assert!(read_exact_at(&legacy, &mut [0u8], size as u64).is_err());
            if size == 0 {
                let hex: String = actual.iter().map(|b| format!("{b:02x}")).collect();
                assert_eq!(
                    hex,
                    "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
                );
            }
        }
        Ok(())
    }
}
