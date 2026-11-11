//! Input identity and canonical split-GGUF validation. No model weights load here.
use anyhow::{Context, Result, ensure};
use candle_core::quantized::gguf_file::Value;
use dinf_engine::gguf_loader::read_gguf_metadata_map;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelFile {
    pub name: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Pin {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(super) struct Stamp {
    bytes: u64,
    modified_ns: u128,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
}

pub(super) fn stamp(path: &Path) -> Result<Stamp> {
    let metadata = path.metadata()?;
    ensure!(metadata.is_file(), "not a regular file: {}", path.display());
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;
    Ok(Stamp {
        bytes: metadata.len(),
        modified_ns: metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos(),
        #[cfg(unix)]
        device: metadata.dev(),
        #[cfg(unix)]
        inode: metadata.ino(),
        #[cfg(unix)]
        changed_seconds: metadata.ctime(),
        #[cfg(unix)]
        changed_nanoseconds: metadata.ctime_nsec(),
    })
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct CheckedFile {
    pub path: PathBuf,
    pub sha256: String,
    pub identity: Stamp,
}

impl CheckedFile {
    pub fn check(base: &Path, pin: &Pin) -> Result<Self> {
        ensure!(
            valid_hash(&pin.sha256),
            "invalid SHA-256 for {}",
            pin.path.display()
        );
        let path = base.join(&pin.path).canonicalize()?;
        let identity = stamp(&path)?;
        ensure!(
            sha256(&path)? == pin.sha256,
            "checksum differs: {}",
            path.display()
        );
        ensure!(
            stamp(&path)? == identity,
            "file changed while hashing: {}",
            path.display()
        );
        Ok(Self {
            path,
            sha256: pin.sha256.clone(),
            identity,
        })
    }

    pub fn recheck(&self, hash: bool) -> Result<()> {
        ensure!(
            stamp(&self.path)? == self.identity,
            "file identity changed: {}",
            self.path.display()
        );
        if hash {
            ensure!(
                sha256(&self.path)? == self.sha256,
                "file bytes changed: {}",
                self.path.display()
            );
            ensure!(
                stamp(&self.path)? == self.identity,
                "file changed while rehashing"
            );
        }
        Ok(())
    }
}

pub(super) fn json_file<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    ensure!(
        path.metadata()?.len() <= 32 * 1024 * 1024,
        "JSON exceeds 32 MiB input limit"
    );
    Ok(serde_json::from_reader(std::io::BufReader::new(
        File::open(path)?,
    ))?)
}

pub(super) fn write_json_new(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    std::io::Write::write_all(&mut file, b"\n")?;
    file.sync_all()?;
    Ok(())
}

pub(super) fn model_manifest(model: &Path) -> Result<Vec<ModelFile>> {
    let metadata = read_gguf_metadata_map(model)?;
    let split = metadata.keys().any(|key| key.starts_with("split."));
    let count = if split {
        usize::try_from(split_integer(&metadata, "split.count")?)?
    } else {
        1
    };
    ensure!((1..=128).contains(&count), "invalid shard count");
    let name = model
        .file_name()
        .and_then(|s| s.to_str())
        .context("invalid model filename")?;
    let suffix = format!("-00001-of-{count:05}.gguf");
    let prefix = if split {
        name.strip_suffix(&suffix)
            .context("noncanonical split entry")?
    } else {
        ""
    };
    let parent = model.parent().unwrap_or(Path::new("."));
    let mut files = Vec::new();
    for index in 0..count {
        let name = if split {
            format!("{prefix}-{:05}-of-{count:05}.gguf", index + 1)
        } else {
            name.to_owned()
        };
        let path = parent.join(&name);
        let identity = stamp(&path)?;
        let sha256 = sha256(&path)?;
        ensure!(stamp(&path)? == identity, "model changed while hashing");
        files.push(ModelFile {
            name,
            size_bytes: identity.bytes,
            sha256,
        });
    }
    validate_model_files(model, &files[0].sha256, Some(&files), false)?;
    Ok(files)
}

pub(super) fn valid_hash(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

pub(super) fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_.-".contains(&c))
}

fn split_integer(metadata: &std::collections::HashMap<String, Value>, key: &str) -> Result<u64> {
    let value = match metadata.get(key) {
        Some(Value::U8(v)) => u64::from(*v),
        Some(Value::U16(v)) => u64::from(*v),
        Some(Value::U32(v)) => u64::from(*v),
        Some(Value::U64(v)) => *v,
        Some(Value::I8(v)) if *v >= 0 => *v as u64,
        Some(Value::I16(v)) if *v >= 0 => *v as u64,
        Some(Value::I32(v)) if *v >= 0 => *v as u64,
        Some(Value::I64(v)) if *v >= 0 => *v as u64,
        _ => anyhow::bail!("missing or invalid {key}"),
    };
    Ok(value)
}

pub(super) fn validate_model_files(
    model: &Path,
    entry_hash: &str,
    files: Option<&[ModelFile]>,
    hash_files: bool,
) -> Result<()> {
    ensure!(valid_hash(entry_hash), "invalid entry checksum");
    let entry_metadata = read_gguf_metadata_map(model)?;
    let is_split = entry_metadata.keys().any(|key| key.starts_with("split."));
    let entry_name = model
        .file_name()
        .and_then(|name| name.to_str())
        .context("invalid entry name")?;
    // The engine also discovers shards from the filename. Never allow that
    // path to read files that a legacy, entry-only fixture did not hash.
    let discovered_split = entry_name
        .split_once("-00001-of-")
        .and_then(|(_, tail)| tail.split_once('.'))
        .and_then(|(count, _)| count.parse::<usize>().ok())
        .is_some_and(|count| count >= 2);
    ensure!(
        !discovered_split || is_split,
        "split-style filename requires split metadata and a manifest"
    );
    let Some(files) = files else {
        ensure!(
            !is_split,
            "split model requires a complete model_files manifest"
        );
        ensure!(
            !hash_files || sha256(model)? == entry_hash,
            "reference model checksum mismatch"
        );
        return Ok(());
    };
    ensure!(
        (1..=128).contains(&files.len()),
        "invalid model_files count"
    );
    ensure!(
        files[0].name == entry_name && files[0].sha256 == entry_hash,
        "manifest entry differs from model/model_sha256"
    );
    ensure!(
        is_split == (files.len() > 1),
        "manifest count disagrees with split metadata"
    );
    let suffix = format!("-00001-of-{:05}.gguf", files.len());
    let prefix = if is_split {
        entry_name
            .strip_suffix(&suffix)
            .filter(|prefix| !prefix.is_empty() && !prefix.contains("-00001-of-"))
            .context("split entry must use canonical PREFIX-00001-of-NNNNN.gguf naming")?
    } else {
        ""
    };
    let parent = model
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let canonical_parent = parent.canonicalize()?;
    let mut tensor_count = 0u64;
    let mut declared_count = None;
    for (index, file) in files.iter().enumerate() {
        ensure!(
            valid_name(&file.name) && valid_hash(&file.sha256) && file.size_bytes > 0,
            "invalid shard name, size or checksum at index {index}"
        );
        if is_split {
            ensure!(
                file.name == format!("{prefix}-{:05}-of-{:05}.gguf", index + 1, files.len()),
                "missing, reordered or noncanonical shard at index {index}"
            );
        }
        let path = parent.join(&file.name);
        ensure!(
            !is_split || path.canonicalize()?.parent() == Some(canonical_parent.as_path()),
            "shard resolves outside the model directory"
        );
        ensure!(
            path.is_file() && path.metadata()?.len() == file.size_bytes,
            "shard size differs: {}",
            file.name
        );
        let metadata = read_gguf_metadata_map(&path)?;
        if is_split {
            ensure!(
                metadata
                    .keys()
                    .filter(|key| key.starts_with("split."))
                    .all(|key| matches!(
                        key.as_str(),
                        "split.no" | "split.count" | "split.tensors.count"
                    )),
                "unknown split metadata in {}",
                file.name
            );
            ensure!(
                split_integer(&metadata, "split.no")? == index as u64
                    && split_integer(&metadata, "split.count")? == files.len() as u64,
                "split index/count differs in {}",
                file.name
            );
            let count = split_integer(&metadata, "split.tensors.count")?;
            ensure!(
                count > 0 && declared_count.is_none_or(|previous| previous == count),
                "split tensor total differs in {}",
                file.name
            );
            declared_count = Some(count);
            // The bounded metadata reader above has parsed the whole header.
            // GGUF v2/v3 place the actual local tensor count at bytes8..16.
            let mut header = [0u8; 16];
            File::open(&path)?.read_exact(&mut header)?;
            ensure!(
                &header[..4] == b"GGUF"
                    && matches!(u32::from_le_bytes(header[4..8].try_into()?), 2 | 3),
                "split reference requires GGUF v2/v3"
            );
            tensor_count = tensor_count
                .checked_add(u64::from_le_bytes(header[8..16].try_into()?))
                .context("split tensor count overflow")?;
        } else {
            ensure!(
                !metadata.keys().any(|key| key.starts_with("split.")),
                "unexpected split metadata"
            );
        }
        ensure!(
            !hash_files || sha256(&path)? == file.sha256,
            "shard checksum differs: {}",
            file.name
        );
    }
    ensure!(
        !is_split || declared_count == Some(tensor_count),
        "actual split tensor total differs"
    );
    Ok(())
}

pub(super) fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::gguf_file;
    use std::io::Write;

    #[test]
    fn pinned_file_rejects_changed_bytes_and_bad_digest() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("input.json");
        std::fs::write(&path, b"first")?;
        let pin = Pin {
            path: path.clone(),
            sha256: sha256(&path)?,
        };
        let checked = CheckedFile::check(dir.path(), &pin)?;
        checked.recheck(true)?;
        std::fs::write(&path, b"other")?;
        assert!(checked.recheck(true).is_err());
        assert!(CheckedFile::check(dir.path(), &pin).is_err());
        assert!(!valid_hash(&"F".repeat(64)));
        Ok(())
    }

    #[test]
    fn split_style_name_cannot_hide_unhashed_siblings() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("model-00001-of-00002.gguf");
        gguf_file::write(&mut File::create(&path)?, &[], &[])?;
        assert!(model_manifest(&path).is_err());
        let invalid = ModelFile {
            name: "../other.gguf".into(),
            size_bytes: 1,
            sha256: "0".repeat(64),
        };
        assert!(!valid_name(&invalid.name));
        assert!(!valid_name("/absolute.gguf"));
        assert!(!valid_name(".."));
        // Keep an independent same-size payload corruption case: the pin
        // checks content, not just length or a filename's quant label.
        let path = dir.path().join("plain.gguf");
        gguf_file::write(&mut File::create(&path)?, &[], &[])?;
        let files = model_manifest(&path)?;
        let mut bytes = std::fs::read(&path)?;
        bytes.push(0);
        File::create(&path)?.write_all(&bytes)?;
        assert!(validate_model_files(&path, &files[0].sha256, Some(&files), true).is_err());
        Ok(())
    }
}
