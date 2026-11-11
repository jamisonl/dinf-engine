//! Bounded, safe reads of local safetensors snapshots; no Python or pickle.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{Init, VarBuilder, var_builder::SimpleBackend};
use serde::Deserialize;

const MAX_METADATA_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct Metadata {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

struct Weight {
    file: Arc<Mutex<File>>,
    dtype: DType,
    shape: Vec<usize>,
    offset: u64,
    bytes: usize,
}

struct Weights {
    tensors: HashMap<String, Weight>,
}

/// Read a small JSON configuration without allowing unbounded metadata.
pub(super) fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    ensure!(
        file.metadata()?.len() <= MAX_METADATA_BYTES,
        "oversized model metadata"
    );
    let mut bytes = Vec::new();
    file.take(MAX_METADATA_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_METADATA_BYTES,
        "oversized model metadata"
    );
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

/// Load headers only. Tensor bytes are read one at a time when requested by the
/// component, avoiding a second in-memory copy of the whole model. Files remain
/// open, so replacing a path cannot change a loaded snapshot's file identity.
pub(super) fn load_component(
    directory: &Path,
    device: &Device,
    force_f32: bool,
) -> Result<VarBuilder<'static>> {
    let candidates = ["model.safetensors", "diffusion_pytorch_model.safetensors"];
    let indices = [
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
    ];
    let single = candidates
        .iter()
        .filter(|p| directory.join(p).is_file())
        .copied()
        .collect::<Vec<_>>();
    let index = indices
        .iter()
        .filter(|p| directory.join(p).is_file())
        .copied()
        .collect::<Vec<_>>();
    ensure!(
        single.len() + index.len() == 1,
        "expected exactly one standard safetensors file or index in {}",
        directory.display()
    );
    let routes = if let Some(index) = index.first() {
        let parsed: Index = read_json(&directory.join(index))?;
        ensure!(!parsed.weight_map.is_empty(), "empty model tensor index");
        Some(parsed.weight_map)
    } else {
        None
    };
    let names: BTreeSet<&str> = match &routes {
        Some(routes) => routes.values().map(String::as_str).collect(),
        None => single.into_iter().collect(),
    };
    let mut tensors = HashMap::new();
    let mut dtype = DType::BF16;
    for name in names {
        ensure!(
            !name.is_empty()
                && name.ends_with(".safetensors")
                && !name.contains(['/', '\\'])
                && name != "..",
            "invalid safetensors shard name"
        );
        let mut file = File::open(directory.join(name))
            .with_context(|| format!("open tensor shard {name}"))?;
        let file_len = file.metadata()?.len();
        let mut length = [0_u8; 8];
        file.read_exact(&mut length)?;
        let header_len = u64::from_le_bytes(length);
        ensure!(
            header_len > 0
                && header_len <= MAX_METADATA_BYTES
                && header_len <= file_len.saturating_sub(8),
            "invalid safetensors header size"
        );
        let mut header = vec![0; header_len as usize];
        file.read_exact(&mut header)?;
        let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)?;
        let file = Arc::new(Mutex::new(file));
        let data_start = 8 + header_len;
        let mut ranges = Vec::new();
        for (key, value) in raw {
            if key == "__metadata__" {
                continue;
            }
            let meta: Metadata = serde_json::from_value(value)?;
            let tensor_dtype = match meta.dtype.as_str() {
                "BF16" => DType::BF16,
                "F32" => {
                    dtype = DType::F32;
                    DType::F32
                }
                other => bail!(
                    "unsupported tensor precision {other} for {key}; use unquantized BF16/F32 reference weights"
                ),
            };
            let elements = meta
                .shape
                .iter()
                .try_fold(1_usize, |n, &d| n.checked_mul(d))
                .context("tensor size overflow")?;
            let bytes = elements
                .checked_mul(tensor_dtype.size_in_bytes())
                .context("tensor byte size overflow")?;
            let [start, end] = meta.data_offsets;
            ensure!(
                end >= start && end - start == bytes as u64 && end <= file_len - data_start,
                "invalid tensor byte range for {key}"
            );
            ensure!(
                routes
                    .as_ref()
                    .is_none_or(|routes| routes.get(&key).is_some_and(|v| v == name)),
                "tensor index disagrees with shard for {key}"
            );
            ranges.push((start, end));
            let old = tensors.insert(
                key.clone(),
                Weight {
                    file: file.clone(),
                    dtype: tensor_dtype,
                    shape: meta.shape,
                    offset: data_start + start,
                    bytes,
                },
            );
            ensure!(old.is_none(), "duplicate tensor {key}");
        }
        ranges.sort_unstable();
        let mut end = 0;
        for (start, next) in ranges {
            ensure!(
                start == end,
                "non-contiguous or overlapping safetensors payload"
            );
            end = next;
        }
        ensure!(
            end == file_len - data_start,
            "unindexed safetensors payload"
        );
    }
    ensure!(!tensors.is_empty(), "no tensors found in component");
    ensure!(
        routes
            .as_ref()
            .is_none_or(|routes| routes.len() == tensors.len()),
        "tensor index refers to missing tensors"
    );
    // F32 anywhere promotes the component; F32 -> BF16 is never implicit.
    if force_f32 {
        dtype = DType::F32;
    }
    Ok(VarBuilder::from_backend(
        Box::new(Weights { tensors }),
        dtype,
        device.clone(),
    ))
}

impl SimpleBackend for Weights {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let weight = self
            .tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::CannotFindTensor { path: name.into() })?;
        if shape.dims() != weight.shape {
            candle_core::bail!("unexpected shape for diffusion tensor {name}");
        }
        self.get_unchecked(name, dtype, device)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let weight = self
            .tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::CannotFindTensor { path: name.into() })?;
        if !matches!(dtype, DType::F32 | DType::BF16)
            || (weight.dtype == DType::F32 && dtype != DType::F32)
        {
            candle_core::bail!("refusing precision reduction for diffusion tensor {name}");
        }
        let mut bytes = vec![0; weight.bytes];
        {
            let mut file = weight
                .file
                .lock()
                .map_err(|_| candle_core::Error::Msg("tensor file lock poisoned".into()))?;
            file.seek(SeekFrom::Start(weight.offset))?;
            file.read_exact(&mut bytes)?;
        }
        Tensor::from_raw_buffer(&bytes, weight.dtype, &weight.shape, device)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_precision_is_preserved() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let tensor = Tensor::new(&[1_f32, 2.], &Device::Cpu)?.to_dtype(DType::BF16)?;
        candle_core::safetensors::save(
            &HashMap::from([("w".to_owned(), tensor)]),
            dir.path().join("model.safetensors"),
        )?;
        let vb = load_component(dir.path(), &Device::Cpu, false)?;
        assert_eq!(vb.dtype(), DType::BF16);
        assert_eq!(
            vb.get(2, "w")?.to_dtype(DType::F32)?.to_vec1::<f32>()?,
            vec![1., 2.]
        );
        assert!(vb.to_dtype(DType::F16).get(2, "w").is_err());
        assert_eq!(
            load_component(dir.path(), &Device::Cpu, true)?.dtype(),
            DType::F32
        );
        Ok(())
    }

    #[test]
    fn f32_cannot_be_silently_downcast() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let tensor = Tensor::new(&[1.0001_f32], &Device::Cpu)?;
        candle_core::safetensors::save(
            &HashMap::from([("w".to_owned(), tensor)]),
            dir.path().join("model.safetensors"),
        )?;
        let vb = load_component(dir.path(), &Device::Cpu, false)?;
        assert_eq!(vb.dtype(), DType::F32);
        assert!(vb.to_dtype(DType::BF16).get(1, "w").is_err());
        Ok(())
    }

    #[test]
    fn rejects_unsafe_index_and_quantized_weights() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"w":"../outside.safetensors"}}"#,
        )?;
        assert!(
            load_component(dir.path(), &Device::Cpu, false)
                .err()
                .unwrap()
                .to_string()
                .contains("shard name")
        );
        let dir = tempfile::tempdir()?;
        let tensor = Tensor::new(&[1_f32], &Device::Cpu)?.to_dtype(DType::F16)?;
        candle_core::safetensors::save(
            &HashMap::from([("w".to_owned(), tensor)]),
            dir.path().join("model.safetensors"),
        )?;
        assert!(
            load_component(dir.path(), &Device::Cpu, false)
                .err()
                .unwrap()
                .to_string()
                .contains("precision")
        );
        Ok(())
    }
}
