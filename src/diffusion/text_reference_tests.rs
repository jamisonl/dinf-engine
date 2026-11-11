//! Opt-in full Image2.1 text-conditioning comparison; no public API additions.
use super::*;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
};

fn file_sha(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[derive(serde::Serialize)]
struct ErrorMetrics {
    max_abs: f64,
    rms: f64,
    relative_l2: f64,
    cosine: f64,
    passed: bool,
}

fn metrics(actual: &[f32], wanted: &[f32]) -> Result<ErrorMetrics> {
    ensure!(
        !actual.is_empty() && actual.len() == wanted.len(),
        "metric shape differs"
    );
    let (mut error, mut aa, mut rr, mut dot, mut max_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&a, &r) in actual.iter().zip(wanted) {
        let (a, r) = (f64::from(a), f64::from(r));
        let d = a - r;
        error += d * d;
        aa += a * a;
        rr += r * r;
        dot += a * r;
        max_abs = max_abs.max(d.abs());
    }
    ensure!(aa > 0. && rr > 0., "zero hidden norm");
    let relative_l2 = (error / rr).sqrt();
    let cosine = dot / (aa * rr).sqrt();
    Ok(ErrorMetrics {
        max_abs,
        rms: (error / actual.len() as f64).sqrt(),
        relative_l2,
        cosine,
        passed: relative_l2 <= 0.01 && cosine >= 0.9999,
    })
}

#[test]
#[ignore = "requires pinned local Image2.1 text weights and independent exact-token reference"]
fn real_image_text_conditioning_matches_reference() -> Result<()> {
    let reference = PathBuf::from(
        std::env::var("DINF_IMAGE_TEXT_REFERENCE").context("set DINF_IMAGE_TEXT_REFERENCE")?,
    );
    let snapshot = PathBuf::from(
        std::env::var("DINF_IMAGE_TEXT_SNAPSHOT").context("set DINF_IMAGE_TEXT_SNAPSHOT")?,
    );
    let prefix = PathBuf::from(
        std::env::var("DINF_IMAGE_TEXT_ACTUAL_PREFIX")
            .context("set new DINF_IMAGE_TEXT_ACTUAL_PREFIX")?,
    );
    let backend = std::env::var("DINF_IMAGE_TEXT_BACKEND").unwrap_or_else(|_| "cpu".into());
    let mode = std::env::var("DINF_IMAGE_TEXT_MODE").unwrap_or_else(|_| "f32".into());
    let (expected_dtype, expected_name, reference_backend, force_f32) = match mode.as_str() {
        "f32" => (DType::F32, "f32", "cpu", true),
        #[cfg(target_os = "macos")]
        "production_bf16" if backend == "metal" => (DType::BF16, "bf16", "mps", false),
        _ => anyhow::bail!("unsupported text mode/backend"),
    };
    let spec: serde_json::Value = read_json(&reference)?;
    ensure!(
        spec["schema_version"] == 1
            && spec["mode"] == mode
            && spec["hidden_dtype"] == expected_name
            && spec["dtype"] == expected_name
            && spec["compute_dtype"] == expected_name
            && spec["storage_dtype"] == "f32le"
            && spec["backend"] == reference_backend
            && spec["producer"] == "Transformers Qwen3VL text-only forward"
            && spec["transformers"] == "5.22.1"
            && spec["torch"]
                .as_str()
                .is_some_and(|v| v.starts_with("2.15.0")),
        "unsupported reference"
    );
    let prompt = spec["prompt"].as_str().context("reference prompt")?;
    let tokenizer = Tokenizer::from_file(snapshot.join("processor/tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let (ids, drop_index) = prompt_tokens(&tokenizer, prompt)?;
    let expected = spec["token_ids"]
        .as_array()
        .context("reference IDs")?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|x| u32::try_from(x).ok())
                .context("token ID")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        ids == expected && spec["drop_index"].as_u64() == Some(drop_index as u64),
        "token IDs/drop index differ from production prompt encoding"
    );
    let weights = spec["snapshot_files"]
        .as_array()
        .context("snapshot files")?;
    let mut paths = std::collections::BTreeSet::new();
    for entry in weights {
        let relative = entry["path"].as_str().context("snapshot path")?;
        let p = Path::new(relative);
        ensure!(
            p.components()
                .all(|x| matches!(x, std::path::Component::Normal(_))),
            "unsafe snapshot path"
        );
        ensure!(paths.insert(relative.to_owned()), "duplicate file pin");
        ensure!(
            file_sha(&snapshot.join(p))? == entry["sha256"].as_str().context("file SHA")?,
            "snapshot hash differs: {relative}"
        );
    }
    let index: serde_json::Value =
        read_json(&snapshot.join("text_encoder/model.safetensors.index.json"))?;
    let mut required = index["weight_map"]
        .as_object()
        .context("weight map")?
        .values()
        .map(|v| {
            v.as_str()
                .map(|x| format!("text_encoder/{x}"))
                .context("shard name")
        })
        .collect::<Result<std::collections::BTreeSet<_>>>()?;
    required.extend(
        [
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.json",
            "processor/tokenizer.json",
        ]
        .map(str::to_owned),
    );
    ensure!(
        paths == required,
        "reference must pin exact text shard/config/tokenizer set"
    );
    let relative = spec["hidden"]["file"]
        .as_str()
        .context("reference hidden file")?;
    ensure!(
        Path::new(relative).file_name().and_then(|x| x.to_str()) == Some(relative),
        "reference raw file must be basename"
    );
    let raw_path = reference
        .parent()
        .context("reference parent")?
        .join(relative);
    ensure!(
        file_sha(&raw_path)? == spec["hidden"]["sha256"].as_str().context("hidden SHA")?,
        "reference hidden SHA differs"
    );
    let raw = fs::read(&raw_path)?;
    ensure!(
        raw.len() == ids.len() * 4096 * 4,
        "reference hidden size differs"
    );
    ensure!(
        spec["hidden"]["shape"] == serde_json::json!([1, ids.len(), 4096]),
        "reference hidden shape differs"
    );
    let wanted = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    ensure!(wanted.iter().all(|v| v.is_finite()), "reference nonfinite");
    let device = match backend.as_str() {
        "cpu" => Device::Cpu,
        #[cfg(target_os = "macos")]
        "metal" => Device::new_metal(0)?,
        _ => anyhow::bail!("unsupported backend"),
    };
    // F32 is an explicit diagnostic on either backend. The separate
    // production_bf16 mode follows the pipeline's Metal force_f32=false path.
    let vb = load_component(&snapshot.join("text_encoder"), &device, force_f32)?;
    let compute_dtype = vb.dtype();
    ensure!(
        compute_dtype == expected_dtype,
        "text fixture compute dtype differs"
    );
    let actual_backend = if device.is_cpu() {
        "cpu"
    } else if device.is_metal() {
        "metal"
    } else {
        anyhow::bail!("unexpected actual backend")
    };
    ensure!(actual_backend == backend, "actual backend differs");
    let model = TextEncoder::load(vb, read_json(&snapshot.join("text_encoder/config.json"))?)?;
    let hidden = model.forward(&Tensor::from_vec(ids.clone(), (1, ids.len()), &device)?)?;
    device.synchronize()?;
    ensure!(
        hidden.dims() == [1, ids.len(), 4096],
        "actual shape differs"
    );
    let hidden_dtype = hidden.dtype();
    ensure!(
        hidden_dtype == expected_dtype,
        "hidden compute dtype differs"
    );
    let actual = hidden
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    ensure!(actual.iter().all(|v| v.is_finite()), "actual nonfinite");
    let actual_bytes = actual
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let actual_path = prefix.with_extension("hidden.f32");
    OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&actual_path)?
        .write_all(&actual_bytes)?;
    let full = metrics(&actual, &wanted)?;
    // This suffix is the tensor the production pipeline sends to the denoiser.
    // Keep its gate separate so system-prefix rows cannot dilute its error.
    let offset = drop_index * 4096;
    let conditioning = metrics(&actual[offset..], &wanted[offset..])?;
    let passed = full.passed && conditioning.passed;
    let report = serde_json::json!({
        "passed": passed,
        "backend": actual_backend,
        "mode": mode,
        "dtype": expected_name,
        "compute_dtype": format!("{compute_dtype:?}").to_lowercase(),
        "hidden_dtype": format!("{hidden_dtype:?}").to_lowercase(),
        "storage_dtype": "f32le",
        "reference_backend": spec["backend"],
        "reference_compute_dtype": spec["compute_dtype"],
        "reference_storage_dtype": spec["storage_dtype"],
        "shape": hidden.dims(),
        "token_ids": ids,
        "drop_index": drop_index,
        "reference_sha256": file_sha(&reference)?,
        "actual_path": actual_path,
        "actual_sha256": file_sha(&actual_path)?,
        "finite": true,
        "max_abs": full.max_abs,
        "rms": full.rms,
        "relative_l2": full.relative_l2,
        "cosine": full.cosine,
        "full_hidden": full,
        "conditioning_shape": [1, ids.len() - drop_index, 4096],
        "conditioning_suffix": conditioning,
        "scope": "exact token IDs; complete pre-final-RMSNorm text hidden states; no image quality claim",
    });
    OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(prefix.with_extension("json"))?
        .write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    ensure!(
        passed,
        "Image2.1 text full/suffix conditioning gate failed; see saved report"
    );
    Ok(())
}
