//! Compare complete projector embeddings on identical normalized pixels.
//! Usage: TEXT.gguf MMPROJ.gguf cpu|metal REFERENCE.json [ACTUAL_PREFIX]
//! The pinned fixtures/vision_reference.cpp producer supplies little-endian
//! HWC pixels. Gemma/Scout use those values directly, transposed to CHW.
//! Qwen requires a 256x256 or 768x768 single-image fixture with exact no-resize pixel
//! equivalence before its public RGB encoder can run. No text weights load.
//! Fixed component gates: relative L2 <=0.01 and cosine >=0.9999. These do
//! not qualify preprocessing policy or final image-description quality.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use candle_core::quantized::gguf_file::Value;
use candle_core::{DType, Device, Tensor};
use dinf_engine::architectures::qwen35_vision::{Qwen35VisionEncoder, VisionImage};
use dinf_engine::gguf_loader::read_gguf_metadata_map;
use dinf_engine::read_gguf_metadata;
use dinf_engine::vision::{Gemma4VisionEncoder, Llama4VisionEncoder};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

const REVISION: &str = "15fa3c493bfcd040b5f4dcb29e1c998a0846de16";
const MAX_VALUES: usize = 64 * 1024 * 1024;
const MAX_RELATIVE_L2: f64 = 0.01;
const MIN_COSINE: f64 = 0.9999;

#[derive(Deserialize)]
struct ImageReference {
    index: usize,
    shape_hwc: [usize; 3],
    pixels_file: String,
    embeddings_file: String,
    embedding_shape: [usize; 2],
}

#[derive(Deserialize)]
struct Reference {
    schema_version: u32,
    producer: String,
    revision: String,
    #[serde(default)]
    oracle_variant: Option<String>,
    #[serde(default)]
    oracle_patch_sha256: Option<String>,
    backend_requested: String,
    text_weights_loaded: bool,
    text_model_sha256: String,
    projector_sha256: String,
    text_architecture: String,
    projector_type: String,
    embedding_width: usize,
    input_rgb_file: String,
    input_shape_hwc: [usize; 3],
    pixel_format: String,
    embedding_format: String,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    images: Vec<ImageReference>,
}

#[derive(Serialize)]
struct Metrics {
    max_abs: f64,
    rms: f64,
    relative_l2: f64,
    cosine: f64,
}

fn metrics(expected: &[f32], actual: &[f32]) -> Result<Metrics> {
    ensure!(
        !expected.is_empty() && expected.len() == actual.len(),
        "embedding length differs"
    );
    ensure!(
        expected.iter().chain(actual).all(|v| v.is_finite()),
        "non-finite embeddings"
    );
    let (mut max_abs, mut difference, mut reference_norm, mut actual_norm, mut dot) =
        (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for (&r, &a) in expected.iter().zip(actual) {
        let (r, a) = (f64::from(r), f64::from(a));
        max_abs = max_abs.max((a - r).abs());
        difference += (a - r).powi(2);
        reference_norm += r * r;
        actual_norm += a * a;
        dot += r * a;
    }
    ensure!(
        reference_norm > 0.0 && actual_norm > 0.0,
        "zero embedding norm"
    );
    Ok(Metrics {
        max_abs,
        rms: (difference / expected.len() as f64).sqrt(),
        relative_l2: (difference / reference_norm).sqrt(),
        cosine: (dot / (reference_norm.sqrt() * actual_norm.sqrt())).clamp(-1.0, 1.0),
    })
}

fn shape_len(shape: &[usize]) -> Result<usize> {
    let len = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .context("shape overflow")?;
    ensure!(
        len > 0 && len <= MAX_VALUES,
        "reference tensor exceeds the 256MiB bound"
    );
    Ok(len)
}

fn local_file(reference: &Path, name: &str) -> Result<PathBuf> {
    let parent = reference
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = Path::new(name);
    ensure!(
        !name
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir)),
        "reference path traversal"
    );
    let canonical_parent = parent.canonicalize()?;
    // The producer records its output prefix, which can be absolute or
    // relative to the working directory. Also accept portable basenames.
    for path in [name.to_path_buf(), parent.join(name)] {
        if path.is_file() && path.canonicalize()?.parent() == Some(canonical_parent.as_path()) {
            return Ok(path);
        }
    }
    anyhow::bail!("reference data must stay in its JSON directory")
}

fn floats(path: &Path, count: usize) -> Result<Vec<f32>> {
    ensure!(
        count > 0 && count <= MAX_VALUES && path.metadata()?.len() == (count as u64) * 4,
        "reference float file size differs"
    );
    let bytes = std::fs::read(path)?;
    let values: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "non-finite reference floats"
    );
    Ok(values)
}

fn write_actual(path: &Path, values: &[f32]) -> Result<()> {
    ensure!(
        !values.is_empty() && values.iter().all(|v| v.is_finite()),
        "empty/non-finite actual embeddings"
    );
    let mut file = std::io::BufWriter::new(std::fs::File::create_new(path)?);
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let len = file.read(&mut buffer)?;
        if len == 0 {
            break;
        }
        digest.update(&buffer[..len]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn valid_hash(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn oracle_variant<'a>(variant: Option<&'a str>, patch: Option<&str>) -> Result<&'a str> {
    let variant = variant.unwrap_or("upstream");
    ensure!(
        match variant {
            "upstream" => patch.is_none(),
            "hf-contract-v1" => patch.is_some_and(valid_hash),
            _ => false,
        },
        "unsupported oracle variant or patch SHA256"
    );
    Ok(variant)
}

fn check_hash(path: &Path, expected: &str) -> Result<()> {
    ensure!(valid_hash(expected), "invalid model SHA256");
    ensure!(
        sha256(path)? == expected,
        "model/projector SHA256 differs: {}",
        path.display()
    );
    Ok(())
}

fn chw(values: Vec<f32>, shape: [usize; 3], device: &Device) -> Result<Tensor> {
    ensure!(
        shape[2] == 3 && values.len() == shape_len(&shape)?,
        "invalid HWC RGB shape"
    );
    Ok(
        Tensor::from_vec(values, (shape[0], shape[1], shape[2]), device)?
            .permute((2, 0, 1))?
            .contiguous()?,
    )
}

enum Encoder {
    Gemma(Gemma4VisionEncoder),
    Scout(Llama4VisionEncoder),
    Qwen(Qwen35VisionEncoder),
}

fn qwen_image_side(input: [usize; 3], image: [usize; 3], image_count: usize) -> Result<usize> {
    ensure!(
        matches!(input[0], 256 | 768)
            && input == [input[0], input[0], 3]
            && image == input
            && image_count == 1,
        "Qwen reference needs one 256x256 or 768x768 image with no resize"
    );
    Ok(input[0])
}

fn qwen_pixels(
    reference: &Reference,
    image: &ImageReference,
    raw: &[u8],
    pixels: &[f32],
    projector: &Path,
) -> Result<VisionImage> {
    let side = qwen_image_side(
        reference.input_shape_hwc,
        image.shape_hwc,
        reference.images.len(),
    )?;
    ensure!(
        raw.len() == side * side * 3 && pixels.len() == raw.len(),
        "Qwen raw/normalized image sizes differ"
    );
    let metadata = read_gguf_metadata_map(projector)?;
    for (name, expected) in [
        ("clip.vision.image_mean", reference.image_mean),
        ("clip.vision.image_std", reference.image_std),
    ] {
        let values = metadata
            .get(name)
            .context("missing Qwen normalization")?
            .to_vec()?;
        ensure!(values.len() == 3, "invalid Qwen normalization");
        for (value, expected) in values.iter().zip(expected) {
            let actual = match value {
                Value::F32(v) => *v,
                Value::F64(v) => *v as f32,
                _ => anyhow::bail!("invalid Qwen normalization type"),
            };
            ensure!(
                actual.to_bits() == expected.to_bits(),
                "Qwen normalization differs from oracle"
            );
        }
    }
    let raw_image = image::RgbImage::from_raw(side as u32, side as u32, raw.to_vec())
        .context("invalid RGB bytes")?;
    // This is the same no-resize call and F32 normalization as the public
    // encoder. Prove each input value matches before comparing its math.
    let resized = image::imageops::resize(
        &raw_image,
        side as u32,
        side as u32,
        image::imageops::FilterType::CatmullRom,
    );
    ensure!(
        resized.as_raw() == raw,
        "Qwen same-size resize changed pixels"
    );
    for (i, (&byte, &expected)) in raw.iter().zip(pixels).enumerate() {
        let actual =
            (f32::from(byte) / 255.0 - reference.image_mean[i % 3]) / reference.image_std[i % 3];
        ensure!(
            actual.to_bits() == expected.to_bits(),
            "Qwen preprocessed pixel differs at {i}"
        );
    }
    VisionImage::new(side, side, raw.to_vec())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        matches!(args.len(), 4 | 5),
        "usage: TEXT.gguf MMPROJ.gguf cpu|metal REFERENCE.json [ACTUAL_PREFIX]"
    );
    let (text, projector, path) = (
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[3]),
    );
    let reference: Reference = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(
        reference.schema_version == 1
            && reference.producer == "llama.cpp clip projector"
            && reference.revision == REVISION,
        "unsupported reference producer/schema/revision"
    );
    let variant = oracle_variant(
        reference.oracle_variant.as_deref(),
        reference.oracle_patch_sha256.as_deref(),
    )?;
    ensure!(
        !reference.text_weights_loaded && reference.backend_requested == args[2],
        "reference backend/scope differs"
    );
    ensure!(
        reference.pixel_format == "f32le-HWC-RGB"
            && reference.embedding_format == "f32le-token-major",
        "unsupported reference float layout"
    );
    ensure!(
        (1..=16).contains(&reference.images.len()),
        "unsupported tile count"
    );
    if reference.text_architecture == "qwen35" {
        qwen_image_side(
            reference.input_shape_hwc,
            reference.images[0].shape_hwc,
            reference.images.len(),
        )?;
    }
    ensure!(
        reference.image_mean.iter().all(|v| v.is_finite())
            && reference
                .image_std
                .iter()
                .all(|v| v.is_finite() && *v > 0.0),
        "invalid normalization"
    );
    for model in [text, projector] {
        ensure!(
            !read_gguf_metadata_map(model)?
                .keys()
                .any(|k| k.starts_with("split.")),
            "vision reference requires unsplit files"
        );
    }
    check_hash(text, &reference.text_model_sha256)?;
    check_hash(projector, &reference.projector_sha256)?;
    let metadata = read_gguf_metadata(text)?;
    ensure!(
        reference.text_architecture == metadata.architecture
            && reference.embedding_width == metadata.hidden_dim,
        "reference text geometry differs"
    );
    let raw_path = local_file(path, &reference.input_rgb_file)?;
    ensure!(
        reference.input_shape_hwc[2] == 3
            && raw_path.metadata()?.len() == shape_len(&reference.input_shape_hwc)? as u64,
        "source RGB shape differs"
    );
    let raw = std::fs::read(&raw_path)?;
    let device = match args[2].as_str() {
        "cpu" => Device::Cpu,
        #[cfg(target_os = "macos")]
        "metal" => Device::new_metal(0)?,
        _ => anyhow::bail!("backend unavailable"),
    };
    let encoder = match (
        metadata.architecture.as_str(),
        reference.projector_type.as_str(),
    ) {
        ("gemma4", "gemma4v") => Encoder::Gemma(Gemma4VisionEncoder::load(projector, &device)?),
        ("llama4", "llama4") => Encoder::Scout(Llama4VisionEncoder::load(projector, &device)?),
        ("qwen35", "qwen3vl_merger") => {
            Encoder::Qwen(Qwen35VisionEncoder::from_gguf(projector, &device)?)
        }
        _ => anyhow::bail!("unsupported projector/text pair"),
    };
    let mut reports = Vec::new();
    let mut passed = true;
    for (index, image) in reference.images.iter().enumerate() {
        ensure!(
            image.index == index
                && image.shape_hwc[2] == 3
                && image.embedding_shape[1] == reference.embedding_width,
            "image index or dimensions differ"
        );
        let pixels_path = local_file(path, &image.pixels_file)?;
        let embeddings_path = local_file(path, &image.embeddings_file)?;
        let pixels = floats(&pixels_path, shape_len(&image.shape_hwc)?)?;
        let expected = floats(&embeddings_path, shape_len(&image.embedding_shape)?)?;
        let actual = match &encoder {
            Encoder::Gemma(encoder) => encoder.encode(&chw(pixels, image.shape_hwc, &device)?)?,
            Encoder::Scout(encoder) => encoder.encode(&chw(pixels, image.shape_hwc, &device)?)?,
            Encoder::Qwen(encoder) => {
                let input = qwen_pixels(&reference, image, &raw, &pixels, projector)?;
                ensure!(
                    encoder.image_tokens_for_dimensions(input.width, input.height)?
                        == image.embedding_shape[0],
                    "Qwen output grid differs"
                );
                encoder.encode(&input)?.embeddings
            }
        };
        ensure!(
            actual.dims() == image.embedding_shape,
            "engine embedding shape differs"
        );
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let metrics = metrics(&expected, &actual)?;
        let actual_file = args
            .get(4)
            .map(|prefix| format!("{prefix}.image{index}.f32"));
        let actual_sha256 = if let Some(path) = actual_file.as_deref() {
            write_actual(Path::new(path), &actual)?;
            Some(sha256(Path::new(path))?)
        } else {
            None
        };
        let accepted = metrics.relative_l2 <= MAX_RELATIVE_L2 && metrics.cosine >= MIN_COSINE;
        passed &= accepted;
        reports.push(json!({"index":index,"shape":image.embedding_shape,"finite":true,"passed":accepted,
            "metrics":metrics,"actual_file":actual_file,"actual_sha256":actual_sha256,"pixels_sha256":sha256(&pixels_path)?,"reference_embeddings_sha256":sha256(&embeddings_path)?}));
    }
    device.synchronize()?;
    println!(
        "{}",
        json!({"passed":passed,"scope":"projector_component_identical_pixels","backend":args[2],
        "reference_revision":reference.revision,"reference_json_sha256":sha256(path)?,
        "oracle_variant":variant,"oracle_patch_sha256":reference.oracle_patch_sha256,
        "text_model_sha256":reference.text_model_sha256,"projector_sha256":reference.projector_sha256,
        "input_rgb_sha256":sha256(&raw_path)?,"input_shape_hwc":reference.input_shape_hwc,"text_weights_loaded":false,"independent_description_quality":false,
        "limits":{"max_relative_l2":MAX_RELATIVE_L2,"min_cosine":MIN_COSINE},"images":reports})
    );
    ensure!(passed, "projector reference comparison failed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_patch_identity_distinguishes_stock_and_corrected_producers() -> Result<()> {
        let hash = "a".repeat(64);
        assert_eq!(oracle_variant(None, None)?, "upstream");
        assert_eq!(oracle_variant(Some("upstream"), None)?, "upstream");
        assert_eq!(
            oracle_variant(Some("hf-contract-v1"), Some(&hash))?,
            "hf-contract-v1"
        );
        assert!(oracle_variant(None, Some(&hash)).is_err());
        assert!(oracle_variant(Some("hf-contract-v1"), None).is_err());
        assert!(oracle_variant(Some("hf-contract-v1"), Some("invalid")).is_err());
        assert!(oracle_variant(Some("unknown"), Some(&hash)).is_err());
        Ok(())
    }

    #[test]
    fn qwen_reference_requires_an_admitted_square_without_resize() -> Result<()> {
        for side in [256, 768] {
            assert_eq!(qwen_image_side([side, side, 3], [side, side, 3], 1)?, side);
        }
        for side in [0, 64, 512, 1024] {
            assert!(qwen_image_side([side, side, 3], [side, side, 3], 1).is_err());
        }
        assert!(qwen_image_side([768, 768, 3], [256, 256, 3], 1).is_err());
        assert!(qwen_image_side([768, 256, 3], [768, 256, 3], 1).is_err());
        assert!(qwen_image_side([768, 768, 4], [768, 768, 4], 1).is_err());
        assert!(qwen_image_side([768, 768, 3], [768, 768, 3], 2).is_err());
        Ok(())
    }

    #[test]
    fn metrics_use_reference_norm_and_reject_invalid_rows() -> Result<()> {
        let equal = metrics(&[3.0, 4.0], &[3.0, 4.0])?;
        assert_eq!((equal.relative_l2, equal.cosine), (0.0, 1.0));
        let scaled = metrics(&[3.0, 4.0], &[6.0, 8.0])?;
        assert_eq!(
            (scaled.max_abs, scaled.relative_l2, scaled.cosine),
            (4.0, 1.0, 1.0)
        );
        assert!((scaled.rms - 12.5f64.sqrt()).abs() < 1e-12);
        assert_eq!(metrics(&[1.0, 0.0], &[0.0, 1.0])?.cosine, 0.0);
        assert_eq!(metrics(&[1.0], &[-1.0])?.cosine, -1.0);
        for (r, a) in [
            (vec![], vec![]),
            (vec![0.0], vec![0.0]),
            (vec![1.0], vec![1.0, 2.0]),
            (vec![1.0], vec![f32::NAN]),
        ] {
            assert!(metrics(&r, &a).is_err());
        }
        Ok(())
    }

    #[test]
    fn hwc_to_chw_only_transposes_values() -> Result<()> {
        let tensor = chw(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3], &Device::Cpu)?;
        assert_eq!(
            tensor.to_vec3::<f32>()?,
            vec![
                vec![vec![1.0, 4.0]],
                vec![vec![2.0, 5.0]],
                vec![vec![3.0, 6.0]]
            ]
        );
        Ok(())
    }

    #[test]
    fn reference_files_reject_wrong_sizes_nonfinite_data_and_parent_paths() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let reference = directory.path().join("reference.json");
        let path = directory.path().join("pixels.f32");
        std::fs::write(&path, 1.0f32.to_le_bytes())?;
        assert_eq!(floats(&path, 1)?, [1.0]);
        assert!(floats(&path, 2).is_err());
        assert!(local_file(&reference, "../pixels.f32").is_err());
        assert_eq!(local_file(&reference, "pixels.f32")?, path);
        std::fs::write(&path, f32::NAN.to_le_bytes())?;
        assert!(floats(&path, 1).is_err());
        Ok(())
    }

    #[test]
    fn actual_dump_preserves_f32_bits_and_refuses_overwrite() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("actual.f32");
        let values = [1.25, -0.0, -17.5];
        write_actual(&path, &values)?;
        let saved = floats(&path, values.len())?;
        assert!(
            saved
                .iter()
                .zip(values)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert!(write_actual(&path, &[2.0]).is_err());
        assert!(write_actual(&directory.path().join("nan.f32"), &[f32::NAN]).is_err());
        assert!(!directory.path().join("nan.f32").exists());
        Ok(())
    }

    #[test]
    fn identity_check_hashes_changed_file_contents() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("weights.bin");
        std::fs::write(&path, [1, 2, 3, 4])?;
        let hash = sha256(&path)?;
        check_hash(&path, &hash)?;
        std::fs::write(&path, [1, 2, 3, 5])?;
        assert!(check_hash(&path, &hash).is_err());
        assert!(check_hash(&path, "not-a-sha256").is_err());
        Ok(())
    }
}
