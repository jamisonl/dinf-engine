//! Full real-draft component oracle. This file is included only under cfg(test).
//! Set DINF_DFLASH_REFERENCE, DINF_DFLASH_TARGET, and DINF_DFLASH_DRAFT.
//! The JSON producer uses only draft + target embedding/head weights and exact
//! synthetic feature rows; this gate makes no target-text quality claim.
use super::*;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

const MAX_RELATIVE_L2: f64 = 0.01;
const MIN_COSINE: f64 = 0.9999;

#[derive(Default, Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum InputMode {
    #[default]
    ProductionEmbedding,
    OracleNoiseEmbeddings,
}

#[derive(Deserialize)]
struct Reference {
    schema_version: u32,
    #[serde(default)]
    input_mode: InputMode,
    producer: String,
    revision: String,
    backend: String,
    target_sha256: String,
    draft_sha256: String,
    derived_sha256: String,
    rows: usize,
    block: usize,
    anchor: u32,
    mask: u32,
    width: usize,
    layers: usize,
    vocab: usize,
    top_k: usize,
    prefix: String,
    cache_dtype: String,
    features: String,
    full_target_loaded: bool,
    // Added by the byte-copy validator after its full/range hashes pass.
    derived_path: std::path::PathBuf,
    copy_manifest_sha256: String,
    copy_manifest_path: std::path::PathBuf,
}
fn sha(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut b = vec![0; 1024 * 1024];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn check_sha(path: &Path, expected: &str) -> Result<()> {
    anyhow::ensure!(
        expected.len() == 64
            && expected
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "invalid SHA256"
    );
    anyhow::ensure!(
        sha(path)? == expected,
        "fixture/model hash differs: {}",
        path.display()
    );
    Ok(())
}
fn values(path: &Path, n: usize) -> Result<Vec<f32>> {
    anyhow::ensure!(
        n > 0 && n <= 64 * 1024 * 1024 && path.metadata()?.len() == n as u64 * 4,
        "invalid float size"
    );
    let b = std::fs::read(path)?;
    let v: Vec<_> = b
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    anyhow::ensure!(v.iter().all(|x| x.is_finite()), "nonfinite floats");
    Ok(v)
}
fn dump(path: &Path, v: &[f32]) -> Result<String> {
    anyhow::ensure!(v.iter().all(|x| x.is_finite()), "nonfinite actual");
    let mut f = std::io::BufWriter::new(std::fs::File::create_new(path)?);
    for x in v {
        f.write_all(&x.to_le_bytes())?;
    }
    f.flush()?;
    sha(path)
}
fn flat(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
}
fn metrics(r: &[f32], a: &[f32]) -> Result<serde_json::Value> {
    anyhow::ensure!(
        !r.is_empty() && r.len() == a.len() && r.iter().chain(a).all(|x| x.is_finite()),
        "invalid vectors"
    );
    let (mut d, mut rn, mut an, mut dot, mut max) = (0.0f64, 0.0, 0.0, 0.0, 0.0f64);
    for (&r, &a) in r.iter().zip(a) {
        let (r, a) = (f64::from(r), f64::from(a));
        d += (a - r).powi(2);
        rn += r * r;
        an += a * a;
        dot += r * a;
        max = max.max((a - r).abs());
    }
    anyhow::ensure!(rn > 0.0 && an > 0.0, "zero norm");
    let relative_l2 = (d / rn).sqrt();
    let cosine = (dot / (rn * an).sqrt()).clamp(-1.0, 1.0);
    Ok(
        json!({"relative_l2":relative_l2,"cosine":cosine,"max_abs":max,"rms":(d/r.len() as f64).sqrt(),"passed":relative_l2<=MAX_RELATIVE_L2&&cosine>=MIN_COSINE}),
    )
}
fn logit_metrics(r: &[f32], a: &[f32], vocab: usize) -> Result<serde_json::Value> {
    anyhow::ensure!(
        !r.is_empty() && vocab > 0 && r.len() == a.len() && r.len().is_multiple_of(vocab),
        "logit dimensions"
    );
    let (mut matched, mut total, mut max_kl) = (0usize, 0.0f64, 0.0f64);
    for (r, a) in r.chunks_exact(vocab).zip(a.chunks_exact(vocab)) {
        anyhow::ensure!(r.iter().chain(a).all(|v| v.is_finite()), "nonfinite logits");
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold(0, |best, (i, x)| if *x > v[best] { i } else { best })
        };
        matched += usize::from(argmax(r) == argmax(a));
        let rm = f64::from(r[argmax(r)]);
        let am = f64::from(a[argmax(a)]);
        let rz: f64 = r.iter().map(|v| (f64::from(*v) - rm).exp()).sum();
        let az: f64 = a.iter().map(|v| (f64::from(*v) - am).exp()).sum();
        let mut kl = 0.0;
        for (&r, &a) in r.iter().zip(a) {
            let lp = f64::from(r) - rm - rz.ln();
            let lq = f64::from(a) - am - az.ln();
            kl += lp.exp() * (lp - lq);
        }
        kl = kl.max(0.0);
        total += kl;
        max_kl = max_kl.max(kl);
    }
    let rows = r.len() / vocab;
    let mean_kl = total / rows as f64;
    Ok(
        json!({"matched":matched,"rows":rows,"mean_kl":mean_kl,"max_kl":max_kl,"passed":matched as f64/rows as f64>=0.95&&mean_kl<=0.02&&max_kl<=0.1}),
    )
}
// Keep this diagnostic independent of the production lookup fix. It uploads
// only selected packed rows and measures the existing Candle Metal row gather;
// it never allocates another full embedding table.
fn selected_packed_q4k_rows(
    embedding: &QTensor,
    tokens: &[u32],
    device: &Device,
) -> Result<Arc<QTensor>> {
    anyhow::ensure!(
        embedding.device().is_cpu() && embedding.dtype() == GgmlDType::Q4K,
        "row probe needs packed CPU Q4K"
    );
    let (rows, width) = embedding.shape().dims2()?;
    let row_bytes = width / GgmlDType::Q4K.block_size() * GgmlDType::Q4K.type_size();
    let data = embedding.data()?;
    anyhow::ensure!(
        data.len() == rows * row_bytes,
        "row probe packed size differs"
    );
    let mut selected = Vec::with_capacity(tokens.len() * row_bytes);
    for &token in tokens {
        anyhow::ensure!(
            (token as usize) < rows,
            "row probe token exceeds vocabulary"
        );
        let offset = token as usize * row_bytes;
        selected.extend_from_slice(&data[offset..offset + row_bytes]);
    }
    let storage =
        candle_core::quantized::QStorage::from_data(selected.into(), device, GgmlDType::Q4K)?;
    Ok(Arc::new(QTensor::new(storage, (tokens.len(), width))?))
}

fn fixture_file(r: &Reference, suffix: &str, directory: &Path) -> Result<std::path::PathBuf> {
    let path = std::path::PathBuf::from(format!("{}.{}", r.prefix, suffix));
    anyhow::ensure!(
        path.canonicalize()?.parent() == Some(directory),
        "reference path outside JSON directory"
    );
    Ok(path)
}

#[test]
#[ignore = "requires pinned full DFlash2/target embedding weights and independent component fixture"]
fn real_dflash_matches_independent_component_reference() -> Result<()> {
    let reference_path = std::path::PathBuf::from(std::env::var("DINF_DFLASH_REFERENCE")?);
    let target = std::path::PathBuf::from(std::env::var("DINF_DFLASH_TARGET")?);
    let draft_path = std::path::PathBuf::from(std::env::var("DINF_DFLASH_DRAFT")?);
    let r: Reference = serde_json::from_slice(&std::fs::read(&reference_path)?)?;
    anyhow::ensure!(
        r.schema_version == 1
            && r.producer == "llama.cpp DFlash2 component"
            && r.revision == "de5122d879d00d2895c36be7e0cee94105aeeabd",
        "unsupported oracle"
    );
    anyhow::ensure!(
        !r.full_target_loaded
            && r.features == "synthetic_exact_dyadic_v1"
            && r.cache_dtype == "F16",
        "wrong component scope"
    );
    anyhow::ensure!(
        (1..=4096).contains(&r.rows)
            && (2..=256).contains(&r.block)
            && (2..=64).contains(&r.layers)
            && (1..=16384).contains(&r.width)
            && r.vocab > 7
            && r.vocab <= 1_000_000
            && (1..=64).contains(&r.top_k)
            && r.anchor == 7
            && r.width >= r.top_k + r.top_k * r.top_k,
        "invalid geometry"
    );
    for (p, h) in [
        (&target, &r.target_sha256),
        (&draft_path, &r.draft_sha256),
        (&r.derived_path, &r.derived_sha256),
        (&r.copy_manifest_path, &r.copy_manifest_sha256),
    ] {
        check_sha(p, h)?;
    }
    let directory = reference_path
        .canonicalize()?
        .parent()
        .unwrap()
        .to_path_buf();
    let features = values(
        &fixture_file(&r, "features.f32", &directory)?,
        r.rows * r.layers * r.width,
    )?;
    anyhow::ensure!(
        features
            .iter()
            .enumerate()
            .all(|(i, v)| v.to_bits()
                == ((((i * 17 + 23) % 257) as i32 - 128) as f32 / 128.0).to_bits()),
        "synthetic feature bits differ"
    );
    let expected_input_embeddings = values(
        &fixture_file(&r, "input_embeddings.f32", &directory)?,
        r.block * r.width,
    )?;
    let expected_hidden = values(
        &fixture_file(&r, "hidden.f32", &directory)?,
        r.block * r.width,
    )?;
    let expected_logits = values(
        &fixture_file(&r, "logits.f32", &directory)?,
        r.block * r.vocab,
    )?;
    let lattice = values(
        &fixture_file(&r, "lattice.f32", &directory)?,
        r.block * r.width,
    )?;
    let device = match r.backend.as_str() {
        "cpu" => Device::Cpu,
        #[cfg(target_os = "macos")]
        "metal" => Device::new_metal(0)?,
        _ => anyhow::bail!("unsupported backend"),
    };
    let mut f = std::fs::File::open(&target)?;
    let content = gguf_file::Content::read(&mut f)?;
    let embedding_device = if device.is_metal()
        && content.tensor_infos["token_embd.weight"].ggml_dtype == GgmlDType::Q4K
    {
        &Device::Cpu
    } else {
        &device
    };
    let embedding = Arc::new(content.tensor(&mut f, "token_embd.weight", embedding_device)?);
    let output = Arc::new(content.tensor(&mut f, "output.weight", &device)?);
    let output = match DraftQMatMul::from_qtensor(output, &device)? {
        DraftQMatMul::Candle(m) => super::super::qwen35::Qwen35QMatMul::Candle(m),
        #[cfg(target_os = "macos")]
        DraftQMatMul::MetalTensor(m) => super::super::qwen35::Qwen35QMatMul::MetalTensor(m),
    };
    let draft = DFlash2Draft::from_gguf(
        &draft_path,
        &device,
        SpeculativeSharedWeights {
            token_embedding: embedding,
            output,
        },
    )?;
    anyhow::ensure!(
        draft.hidden_size == r.width
            && draft.target_layers.len() == r.layers
            && draft.block_size >= r.block
            && draft.mask_token_id == r.mask
            && draft.selector_top_k == r.top_k
            && r.rows + r.block <= draft.context_length(),
        "draft fixture geometry differs"
    );
    let fused = Tensor::from_vec(features, (r.rows, r.layers * r.width), &device)?;
    let layer_inputs = (0..r.layers)
        .map(|i| fused.narrow(1, i * r.width, r.width)?.contiguous())
        .collect::<candle_core::Result<Vec<_>>>()?;
    let mut cache = draft.new_cache();
    draft.inject(&mut cache, &layer_inputs, 0)?;
    let mut tokens = vec![r.mask; r.block];
    tokens[0] = r.anchor;
    let production_input = super::super::qwen35::packed_embedding_rows(
        &draft.token_embedding,
        tokens.as_slice(),
        &device,
    )?
    .reshape((1, r.block, r.width))?;
    let actual_input = flat(&production_input)?;
    let input_equal = actual_input
        .iter()
        .zip(&expected_input_embeddings)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let input_metrics = metrics(&expected_input_embeddings, &actual_input)?;
    let output_prefix = format!("{}.engine-{}", r.prefix, r.backend);
    let input_path = std::path::PathBuf::from(format!("{output_prefix}.input_embeddings.f32"));
    let input_hash = dump(&input_path, &actual_input)?;
    println!(
        "DFLASH_EMBEDDINGS {}",
        json!({
            "input_mode":r.input_mode,"bitwise_equal":input_equal,
            "different_bits":actual_input.iter().zip(&expected_input_embeddings).filter(|(a,b)|a.to_bits()!=b.to_bits()).count(),
            "metrics":input_metrics,"path":input_path,"sha256":input_hash,
        })
    );
    let original_device_gather = if device.is_metal()
        && draft.token_embedding.device().is_cpu()
        && draft.token_embedding.dtype() == GgmlDType::Q4K
    {
        let probe = selected_packed_q4k_rows(&draft.token_embedding, &tokens, &device)?;
        let probe_ids = (0..r.block as u32).collect::<Vec<_>>();
        let probe_values = flat(&probe.embedding(&Tensor::new(probe_ids.as_slice(), &device)?)?)?;
        let path = std::path::PathBuf::from(format!("{output_prefix}.original_metal_gather.f32"));
        Some(json!({"path":path,"sha256":dump(&path,&probe_values)?,
            "metrics":metrics(&expected_input_embeddings,&probe_values)?,
            "different_bits":probe_values.iter().zip(&expected_input_embeddings).filter(|(a,b)|a.to_bits()!=b.to_bits()).count(),
            "packed_bytes":probe.data()?.len(),"diagnostic_only":true}))
    } else {
        None
    };
    if r.input_mode == InputMode::ProductionEmbedding {
        anyhow::ensure!(
            input_equal,
            "anchor/mask input embedding bits differ; numeric comparison is not on identical inputs"
        );
    }
    let run_noise = |mut hidden: Tensor| -> Result<(Tensor, Tensor)> {
        for (i, block) in draft.blocks.iter().enumerate() {
            hidden = block.forward_noise(
                &hidden,
                cache.layers[i].as_ref(),
                r.rows,
                &draft.rope_cos,
                &draft.rope_sin,
                draft.heads,
                draft.kv_heads,
                draft.head_dim,
            )?;
        }
        let hidden = draft.output_norm.forward(&hidden)?;
        let logits = draft.output_logits(&hidden.squeeze(0)?)?;
        Ok((hidden, logits))
    };
    // Verify the wrapper on its own production rows, even when the independent
    // math comparison deliberately supplies the oracle's saved input rows.
    let (production_hidden, production_logits) = run_noise(production_input)?;
    let production_path =
        draft.select_path(r.anchor, &production_hidden, &production_logits, r.block)?;
    anyhow::ensure!(
        production_path == draft.draft(&cache, r.anchor, r.rows, r.block)?,
        "inspection differs from production draft path"
    );
    let (hidden, logits) = if r.input_mode == InputMode::OracleNoiseEmbeddings {
        run_noise(Tensor::from_slice(
            &expected_input_embeddings,
            (1, r.block, r.width),
            &device,
        )?)?
    } else {
        (production_hidden, production_logits)
    };
    let actual_path = draft.select_path(r.anchor, &hidden, &logits, r.block)?;
    let selector = draft.selector_hidden.forward(&hidden)?.squeeze(0)?;
    let mut expected_scores = Vec::new();
    let mut actual_scores = Vec::new();
    let mut reference_tokens = Vec::new();
    let mut predecessor = 0usize;
    for position in 1..r.block {
        let row = &lattice[position * r.width..(position + 1) * r.width];
        let parse_ids = |slice: &[f32]| -> Result<Vec<u32>> {
            anyhow::ensure!(
                slice
                    .iter()
                    .all(|v| *v >= 0.0 && *v < r.vocab as f32 && v.fract() == 0.0),
                "invalid selector ID"
            );
            Ok(slice.iter().map(|v| *v as u32).collect())
        };
        let ids = parse_ids(&row[..r.top_k])?;
        let prior = if position == 1 {
            vec![r.anchor]
        } else {
            parse_ids(&lattice[(position - 1) * r.width..(position - 1) * r.width + r.top_k])?
        };
        let successors = draft
            .selector_successor
            .embedding(&Tensor::new(ids.as_slice(), &device)?)?
            .to_device(&device)?;
        let predecessors = draft
            .selector_predecessor
            .embedding(&Tensor::new(prior.as_slice(), &device)?)?
            .to_device(&device)?;
        let conditioned = predecessors.broadcast_mul(&selector.i(position)?)?;
        let unary = logits
            .i(position)?
            .index_select(&Tensor::new(ids.as_slice(), &device)?, 0)?;
        let mut scores = conditioned
            .matmul(&successors.transpose(0, 1)?)?
            .broadcast_add(&unary)?;
        if position == 1 {
            scores = scores.broadcast_as((r.top_k, r.top_k))?;
        }
        actual_scores.extend(flat(&scores)?);
        expected_scores.extend_from_slice(&row[r.top_k..r.top_k + r.top_k * r.top_k]);
        let path_scores =
            &row[r.top_k + predecessor * r.top_k..r.top_k + (predecessor + 1) * r.top_k];
        predecessor =
            path_scores.iter().enumerate().fold(
                0,
                |best, (i, v)| if *v > path_scores[best] { i } else { best },
            );
        reference_tokens.push(ids[predecessor]);
    }
    let hidden = flat(&hidden)?;
    let logits = flat(&logits)?;
    let hidden_metrics = metrics(&expected_hidden, &hidden)?;
    let score_metrics = metrics(&expected_scores, &actual_scores)?;
    let logits_metrics = logit_metrics(&expected_logits, &logits, r.vocab)?;
    let dumps = [
        ("hidden", &hidden),
        ("logits", &logits),
        ("selector", &actual_scores),
    ]
    .into_iter()
    .map(|(kind, v)| {
        let path = std::path::PathBuf::from(format!("{output_prefix}.{kind}.f32"));
        Ok(json!({"kind":kind,"sha256":dump(&path,v)?,"path":path}))
    })
    .collect::<Result<Vec<_>>>()?;
    let controlled_component_passed = hidden_metrics["passed"] == true
        && score_metrics["passed"] == true
        && logits_metrics["passed"] == true
        && actual_path == reference_tokens;
    // A controlled-input pass alone does not qualify production embeddings.
    let passed = controlled_component_passed && input_equal;
    device.synchronize()?;
    println!(
        "DFLASH_REFERENCE {}",
        json!({"passed":passed,"controlled_component_passed":controlled_component_passed,"input_mode":r.input_mode,"production_embedding_bitwise_equal":input_equal,"production_embedding_metrics":input_metrics,"production_embedding_dump":{"path":input_path,"sha256":input_hash},"production_wrapper_tokens":production_path,"original_device_gather":original_device_gather,"backend":r.backend,"rows":r.rows,"block":r.block,"hidden":hidden_metrics,"selector_same_candidate_ids":score_metrics,"logits":logits_metrics,"actual_tokens":actual_path,"reference_tokens":reference_tokens,"dumps":dumps,"component_only":true,"actual_backend":if device.is_metal(){"metal"}else{"cpu"},"limits":{"relative_l2":MAX_RELATIVE_L2,"cosine":MIN_COSINE,"argmax_fraction":0.95,"mean_kl":0.02,"max_kl":0.1}})
    );
    anyhow::ensure!(passed, "independent DFlash component gate failed");
    Ok(())
}

#[test]
fn component_metrics_use_reference_norm_and_fixed_logit_gates() -> Result<()> {
    assert_eq!(metrics(&[3.0, 4.0], &[6.0, 8.0])?["relative_l2"], 1.0);
    assert_eq!(metrics(&[3.0, 4.0], &[3.0, 4.0])?["passed"], true);
    assert!(metrics(&[0.0], &[0.0]).is_err());
    assert!(metrics(&[1.0], &[f32::NAN]).is_err());
    assert_eq!(
        logit_metrics(&[1.0, 2.0, 3.0, 0.0], &[1.0, 2.0, 3.0, 0.0], 2)?["passed"],
        true
    );
    assert_eq!(logit_metrics(&[1.0, 2.0], &[2.0, 1.0], 2)?["passed"], false);
    assert!(logit_metrics(&[], &[], 0).is_err());
    assert!(logit_metrics(&[1.0], &[f32::INFINITY], 1).is_err());
    Ok(())
}
#[test]
fn component_float_fixture_preserves_bits_and_refuses_overwrite() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("sample.f32");
    let original = [1.25, -0.0, -17.5];
    let h = dump(&path, &original)?;
    check_sha(&path, &h)?;
    let saved = values(&path, original.len())?;
    assert!(
        saved
            .iter()
            .zip(original)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(dump(&path, &[2.0]).is_err());
    assert!(values(&path, 4).is_err());
    assert!(dump(&directory.path().join("nan.f32"), &[f32::NAN]).is_err());
    Ok(())
}

#[test]
fn component_input_mode_requires_an_explicit_known_contract() -> Result<()> {
    assert!(matches!(
        InputMode::default(),
        InputMode::ProductionEmbedding
    ));
    assert!(matches!(
        serde_json::from_str::<InputMode>("\"oracle_noise_embeddings\"")?,
        InputMode::OracleNoiseEmbeddings
    ));
    assert!(serde_json::from_str::<InputMode>("\"unverified_embeddings\"").is_err());
    Ok(())
}

#[test]
fn component_row_probe_copies_only_selected_q4k_blocks() -> Result<()> {
    let mut raw = vec![0u8; 2 * 144];
    raw[..144].fill(17);
    raw[144..].fill(23);
    let storage =
        candle_core::quantized::QStorage::from_data(raw.into(), &Device::Cpu, GgmlDType::Q4K)?;
    let original = QTensor::new(storage, (2, 256))?;
    let selected = selected_packed_q4k_rows(&original, &[1, 0, 1], &Device::Cpu)?;
    assert_eq!(selected.shape().dims(), &[3, 256]);
    let bytes = selected.data()?;
    assert_eq!(bytes.len(), 3 * 144);
    assert!(bytes[..144].iter().all(|b| *b == 23));
    assert!(bytes[144..288].iter().all(|b| *b == 17));
    assert!(bytes[288..].iter().all(|b| *b == 23));
    assert!(selected_packed_q4k_rows(&original, &[2], &Device::Cpu).is_err());
    Ok(())
}
