//! Inkling 2 vision: encode images with the projector and, optionally, ask
//! the decoder about each one.
//! Usage: MMPROJ.gguf cpu|metal OUT_DIR IMAGE... [--model MODEL.gguf]
//!        [--prompt TEXT] [--tokens N]
//! For each image it writes `STEM.rgb` (the encoded pixels, u8 HWC at the
//! planned size), `STEM.f32` (the `[tokens, decoder]` embeddings, little
//! endian) and `STEM.json` (the plan), which
//! `fixtures/inkling2_vision_reference.py` checks against the official
//! weights. With `--model` it prints the greedy answer to the prompt.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use dinf_engine::continuous::GenerationSession;
use dinf_engine::sampling::SamplingParams;
use dinf_engine::tokenizer::GgufTokenizer;
use dinf_engine::vision::Inkling2VisionEncoder;
use dinf_engine::{LayerRange, PipelineStage, read_gguf_metadata};
use image::imageops::FilterType;
use serde_json::json;

const CONTENT_IMAGE: &str = "<|content_image|>";
const IMAGE_TOKEN: &str = "<|unused_200054|>";
const END_MESSAGE: &str = "<|end_message|>";
const END_SAMPLING: &str = "<|content_model_end_sampling|>";
const MAX_IMAGE_TOKENS: usize = 1024;

struct Encoded {
    name: String,
    rows: Tensor,
}

fn encode(encoder: &Inkling2VisionEncoder, path: &Path, out: &Path) -> Result<Encoded> {
    let image = image::open(path)
        .with_context(|| format!("reading {}", path.display()))?
        .to_rgb8();
    let plan = encoder.plan(image.width() as usize, image.height() as usize)?;
    let resized = if image.dimensions() == (plan.width as u32, plan.height as u32) {
        image
    } else {
        image::imageops::resize(
            &image,
            plan.width as u32,
            plan.height as u32,
            FilterType::Triangle,
        )
    };
    let started = std::time::Instant::now();
    let rows = encoder.encode(resized.as_raw(), &plan)?;
    let values = rows.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let seconds = started.elapsed().as_secs_f64();
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "non-finite image embeddings"
    );
    let name = path
        .file_stem()
        .context("image path has no name")?
        .to_string_lossy()
        .into_owned();
    std::fs::write(out.join(format!("{name}.rgb")), resized.as_raw())?;
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    std::fs::write(out.join(format!("{name}.f32")), bytes)?;
    let plan_json = json!({
        "image": path, "width": plan.width, "height": plan.height,
        "padded_width": plan.padded_width, "padded_height": plan.padded_height,
        "tokens": plan.tokens, "decoder_dim": encoder.decoder_dim(),
    });
    std::fs::write(
        out.join(format!("{name}.json")),
        serde_json::to_vec_pretty(&plan_json)?,
    )?;
    println!(
        "{}",
        json!({"image": name, "tokens": plan.tokens, "encode_seconds": seconds,
            "size": [plan.width, plan.height]})
    );
    Ok(Encoded { name, rows })
}

fn answer(
    stage: &mut PipelineStage,
    tokenizer: &GgufTokenizer,
    image: &Tensor,
    prompt: &str,
    count: usize,
) -> Result<String> {
    let marker = format!("{CONTENT_IMAGE}{IMAGE_TOKEN}{END_MESSAGE}");
    let text_tokens = tokenizer
        .encode_chat_with_image_marker(&[("user".into(), format!("{marker}{prompt}"))], &marker)?;
    let id = |token: &str| {
        tokenizer
            .token_id(token)
            .with_context(|| format!("tokenizer lacks {token}"))
    };
    let image_id = id(IMAGE_TOKEN)?;
    let places = text_tokens
        .iter()
        .enumerate()
        .filter_map(|(index, &token)| (token == image_id).then_some(index))
        .collect::<Vec<_>>();
    ensure!(places.len() == 1, "expected one image token");
    let at = places[0];
    ensure!(
        text_tokens[at - 1] == id(CONTENT_IMAGE)? && text_tokens[at + 1] == id(END_MESSAGE)?,
        "image part boundaries differ"
    );
    let text = stage.embed_tokens(&text_tokens)?;
    let rows = image.to_dtype(text.dtype())?;
    let suffix = text_tokens.len() - at - 1;
    let embeddings = Tensor::cat(
        &[
            text.narrow(0, 0, at)?,
            rows.clone(),
            text.narrow(0, at + 1, suffix)?,
        ],
        0,
    )?;
    let mut tokens = text_tokens[..at].to_vec();
    tokens.extend(std::iter::repeat_n(image_id, rows.dim(0)?));
    tokens.extend_from_slice(&text_tokens[at + 1..]);
    let params = SamplingParams {
        temperature: 0.0,
        max_tokens: count,
        ..Default::default()
    };
    let stop = id(END_SAMPLING)?;
    let mut session = GenerationSession::new_with_embeddings_and_spans(
        stage,
        tokens,
        embeddings,
        vec![],
        params,
        vec![stop],
    )?;
    while !session.is_finished() {
        session.step(stage)?;
    }
    let generated = session.generated().to_vec();
    session.discard_cache(stage);
    Ok(tokenizer.decode(&generated))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut take = |flag: &str| -> Option<String> {
        let at = args.iter().position(|arg| arg == flag)?;
        let value = args.get(at + 1).cloned();
        args.drain(at..(at + 2).min(args.len()));
        value
    };
    let model = take("--model").map(PathBuf::from);
    let prompt = take("--prompt").unwrap_or_else(|| "Describe this image.".into());
    let count = take("--tokens")
        .map(|n| n.parse())
        .transpose()?
        .unwrap_or(160);
    ensure!(
        args.len() >= 4,
        "usage: MMPROJ cpu|metal|cuda OUT_DIR IMAGE... [--model MODEL] [--prompt TEXT] [--tokens N]"
    );
    let device = match args[1].as_str() {
        "cpu" => Device::Cpu,
        #[cfg(target_os = "macos")]
        "metal" => Device::new_metal(0)?,
        #[cfg(feature = "cuda")]
        "cuda" => Device::new_cuda(0)?,
        other => anyhow::bail!("backend {other} unavailable"),
    };
    let out = PathBuf::from(&args[2]);
    std::fs::create_dir_all(&out)?;
    let encoder = Inkling2VisionEncoder::load(Path::new(&args[0]), &device, MAX_IMAGE_TOKENS)?;
    let images = args[3..]
        .iter()
        .map(|path| encode(&encoder, Path::new(path), &out))
        .collect::<Result<Vec<_>>>()?;
    drop(encoder);
    let Some(model) = model else {
        return Ok(());
    };
    let metadata = read_gguf_metadata(&model)?;
    let tokenizer = GgufTokenizer::from_gguf(&model)?;
    let mut stage = PipelineStage::from_gguf(
        &model,
        LayerRange::try_solo(metadata.num_layers)?,
        &device,
        None,
    )?;
    for image in &images {
        let started = std::time::Instant::now();
        let text = answer(&mut stage, &tokenizer, &image.rows, &prompt, count)?;
        println!(
            "{}",
            json!({"image": image.name, "prompt": prompt, "seconds": started.elapsed().as_secs_f64(),
                "answer": text})
        );
    }
    Ok(())
}
