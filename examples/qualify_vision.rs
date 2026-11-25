//! Paired projector + text-model checks using the engine's embedding API.
//! Usage: MODEL.gguf MMPROJ.gguf cpu|metal [TOKENS=16]. Admit both models first.
//! Covers fixed images, finite projections, greedy replay and request isolation.
//! It has no independent image-description or projection-quality oracle.
//!
//! DINF_QUALIFY_VISION_DRAFT=DRAFT.gguf also attaches a DFlash draft and
//! checks that image requests draft once the image is prefilled, through both
//! the whole-prompt embedding path and the sparse image-span path.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use dinf_engine::architectures::qwen35_vision::{Qwen35VisionEncoder, VisionGrid, VisionImage};
use dinf_engine::architectures::{
    AttentionSpan, ImageEmbeddingSpan, MropePositionIds, StageInput, StageOutput,
};
use dinf_engine::continuous::GenerationSession;
use dinf_engine::sampling::SamplingParams;
use dinf_engine::tokenizer::GgufTokenizer;
use dinf_engine::vision::{Gemma4VisionEncoder, Llama4VisionEncoder};
use dinf_engine::{LayerRange, PipelineStage, read_gguf_metadata};
use image::imageops::FilterType;
use serde_json::json;
use sha2::{Digest, Sha256};

enum Projector {
    Qwen(Qwen35VisionEncoder),
    Llama(Llama4VisionEncoder),
    Gemma(Gemma4VisionEncoder),
}

impl Projector {
    fn load(path: &Path, stage: &PipelineStage) -> Result<Self> {
        Ok(match stage.family() {
            "qwen35" => Self::Qwen(Qwen35VisionEncoder::from_gguf(path, stage.device())?),
            "llama4" => Self::Llama(Llama4VisionEncoder::load(path, stage.device())?),
            "gemma4" => Self::Gemma(Gemma4VisionEncoder::load(path, stage.device())?),
            family => anyhow::bail!("unsupported vision family {family}"),
        })
    }

    fn markers(&self) -> (&str, &str, &str) {
        match self {
            Self::Qwen(_) => ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"),
            Self::Llama(_) => ("<|image_start|>", "<|image|>", "<|image_end|>"),
            Self::Gemma(_) => ("<|image>", "<|image|>", "<image|>"),
        }
    }

    fn encode(
        &self,
        image: &image::RgbImage,
        device: &Device,
    ) -> Result<(Tensor, Option<VisionGrid>)> {
        let (encoded, grid, expected) = match self {
            Self::Qwen(encoder) => {
                let image = VisionImage::new(
                    image.width() as usize,
                    image.height() as usize,
                    image.as_raw().clone(),
                )?;
                let expected = encoder.image_tokens_for_dimensions(image.width, image.height)?;
                let output = encoder.encode(&image)?;
                (output.embeddings, Some(output.grid), expected)
            }
            Self::Llama(encoder) => {
                let size = encoder.image_size();
                let resized =
                    image::imageops::resize(image, size as u32, size as u32, FilterType::Triangle);
                (
                    encoder.encode(&chw(&resized, true, device)?)?,
                    None,
                    encoder.image_tokens(),
                )
            }
            Self::Gemma(encoder) => {
                let (width, height) =
                    encoder.target_image_size(image.width() as usize, image.height() as usize)?;
                let expected = encoder
                    .image_tokens_for_size(image.width() as usize, image.height() as usize)?;
                let resized = image::imageops::resize(
                    image,
                    width as u32,
                    height as u32,
                    FilterType::CatmullRom,
                );
                (
                    encoder.encode(&chw(&resized, false, device)?)?,
                    None,
                    expected,
                )
            }
        };
        ensure!(
            encoded.dim(0)? == expected && expected > 0,
            "wrong projected image row count"
        );
        Ok((encoded, grid))
    }
}

fn chw(image: &image::RgbImage, normalized: bool, device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(image.as_raw().len());
    for channel in 0..3 {
        values.extend(image.pixels().map(|p| {
            if normalized {
                f32::from(p[channel]) / 127.5 - 1.0
            } else {
                f32::from(p[channel]) / 255.0
            }
        }));
    }
    Ok(Tensor::from_vec(
        values,
        (3, image.height() as usize, image.width() as usize),
        device,
    )?)
}

fn fixture(changed: bool) -> image::RgbImage {
    image::RgbImage::from_fn(64, 64, |x, y| {
        let color = if x < 2 || y < 2 || x >= 62 || y >= 62 {
            [0, 0, 0]
        } else if (30..34).contains(&x) || (30..34).contains(&y) {
            [255, 255, 255]
        } else {
            match (x < 32, y < 32) {
                (true, true) => [255, 0, 0],
                (false, true) => [0, 255, 0],
                (true, false) => [0, 0, 255],
                (false, false) => [255, 255, 0],
            }
        };
        image::Rgb(if changed {
            color.map(|c| 255 - c)
        } else {
            color
        })
    })
}

fn finite_values(tensor: &Tensor) -> Result<Vec<f32>> {
    let values = tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    ensure!(
        !values.is_empty() && values.iter().all(|v| v.is_finite()),
        "empty/non-finite image projection"
    );
    Ok(values)
}

struct Prompt {
    image_start: usize,
    image_rows: usize,
    tokens: Vec<u32>,
    embeddings: Tensor,
    positions: Option<MropePositionIds>,
    spans: Vec<AttentionSpan>,
}

impl Prompt {
    fn session(&self, stage: &PipelineStage, count: usize) -> Result<GenerationSession> {
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: count,
            ..Default::default()
        };
        match &self.positions {
            Some(positions) => GenerationSession::new_with_positioned_embeddings_and_spans(
                stage,
                self.tokens.clone(),
                self.embeddings.clone(),
                positions.clone(),
                self.spans.clone(),
                params,
                vec![],
            ),
            None => GenerationSession::new_with_embeddings_and_spans(
                stage,
                self.tokens.clone(),
                self.embeddings.clone(),
                self.spans.clone(),
                params,
                vec![],
            ),
        }
    }
}

fn positions(prefix: usize, suffix: usize, grid: VisionGrid) -> Result<MropePositionIds> {
    let mut axes = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..prefix {
        for axis in &mut axes {
            axis.push(u32::try_from(i)?);
        }
    }
    let base = u32::try_from(prefix)?;
    for row in 0..grid.merged_rows {
        for col in 0..grid.merged_cols {
            axes[0].push(base);
            axes[1].push(
                base.checked_add(u32::try_from(row)?)
                    .context("position overflow")?,
            );
            axes[2].push(
                base.checked_add(u32::try_from(col)?)
                    .context("position overflow")?,
            );
        }
    }
    for i in 0..suffix {
        let pos = u32::try_from(prefix + grid.merged_rows.max(grid.merged_cols) + i)?;
        for axis in &mut axes {
            axis.push(pos);
        }
    }
    let [temporal, height, width] = axes;
    MropePositionIds::new(temporal, height, width)
}

fn prepare(
    stage: &PipelineStage,
    tokenizer: &GgufTokenizer,
    projector: &Projector,
    image: Tensor,
    grid: Option<VisionGrid>,
) -> Result<Prompt> {
    let (start, marker, end) = projector.markers();
    let ids: Vec<_> = [start, marker, end]
        .into_iter()
        .map(|s| {
            tokenizer
                .token_id(s)
                .with_context(|| format!("missing vision marker {s}"))
        })
        .collect::<Result<_>>()?;
    let text =
        format!("{start}{marker}{end}\nDescribe the colors and geometric layout of this image.");
    let text_tokens = tokenizer.encode_chat(&[("user".into(), text)])?;
    let places: Vec<_> = text_tokens
        .iter()
        .enumerate()
        .filter_map(|(i, &id)| (id == ids[1]).then_some(i))
        .collect();
    ensure!(places.len() == 1, "expected exactly one image placeholder");
    let offset = places[0];
    ensure!(
        offset > 0
            && text_tokens.get(offset - 1) == Some(&ids[0])
            && text_tokens.get(offset + 1) == Some(&ids[2]),
        "image marker boundaries differ"
    );
    let (rows, width) = image.dims2()?;
    ensure!(
        width == stage.hidden_dim(),
        "projector/text model width mismatch"
    );
    let text_embeddings = stage.embed_tokens(&text_tokens)?;
    let suffix = text_tokens.len() - offset - 1;
    let embeddings = Tensor::cat(
        &[
            text_embeddings.narrow(0, 0, offset)?,
            image.to_dtype(text_embeddings.dtype())?,
            text_embeddings.narrow(0, offset + 1, suffix)?,
        ],
        0,
    )?;
    let mut tokens = text_tokens[..offset].to_vec();
    tokens.extend(std::iter::repeat_n(ids[1], rows));
    tokens.extend_from_slice(&text_tokens[offset + 1..]);
    let positions = grid
        .map(|grid| positions(offset, suffix, grid))
        .transpose()?;
    let spans = if matches!(projector, Projector::Gemma(_)) {
        vec![AttentionSpan {
            start: offset,
            len: rows,
        }]
    } else {
        vec![]
    };
    ensure!(embeddings.dim(0)? == tokens.len(), "prompt shape differs");
    Ok(Prompt {
        image_start: offset,
        image_rows: rows,
        tokens,
        embeddings,
        positions,
        spans,
    })
}

fn clean(stage: &PipelineStage) -> Result<()> {
    stage.device().synchronize()?;
    ensure!(
        stage.total_kv_cache_tokens() == 0 && stage.total_kv_cache_allocation_tokens() == 0,
        "retained cache"
    );
    Ok(())
}

fn finish(
    stage: &mut PipelineStage,
    prompt: &Prompt,
    session: &mut GenerationSession,
    count: usize,
) -> Result<Vec<u32>> {
    ensure!(
        session.is_finished() && session.generated().len() == count,
        "short generation"
    );
    ensure!(
        session.next_position(stage) == prompt.tokens.len() + count - 1,
        "wrong cache position"
    );
    let tokens = session.generated().to_vec();
    session.discard_cache(stage);
    Ok(tokens)
}

fn run(stage: &mut PipelineStage, prompt: &Prompt, count: usize) -> Result<Vec<u32>> {
    let mut session = prompt.session(stage, count)?;
    while !session.is_finished() {
        session.step(stage)?;
    }
    let tokens = finish(stage, prompt, &mut session, count)?;
    clean(stage)?;
    Ok(tokens)
}

fn prompt_logits(stage: &mut PipelineStage, prompt: &Prompt) -> Result<Vec<f32>> {
    let input = match &prompt.positions {
        Some(positions) => StageInput::positioned_activations_with_spans(
            prompt.embeddings.clone(),
            prompt.spans.clone(),
            positions.clone(),
        ),
        None => StageInput::activations_with_spans(prompt.embeddings.clone(), prompt.spans.clone()),
    };
    let mut state = stage.new_session_state();
    let StageOutput::Logits(logits) =
        stage.prefill_with_session(&mut state, input, prompt.tokens.len(), 0)?
    else {
        anyhow::bail!("solo model returned activations");
    };
    let values = finite_values(&logits)?;
    drop(logits);
    drop(state);
    clean(stage)?;
    Ok(values)
}

/// Generate greedily from the image prompt with drafting on or off, through
/// the whole-prompt (`sparse == false`) or sparse image-span path.
fn run_draft(
    stage: &mut PipelineStage,
    prompt: &Prompt,
    count: usize,
    sparse: bool,
    draft: bool,
) -> Result<(Vec<u32>, serde_json::Value)> {
    let params = SamplingParams {
        temperature: 0.0,
        max_tokens: count,
        ..Default::default()
    };
    let mut session = if sparse {
        let span = ImageEmbeddingSpan {
            start: prompt.image_start,
            embeddings: prompt
                .embeddings
                .narrow(0, prompt.image_start, prompt.image_rows)?,
            bidirectional: !prompt.spans.is_empty(),
            digest: None,
        };
        match &prompt.positions {
            Some(positions) => GenerationSession::new_with_positioned_image_spans(
                stage,
                prompt.tokens.clone(),
                vec![span],
                positions.clone(),
                params,
                vec![],
            )?,
            None => GenerationSession::new_with_image_spans(
                stage,
                prompt.tokens.clone(),
                vec![span],
                params,
                vec![],
            )?,
        }
    } else {
        prompt.session(stage, count)?
    };
    session.set_speculative_decoding_enabled(stage, draft);
    while !session.is_finished() {
        session.step(stage)?;
    }
    let stats = session.speculative_stats();
    let report = json!({"passes":stats.passes,"drafted":stats.drafted_tokens,
        "accepted":stats.accepted_tokens,"fell_back":stats.fell_back});
    let tokens = finish(stage, prompt, &mut session, count)?;
    clean(stage)?;
    Ok((tokens, report))
}

fn check_draft(
    stage: &mut PipelineStage,
    prompt: &Prompt,
    draft: &Path,
    count: usize,
) -> Result<serde_json::Value> {
    let _ = draft;
    let mut options = stage.options().clone();
    // Greedy requests draft only when approximate verification is allowed.
    options.speculative.allow_approximate = true;
    options.speculative.fallback = false;
    stage.set_options(options)?;
    let mut report = serde_json::Map::new();
    for sparse in [false, true] {
        let (reference, _) = run_draft(stage, prompt, count, sparse, false)?;
        let (drafted, stats) = run_draft(stage, prompt, count, sparse, true)?;
        ensure!(
            stats["passes"].as_u64().unwrap_or(0) > 0
                && stats["accepted"].as_u64().unwrap_or(0) > 0,
            "image request did not draft after the image (sparse={sparse}): {stats}"
        );
        let matching = reference
            .iter()
            .zip(&drafted)
            .take_while(|(a, b)| a == b)
            .count();
        report.insert(
            if sparse {
                "sparse_spans"
            } else {
                "whole_prompt"
            }
            .into(),
            json!({"stats":stats,"tokens_matching_scalar_prefix":matching,"tokens":count}),
        );
    }
    Ok(report.into())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        (3..=4).contains(&args.len()),
        "usage: MODEL MMPROJ cpu|metal [TOKENS=16]"
    );
    let count = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(16);
    ensure!(count >= 4, "use at least 4 tokens");
    let device = match args[2].as_str() {
        "cpu" => Device::Cpu,
        #[cfg(target_os = "macos")]
        "metal" => Device::new_metal(0)?,
        _ => anyhow::bail!("backend unavailable"),
    };
    let model = Path::new(&args[0]);
    let metadata = read_gguf_metadata(model)?;
    let tokenizer = GgufTokenizer::from_gguf(model)?;
    let mut stage = PipelineStage::from_gguf(
        model,
        LayerRange::try_solo(metadata.num_layers)?,
        &device,
        None,
    )?;
    // A draft attaches before any inference. Greedy requests keep the scalar
    // path until `check_draft` allows approximate verification.
    let draft = std::env::var_os("DINF_QUALIFY_VISION_DRAFT").map(std::path::PathBuf::from);
    if let Some(draft) = &draft {
        stage.attach_speculative_draft(draft)?;
    }
    let projector = Projector::load(Path::new(&args[1]), &stage)?;
    let image = fixture(false);
    let (first, grid) = projector.encode(&image, &device)?;
    let values = finite_values(&first)?;
    let (repeat, repeated_grid) = projector.encode(&image, &device)?;
    ensure!(
        grid == repeated_grid && values == finite_values(&repeat)?,
        "projection replay changed"
    );
    drop(repeat);
    let (changed, changed_grid) = projector.encode(&fixture(true), &device)?;
    let changed_values = finite_values(&changed)?;
    ensure!(
        values.len() == changed_values.len() && values != changed_values,
        "projector ignored changed image"
    );
    let rows = first.dim(0)?;
    let first = prepare(&stage, &tokenizer, &projector, first, grid)?;
    let changed = prepare(&stage, &tokenizer, &projector, changed, changed_grid)?;
    drop(projector);
    device.synchronize()?;
    ensure!(
        first.tokens.len().max(changed.tokens.len()) + count <= metadata.context_length,
        "request exceeds context"
    );
    let first_logits = prompt_logits(&mut stage, &first)?;
    let changed_logits = prompt_logits(&mut stage, &changed)?;
    ensure!(
        first_logits.len() == metadata.vocab_size && changed_logits.len() == metadata.vocab_size,
        "wrong vocabulary dimension"
    );
    let image_logit_changed_count = first_logits
        .iter()
        .zip(&changed_logits)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    ensure!(
        image_logit_changed_count > 0,
        "decoder logits ignored changed image"
    );
    ensure!(
        first_logits == prompt_logits(&mut stage, &first)?,
        "prompt logits changed after another image"
    );
    let image_logit_max_delta = first_logits
        .iter()
        .zip(&changed_logits)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0f64, f64::max);
    let image_logit_rms_delta = (first_logits
        .iter()
        .zip(&changed_logits)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum::<f64>()
        / first_logits.len() as f64)
        .sqrt();
    let baseline = run(&mut stage, &first, count)?;
    let other = run(&mut stage, &changed, count)?;
    ensure!(
        baseline == run(&mut stage, &first, count)?,
        "another image changed replay"
    );
    let mut target = first.session(&stage, count)?;
    let mut cancelled = changed.session(&stage, count)?;
    for _ in 0..2 {
        target.step(&mut stage)?;
        cancelled.step(&mut stage)?;
    }
    cancelled.discard_cache(&mut stage);
    while !target.is_finished() {
        target.step(&mut stage)?;
    }
    ensure!(
        baseline == finish(&mut stage, &first, &mut target, count)?,
        "interleaved cancellation changed tokens"
    );
    clean(&stage)?;
    let draft_report = match &draft {
        Some(draft) => Some(check_draft(&mut stage, &first, draft, count)?),
        None => None,
    };
    let projection_bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    println!(
        "{}",
        json!({"passed":true,"model":args[0],"projector":args[1],"backend":args[2],
        "image_tokens":rows,"prompt_tokens":first.tokens.len(),"generated_tokens":baseline,
        "changed_image_logit_max_delta":image_logit_max_delta,
        "changed_image_logit_rms_delta":image_logit_rms_delta,
        "changed_image_logit_count":image_logit_changed_count,
        "text":tokenizer.decode(&baseline),"changed_image_tokens":other,
        "projection_sha256":format!("{:x}", Sha256::digest(projection_bytes)),
        "image_sha256":format!("{:x}", Sha256::digest(image.as_raw())),
        "checks":["finite_projection","projection_replay","projector_image_sensitivity","decoder_image_sensitivity","prompt_logit_replay","paired_text_generation","replay","interleaved_cancel"],
        "embedded_prefill":"atomic","independent_quality_oracle":false,"draft":draft_report})
    );
    Ok(())
}
