use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::{DType, Device, Tensor};
use dinf_engine::architectures::{StageInput, StageOutput};
use dinf_engine::turboquant::{QuantizedKvCache, TurboQuantConfig};
use dinf_engine::{
    Engine, GenerationOutput, GenerationParams, LayerRange, PipelineStage, StageMemoryProfile,
};

struct TinyFixture<'a> {
    architecture: &'a str,
    hidden_dim: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    metadata_head_dim: u64,
    omit: Vec<String>,
    mismatched_projection: Option<String>,
    rope_freq_shape: Option<Vec<usize>>,
}

impl<'a> TinyFixture<'a> {
    fn new(
        architecture: &'a str,
        hidden_dim: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        metadata_head_dim: u64,
    ) -> Self {
        Self {
            architecture,
            hidden_dim,
            num_attention_heads,
            num_kv_heads,
            metadata_head_dim,
            omit: Vec::new(),
            mismatched_projection: None,
            rope_freq_shape: None,
        }
    }

    fn omit(mut self, names: &[&str]) -> Self {
        self.omit = names.iter().map(|name| (*name).to_string()).collect();
        self
    }

    fn mismatch(mut self, name: &str) -> Self {
        self.mismatched_projection = Some(name.to_string());
        self
    }

    fn rope_freq_shape(mut self, shape: Vec<usize>) -> Self {
        self.rope_freq_shape = Some(shape);
        self
    }
}

fn write_tiny_fixture(dir: &Path, fixture: TinyFixture<'_>) -> anyhow::Result<PathBuf> {
    let TinyFixture {
        architecture,
        hidden_dim,
        num_attention_heads,
        num_kv_heads,
        metadata_head_dim,
        omit,
        mismatched_projection,
        rope_freq_shape,
    } = fixture;
    let path = dir.join(format!("{architecture}.gguf"));
    let prefix = architecture.to_string();
    let key_length = if metadata_head_dim <= u32::MAX as u64 {
        gguf_file::Value::U32(metadata_head_dim as u32)
    } else {
        gguf_file::Value::U64(metadata_head_dim)
    };
    let metadata = vec![
        (
            "general.architecture".to_string(),
            gguf_file::Value::String(prefix.clone()),
        ),
        (format!("{prefix}.block_count"), gguf_file::Value::U32(1)),
        (
            format!("{prefix}.embedding_length"),
            gguf_file::Value::U32(hidden_dim as u32),
        ),
        (
            format!("{prefix}.attention.head_count"),
            gguf_file::Value::U32(num_attention_heads as u32),
        ),
        (
            format!("{prefix}.attention.head_count_kv"),
            gguf_file::Value::U32(num_kv_heads as u32),
        ),
        (format!("{prefix}.attention.key_length"), key_length),
        (
            format!("{prefix}.feed_forward_length"),
            gguf_file::Value::U32((hidden_dim * 2) as u32),
        ),
        (
            format!("{prefix}.context_length"),
            gguf_file::Value::U32(16),
        ),
        (format!("{prefix}.vocab_size"), gguf_file::Value::U32(8)),
        (
            format!("{prefix}.rope.freq_base"),
            gguf_file::Value::F64(10_000.0),
        ),
        (
            format!("{prefix}.attention.layer_norm_rms_epsilon"),
            gguf_file::Value::F64(1e-5),
        ),
    ];

    // Malformed-dimension fixtures deliberately retain tiny projection tensors
    // so the constructor can reject the metadata before allocating RoPE.
    let tensor_head_dim = usize::try_from(metadata_head_dim)
        .ok()
        .filter(|&value| value > 0 && value <= 1024)
        .unwrap_or(4);
    let query_width = num_attention_heads * tensor_head_dim;
    let kv_width = num_kv_heads * tensor_head_dim;
    let mut specs = vec![
        ("token_embd.weight", vec![8, hidden_dim]),
        ("output_norm.weight", vec![hidden_dim]),
        ("blk.0.attn_norm.weight", vec![hidden_dim]),
        ("blk.0.ffn_norm.weight", vec![hidden_dim]),
        ("blk.0.attn_q.weight", vec![query_width, hidden_dim]),
        ("blk.0.attn_k.weight", vec![kv_width, hidden_dim]),
        ("blk.0.attn_v.weight", vec![kv_width, hidden_dim]),
        ("blk.0.attn_output.weight", vec![hidden_dim, query_width]),
        ("blk.0.ffn_gate.weight", vec![hidden_dim * 2, hidden_dim]),
        ("blk.0.ffn_up.weight", vec![hidden_dim * 2, hidden_dim]),
        ("blk.0.ffn_down.weight", vec![hidden_dim, hidden_dim * 2]),
        ("output.weight", vec![8, hidden_dim]),
    ];
    if let Some(target) = mismatched_projection
        && let Some((_, shape)) = specs.iter_mut().find(|(name, _)| *name == target.as_str())
    {
        shape[0] = shape[0].saturating_sub(1);
    }
    if let Some(shape) = rope_freq_shape {
        specs.push(("rope_freqs.weight", shape));
    }

    let mut tensors = Vec::new();
    for (name, shape) in specs {
        if omit.iter().any(|omitted| omitted == name) {
            continue;
        }
        let tensor = Tensor::ones(shape.as_slice(), DType::F32, &Device::Cpu)?;
        tensors.push((
            name.to_string(),
            QTensor::quantize(&tensor, GgmlDType::F32)?,
        ));
    }
    let metadata_refs = metadata
        .iter()
        .map(|(key, value)| (key.as_str(), value))
        .collect::<Vec<_>>();
    let tensor_refs = tensors
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect::<Vec<_>>();
    gguf_file::write(
        &mut std::fs::File::create(&path)?,
        &metadata_refs,
        &tensor_refs,
    )?;
    Ok(path)
}

fn assert_finite_stage_output(output: StageOutput) -> anyhow::Result<()> {
    let tensor = match output {
        StageOutput::Logits(tensor) | StageOutput::Activations(tensor) => tensor,
    };
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    assert!(!values.is_empty());
    assert!(values.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn builder_exposes_stage_and_cache_options_without_loading() {
    let _builder = Engine::builder("model.gguf", Device::Cpu)
        .layers(4..12)
        .kv_quant_bits(8)
        .max_kv_tokens(4096);
}

#[test]
fn generation_and_memory_results_are_plain_public_values() {
    let params = GenerationParams::default();
    assert!(params.max_tokens > 0);

    let output = GenerationOutput {
        tokens: vec![1, 2],
        text: "ok".to_string(),
    };
    assert_eq!(output.tokens.len(), 2);

    let profile = StageMemoryProfile::default();
    assert!(profile.layer_gpu_bytes.is_empty());
}

#[test]
fn fallible_public_configuration_rejects_untrusted_values() {
    assert!(LayerRange::try_new(4, 4, 12).is_err());
    assert!(LayerRange::try_new(4, 13, 12).is_err());
    assert!(TurboQuantConfig::try_new(7, 128, 0).is_err());

    let params = GenerationParams {
        top_p: f64::NAN,
        ..GenerationParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn unqualified_inference_quantization_fails_before_model_io() {
    for bits in [3, 4] {
        let error = Engine::builder("model-does-not-exist.gguf", Device::Cpu)
            .kv_quant_bits(bits)
            .build()
            .err()
            .expect("low-bit inference must be rejected");
        assert!(error.to_string().contains("not quality-qualified"));
        let error = dinf_engine::PipelineStage::from_gguf(
            std::path::Path::new("model-does-not-exist.gguf"),
            LayerRange::solo(1),
            &Device::Cpu,
            Some(bits),
        )
        .err()
        .expect("stage must reject low-bit inference");
        assert!(error.to_string().contains("not quality-qualified"));
    }
}

#[test]
fn legacy_infallible_configuration_surface_remains_available() {
    let range = LayerRange::new(4, 8, 12);
    assert_eq!(range.start, 4);
    assert_eq!(range.end, 8);
    assert_eq!(range.total_layers, 12);

    let config = TurboQuantConfig::new(4, 128, 0);
    assert_eq!(config.bits, 4);
    assert_eq!(config.head_dim, 128);
    assert_eq!(TurboQuantConfig::new(4, 32_768, 0).head_dim, 32_768);

    // Legacy construction still defers validation to loaders and profilers.
    let invalid = LayerRange::new(8, 4, 12);
    assert_eq!(invalid.layer_count(), 0);
    assert!(invalid.validate().is_err());
}

#[test]
fn mutated_legacy_quant_config_is_rejected_before_cache_mutation() {
    fn assert_rejected(config: TurboQuantConfig, input_head_dim: usize) {
        let config = Arc::new(config);
        let mut cache = QuantizedKvCache::new(config, 1);
        let input = Tensor::zeros((1, 1, 1, input_head_dim), DType::F32, &Device::Cpu)
            .expect("CPU test tensor");
        assert!(
            cache.update(&input, &input).is_err(),
            "a config whose public fields were mutated must be rejected"
        );
        assert_eq!(
            cache.seq_len(),
            0,
            "rejected updates must not publish state"
        );
    }

    let mut bits = TurboQuantConfig::new(4, 64, 42);
    bits.bits = 8;
    assert_rejected(bits, 64);

    let mut head_dim = TurboQuantConfig::new(8, 64, 42);
    head_dim.head_dim = 128;
    assert_rejected(head_dim, 128);

    let mut invalid_bits = TurboQuantConfig::new(4, 64, 42);
    invalid_bits.bits = 2;
    assert_rejected(invalid_bits, 64);

    // This width still fits within the old sign allocation, so stale signs
    // alone would not necessarily panic; the cached scale/packed geometry
    // must also be checked before quantization begins.
    let mut smaller_head = TurboQuantConfig::new(4, 64, 42);
    smaller_head.head_dim = 32;
    assert_rejected(smaller_head, 32);
}

#[test]
fn malformed_numeric_model_metadata_fails_before_weights_or_tokenizer() -> anyhow::Result<()> {
    use candle_core::quantized::gguf_file::{self, Value};

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("metadata-only.gguf");
    for (key, value, expected) in [
        (
            "llama.context_length",
            Value::U64(1_000_000_000),
            "context length",
        ),
        (
            "llama.context_length",
            Value::U64(u64::MAX),
            "context length",
        ),
        ("llama.rope.freq_base", Value::F64(0.0), "rope.freq_base"),
        ("llama.rope.freq_base", Value::F64(-1.0), "rope.freq_base"),
        ("llama.rope.freq_base", Value::F64(1e300), "rope.freq_base"),
        (
            "llama.attention.layer_norm_rms_epsilon",
            Value::F64(-1.0),
            "layer_norm_rms_epsilon",
        ),
        (
            "llama.attention.layer_norm_rms_epsilon",
            Value::F64(1e-300),
            "layer_norm_rms_epsilon",
        ),
    ] {
        let mut metadata = vec![
            ("general.architecture", Value::String("llama".into())),
            ("llama.block_count", Value::U32(1)),
            ("llama.embedding_length", Value::U32(32)),
            ("llama.attention.head_count", Value::U32(4)),
        ];
        metadata.push((key, value));
        let refs = metadata
            .iter()
            .map(|(key, value)| (*key, value))
            .collect::<Vec<_>>();
        gguf_file::write(&mut std::fs::File::create(&path)?, &refs, &[])?;
        let errors = [
            dinf_engine::read_gguf_metadata(&path).expect_err("metadata must fail"),
            Engine::builder(&path, Device::Cpu)
                .build()
                .err()
                .expect("builder must fail"),
            dinf_engine::PipelineStage::from_gguf(&path, LayerRange::solo(1), &Device::Cpu, None)
                .err()
                .expect("stage must fail"),
            dinf_engine::gguf_loader::estimate_stage_memory_profile(
                &path,
                "invalid",
                1,
                &Device::Cpu,
            )
            .expect_err("admission must fail"),
        ];
        for error in errors {
            assert!(format!("{error:#}").contains(expected), "{key}: {error:#}");
        }
    }
    Ok(())
}

#[test]
fn tiny_llama_mistral_and_qwen2_f32_models_load_and_forward_finitely() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    for architecture in ["llama", "mistral", "qwen2"] {
        let path = write_tiny_fixture(dir.path(), TinyFixture::new(architecture, 8, 2, 1, 4))?;
        let mut stage = PipelineStage::from_gguf(&path, LayerRange::solo(1), &Device::Cpu, None)?;
        let output = stage.prefill(StageInput::Tokens(vec![1, 2]), 2, 0)?;
        assert_finite_stage_output(output)?;
    }

    // Frequency tensors are accepted in any rank as long as their element
    // count matches half the explicit head dimension.
    let alternate_freq_shape = write_tiny_fixture(
        dir.path(),
        TinyFixture::new("llama", 8, 2, 1, 4).rope_freq_shape(vec![2, 1]),
    )?;
    let mut stage = PipelineStage::from_gguf(
        &alternate_freq_shape,
        LayerRange::solo(1),
        &Device::Cpu,
        None,
    )?;
    assert_finite_stage_output(stage.prefill(StageInput::Tokens(vec![1]), 1, 0)?)?;
    Ok(())
}

#[test]
fn explicit_head_dimension_and_gqa_are_supported() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    // hidden_dim / num_attention_heads = 8, but the explicit GGUF head
    // dimension is 4. Query width is 8 and KV width is 4 (GQA).
    let path = write_tiny_fixture(dir.path(), TinyFixture::new("llama", 16, 2, 1, 4))?;
    let mut stage = PipelineStage::from_gguf(&path, LayerRange::solo(1), &Device::Cpu, None)?;
    assert_finite_stage_output(stage.prefill(StageInput::Tokens(vec![1, 2]), 2, 0)?)?;
    Ok(())
}

#[test]
fn malformed_attention_geometry_fails_before_rope_or_embedding_work() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;

    let huge_head = write_tiny_fixture(dir.path(), TinyFixture::new("llama", 8, 2, 1, 131_072))?;
    // The fixture's projection tensors remain tiny; validation must reject the
    // metadata at construction instead of allocating [context, head_dim/2]
    // RoPE tables and failing only on the first forward.
    let error = match PipelineStage::from_gguf(&huge_head, LayerRange::solo(1), &Device::Cpu, None)
    {
        Ok(_) => anyhow::bail!("huge head dimension unexpectedly loaded"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("attn_q.weight"), "{error:#}");

    let overflow_head =
        write_tiny_fixture(dir.path(), TinyFixture::new("llama", 8, 4, 1, u64::MAX - 1))?;
    let error =
        match PipelineStage::from_gguf(&overflow_head, LayerRange::solo(1), &Device::Cpu, None) {
            Ok(_) => anyhow::bail!("attention width overflow unexpectedly loaded"),
            Err(error) => error,
        };
    assert!(error.to_string().contains("overflow"), "{error:#}");

    let invalid_gqa = write_tiny_fixture(dir.path(), TinyFixture::new("llama", 12, 3, 2, 4))?;
    let error =
        match PipelineStage::from_gguf(&invalid_gqa, LayerRange::solo(1), &Device::Cpu, None) {
            Ok(_) => anyhow::bail!("non-divisible GQA unexpectedly loaded"),
            Err(error) => error,
        };
    assert!(error.to_string().contains("multiple"), "{error:#}");

    for (head_dim, expected) in [(0, "positive"), (3, "even")] {
        let path = write_tiny_fixture(dir.path(), TinyFixture::new("llama", 8, 2, 1, head_dim))?;
        let error = match PipelineStage::from_gguf(&path, LayerRange::solo(1), &Device::Cpu, None) {
            Ok(_) => anyhow::bail!("invalid head dimension {head_dim} unexpectedly loaded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains(expected), "{error:#}");
    }

    let invalid_before_embedding = write_tiny_fixture(
        dir.path(),
        TinyFixture::new("llama", 8, 2, 1, 4)
            .omit(&["token_embd.weight"])
            .mismatch("blk.0.attn_q.weight"),
    )?;
    let error = match PipelineStage::from_gguf(
        &invalid_before_embedding,
        LayerRange::solo(1),
        &Device::Cpu,
        None,
    ) {
        Ok(_) => anyhow::bail!("invalid projection geometry unexpectedly loaded"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(message.contains("attn_q.weight"), "{message:#}");
    assert!(!message.contains("token_embd.weight"), "{message:#}");
    assert!(!message.contains("rope_freqs.weight"), "{message:#}");

    let malformed_freqs = write_tiny_fixture(
        dir.path(),
        TinyFixture::new("llama", 8, 2, 1, 4).rope_freq_shape(vec![1]),
    )?;
    let error =
        match PipelineStage::from_gguf(&malformed_freqs, LayerRange::solo(1), &Device::Cpu, None) {
            Ok(_) => anyhow::bail!("malformed rope frequencies unexpectedly loaded"),
            Err(error) => error,
        };
    assert!(error.to_string().contains("rope_freqs.weight"), "{error:#}");
    Ok(())
}

#[test]
fn missing_or_mismatched_projection_is_rejected_at_construction() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    for omitted in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
    ] {
        let path = write_tiny_fixture(
            dir.path(),
            TinyFixture::new("llama", 8, 2, 1, 4).omit(&[omitted]),
        )?;
        let error = match PipelineStage::from_gguf(&path, LayerRange::solo(1), &Device::Cpu, None) {
            Ok(_) => anyhow::bail!("missing attention projection unexpectedly loaded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains(omitted), "{error:#}");
    }

    for target in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
    ] {
        let path = write_tiny_fixture(
            dir.path(),
            TinyFixture::new("llama", 8, 2, 1, 4).mismatch(target),
        )?;
        let error = match PipelineStage::from_gguf(&path, LayerRange::solo(1), &Device::Cpu, None) {
            Ok(_) => anyhow::bail!("mismatched attention projection unexpectedly loaded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains(target), "{error:#}");
        assert!(error.to_string().contains("shape"), "{error:#}");
    }
    Ok(())
}
