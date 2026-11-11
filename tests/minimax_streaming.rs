//! Compare the bounded source loader with the old whole-stage constructor.
//! Tiny quantized fixtures cover split files, stage boundaries, and tied heads.

use anyhow::Result;
use candle_core::quantized::{
    GgmlDType, QTensor,
    gguf_file::{self, Value},
};
use candle_core::{Device, Tensor};
use dinf_engine::architectures::minimax_m2::MiniMaxM2Stage;
use dinf_engine::architectures::{ModelArchitecture, StageInput, StageOutput};
use dinf_engine::gguf_loader::{LayerRange, estimate_stage_memory_profile, load_stage_tensors};
use std::path::{Path, PathBuf};

fn fixture(dir: &Path, split: bool, tied: bool) -> Result<PathBuf> {
    let mut metadata = vec![(
        "general.architecture".to_string(),
        Value::String("minimax-m2".into()),
    )];
    for (key, value) in [
        ("block_count", 3),
        ("embedding_length", 256),
        ("attention.head_count", 4),
        ("attention.head_count_kv", 2),
        ("attention.key_length", 64),
        ("feed_forward_length", 256),
        ("expert_feed_forward_length", 256),
        ("expert_count", 4),
        ("expert_used_count", 2),
        ("expert_gating_func", 2),
        ("context_length", 32),
        ("vocab_size", 64),
        ("rope.dimension_count", 32),
    ] {
        metadata.push((format!("minimax-m2.{key}"), Value::U32(value)));
    }
    let mut tensors = Vec::new();
    let mut add = |name: String, shape: &[usize], norm: bool, dtype: GgmlDType| -> Result<()> {
        let count = shape.iter().product();
        let salt = tensors.len() as f32 * 0.3;
        let values: Vec<f32> = (0..count)
            .map(|i| {
                if norm {
                    1.
                } else {
                    ((i as f32 * 0.71 + salt).sin()) * 0.05
                }
            })
            .collect();
        let tensor = Tensor::from_vec(values, shape, &Device::Cpu)?;
        tensors.push((name, QTensor::quantize(&tensor, dtype)?));
        Ok(())
    };
    add(
        "token_embd.weight".into(),
        &[64, 256],
        false,
        GgmlDType::F32,
    )?;
    add("output_norm.weight".into(), &[256], true, GgmlDType::F32)?;
    if !tied {
        add("output.weight".into(), &[64, 256], false, GgmlDType::F32)?;
    }
    for layer in 0..3 {
        for (name, shape, norm) in [
            ("attn_norm.weight", vec![256], true),
            ("ffn_norm.weight", vec![256], true),
            ("attn_q_norm.weight", vec![256], true),
            ("attn_k_norm.weight", vec![128], true),
            ("attn_q.weight", vec![256, 256], false),
            ("attn_k.weight", vec![128, 256], false),
            ("attn_v.weight", vec![128, 256], false),
            ("attn_output.weight", vec![256, 256], false),
            ("ffn_gate_inp.weight", vec![4, 256], false),
            ("exp_probs_b.bias", vec![4], false),
        ] {
            add(format!("blk.{layer}.{name}"), &shape, norm, GgmlDType::F32)?;
        }
        for name in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
            add(
                format!("blk.{layer}.{name}.weight"),
                &[4, 256, 256],
                false,
                GgmlDType::Q2K,
            )?;
        }
    }
    let shards = if split { 3 } else { 1 };
    for shard in 0..shards {
        let path = if split {
            dir.join(format!("model-{:05}-of-00003.gguf", shard + 1))
        } else {
            dir.join("model.gguf")
        };
        // Shard zero deliberately has metadata but no tensors, like Unsloth.
        let selected: Vec<_> = tensors
            .iter()
            .enumerate()
            .filter(|(i, _)| !split || shard == 1 + i % 2)
            .map(|(_, (name, tensor))| (name.as_str(), tensor))
            .collect();
        let md: Vec<_> = metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
        gguf_file::write(&mut std::fs::File::create(path)?, &md, &selected)?;
    }
    Ok(dir.join(if split {
        "model-00001-of-00003.gguf"
    } else {
        "model.gguf"
    }))
}

fn run(stage: &MiniMaxM2Stage, range: &LayerRange, device: &Device) -> Result<Vec<Vec<f32>>> {
    let mut cache = (0..range.layer_count()).map(|_| None).collect::<Vec<_>>();
    let mut rows = Vec::new();
    for (seq_len, pos) in [(3, 0), (1, 3), (1, 4)] {
        let input = if range.is_head() {
            StageInput::Tokens(vec![7; seq_len])
        } else {
            StageInput::activations(Tensor::from_vec(
                (0..seq_len * 256)
                    .map(|i| (i as f32 * 0.3).cos() * 0.1)
                    .collect::<Vec<_>>(),
                (seq_len, 256),
                device,
            )?)
        };
        let output = if pos == 0 {
            stage.forward_prefill(input, &mut cache, seq_len, pos)?
        } else {
            stage.forward(input, &mut cache, seq_len, pos)?
        };
        let tensor = match output {
            StageOutput::Logits(t) | StageOutput::Activations(t) => t,
        };
        let values = tensor.flatten_all()?.to_vec1::<f32>()?;
        assert!(values.iter().all(|v| v.is_finite()));
        rows.push(values);
    }
    Ok(rows)
}

fn compare_loaders(device: &Device) -> Result<()> {
    for split in [false, true] {
        for tied in [false, true] {
            let dir = tempfile::tempdir()?;
            let path = fixture(dir.path(), split, tied)?;
            let profile = estimate_stage_memory_profile(&path, "test", 3, device)?;
            let source_per_layer = 3 * (4 * 256 * 256 / 256 * 84) as u64;
            let layer_sources = load_stage_tensors(&path, LayerRange::new(1, 2, 3), &Device::Cpu)?;
            let floating_sources = layer_sources
                .tensors
                .values()
                .filter(|tensor| tensor.dtype() == GgmlDType::F32)
                .map(|tensor| tensor.storage_size_in_bytes() as u64)
                .collect::<Vec<_>>();
            let floating_source_bytes = floating_sources.iter().sum::<u64>();
            let largest_floating_source = floating_sources.iter().copied().max().unwrap();
            // CPU retains the split experts; GPU releases every CPU source.
            assert_eq!(
                profile.layer_cpu_bytes,
                vec![if device.is_cpu() { source_per_layer } else { 0 }; 3]
            );
            assert_eq!(
                profile.layer_cpu_transient_bytes,
                vec![
                    source_per_layer
                        + if device.is_cpu() {
                            // Dense F32 copies coexist with this layer's
                            // source QTensor pool until construction ends.
                            floating_source_bytes
                        } else {
                            // Packed experts retain CPU sources. Metal also
                            // overlaps encoded and decoded host vectors for
                            // the largest dense F32 projection.
                            (source_per_layer / 3).max(if device.is_metal() {
                                largest_floating_source * 2
                            } else {
                                0
                            })
                        };
                    3
                ]
            );
            for (start, end) in [(0, 3), (0, 1), (1, 2), (2, 3), (1, 3)] {
                let range = LayerRange::try_new(start, end, 3)?;
                let old = MiniMaxM2Stage::from_stage_tensors(
                    load_stage_tensors(&path, range.clone(), device)?,
                    None,
                )?;
                let expected = run(&old, &range, device)?;
                drop(old);
                device.synchronize()?;
                let streamed =
                    MiniMaxM2Stage::from_streaming_gguf(&path, range.clone(), device, None)?;
                assert_eq!(
                    expected,
                    run(&streamed, &range, device)?,
                    "{device:?} split={split} tied={tied} range={range:?}"
                );
            }
            assert!(
                MiniMaxM2Stage::from_streaming_gguf(&path, LayerRange::new(1, 1, 3), device, None)
                    .is_err()
            );
            assert!(
                MiniMaxM2Stage::from_streaming_gguf(&path, LayerRange::new(0, 1, 4), device, None)
                    .is_err()
            );
        }
    }
    Ok(())
}

#[test]
fn minimax_streaming_cpu_matches_whole_stage() -> Result<()> {
    compare_loaders(&Device::Cpu)
}

#[cfg(target_os = "macos")]
#[test]
fn minimax_streaming_metal_matches_whole_stage() -> Result<()> {
    match Device::new_metal(0) {
        Ok(device) => compare_loaders(&device),
        Err(error) if std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_none() => {
            eprintln!("Skipping Metal loader test: {error}");
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}
