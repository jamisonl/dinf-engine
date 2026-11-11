use super::*;

fn matrix(rows: usize, width: usize, seed: usize, device: &Device) -> Result<DeepSeek4QMatMul> {
    let bytes = (0..rows * width)
        .flat_map(|i| ((((i * 17 + seed * 7) % 71) as f32 - 35.) / 100.).to_le_bytes())
        .collect::<Vec<_>>();
    DeepSeek4QMatMul::new_on_device(
        Arc::new(RawTensor::new(
            vec![rows, width],
            DeepSeek4DType::F32,
            Tensor::from_slice(&bytes, bytes.len(), device)?,
        )?),
        device,
    )
}

fn fixture(experts: Vec<V4Expert>, device: &Device) -> Result<V4Moe> {
    let width = experts[0].gate.in_dim();
    Ok(V4Moe {
        router: matrix(experts.len(), width, 0, device)?,
        hash_selected_experts: None,
        bias: Some(Tensor::new(&[0.3f32, -0.2, 0.1, -0.5], device)?),
        experts,
        shared_gate: matrix(16, width, 11, device)?,
        shared_up: matrix(16, width, 12, device)?,
        shared_down: matrix(width, 16, 13, device)?,
        active_experts: 3,
        scale: 1.2,
        normalize_weights: true,
        clamp: 0.5,
        shared_clamp: 0.25,
        group_experts: true,
    })
}

fn dense_experts(device: &Device) -> Result<Vec<V4Expert>> {
    (0..4)
        .map(|expert| {
            Ok(V4Expert {
                gate: matrix(16, 8, expert, device)?,
                up: matrix(16, 8, expert + 5, device)?,
                down: matrix(8, 16, expert + 8, device)?,
            })
        })
        .collect()
}

fn input(rows: usize, width: usize, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        (0..rows * width)
            .map(|i| (((i * 19 + 3) % 101) as f32 - 50.) / 63.7)
            .collect::<Vec<_>>(),
        (1, rows, width),
        device,
    )?)
}

fn same_bits(actual: &Tensor, expected: &Tensor) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    assert!(actual.iter().chain(&expected).all(|v| v.is_finite()));
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    Ok(())
}

fn check_grouping(mut moe: V4Moe, device: &Device) -> Result<()> {
    let width = moe.experts[0].gate.in_dim();
    for clamp in [0., 0.5] {
        moe.clamp = clamp;
        // Reverse expert order, duplicates, an empty row, uneven groups,
        // signed weights, and a singleton. Keep each contribution in order.
        let routes = vec![
            vec![(2, 0.13), (0, 0.71), (2, -0.2)],
            vec![(0, 0.31), (1, 0.43)],
            vec![],
            vec![(2, -0.67), (0, 1.23)],
        ];
        let full = input(6, width, device)?;
        let offset = full.narrow(1, 1, 4)?;
        let strided = offset.transpose(1, 2)?.contiguous()?.transpose(1, 2)?;
        for x in [offset, strided] {
            same_bits(
                &moe.grouped_experts_forward(&x, &routes)?,
                &moe.scalar_experts_forward(&x, &routes)?,
            )?;
            let unique = vec![vec![(3, 0.5)], vec![(0, 0.1)], vec![], vec![(2, 0.4)]];
            same_bits(
                &moe.grouped_experts_forward(&x, &unique)?,
                &moe.scalar_experts_forward(&x, &unique)?,
            )?;
            same_bits(
                &moe.grouped_experts_forward(&x, &vec![vec![]; 4])?,
                &Tensor::zeros((1, 4, width), DType::F32, device)?,
            )?;
        }
    }
    // Include selection bias, normalization modes, hash routing, and both
    // shared-prefill and shared-decode projections in the comparison.
    for hash in [false, true] {
        moe.hash_selected_experts = if hash {
            Some(Tensor::new(
                &[[2f32, 0., 2.], [0., 1., 2.], [3., 2., 0.], [2., 1., 99.]],
                device,
            )?)
        } else {
            None
        };
        for normalize in [false, true] {
            moe.normalize_weights = normalize;
            for decode_rows in [false, true] {
                for rows in [1, 4, 8] {
                    let tokens = (0..rows).map(|i| ((i * 3) % 4) as u32).collect::<Vec<_>>();
                    let x = input(rows, width, device)?;
                    moe.group_experts = false;
                    let expected = moe.forward_impl(&x, Some(&tokens), decode_rows)?;
                    moe.group_experts = true;
                    same_bits(
                        &moe.forward_impl(&x, Some(&tokens), decode_rows)?,
                        &expected,
                    )?;
                }
            }
        }
    }
    assert!(moe.forward(&input(4, width, device)?, None).is_err());
    assert!(moe.forward(&input(4, width, device)?, Some(&[0])).is_err());
    Ok(())
}

#[test]
fn grouped_expert_policy() {
    use std::ffi::OsStr;
    assert!(enabled(None, true));
    assert!(!enabled(None, false));
    for native in [true, false] {
        assert!(enabled(Some(OsStr::new("1")), native));
        for value in ["0", "false", "invalid", ""] {
            assert!(!enabled(Some(OsStr::new(value)), native));
        }
    }
    for enabled in [false, true] {
        for decode_rows in [false, true] {
            for rows in [0, 1, 2, 4, 7, 8, 9, 32, 1024] {
                assert_eq!(
                    should_group(enabled, decode_rows, rows),
                    enabled && !decode_rows && rows > 1,
                );
            }
        }
    }
}

#[test]
fn grouped_experts_exact_cpu() -> Result<()> {
    let device = Device::Cpu;
    check_grouping(fixture(dense_experts(&device)?, &device)?, &device)
}

#[cfg(target_os = "macos")]
#[test]
fn ordered_route_sum_matches_separate_rounding() -> Result<()> {
    use super::super::super::indexed_moe_metal::fused_ordered_weighted_route_sum;
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    for top_k in [1, 3, 8, 9, 16, 64, 128] {
        let full = input(5 * top_k, 19, &device)?.reshape((5, top_k, 19))?;
        let routes = full.narrow(0, 1, 3)?;
        let weights = Tensor::from_vec(
            (0..5 * top_k)
                .map(|i| match i % 7 {
                    0 => -0.,
                    1 => 1e10,
                    2 => -1e10,
                    _ => (i % 19) as f32 / -13.7,
                })
                .collect::<Vec<_>>(),
            (5, top_k),
            &device,
        )?
        .narrow(0, 1, 3)?;
        let mut reference = Tensor::zeros((3, 19), DType::F32, &device)?;
        for i in 0..top_k {
            let weighted = routes
                .narrow(1, i, 1)?
                .squeeze(1)?
                .broadcast_mul(&weights.narrow(1, i, 1)?)?;
            reference = (reference + weighted)?;
        }
        same_bits(
            &fused_ordered_weighted_route_sum(&routes, &weights)?,
            &reference,
        )?;
        assert!(fused_ordered_weighted_route_sum(&routes, &weights.narrow(0, 0, 1)?).is_err());
        assert!(fused_ordered_weighted_route_sum(&routes.to_dtype(DType::F16)?, &weights).is_err());
        assert!(fused_ordered_weighted_route_sum(&routes.transpose(0, 1)?, &weights).is_err());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn grouped_experts_exact_metal() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    check_grouping(fixture(dense_experts(&device)?, &device)?, &device)?;
    use candle_core::quantized::{GgmlDType, QTensor};
    let packed = |rows, width, seed| -> Result<DeepSeek4QMatMul> {
        let source = input(rows, width, &Device::Cpu)?.affine(0.1 + seed as f64 / 30., 0.)?;
        let tensor = QTensor::quantize(&source, GgmlDType::Q8_0)?;
        let bytes = tensor.data()?.to_vec();
        DeepSeek4QMatMul::new(Arc::new(RawTensor::new(
            vec![rows, width],
            DeepSeek4DType::Q8_0,
            Tensor::from_slice(&bytes, bytes.len(), &device)?,
        )?))
    };
    let experts = (0..4)
        .map(|i| {
            Ok(V4Expert {
                gate: packed(256, 1024, i)?,
                up: packed(256, 1024, i + 1)?,
                down: packed(1024, 256, i + 2)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    check_grouping(fixture(experts, &device)?, &device)
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires local V4 weight shard or verified Maya; small expert groups, not full inference"]
fn bench_real_grouped_experts() -> Result<()> {
    use crate::deepseek4_gguf::{RawGgufContent, RawGgufSource, read_architecture};
    let path = std::env::var("DINF_DEEPSEEK4_BENCH_GGUF")?;
    let layer = std::env::var("DINF_DEEPSEEK4_BENCH_LAYER")
        .unwrap_or_else(|_| "0".into())
        .parse::<usize>()?;
    let maya = if read_architecture(std::path::Path::new(&path))? == "deepseek41" {
        Some(RawGgufSource::open_maya(std::path::Path::new(&path))?)
    } else {
        None
    };
    eprintln!(
        "grouped expert weight source={path} verified_maya={}",
        maya.is_some()
    );
    let device = crate::test_metal_device().expect("benchmark requires Metal");
    let load = |part: &str, expert: usize| -> Result<DeepSeek4QMatMul> {
        let name = format!("blk.{layer}.ffn_{part}_exps.weight");
        if let Some(source) = &maya {
            // Keep the verified descriptor and read just this bounded expert slice.
            return DeepSeek4QMatMul::new(Arc::new(
                source.read_expert_slice(&name, expert, &device)?,
            ));
        }
        let mut file = std::fs::File::open(&path)?;
        let mut content = RawGgufContent::read(&mut file)?;
        let info = content
            .tensor_infos
            .get_mut(&name)
            .context("missing expert tensor")?;
        anyhow::ensure!(
            info.shape.len() == 3 && info.shape[0] > expert,
            "invalid expert shape"
        );
        info.shape = info.shape[1..].to_vec();
        let bytes = info.byte_len()?;
        anyhow::ensure!(bytes <= 64 * 1024 * 1024, "expert exceeds 64 MiB");
        info.offset += (expert * bytes) as u64;
        DeepSeek4QMatMul::new(Arc::new(content.load_tensor(&mut file, &name, &device)?))
    };
    let experts = (0..4)
        .map(|i| {
            Ok(V4Expert {
                gate: load("gate", i)?,
                up: load("up", i)?,
                down: load("down", i)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let moe = fixture(experts, &device)?;
    for rows in [2, 8, 32] {
        let x = input(rows, moe.experts[0].gate.in_dim(), &device)?;
        let routes = (0..rows)
            .map(|i| vec![((i * 3) % 4, 0.7), ((i + 1) % 4, 0.3)])
            .collect::<Vec<_>>();
        same_bits(
            &moe.grouped_experts_forward(&x, &routes)?,
            &moe.scalar_experts_forward(&x, &routes)?,
        )?;
        let mut times = [Vec::new(), Vec::new()];
        for repeat in 0..25 {
            for grouped in [repeat % 2, 1 - repeat % 2] {
                let start = std::time::Instant::now();
                let output = if grouped == 1 {
                    moe.grouped_experts_forward(&x, &routes)?
                } else {
                    moe.scalar_experts_forward(&x, &routes)?
                };
                device.synchronize()?;
                if repeat >= 4 {
                    times[grouped].push(start.elapsed().as_secs_f64() * 1e6);
                }
                std::hint::black_box(output);
            }
        }
        for values in &mut times {
            values.sort_by(f64::total_cmp);
        }
        eprintln!(
            "grouped experts layer={layer} rows={rows} scalar_us={:.2} grouped_us={:.2} speedup={:.3} exact=true",
            times[0][10],
            times[1][10],
            times[0][10] / times[1][10]
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "loads the full local V4 model; stop other resident models first"]
fn bench_full_v4_grouped_prefill() -> Result<()> {
    use crate::gguf_loader::{LayerRange, load_stage_tensors, read_gguf_metadata};
    use crate::tokenizer::GgufTokenizer;
    let path = std::path::PathBuf::from(std::env::var("DINF_DEEPSEEK4_FULL_GGUF")?);
    let functions = std::env::var("DINF_DEEPSEEK4_PREFILL_FUNCTIONS")
        .unwrap_or_else(|_| "16".into())
        .parse::<usize>()?;
    anyhow::ensure!((1..=96).contains(&functions), "functions must be 1..96");
    let md = read_gguf_metadata(&path)?;
    anyhow::ensure!(md.architecture == "deepseek4", "requires a V4 model");
    let tokenizer = GgufTokenizer::from_gguf(&path)?;
    let mut prompt = String::from(
        "Review this Rust module for overflow and edge cases. Identify bugs, explain your reasoning, then write unit tests.\n",
    );
    for i in 0..functions {
        prompt.push_str(&format!(
            "pub fn entry_{i}(x: u64) -> u64 {{ x.wrapping_mul({}).wrapping_add({i}) }}\n",
            i + 1
        ));
    }
    let tokens = tokenizer.encode(&prompt)?;
    anyhow::ensure!(
        tokens.len() > 32 && tokens.len() <= md.context_length,
        "invalid prompt length"
    );
    let device = crate::test_metal_device().expect("benchmark requires Metal");
    device.synchronize()?;
    let mut model = DeepSeek4Stage::from_stage_tensors(
        load_stage_tensors(&path, LayerRange::solo(md.num_layers), &device)?,
        None,
    )?;
    let run = |model: &mut DeepSeek4Stage,
               tokens: &[u32],
               grouped: bool|
     -> Result<(f64, Vec<Vec<u32>>)> {
        for block in &mut model.blocks {
            block.moe.group_experts = grouped;
        }
        let mut cache = (0..md.num_layers).map(|_| None).collect::<Vec<_>>();
        device.synchronize()?;
        let started = std::time::Instant::now();
        let mut values = Vec::new();
        for (i, chunk) in tokens.chunks(32).enumerate() {
            let output = model.forward(
                StageInput::Tokens(chunk.to_vec()),
                &mut cache,
                chunk.len(),
                i * 32,
            )?;
            let StageOutput::Logits(logits) = output else {
                anyhow::bail!("missing logits");
            };
            let logits = logits.to_vec1::<f32>()?;
            anyhow::ensure!(
                logits.len() == md.vocab_size && logits.iter().all(|v| v.is_finite()),
                "invalid logits"
            );
            values.push(logits.into_iter().map(f32::to_bits).collect::<Vec<_>>());
        }
        let seconds = started.elapsed().as_secs_f64();
        for layer in &cache {
            let Some(LayerCache::DeepSeek4(cache)) = layer else {
                anyhow::bail!("missing V4 cache");
            };
            anyhow::ensure!(cache.position() == tokens.len(), "cache position differs");
        }
        drop(cache);
        device.synchronize()?;
        Ok((seconds, values))
    };
    // Compile both paths first. Every timed call starts with an empty cache;
    // only the immutable loaded weights and shaders survive between calls.
    let (_, warm_reference) = run(&mut model, &tokens[..32], false)?;
    let (_, warm_candidate) = run(&mut model, &tokens[..32], true)?;
    anyhow::ensure!(warm_reference == warm_candidate, "warmup logits differ");
    let mut reference = None;
    let mut times = [Vec::new(), Vec::new()];
    for repeat in 0..2 {
        for grouped in [repeat % 2, 1 - repeat % 2] {
            let (seconds, logits) = run(&mut model, &tokens, grouped == 1)?;
            if let Some(reference) = &reference {
                anyhow::ensure!(reference == &logits, "full-model chunk logits differ");
            } else {
                reference = Some(logits);
            }
            times[grouped].push(seconds);
            eprintln!(
                "full V4 prefill repeat={repeat} grouped={} tokens={} chunk=32 seconds={seconds:.4} exact_logits=true",
                grouped == 1,
                tokens.len()
            );
        }
    }
    let scalar = times[0].iter().sum::<f64>();
    let grouped = times[1].iter().sum::<f64>();
    eprintln!(
        "full V4 prefill summary: tokens={} scalar_seconds={:?} grouped_seconds={:?} speedup={:.4} exact_logits=true",
        tokens.len(),
        times[0],
        times[1],
        scalar / grouped
    );
    Ok(())
}
