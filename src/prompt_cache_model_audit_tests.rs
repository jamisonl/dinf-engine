// Included in prompt_cache::tests. Explicit real-model fixture; not auto-loaded.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "loads DINF_KV_AUDIT_MODEL and requires Metal"]
fn cached_model_prompt_forks_match_teacher_forced_cold_logits() -> Result<()> {
    use crate::{LayerRange, read_gguf_metadata, tokenizer::GgufTokenizer};
    use candle_core::{DType, Device};
    let path = std::env::var("DINF_KV_AUDIT_MODEL")?;
    let path = std::path::Path::new(&path);
    let device = Device::new_metal(0)?;
    let meta = read_gguf_metadata(path)?;
    let mut stage =
        PipelineStage::from_gguf(path, LayerRange::solo(meta.num_layers), &device, None)?;
    let tokenizer = GgufTokenizer::from_gguf(path)?;
    let source = tokenizer.encode_chat(&[("user".into(),"Public cache ownership fixture: each request owns token positions and immutable prefixes. ".repeat(4096))])?;
    fn prefill(
        stage: &mut PipelineStage,
        state: &mut PipelineStageSessionState,
        tokens: &[u32],
        offset: usize,
    ) -> Result<Tensor> {
        let mut logits = None;
        for (index, chunk) in tokens.chunks(128).enumerate() {
            let start = offset + index * 128;
            if (index + 1) * 128 >= tokens.len() {
                let StageOutput::Logits(row) = stage.prefill_with_session(
                    state,
                    StageInput::Tokens(chunk.to_vec()),
                    chunk.len(),
                    start,
                )?
                else {
                    anyhow::bail!("solo logits required")
                };
                logits = Some(row);
            } else {
                stage.prefill_cache_only_with_session(
                    state,
                    StageInput::Tokens(chunk.to_vec()),
                    chunk.len(),
                    start,
                )?;
            }
            stage.device().synchronize()?;
        }
        logits.ok_or_else(|| anyhow::anyhow!("empty public prompt"))
    }
    for size in [1024, 4096, 8192] {
        stage.clear_prompt_cache();
        let prompt = &source[..size];
        let mut live = stage.new_session_state();
        let original = prefill(&mut stage, &mut live, prompt, 0)?;
        let ticket = stage
            .capture_prompt("public-logit-owner", prompt, false, &live, &original, false)?
            .ok_or_else(|| anyhow::anyhow!("real {size}-token snapshot did not fit 512MiB"))?;
        stage.publish_prompt(&ticket);
        // Mutate the original live state after capture. Snapshot must remain
        // private; logits and recurrent/attention state cannot follow its tail.
        prefill(&mut stage, &mut live, &source[size..size + 1], size)?;
        stage.clear_session_state(&mut live);
        let (mut warm, saved, reused) = stage
            .reuse_prompt("public-logit-owner", prompt, false)?
            .unwrap();
        assert_eq!(reused, size);
        let original = original
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let saved = saved
            .unwrap()
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        assert_eq!(
            saved, original,
            "copied next-token logits changed after live append"
        );
        let suffix = &source[size..size + 8];
        let warm_logits = prefill(&mut stage, &mut warm, suffix, size)?;
        let mut cold = stage.new_session_state();
        prefill(&mut stage, &mut cold, prompt, 0)?;
        let cold_logits = prefill(&mut stage, &mut cold, suffix, size)?;
        let warm_logits = warm_logits
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let cold_logits = cold_logits
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        assert_eq!(warm_logits.len(), cold_logits.len());
        let max_abs = warm_logits
            .iter()
            .zip(&cold_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            warm_logits
                .iter()
                .chain(&cold_logits)
                .all(|x| x.is_finite())
        );
        assert!(
            max_abs <= 1e-4,
            "warm/cold teacher-forced logits differ by {max_abs}"
        );
        eprintln!(
            "REAL_CACHE_LOGITS device=Metal cache=F16 size={size} reused={reused} finite_logits={} copied_logits_exact=true teacher_forced_suffix=8 max_abs_diff={max_abs} tolerance=0.0001 retained_bytes={}",
            warm_logits.len(),
            stage.prompt_cache_bytes()
        );
        stage.clear_session_state(&mut warm);
        stage.clear_session_state(&mut cold);
        stage.clear_prompt_cache();
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
    }
    Ok(())
}
