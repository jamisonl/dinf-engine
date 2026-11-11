use super::*;
use candle_core::{DType, Tensor};

fn stage() -> PipelineStage {
    let mut stage = session_state_tests::test_stage();
    stage.vocab_size = 2;
    stage
}

fn assert_position(stage: &PipelineStage, tokens: usize, allocation: usize) {
    assert_eq!(stage.next_position(), tokens);
    assert_eq!(stage.total_kv_cache_tokens(), tokens);
    assert_eq!(stage.total_kv_cache_allocation_tokens(), allocation);
}

#[test]
fn invalid_token_input_preserves_cache_before_eviction() -> Result<()> {
    let mut stage = stage();
    stage.set_max_kv_tokens(1);
    stage.prefill(StageInput::Tokens(vec![1]), 1, 0)?;
    let allocated = stage.total_kv_cache_allocation_tokens();
    for input in [vec![2], vec![u32::MAX], vec![0, 1], vec![]] {
        assert!(
            stage
                .forward(StageInput::Tokens(input.clone()), 1, 1)
                .is_err()
        );
        assert_position(&stage, 1, allocated);
        assert!(stage.prefill(StageInput::Tokens(input), 1, 1).is_err());
        assert_position(&stage, 1, allocated);
    }
    stage.forward(StageInput::Tokens(vec![1]), 1, 1)?;
    assert_eq!(stage.next_position(), 2);
    Ok(())
}

#[test]
fn invalid_session_tokens_preserve_all_lanes() -> Result<()> {
    let mut stage = stage();
    let mut a = stage.new_session_state();
    let mut b = stage.new_session_state();
    stage.prefill_with_session(&mut a, StageInput::Tokens(vec![0]), 1, 0)?;
    stage.prefill_with_session(&mut b, StageInput::Tokens(vec![1]), 1, 0)?;
    let allocated = stage.total_kv_cache_allocation_tokens();
    for token in [2, u32::MAX] {
        assert!(
            stage
                .forward_with_session(&mut a, StageInput::Tokens(vec![token]), 1, 1)
                .is_err()
        );
        assert!(
            stage
                .forward_decode_batch_with_sessions(&mut [&mut a, &mut b], &[0, token])
                .is_err()
        );
        assert_eq!((a.next_position, b.next_position), (1, 1));
        assert_eq!(stage.total_kv_cache_tokens(), 2);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), allocated);
    }
    stage.forward_decode_batch_with_sessions(&mut [&mut a, &mut b], &[1, 0])?;
    assert_eq!((a.next_position, b.next_position), (2, 2));
    Ok(())
}

#[test]
fn invalid_generation_prompt_preserves_reusable_cache() -> Result<()> {
    let mut stage = stage();
    stage.prefill(StageInput::Tokens(vec![1]), 1, 0)?;
    let allocated = stage.total_kv_cache_allocation_tokens();
    let params = SamplingParams {
        temperature: 0.0,
        max_tokens: 1,
        ..Default::default()
    };
    for token in [2, u32::MAX] {
        assert!(
            stage
                .generate(vec![0, token], &params, &[], 0, |_| true)
                .is_err()
        );
        assert_position(&stage, 1, allocated);
        assert!(
            stage
                .generate_with_embeddings(
                    vec![0, token],
                    Tensor::zeros((2, 1), DType::F32, &Device::Cpu)?,
                    &params,
                    &[],
                    |_| true,
                )
                .is_err()
        );
        assert_position(&stage, 1, allocated);
    }
    Ok(())
}

#[test]
fn invalid_later_prompt_token_is_checked_before_first_chunk() -> Result<()> {
    let mut stage = stage();
    let mut session = continuous::GenerationSession::new(
        &stage,
        vec![0, 2],
        SamplingParams {
            temperature: 0.0,
            max_tokens: 1,
            ..Default::default()
        },
        vec![],
    );
    assert!(session.step_with_prefill_budget(&mut stage, 1).is_err());
    assert_eq!(stage.total_kv_cache_tokens(), 0);
    assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
    Ok(())
}

struct EmbeddingArchitecture(Tensor);

impl ModelArchitecture for EmbeddingArchitecture {
    fn forward(
        &self,
        _: StageInput,
        _: &mut [Option<LayerCache>],
        _: usize,
        _: usize,
    ) -> Result<StageOutput> {
        anyhow::bail!("embedding-only test")
    }
    fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor> {
        Ok(self
            .0
            .index_select(&Tensor::new(ids, self.0.device())?, 0)?)
    }
    fn family(&self) -> &str {
        "embedding-test"
    }
    fn hidden_dim(&self) -> usize {
        1
    }
    fn layer_count(&self) -> usize {
        1
    }
    fn vocab_size(&self) -> usize {
        2
    }
}

fn embedding_bounds(device: Device) -> Result<()> {
    let mut stage = stage();
    stage.arch = Box::new(EmbeddingArchitecture(Tensor::new(
        &[[3f32], [7.]],
        &device,
    )?));
    stage.device = device;
    assert_eq!(stage.embed_tokens(&[1])?.to_vec2::<f32>()?, vec![vec![7.]]);
    for token in [2, u32::MAX] {
        let error = stage.embed_tokens(&[token]).unwrap_err();
        assert!(error.to_string().contains("outside the model vocabulary"));
    }
    Ok(())
}

#[test]
fn cpu_embedding_rejects_invalid_ids() -> Result<()> {
    embedding_bounds(Device::Cpu)
}

#[test]
#[cfg(target_os = "macos")]
fn metal_embedding_rejects_invalid_ids() -> Result<()> {
    let Some(device) = test_metal_device() else {
        return Ok(());
    };
    embedding_bounds(device)
}
