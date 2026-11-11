use super::super::hc_bench::{Fusion, HcBenchmark};
use super::*;
use crate::gguf_loader::read_gguf_metadata;
use crate::tokenizer::GgufTokenizer;

#[test]
fn fused_hc_matches_ordinary_and_deferred_controls() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let mut mix = HcMix {
        projection: GlmQMatMul::Candle(QMatMul::Tensor(Tensor::from_vec(
            (0..24 * 32)
                .map(|i| ((i * 17 % 71) as f32 - 35.) / 300.)
                .collect::<Vec<_>>(),
            (24, 32),
            &device,
        )?)),
        base: Tensor::from_vec(
            (0..24).map(|i| (i as f32 - 12.) / 7.).collect::<Vec<_>>(),
            24,
            &device,
        )?,
        scale: Tensor::new(&[0.7f32, 0.8, 0.6], &device)?,
        norm: RmsNorm::ones(32, 1e-6, &device)?,
        streams: 4,
        hidden: 8,
        sinkhorn_iterations: 20,
        eps: 1e-6,
        fused_sinkhorn: false,
        fused_post: false,
        fused_pre: false,
        reuse_projection: false,
    };
    let bits = |x: &Tensor| -> Result<Vec<u32>> {
        Ok(x.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    };
    for rows in [1, 3, 32] {
        let input = Tensor::from_vec(
            (0..(rows + 2) * 32)
                .map(|i| ((i * 13 % 31) as f32 - 15.) / 7.3)
                .collect::<Vec<_>>(),
            (1, rows + 2, 4, 8),
            &device,
        )?
        .narrow(1, 1, rows)?;
        for iterations in [1, 20, 128, 129] {
            mix.sinkhorn_iterations = iterations;
            mix.fused_sinkhorn = false;
            mix.fused_post = false;
            mix.fused_pre = false;
            let (update, post, comb) = mix.pre(&input)?;
            let expected = bits(&mix.post(&update, &input, &post, &comb)?)?;
            mix.fused_sinkhorn = true;
            mix.fused_post = true;
            mix.fused_pre = true;
            let (update2, post2, comb2) = mix.pre(&input)?;
            assert_eq!(bits(&update2)?, bits(&update)?);
            assert_eq!(bits(&post2)?, bits(&post)?);
            assert_eq!(bits(&comb2)?, bits(&comb)?);
            assert_eq!(
                bits(&mix.post(&update2, &input, &post2, &comb2)?)?,
                expected
            );
            for reuse in [false, true] {
                mix.reuse_projection = reuse;
                let (deferred_update, mixes) = mix.pre_update(&input)?;
                assert_eq!(mixes.is_some(), reuse);
                let (deferred_post, deferred_comb) = mix.post_controls(&input, mixes.as_ref())?;
                assert_eq!(
                    bits(&mix.post(&deferred_update, &input, &deferred_post, &deferred_comb)?)?,
                    expected
                );
            }
        }
    }
    Ok(())
}

impl HcBenchmark for Glm5NextStage {
    fn set_hc_fusion(&mut self, kind: Fusion, fused: bool) {
        for block in &mut self.blocks {
            for mix in [&mut block.attn_hc, &mut block.ffn_hc] {
                match kind {
                    Fusion::Sinkhorn => mix.fused_sinkhorn = fused,
                    Fusion::Post => mix.fused_post = fused,
                    Fusion::Pre => mix.fused_pre = fused,
                    Fusion::ReuseProjection => mix.reuse_projection = fused,
                }
            }
        }
    }

    fn check_hc_cache(&self, cache: &[Option<LayerCache>], position: usize) -> Result<()> {
        anyhow::ensure!(
            cache.len() == self.blocks.len(),
            "cache layer count differs"
        );
        for (block, state) in self.blocks.iter().zip(cache) {
            match (&block.attention, state) {
                (GlmAttention::Kda(attn), Some(LayerCache::DeltaNet(state))) => {
                    anyhow::ensure!(
                        state.recurrent_state.dims() == [attn.heads, attn.head_dim, attn.head_dim]
                            && state.conv_state.dims() == [3 * attn.inner, attn.kernel],
                        "invalid KDA state shape"
                    );
                }
                (GlmAttention::Dsa(_), Some(LayerCache::Attention(kv))) => {
                    let length = match kv {
                        KvCache::F16 { k, v } => {
                            anyhow::ensure!(k.dim(2)? == v.dim(2)?, "KV lengths differ");
                            k.dim(2)?
                        }
                        KvCache::PreallocatedF16(kv) => kv.len,
                        KvCache::Quantized(_) => anyhow::bail!("benchmark does not quantize state"),
                    };
                    anyhow::ensure!(length == position, "DSA cache position differs");
                }
                _ => anyhow::bail!("missing or wrong GLM5 cache type"),
            }
        }
        Ok(())
    }
}

#[test]
#[ignore = "loads full local GLM5 weights; stop other resident models first"]
fn bench_full_glm5_fused_hc() -> Result<()> {
    bench(Fusion::Sinkhorn)
}

#[test]
#[ignore = "loads full local GLM5 weights; stop other resident models first"]
fn bench_full_glm5_fused_hc_post() -> Result<()> {
    bench(Fusion::Post)
}

#[test]
#[ignore = "loads full local GLM5 weights; stop other resident models first"]
fn bench_full_glm5_fused_hc_pre() -> Result<()> {
    bench(Fusion::Pre)
}

#[test]
#[ignore = "loads full local GLM5 weights; stop other resident models first"]
fn bench_full_glm5_reuse_hc_projection() -> Result<()> {
    bench(Fusion::ReuseProjection)
}

fn bench(kind: Fusion) -> Result<()> {
    let path = std::path::PathBuf::from(std::env::var("DINF_GLM5_FULL_GGUF")?);
    let md = read_gguf_metadata(&path)?;
    anyhow::ensure!(
        matches!(md.architecture.as_str(), "glm5next" | "glm5-next"),
        "requires GLM5"
    );
    let tokenizer = GgufTokenizer::from_gguf(&path)?;
    let device = crate::test_metal_device().expect("benchmark requires Metal");
    let model =
        Glm5NextStage::from_streaming_gguf(&path, LayerRange::solo(md.num_layers), &device, None)?;
    super::super::hc_bench::compare(model, &md, &tokenizer, &device, true, kind)
}
