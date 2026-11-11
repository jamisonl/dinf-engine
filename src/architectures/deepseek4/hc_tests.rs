use super::super::hc_bench::{Fusion, HcBenchmark};
use super::*;
use crate::gguf_loader::{LayerRange, load_stage_tensors, read_gguf_metadata};
use crate::tokenizer::GgufTokenizer;

impl HcBenchmark for DeepSeek4Stage {
    fn set_hc_fusion(&mut self, kind: Fusion, fused: bool) {
        for block in &mut self.blocks {
            for mix in [&mut block.attn_hc, &mut block.ffn_hc] {
                match kind {
                    Fusion::Sinkhorn => mix.fused_sinkhorn = fused,
                    Fusion::Post => mix.fused_post = fused,
                    Fusion::Pre => mix.fused_pre = fused,
                    Fusion::ReuseProjection => panic!("projection reuse benchmark is GLM-only"),
                }
            }
        }
    }

    fn check_hc_cache(&self, cache: &[Option<LayerCache>], position: usize) -> Result<()> {
        anyhow::ensure!(
            cache.len() == self.blocks.len(),
            "cache layer count differs"
        );
        for layer in cache {
            let Some(LayerCache::DeepSeek4(cache)) = layer else {
                anyhow::bail!("missing V4 cache");
            };
            anyhow::ensure!(cache.position() == position, "cache position differs");
        }
        Ok(())
    }
}

#[test]
#[ignore = "loads full local V4 weights; stop other resident models first"]
fn bench_full_v4_fused_hc() -> Result<()> {
    bench(Fusion::Sinkhorn)
}

#[test]
#[ignore = "loads full local V4 weights; stop other resident models first"]
fn bench_full_v4_fused_hc_post() -> Result<()> {
    bench(Fusion::Post)
}

#[test]
#[ignore = "loads full local V4 weights; stop other resident models first"]
fn bench_full_v4_fused_hc_pre() -> Result<()> {
    bench(Fusion::Pre)
}

fn bench(kind: Fusion) -> Result<()> {
    let path = std::path::PathBuf::from(std::env::var("DINF_DEEPSEEK4_FULL_GGUF")?);
    let md = read_gguf_metadata(&path)?;
    anyhow::ensure!(md.architecture == "deepseek4", "requires V4");
    let tokenizer = GgufTokenizer::from_gguf(&path)?;
    let device = crate::test_metal_device().expect("benchmark requires Metal");
    let model = DeepSeek4Stage::from_stage_tensors(
        load_stage_tensors(&path, LayerRange::solo(md.num_layers), &device)?,
        None,
    )?;
    super::super::hc_bench::compare(model, &md, &tokenizer, &device, false, kind)
}
