use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use dinf_engine::diffusion::{
    scheduler::{FlowMatchEuler, SchedulerConfig},
    transformer::{
        ImageShape, KvCacheMode, QwenImageTransformer, TransformerConfig, TransformerKvCache,
    },
    vae::{PublicVaeConfig, QwenImageVae},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn input(base: &Path, spec: &Value, device: &Device, dtype: DType) -> Result<Tensor> {
    let name = spec["file"].as_str().context("input file")?;
    ensure!(
        Path::new(name).file_name().and_then(|x| x.to_str()) == Some(name),
        "input must be basename"
    );
    let shape = spec["shape"]
        .as_array()
        .context("input shape")?
        .iter()
        .map(|x| {
            x.as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .context("positive shape")
        })
        .collect::<Result<Vec<_>>>()?;
    let n = shape
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .context("shape overflow")?;
    ensure!(n > 0 && n <= 4 * 1024 * 1024, "input size limit");
    let raw = fs::read(base.join(name))?;
    ensure!(
        raw.len() == n * 4 && hash(&raw) == spec["sha256"].as_str().context("input hash")?,
        "input bytes differ"
    );
    let values = raw
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    ensure!(values.iter().all(|x| x.is_finite()), "input nonfinite");
    Ok(Tensor::from_vec(values, shape.as_slice(), device)?.to_dtype(dtype)?)
}
fn save(dir: &Path, name: &str, tensor: &Tensor) -> Result<Value> {
    let values = tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    ensure!(values.iter().all(|x| x.is_finite()), "nonfinite {name}");
    let raw = values
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    let file = format!("{name}.f32");
    OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(dir.join(&file))?
        .write_all(&raw)?;
    Ok(json!({"file":file,"shape":tensor.dims(),"sha256":hash(&raw),"finite":true}))
}
fn weights(component: &Path, device: &Device, dtype: DType) -> Result<VarBuilder<'static>> {
    let files = fs::read_dir(component)?
        .map(|e| Ok(e?.path()))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect::<BTreeSet<PathBuf>>();
    ensure!(!files.is_empty(), "missing safetensors");
    // The runner pins complete files and identities before/after this process.
    Ok(unsafe {
        VarBuilder::from_mmaped_safetensors(&files.into_iter().collect::<Vec<_>>(), dtype, device)?
    })
}
fn main() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    ensure!(
        a.len() == 7,
        "usage: image-component SNAPSHOT FIXTURE.json schedule|denoise_single|denoise_cache|denoise_uncached|vae cpu|metal f32|bf16 OUTPUT_DIR"
    );
    let snapshot = Path::new(&a[1]);
    let fp = Path::new(&a[2]);
    let mode = &a[3];
    ensure!(
        [
            "schedule",
            "denoise_single",
            "denoise_cache",
            "denoise_uncached",
            "vae"
        ]
        .contains(&mode.as_str()),
        "bad mode"
    );
    let dtype = match a[5].as_str() {
        "f32" => DType::F32,
        "bf16" => DType::BF16,
        _ => anyhow::bail!("bad dtype"),
    };
    let device = match a[4].as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)?,
        _ => anyhow::bail!("bad backend"),
    };
    ensure!(!device.is_cpu() || dtype == DType::F32, "CPU F32 only");
    let fixture = read(fp)?;
    ensure!(
        fixture["schema_version"] == 1 && fixture["steps"] == 4,
        "unsupported fixture"
    );
    let out = Path::new(&a[6]);
    fs::create_dir(out).context("output must be new")?;
    let mut outputs = serde_json::Map::new();
    let sc: SchedulerConfig =
        serde_json::from_value(read(&snapshot.join("scheduler/scheduler_config.json"))?)?;
    let schedule = FlowMatchEuler::new(&sc, 4, 1024)?;
    outputs.insert(
        "sigmas".into(),
        save(
            out,
            "sigmas",
            &Tensor::new(schedule.sigmas(), &Device::Cpu)?,
        )?,
    );
    outputs.insert(
        "timesteps".into(),
        save(
            out,
            "timesteps",
            &Tensor::new(
                (0..4)
                    .map(|i| schedule.timestep(i))
                    .collect::<candle_core::Result<Vec<_>>>()?
                    .as_slice(),
                &Device::Cpu,
            )?,
        )?,
    );
    if mode != "schedule" {
        let base = fp.parent().context("fixture parent")?;
        let mut latent = input(base, &fixture["latents"], &device, dtype)?;
        ensure!(latent.dims() == [1, 1024, 64], "latent shape");
        outputs.insert("input_latents".into(), save(out, "input_latents", &latent)?);
        if mode == "vae" {
            let config: PublicVaeConfig =
                serde_json::from_value(read(&snapshot.join("vae/config.json"))?)?;
            let model = QwenImageVae::load_decoder(
                weights(&snapshot.join("vae"), &device, dtype)?,
                config,
            )?;
            let raw = model.denormalize_latents(
                &latent
                    .transpose(1, 2)?
                    .reshape((1, 64, 32, 32))?
                    .contiguous()?,
            )?;
            outputs.insert(
                "denormalized_latents".into(),
                save(out, "denormalized_latents", &raw)?,
            );
            outputs.insert(
                "decoded".into(),
                save(out, "decoded", &model.decode(&raw)?)?,
            );
        } else {
            let cond = input(base, &fixture["conditioning"], &device, dtype)?;
            ensure!(cond.dims() == [1, 32, 4096], "conditioning shape");
            outputs.insert(
                "input_conditioning".into(),
                save(out, "input_conditioning", &cond)?,
            );
            let config: TransformerConfig =
                serde_json::from_value(read(&snapshot.join("transformer/config.json"))?)?;
            let model = QwenImageTransformer::load(
                weights(&snapshot.join("transformer"), &device, dtype)?,
                config,
            )?;
            let mut cache = TransformerKvCache::new(model.num_layers());
            let mut mask = vec![false; 32];
            mask.extend(std::iter::repeat_n(true, 256));
            for i in 0..if mode == "denoise_single" { 1 } else { 4 } {
                let timestep =
                    (Tensor::new(&[schedule.timestep(i)?], &device)?.to_dtype(dtype)? / 1000.)?;
                let cm = if mode == "denoise_cache" {
                    Some(if i == 0 {
                        KvCacheMode::Extract
                    } else {
                        KvCacheMode::Cached
                    })
                } else {
                    None
                };
                let y = model.forward(
                    &latent,
                    &cond,
                    &timestep,
                    &[ImageShape::new(1, 32, 32)],
                    &mask,
                    None,
                    cm.map(|_| &mut cache),
                    cm,
                )?;
                let y = y.narrow(1, y.dim(1)? - 1024, 1024)?;
                outputs.insert(
                    format!("prediction_{i}"),
                    save(out, &format!("prediction_{i}"), &y)?,
                );
                latent = schedule.step(i, &latent, &y)?;
                device.synchronize()?;
                outputs.insert(
                    format!("latent_{i}"),
                    save(out, &format!("latent_{i}"), &latent)?,
                );
            }
        }
    }
    device.synchronize()?;
    let report = json!({"schema_version":1,"runtime":"dinf-engine","backend":a[4],"dtype":a[5],"mode":mode,"fixture_sha256":hash(&fs::read(fp)?),"outputs":outputs,"quality_scope":"synthetic conditioning, full component weights; no text encoder or semantic quality claim"});
    OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("result.json"))?
        .write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    println!("{}", report);
    Ok(())
}
