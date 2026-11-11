//! Pinned Maya geometry and owned-source adapters. See NOTICE and LICENSE-MIT.
use super::super::deepseek4_qmatmul::DeepSeek4QMatMul;
use super::{cache, engram};
use crate::deepseek4_gguf::{DeepSeek4DType as T, RawGgufContent, RawTensor};
use anyhow::{Context, Result, ensure};
use candle_core::quantized::gguf_file::Value;
use candle_core::{DType, Device, Tensor};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub(crate) type Weights = HashMap<String, Arc<RawTensor>>;

pub(crate) fn raw<'a>(w: &'a Weights, name: &str) -> Result<&'a Arc<RawTensor>> {
    w.get(name)
        .with_context(|| format!("missing Maya weight {name}"))
}

pub(crate) fn dense(w: &Weights, name: &str, d: &Device) -> Result<Tensor> {
    let raw = raw(w, name)?;
    let dense = raw
        .to_dense(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_device(d)?;
    #[cfg(all(test, target_os = "macos"))]
    crate::deepseek4_gguf::observe_maya_dense_copy(raw, &dense)?;
    Ok(dense)
}

pub(crate) fn linear(w: &Weights, name: &str, d: &Device) -> Result<DeepSeek4QMatMul> {
    DeepSeek4QMatMul::new_v41_on_device(Arc::clone(raw(w, name)?), d)
}

fn integer(value: &Value) -> Result<u64> {
    Ok(match value {
        Value::U8(v) => (*v).into(),
        Value::U16(v) => (*v).into(),
        Value::U32(v) => (*v).into(),
        Value::U64(v) => *v,
        Value::I8(v) if *v >= 0 => *v as u64,
        Value::I16(v) if *v >= 0 => *v as u64,
        Value::I32(v) if *v >= 0 => *v as u64,
        Value::I64(v) if *v >= 0 => *v as u64,
        _ => anyhow::bail!("Maya metadata requires unsigned integer"),
    })
}

fn value<'a>(c: &'a RawGgufContent, suffix: &str) -> Result<&'a Value> {
    c.metadata
        .get(&format!("deepseek41.{suffix}"))
        .with_context(|| format!("missing Maya {suffix}"))
}
fn array(c: &RawGgufContent, suffix: &str, count: usize) -> Result<Vec<u64>> {
    let Value::Array(v) = value(c, suffix)? else {
        anyhow::bail!("Maya {suffix} must be an array");
    };
    ensure!(v.len() == count, "Maya {suffix} array length mismatch");
    v.iter().map(integer).collect()
}
fn u32_array(c: &RawGgufContent, suffix: &str, count: usize) -> Result<Vec<u32>> {
    array(c, suffix, count)?
        .into_iter()
        .map(|v| Ok(u32::try_from(v)?))
        .collect()
}

pub(crate) fn engram_layout(c: &RawGgufContent) -> Result<engram::Layout> {
    if c.metadata.contains_key("deepseek41.ple.layers") {
        return coder_engram_layout(c);
    }
    let primes = u32_array(c, "engram.primes", 48)?;
    let mult = array(c, "engram.multipliers", 8)?;
    let layers = u32_array(c, "engram.layer_ids", 2)?;
    let rows = array(c, "engram.rows", 2)?;
    let tables = engram::LAYER_IDS.map(|l| {
        let name = format!("blk.{l}.engram_embd.weight");
        let t = c
            .tensor_infos
            .get(&name)
            .context("missing Maya Engram table")?;
        ensure!(t.shape.len() == 2, "Engram table rank mismatch");
        Ok(engram::TableGeometry {
            name,
            dtype: t.dtype.ggml_id(),
            dimensions: [t.shape[1] as u64, t.shape[0] as u64],
        })
    });
    let [a, b] = tables;
    let Value::String(encoding) = value(c, "engram.encoding")? else {
        anyhow::bail!("invalid Engram encoding");
    };
    engram::Layout::new(engram::LayoutMetadata {
        architecture: "deepseek41".into(),
        encoding: encoding.clone(),
        vocab_size: u32::try_from(integer(value(c, "vocab_size")?)?)?,
        compressed_vocab_size: u32::try_from(integer(value(c, "engram.compressed_vocab_size")?)?)?,
        pad_id: u32::try_from(integer(value(c, "engram.pad_id")?)?)?,
        layer_ids: layers.try_into().unwrap(),
        rows: rows.try_into().unwrap(),
        token_map: u32_array(c, "engram.token_map", engram::VOCAB_SIZE as usize)?,
        primes: [
            primes[..24].try_into().unwrap(),
            primes[24..].try_into().unwrap(),
        ],
        multipliers: [mult[..4].try_into().unwrap(), mult[4..].try_into().unwrap()],
        tables: [a?, b?],
    })
}

fn coder_engram_layout(c: &RawGgufContent) -> Result<engram::Layout> {
    let primes = u32_array(c, "ple.head_vocab_sizes", 48)?;
    let offsets = array(c, "ple.head_offsets", 48)?;
    let mut sum = 0u64;
    for (&offset, &prime) in offsets.iter().zip(&primes) {
        ensure!(
            offset == sum,
            "Coder Engram head offsets are not contiguous"
        );
        sum = sum
            .checked_add(prime.into())
            .context("Coder Engram row overflow")?;
    }
    let table = c
        .tensor_infos
        .get("per_layer_token_embd.weight")
        .context("missing Coder Engram")?;
    ensure!(
        table.dtype == T::IQ4NL && table.shape == [768022850, 256] && sum == 768022850,
        "unsupported Coder Engram tensor"
    );
    let mult = array(c, "ple.layer_multipliers", 8)?;
    let layers = u32_array(c, "ple.layers", 2)?;
    let map = u32_array(c, "engram.token_map", engram::VOCAB_SIZE as usize)?;
    let pad = usize::try_from(integer(value(c, "engram.pad_token_id")?)?)?;
    ensure!(pad < map.len(), "Coder Engram pad outside vocabulary");
    ensure!(
        integer(value(c, "ple.ngram_size")?)? == 4
            && integer(value(c, "ple.heads_per_ngram")?)? == 8
            && integer(value(c, "embedding_length_per_layer_input")?)? == 256,
        "unsupported Coder Engram hash geometry"
    );
    engram::Layout::new(engram::LayoutMetadata {
        architecture: "deepseek41".into(),
        encoding: "iq4_nl".into(),
        vocab_size: engram::VOCAB_SIZE,
        compressed_vocab_size: engram::COMPRESSED_VOCAB_SIZE,
        pad_id: map[pad],
        layer_ids: layers.try_into().unwrap(),
        rows: engram::TABLE_ROWS.map(u64::from),
        token_map: map,
        primes: [
            primes[..24].try_into().unwrap(),
            primes[24..].try_into().unwrap(),
        ],
        multipliers: [mult[..4].try_into().unwrap(), mult[4..].try_into().unwrap()],
        tables: engram::TABLE_ROWS.map(|rows| engram::TableGeometry {
            name: "per_layer_token_embd.weight".into(),
            dtype: 20,
            dimensions: [256, rows.into()],
        }),
    })
}

/// Validate every required name, dtype, shape and schedule before uploading any
/// resident weight. Unknown tensors are rejected rather than eagerly loaded.
pub(crate) fn validate_source(c: &RawGgufContent) -> Result<()> {
    let coder = c.metadata.contains_key("deepseek41.ple.layers");
    let experts = if coder { 192 } else { 384 };
    let mut expected = HashMap::<String, (T, Vec<usize>)>::new();
    let mut add = |name: String, dtype: T, shape: &[usize]| {
        expected.insert(name, (dtype, shape.to_vec()));
    };
    add("token_embd.weight".into(), T::Q8_0, &[129280, 5120]);
    add("output.weight".into(), T::Q8_0, &[129280, 5120]);
    add("output_norm.weight".into(), T::F32, &[5120]);
    for l in 0..40 {
        let p = format!("blk.{l}.");
        for side in ["attn", "ffn"] {
            add(format!("{p}hc_{side}_fn.weight"), T::F16, &[24, 20480]);
            add(format!("{p}hc_{side}_scale.weight"), T::F32, &[3]);
            add(format!("{p}hc_{side}_base.weight"), T::F32, &[24]);
            add(format!("{p}{side}_norm.weight"), T::F32, &[5120]);
        }
        for (name, shape) in [
            ("attn_q_a.weight", vec![1280, 5120]),
            ("attn_q_b.weight", vec![32768, 1280]),
            ("attn_kv.weight", vec![512, 5120]),
            ("attn_output_a.weight", vec![8192, 4096]),
            ("attn_output_b.weight", vec![5120, 8192]),
            ("ffn_gate_shexp.weight", vec![2304, 5120]),
            ("ffn_up_shexp.weight", vec![2304, 5120]),
            ("ffn_down_shexp.weight", vec![5120, 2304]),
        ] {
            add(format!("{p}{name}"), T::Q8_0, &shape);
        }
        for (name, shape) in [
            ("attn_q_a_norm.weight", vec![1280]),
            ("attn_kv_a_norm.weight", vec![512]),
            ("attn_sinks.weight", vec![64]),
            ("ffn_gate_inp.weight", vec![experts, 5120]),
            ("exp_probs_b.bias", vec![experts]),
        ] {
            add(format!("{p}{name}"), T::F32, &shape);
        }
        for part in ["gate", "up", "down"] {
            let shape = if part == "down" {
                [experts, 5120, 2304]
            } else {
                [experts, 2304, 5120]
            };
            add(format!("{p}ffn_{part}_exps.weight"), T::IQ1S, &shape);
        }
        if cache::OWNERS.contains(&l) {
            add(
                format!("{p}attn_compressor_kv.weight"),
                T::Q8_0,
                &[512, 5120],
            );
            add(format!("{p}attn_compressor_norm.weight"), T::F32, &[512]);
            add(format!("{p}indexer.attn_k.weight"), T::F16, &[128, 512]);
            add(format!("{p}indexer.k_norm.weight"), T::F32, &[128]);
            if l != 20 {
                add(
                    format!("{p}attn_compressor_gate.weight"),
                    T::Q8_0,
                    &[512, 5120],
                );
            }
        }
        if cache::refresh(l) {
            add(
                format!("{p}indexer.attn_q_b.weight"),
                T::Q8_0,
                &[4096, 1280],
            );
            add(format!("{p}indexer.proj.weight"), T::F16, &[32, 5120]);
        }
        if let Some(i) = engram::LAYER_IDS.iter().position(|&id| id as usize == l) {
            if coder {
                if i == 0 {
                    add(
                        "per_layer_token_embd.weight".into(),
                        T::IQ4NL,
                        &[768022850, 256],
                    );
                }
                add(format!("{p}engram_gate.weight"), T::F32, &[4, 5120]);
            } else {
                add(
                    format!("{p}engram_embd.weight"),
                    T::I8,
                    &[engram::TABLE_ROWS[i] as usize, 264],
                );
                add(format!("{p}engram_q_norm.weight"), T::F32, &[4, 5120]);
                add(format!("{p}engram_k_norm.weight"), T::F32, &[4, 5120]);
            }
            add(format!("{p}engram_kv.weight"), T::Q8_0, &[25600, 6144]);
        }
    }
    for (name, (dtype, shape)) in &expected {
        let info = c
            .tensor_infos
            .get(name)
            .with_context(|| format!("missing Maya tensor {name}"))?;
        let valid_dtype = if !coder {
            info.dtype == *dtype
        } else if name.ends_with("_exps.weight") {
            matches!(info.dtype, T::IQ1M | T::IQ2XXS | T::IQ2XS)
        } else if name.contains("hc_") && name.ends_with("_fn.weight") {
            info.dtype == T::F32
        } else if name.ends_with("ffn_gate_inp.weight") {
            info.dtype == T::BF16
        } else if *dtype == T::Q8_0 || *dtype == T::F16 {
            matches!(info.dtype, T::Q8_0 | T::Q5K | T::BF16)
        } else {
            info.dtype == *dtype
        };
        ensure!(
            valid_dtype && info.shape == *shape,
            "invalid native tensor {name}: {:?} {:?}",
            info.dtype,
            info.shape
        );
    }
    let allowed: HashSet<_> = expected.keys().map(String::as_str).collect();
    for (name, info) in &c.tensor_infos {
        if allowed.contains(name.as_str()) {
            continue;
        }
        // The vision router biases are present in the text artifact but unused.
        let optional = name
            .strip_prefix("blk.")
            .and_then(|s| s.strip_suffix(".exp_probs_b_vl.bias"))
            .and_then(|s| s.parse::<usize>().ok())
            .is_some_and(|l| l < 40);
        ensure!(
            optional && info.dtype == T::F32 && info.shape == [experts],
            "unexpected native tensor {name}"
        );
    }
    engram_layout(c)?;
    Ok(())
}
