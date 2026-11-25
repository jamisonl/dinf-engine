//! Disk-backed KV prompt-cache persistence for any supported model architecture.
//!
//! Saves completed prompt-cache snapshots (Attention KV, DeltaNet, Mamba-2,
//! Inkling 2 loop states, optional DSpark / DFlash2 draft caches, and
//! last-token logits) asynchronously to `<root>/<model_key>/` so a restarted
//! process reloads long contexts in seconds instead of a cold prefill.
//! Off by default: [`set_disk_kv_cache_limit_bytes`] turns it on. The root is
//! the folder given to [`set_disk_kv_cache_dir`], else `DINF_DISK_KV_CACHE_DIR`,
//! else `dinf-engine/kv-cache` in the user's cache folder
//! (`~/Library/Caches` on macOS, `$XDG_CACHE_HOME` or `~/.cache` elsewhere,
//! `%LOCALAPPDATA%` on Windows).

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::architectures::dflash2::DFlash2Cache;
use crate::architectures::dspark::DSparkCache;
use crate::architectures::inkling2_loop::Inkling2Cache;
use crate::architectures::speculative_draft::SpeculativeCache;
use crate::architectures::{
    DeltaNetCache, ImageSpanKey, KvCache, LayerCache, Mamba2Cache, PreallocatedKvCache,
    copy_persistent_tensor,
};

const FORMAT_VERSION: u32 = 2;
const FALLBACK_MAX_DISK_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const AUTO_MIN_DISK_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const AUTO_MAX_DISK_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const AUTO_MIN_FREE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DEFAULT_MIN_TOKENS: usize = 4_096;

/// Configured disk KV cache byte limit:
/// `-1` = automatic (scaled from drive capacity), `0` = disabled, `> 0` = explicit byte cap.
/// Off until [`set_disk_kv_cache_limit_bytes`] turns it on.
static CONFIGURED_LIMIT_BYTES: AtomicI64 = AtomicI64::new(0);

/// Folder set by [`set_disk_kv_cache_dir`], if any.
static CONFIGURED_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Sets the process-wide folder for disk KV cache snapshots. `None` restores
/// the default (`DINF_DISK_KV_CACHE_DIR`, else the user's cache folder).
pub fn set_disk_kv_cache_dir(dir: Option<PathBuf>) {
    *CONFIGURED_DIR.lock().unwrap_or_else(|e| e.into_inner()) = dir;
}

/// Sets the process-wide disk KV cache limit in bytes. The cache is off until
/// this is called. `None` turns it on with the automatic drive-scaled limit;
/// `Some(0)` turns it off.
pub fn set_disk_kv_cache_limit_bytes(limit: Option<u64>) {
    let encoded = match limit {
        None => -1,
        Some(bytes) => i64::try_from(bytes).unwrap_or(i64::MAX),
    };
    CONFIGURED_LIMIT_BYTES.store(encoded, Ordering::Relaxed);
}

/// Computes an appropriate automatic disk KV cache limit in bytes from drive
/// capacity (`total_bytes`, `available_bytes`): 5% of total drive size clamped
/// to `[4 GiB, 64 GiB]` (rounded to whole GiB), and never more than 25% of
/// currently free space (0 if under 8 GiB free).
pub fn automatic_disk_kv_cache_limit_bytes(total_bytes: u64, available_bytes: u64) -> u64 {
    if total_bytes == 0 {
        return FALLBACK_MAX_DISK_BYTES;
    }
    if available_bytes < AUTO_MIN_FREE_BYTES {
        return 0;
    }
    let gib = 1024 * 1024 * 1024u64;
    let five_pct = (total_bytes / 20).clamp(AUTO_MIN_DISK_BYTES, AUTO_MAX_DISK_BYTES);
    let safe_cap = available_bytes / 4;
    let capped = five_pct.min(safe_cap);
    (capped / gib) * gib
}

/// Queries the filesystem backing `dir` (or the home folder) and returns the
/// automatic disk KV cache byte limit for this machine.
pub fn default_disk_kv_cache_limit_bytes(dir: Option<&Path>) -> u64 {
    let probe = dir
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."));
    let existing = existing_ancestor(&probe);
    match (fs4::total_space(&existing), fs4::available_space(&existing)) {
        (Ok(total), Ok(available)) => automatic_disk_kv_cache_limit_bytes(total, available),
        _ => FALLBACK_MAX_DISK_BYTES,
    }
}

/// The nearest folder at or above `path` that exists, since the cache folder
/// may not have been created yet.
fn existing_ancestor(path: &Path) -> PathBuf {
    let mut cur = path.to_path_buf();
    while !cur.exists() {
        if !cur.pop() {
            return path.to_path_buf();
        }
    }
    cur
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiskImageSpanKey {
    start: usize,
    len: usize,
    digest: [u8; 32],
}

impl From<&ImageSpanKey> for DiskImageSpanKey {
    fn from(k: &ImageSpanKey) -> Self {
        Self {
            start: k.start,
            len: k.len,
            digest: k.digest,
        }
    }
}

impl From<&DiskImageSpanKey> for ImageSpanKey {
    fn from(k: &DiskImageSpanKey) -> Self {
        Self {
            start: k.start,
            len: k.len,
            digest: k.digest,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorDesc {
    dtype: String,
    shape: Vec<usize>,
    offset: u64,
    byte_len: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VisitDesc {
    kv: Option<(TensorDesc, TensorDesc)>,
    conv: [Option<TensorDesc>; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LayerDesc {
    Attention {
        k: TensorDesc,
        v: TensorDesc,
    },
    DeltaNet {
        conv_state: TensorDesc,
        recurrent_state: TensorDesc,
    },
    Mamba2 {
        conv_state: TensorDesc,
        ssm_state: TensorDesc,
    },
    Inkling2 {
        position: usize,
        visits: Vec<VisitDesc>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum SpeculativeDesc {
    DSpark {
        next_position: usize,
        layers: Vec<Option<(TensorDesc, TensorDesc)>>,
        f_rnn: Option<TensorDesc>,
    },
    DFlash2 {
        next_position: usize,
        layers: Vec<Option<(TensorDesc, TensorDesc)>>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiskKvMeta {
    version: u32,
    scope: String,
    draft: bool,
    tokens: Vec<u32>,
    image_keys: Vec<DiskImageSpanKey>,
    next_position: usize,
    kv_cache_tokens: usize,
    kv_cache_lane_capacity_tokens: usize,
    fixed_cache_allocation_tokens: usize,
    has_embedded_prompt: bool,
    layers: Vec<Option<LayerDesc>>,
    speculative: Option<SpeculativeDesc>,
    logits: TensorDesc,
    total_blob_bytes: u64,
    saved_at_unix_secs: u64,
}

/// Restored prompt-cache snapshot loaded from disk for any supported architecture.
pub(crate) struct RestoredDiskSnapshot {
    pub(crate) cache: Vec<Option<LayerCache>>,
    pub(crate) speculative_cache: Option<SpeculativeCache>,
    pub(crate) logits: Tensor,
    pub(crate) tokens: Vec<u32>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) image_keys: Vec<ImageSpanKey>,
    pub(crate) next_position: usize,
    pub(crate) kv_cache_tokens: usize,
    pub(crate) kv_cache_lane_capacity_tokens: usize,
    pub(crate) fixed_cache_allocation_tokens: usize,
    pub(crate) has_embedded_prompt: bool,
}

fn in_flight() -> &'static Mutex<HashSet<String>> {
    static SET: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    SET.get_or_init(|| Mutex::new(HashSet::new()))
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

pub(crate) fn enabled() -> bool {
    if matches!(
        std::env::var("DINF_DISK_KV_CACHE").ok().as_deref(),
        Some("0" | "false" | "FALSE" | "off" | "OFF")
    ) {
        return false;
    }
    CONFIGURED_LIMIT_BYTES.load(Ordering::Relaxed) != 0
}

pub(crate) fn min_tokens() -> usize {
    std::env::var("DINF_DISK_KV_CACHE_MIN_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MIN_TOKENS)
}

pub(crate) fn max_disk_bytes() -> u64 {
    if let Some(gb) = std::env::var("DINF_DISK_KV_CACHE_MAX_GB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0)
    {
        return gb.saturating_mul(1024 * 1024 * 1024);
    }
    let configured = CONFIGURED_LIMIT_BYTES.load(Ordering::Relaxed);
    if configured >= 0 {
        return configured as u64;
    }
    default_disk_kv_cache_limit_bytes(cache_root().as_deref())
}

pub(crate) fn cache_dir(model_key: &str) -> Option<PathBuf> {
    if !enabled() {
        return None;
    }
    let root = cache_root()?;
    let safe_key: String = model_key
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    Some(root.join(safe_key))
}

fn cache_root() -> Option<PathBuf> {
    let configured = CONFIGURED_DIR
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    if let Some(dir) = configured {
        return Some(dir);
    }
    if let Some(custom) = std::env::var_os("DINF_DISK_KV_CACHE_DIR") {
        return Some(PathBuf::from(custom));
    }
    Some(user_cache_dir()?.join("dinf-engine").join("kv-cache"))
}

/// The user's cache folder for this platform.
fn user_cache_dir() -> Option<PathBuf> {
    let non_empty = |name: &str| {
        std::env::var_os(name)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    if cfg!(windows) {
        return non_empty("LOCALAPPDATA");
    }
    let home = non_empty("HOME");
    if cfg!(target_os = "macos") {
        return Some(home?.join("Library").join("Caches"));
    }
    non_empty("XDG_CACHE_HOME")
        .filter(|p| p.is_absolute())
        .or_else(|| Some(home?.join(".cache")))
}

fn entry_id(scope: &str, draft: bool, tokens: &[u32], image_keys: &[ImageSpanKey]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    let mix = |h: &mut u64, bytes: &[u8]| {
        for &b in bytes {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100000001b3);
        }
    };
    mix(&mut h, scope.as_bytes());
    mix(&mut h, &[u8::from(draft)]);
    mix(&mut h, &(tokens.len() as u64).to_le_bytes());
    for &t in tokens {
        mix(&mut h, &t.to_le_bytes());
    }
    for img in image_keys {
        mix(&mut h, &(img.start as u64).to_le_bytes());
        mix(&mut h, &(img.len as u64).to_le_bytes());
        mix(&mut h, &img.digest);
    }
    format!("kv_{:016x}_{}", h, tokens.len())
}

fn encode_tensor(tensor: &Tensor, offset: &mut u64) -> Result<(TensorDesc, Vec<u8>)> {
    let cpu = tensor.to_device(&Device::Cpu)?.contiguous()?;
    let shape = cpu.dims().to_vec();
    let (dtype_name, bytes) = match cpu.dtype() {
        DType::F16 => {
            let vals = cpu.flatten_all()?.to_vec1::<half::f16>()?;
            let mut buf = Vec::with_capacity(vals.len() * 2);
            for v in vals {
                buf.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            ("f16".to_string(), buf)
        }
        DType::BF16 => {
            let vals = cpu.flatten_all()?.to_vec1::<half::bf16>()?;
            let mut buf = Vec::with_capacity(vals.len() * 2);
            for v in vals {
                buf.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            ("bf16".to_string(), buf)
        }
        DType::F32 => {
            let vals = cpu.flatten_all()?.to_vec1::<f32>()?;
            let mut buf = Vec::with_capacity(vals.len() * 4);
            for v in vals {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            ("f32".to_string(), buf)
        }
        other => anyhow::bail!("unsupported disk KV cache tensor dtype: {other:?}"),
    };
    let byte_len = bytes.len() as u64;
    let desc = TensorDesc {
        dtype: dtype_name,
        shape,
        offset: *offset,
        byte_len,
    };
    *offset = offset.saturating_add(byte_len);
    Ok((desc, bytes))
}

fn decode_tensor(blob: &[u8], desc: &TensorDesc, device: &Device) -> Result<Tensor> {
    let start = usize::try_from(desc.offset).context("tensor offset overflow")?;
    let len = usize::try_from(desc.byte_len).context("tensor length overflow")?;
    let end = start
        .checked_add(len)
        .filter(|&e| e <= blob.len())
        .context("disk KV cache blob truncated")?;
    let slice = &blob[start..end];
    match desc.dtype.as_str() {
        "f16" => {
            anyhow::ensure!(slice.len().is_multiple_of(2), "invalid f16 byte length");
            let mut vals = Vec::with_capacity(slice.len() / 2);
            for chunk in slice.chunks_exact(2) {
                vals.push(half::f16::from_bits(u16::from_le_bytes([
                    chunk[0], chunk[1],
                ])));
            }
            Ok(Tensor::from_vec(vals, desc.shape.as_slice(), device)?)
        }
        "bf16" => {
            anyhow::ensure!(slice.len().is_multiple_of(2), "invalid bf16 byte length");
            let mut vals = Vec::with_capacity(slice.len() / 2);
            for chunk in slice.chunks_exact(2) {
                vals.push(half::bf16::from_bits(u16::from_le_bytes([
                    chunk[0], chunk[1],
                ])));
            }
            Ok(Tensor::from_vec(vals, desc.shape.as_slice(), device)?)
        }
        "f32" => {
            anyhow::ensure!(slice.len().is_multiple_of(4), "invalid f32 byte length");
            let mut vals = Vec::with_capacity(slice.len() / 4);
            for chunk in slice.chunks_exact(4) {
                vals.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(Tensor::from_vec(vals, desc.shape.as_slice(), device)?)
        }
        other => anyhow::bail!("unsupported disk KV cache tensor dtype: {other}"),
    }
}

fn live_kv_tensors(kv: &KvCache) -> Option<Result<(Tensor, Tensor)>> {
    match kv {
        KvCache::F16 { k, v } => Some(Ok((k.clone(), v.clone()))),
        KvCache::PreallocatedF16(cache) => Some(cache.active_f16()),
        KvCache::Quantized(_) => None,
    }
}

fn supports_layer(layer: &LayerCache) -> bool {
    match layer {
        LayerCache::Attention(kv) => !matches!(kv, KvCache::Quantized(_)),
        LayerCache::DeltaNet(_) | LayerCache::Mamba2(_) | LayerCache::Inkling2(_) => true,
        LayerCache::DeepSeek4(_) | LayerCache::DeepSeek41(_) => false,
    }
}

fn prefix_image_keys_match(
    query: &[ImageSpanKey],
    cached: &[DiskImageSpanKey],
    cached_len: usize,
) -> bool {
    let mut matched = 0usize;
    for key in query {
        if key.start >= cached_len {
            break;
        }
        if key.start.saturating_add(key.len) > cached_len {
            return false;
        }
        let Some(entry) = cached.get(matched) else {
            return false;
        };
        if entry.start != key.start || entry.len != key.len || entry.digest != key.digest {
            return false;
        }
        matched += 1;
    }
    matched == cached.len()
}

fn prune_disk_dir(dir: &Path, max_bytes: u64) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut items: Vec<(u64, u64, PathBuf, PathBuf)> = Vec::new();
    let mut total: u64 = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(".meta.json") {
            continue;
        }
        let Ok(raw) = fs::read(&path) else {
            continue;
        };
        let Ok(meta) = serde_json::from_slice::<DiskKvMeta>(&raw) else {
            let _ = fs::remove_file(&path);
            continue;
        };
        let stem = name.trim_end_matches(".meta.json");
        let bin_path = dir.join(format!("{stem}.bin"));
        let bin_size = fs::metadata(&bin_path).map_or(0, |m| m.len());
        total = total.saturating_add(bin_size);
        items.push((meta.saved_at_unix_secs, bin_size, path, bin_path));
    }
    if total <= max_bytes {
        return;
    }
    items.sort_by_key(|(ts, _, _, _)| *ts);
    for (_, size, meta_path, bin_path) in items {
        if total <= max_bytes {
            break;
        }
        let _ = fs::remove_file(&meta_path);
        let _ = fs::remove_file(&bin_path);
        total = total.saturating_sub(size);
    }
}

/// Persists a prompt-cache snapshot for any supported model to disk on a background thread.
#[allow(clippy::too_many_arguments)]
pub(crate) fn save_snapshot_async(
    dir: PathBuf,
    max_bytes: u64,
    scope: String,
    draft: bool,
    tokens: Vec<u32>,
    image_keys: Vec<ImageSpanKey>,
    cache: &[Option<LayerCache>],
    speculative: Option<&SpeculativeCache>,
    logits: &Tensor,
    next_position: usize,
    kv_cache_tokens: usize,
    kv_cache_lane_capacity_tokens: usize,
    fixed_cache_allocation_tokens: usize,
    has_embedded_prompt: bool,
) -> Result<()> {
    if max_bytes == 0 || !cache.iter().flatten().all(supports_layer) {
        return Ok(());
    }
    let id = entry_id(&scope, draft, &tokens, &image_keys);
    let meta_path = dir.join(format!("{id}.meta.json"));
    if meta_path.exists() {
        return Ok(());
    }
    {
        let mut guard = in_flight().lock().unwrap();
        if !guard.insert(id.clone()) {
            return Ok(());
        }
    }
    let prepare = || -> Result<(DiskKvMeta, Vec<Vec<u8>>)> {
        let mut offset = 0u64;
        let mut chunks: Vec<Vec<u8>> = Vec::new();
        let mut layer_descs = Vec::with_capacity(cache.len());
        for slot in cache {
            let desc = match slot {
                None => None,
                Some(LayerCache::Attention(kv)) => {
                    let (k, v) = live_kv_tensors(kv)
                        .context("quantized KV cache is not supported on disk")??;
                    let (kd, kb) = encode_tensor(&k, &mut offset)?;
                    chunks.push(kb);
                    let (vd, vb) = encode_tensor(&v, &mut offset)?;
                    chunks.push(vb);
                    Some(LayerDesc::Attention { k: kd, v: vd })
                }
                Some(LayerCache::DeltaNet(dn)) => {
                    let (cd, cb) = encode_tensor(&dn.conv_state, &mut offset)?;
                    chunks.push(cb);
                    let (rd, rb) = encode_tensor(&dn.recurrent_state, &mut offset)?;
                    chunks.push(rb);
                    Some(LayerDesc::DeltaNet {
                        conv_state: cd,
                        recurrent_state: rd,
                    })
                }
                Some(LayerCache::Mamba2(m2)) => {
                    let (cd, cb) = encode_tensor(&m2.conv_state, &mut offset)?;
                    chunks.push(cb);
                    let (sd, sb) = encode_tensor(&m2.ssm_state, &mut offset)?;
                    chunks.push(sb);
                    Some(LayerDesc::Mamba2 {
                        conv_state: cd,
                        ssm_state: sd,
                    })
                }
                Some(LayerCache::Inkling2(inkling)) => {
                    let (position, visits) = inkling.export_tensors()?;
                    let mut visit_descs = Vec::with_capacity(visits.len());
                    for (kv, conv) in visits {
                        let kv_desc = match kv {
                            Some((k, v)) => {
                                let (kd, kb) = encode_tensor(&k, &mut offset)?;
                                chunks.push(kb);
                                let (vd, vb) = encode_tensor(&v, &mut offset)?;
                                chunks.push(vb);
                                Some((kd, vd))
                            }
                            None => None,
                        };
                        let mut conv_descs: [Option<TensorDesc>; 4] = [None, None, None, None];
                        for (cslot, tensor) in conv_descs.iter_mut().zip(conv) {
                            if let Some(t) = tensor {
                                let (td, tb) = encode_tensor(&t, &mut offset)?;
                                chunks.push(tb);
                                *cslot = Some(td);
                            }
                        }
                        visit_descs.push(VisitDesc {
                            kv: kv_desc,
                            conv: conv_descs,
                        });
                    }
                    Some(LayerDesc::Inkling2 {
                        position,
                        visits: visit_descs,
                    })
                }
                Some(LayerCache::DeepSeek4(_) | LayerCache::DeepSeek41(_)) => {
                    anyhow::bail!("unsupported layer cache for disk persistence")
                }
            };
            layer_descs.push(desc);
        }
        let speculative_desc = match speculative {
            Some(SpeculativeCache::DSpark(ds)) => {
                let (next_pos, layers, f_rnn) = ds.export_tensors()?;
                let mut ds_layers = Vec::with_capacity(layers.len());
                for kv in layers {
                    let desc = match kv {
                        Some((k, v)) => {
                            let (kd, kb) = encode_tensor(&k, &mut offset)?;
                            chunks.push(kb);
                            let (vd, vb) = encode_tensor(&v, &mut offset)?;
                            chunks.push(vb);
                            Some((kd, vd))
                        }
                        None => None,
                    };
                    ds_layers.push(desc);
                }
                let f_rnn_desc = match f_rnn {
                    Some(t) => {
                        let (td, tb) = encode_tensor(&t, &mut offset)?;
                        chunks.push(tb);
                        Some(td)
                    }
                    None => None,
                };
                Some(SpeculativeDesc::DSpark {
                    next_position: next_pos,
                    layers: ds_layers,
                    f_rnn: f_rnn_desc,
                })
            }
            Some(SpeculativeCache::DFlash2(df)) => {
                let (next_pos, layers) = df.export_tensors();
                let mut df_layers = Vec::with_capacity(layers.len());
                for kv in layers {
                    let desc = match kv {
                        Some((k, v)) => {
                            let (kd, kb) = encode_tensor(&k, &mut offset)?;
                            chunks.push(kb);
                            let (vd, vb) = encode_tensor(&v, &mut offset)?;
                            chunks.push(vb);
                            Some((kd, vd))
                        }
                        None => None,
                    };
                    df_layers.push(desc);
                }
                Some(SpeculativeDesc::DFlash2 {
                    next_position: next_pos,
                    layers: df_layers,
                })
            }
            None => None,
        };
        let (logits_desc, logits_bytes) = encode_tensor(logits, &mut offset)?;
        chunks.push(logits_bytes);
        let meta = DiskKvMeta {
            version: FORMAT_VERSION,
            scope,
            draft,
            tokens,
            image_keys: image_keys.iter().map(DiskImageSpanKey::from).collect(),
            next_position,
            kv_cache_tokens,
            kv_cache_lane_capacity_tokens,
            fixed_cache_allocation_tokens,
            has_embedded_prompt,
            layers: layer_descs,
            speculative: speculative_desc,
            logits: logits_desc,
            total_blob_bytes: offset,
            saved_at_unix_secs: now_unix_secs(),
        };
        Ok((meta, chunks))
    };

    let (meta, chunks) = match prepare() {
        Ok(prepared) => prepared,
        Err(error) => {
            in_flight().lock().unwrap().remove(&id);
            return Err(error);
        }
    };
    if meta.total_blob_bytes > max_bytes {
        in_flight().lock().unwrap().remove(&id);
        return Ok(());
    }

    let id_clone = id.clone();
    let spawn = std::thread::Builder::new()
        .name("dinf-kv-disk-save".into())
        .spawn(move || {
            let started = Instant::now();
            let result = (|| -> Result<()> {
                fs::create_dir_all(&dir)?;
                let bin_tmp = dir.join(format!("{id_clone}.bin.tmp"));
                let bin_final = dir.join(format!("{id_clone}.bin"));
                let meta_tmp = dir.join(format!("{id_clone}.meta.json.tmp"));
                let meta_final = dir.join(format!("{id_clone}.meta.json"));
                {
                    let file = File::create(&bin_tmp)?;
                    let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, file);
                    for chunk in &chunks {
                        writer.write_all(chunk)?;
                    }
                    writer.flush()?;
                }
                fs::rename(&bin_tmp, &bin_final)?;
                let meta_bytes = serde_json::to_vec(&meta)?;
                fs::write(&meta_tmp, &meta_bytes)?;
                fs::rename(&meta_tmp, &meta_final)?;
                prune_disk_dir(&dir, max_bytes);
                tracing::info!(
                    tokens = meta.tokens.len(),
                    images = meta.image_keys.len(),
                    blob_mb = meta.total_blob_bytes / (1024 * 1024),
                    save_ms = started.elapsed().as_millis() as u64,
                    "Saved KV snapshot to disk"
                );
                Ok(())
            })();
            if let Err(error) = result {
                tracing::warn!(%error, "Failed to persist KV snapshot to disk");
            }
            in_flight().lock().unwrap().remove(&id_clone);
        });
    if let Err(error) = spawn {
        in_flight().lock().unwrap().remove(&id);
        tracing::warn!(%error, "Could not spawn disk KV cache writer thread");
    }
    Ok(())
}

/// Finds and loads the longest matching KV snapshot from disk for any supported model.
pub(crate) fn load_best_snapshot(
    dir: &Path,
    scope: &str,
    tokens: &[u32],
    image_keys: &[ImageSpanKey],
    draft: bool,
    device: &Device,
) -> Result<Option<RestoredDiskSnapshot>> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(None);
    };
    let mut best: Option<(DiskKvMeta, PathBuf)> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(".meta.json") {
            continue;
        }
        let Ok(raw) = fs::read(&path) else {
            continue;
        };
        let Ok(meta) = serde_json::from_slice::<DiskKvMeta>(&raw) else {
            continue;
        };
        if meta.version != FORMAT_VERSION
            || meta.scope != scope
            || meta.tokens.is_empty()
            || !tokens.starts_with(&meta.tokens)
            || !prefix_image_keys_match(image_keys, &meta.image_keys, meta.tokens.len())
            || (draft && (!meta.draft || meta.speculative.is_none()))
        {
            continue;
        }
        let stem = name.trim_end_matches(".meta.json");
        let bin_path = dir.join(format!("{stem}.bin"));
        if !bin_path.exists() {
            continue;
        }
        let better = best
            .as_ref()
            .is_none_or(|(prev, _)| meta.tokens.len() > prev.tokens.len());
        if better {
            best = Some((meta, bin_path));
        }
    }
    let Some((meta, bin_path)) = best else {
        return Ok(None);
    };
    let started = Instant::now();
    let blob = {
        let file = File::open(&bin_path)?;
        let expected = usize::try_from(meta.total_blob_bytes).unwrap_or(0);
        let mut buf = Vec::with_capacity(expected);
        let mut reader = BufReader::with_capacity(4 * 1024 * 1024, file);
        reader.read_to_end(&mut buf)?;
        buf
    };
    if blob.len() as u64 != meta.total_blob_bytes {
        tracing::warn!(
            path = %bin_path.display(),
            expected = meta.total_blob_bytes,
            actual = blob.len(),
            "Ignoring truncated disk KV snapshot"
        );
        return Ok(None);
    }
    let mut cache = Vec::with_capacity(meta.layers.len());
    for layer_desc in &meta.layers {
        let layer = match layer_desc {
            None => None,
            Some(LayerDesc::Attention { k, v }) => {
                let kt = copy_persistent_tensor(&decode_tensor(&blob, k, device)?)?;
                let vt = copy_persistent_tensor(&decode_tensor(&blob, v, device)?)?;
                Some(LayerCache::Attention(KvCache::PreallocatedF16(
                    PreallocatedKvCache::from_f16(kt, vt)?,
                )))
            }
            Some(LayerDesc::DeltaNet {
                conv_state,
                recurrent_state,
            }) => Some(LayerCache::DeltaNet(DeltaNetCache {
                conv_state: copy_persistent_tensor(&decode_tensor(&blob, conv_state, device)?)?,
                recurrent_state: copy_persistent_tensor(&decode_tensor(
                    &blob,
                    recurrent_state,
                    device,
                )?)?,
                speculative_state: None,
            })),
            Some(LayerDesc::Mamba2 {
                conv_state,
                ssm_state,
            }) => Some(LayerCache::Mamba2(Mamba2Cache {
                conv_state: copy_persistent_tensor(&decode_tensor(&blob, conv_state, device)?)?,
                ssm_state: copy_persistent_tensor(&decode_tensor(&blob, ssm_state, device)?)?,
            })),
            Some(LayerDesc::Inkling2 { position, visits }) => {
                let mut restored_visits = Vec::with_capacity(visits.len());
                for vd in visits {
                    let kv = match &vd.kv {
                        Some((kd, vd_desc)) => Some((
                            decode_tensor(&blob, kd, device)?,
                            decode_tensor(&blob, vd_desc, device)?,
                        )),
                        None => None,
                    };
                    let mut conv: [Option<Tensor>; 4] = [None, None, None, None];
                    for (slot, desc) in conv.iter_mut().zip(&vd.conv) {
                        if let Some(d) = desc {
                            *slot = Some(decode_tensor(&blob, d, device)?);
                        }
                    }
                    restored_visits.push((kv, conv));
                }
                Some(LayerCache::Inkling2(Box::new(
                    Inkling2Cache::import_tensors(*position, restored_visits)?,
                )))
            }
        };
        cache.push(layer);
    }
    let speculative_cache = if draft {
        match &meta.speculative {
            Some(SpeculativeDesc::DSpark {
                next_position,
                layers,
                f_rnn,
            }) => {
                let mut ds_layers = Vec::with_capacity(layers.len());
                for layer in layers {
                    let kv = match layer {
                        Some((kd, vd)) => Some((
                            decode_tensor(&blob, kd, device)?,
                            decode_tensor(&blob, vd, device)?,
                        )),
                        None => None,
                    };
                    ds_layers.push(kv);
                }
                let f_rnn_t = f_rnn
                    .as_ref()
                    .map(|d| decode_tensor(&blob, d, device))
                    .transpose()?;
                Some(SpeculativeCache::DSpark(DSparkCache::import_tensors(
                    *next_position,
                    ds_layers,
                    f_rnn_t,
                )?))
            }
            Some(SpeculativeDesc::DFlash2 {
                next_position,
                layers,
            }) => {
                let mut df_layers = Vec::with_capacity(layers.len());
                for layer in layers {
                    let kv = match layer {
                        Some((kd, vd)) => Some((
                            decode_tensor(&blob, kd, device)?,
                            decode_tensor(&blob, vd, device)?,
                        )),
                        None => None,
                    };
                    df_layers.push(kv);
                }
                Some(SpeculativeCache::DFlash2(DFlash2Cache::import_tensors(
                    *next_position,
                    df_layers,
                )?))
            }
            None => None,
        }
    } else {
        None
    };
    let logits = decode_tensor(&blob, &meta.logits, device)?;
    let restored_images: Vec<ImageSpanKey> =
        meta.image_keys.iter().map(ImageSpanKey::from).collect();
    tracing::info!(
        reused_prompt_tokens = meta.tokens.len(),
        reused_images = restored_images.len(),
        blob_mb = meta.total_blob_bytes / (1024 * 1024),
        load_ms = started.elapsed().as_millis() as u64,
        "Disk KV cache hit"
    );
    Ok(Some(RestoredDiskSnapshot {
        cache,
        speculative_cache,
        logits,
        tokens: meta.tokens,
        image_keys: restored_images,
        next_position: meta.next_position,
        kv_cache_tokens: meta.kv_cache_tokens,
        kv_cache_lane_capacity_tokens: meta.kv_cache_lane_capacity_tokens,
        fixed_cache_allocation_tokens: meta.fixed_cache_allocation_tokens,
        has_embedded_prompt: meta.has_embedded_prompt,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait_in_flight() {
        for _ in 0..100 {
            if !in_flight().lock().unwrap().is_empty() {
                std::thread::sleep(std::time::Duration::from_millis(10));
            } else {
                break;
            }
        }
    }

    #[test]
    fn round_trips_inkling2_and_dspark_snapshot_on_disk() -> Result<()> {
        let dir = std::env::temp_dir().join(format!(
            "dinf_disk_kv_inkling_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let device = Device::Cpu;
        let k = Tensor::from_vec(
            vec![half::f16::from_f32(1.25); 1 * 2 * 4 * 8],
            (1, 2, 4, 8),
            &device,
        )?;
        let v = Tensor::from_vec(
            vec![half::f16::from_f32(-0.5); 1 * 2 * 4 * 8],
            (1, 2, 4, 8),
            &device,
        )?;
        let conv = Tensor::from_vec(vec![0.75f32; 3 * 16], (3, 16), &device)?;
        let inkling = Inkling2Cache::import_tensors(
            4,
            vec![(Some((k.clone(), v.clone())), [Some(conv), None, None, None])],
        )?;
        let dspark = SpeculativeCache::DSpark(DSparkCache::import_tensors(
            4,
            vec![Some((k, v))],
            Some(Tensor::from_vec(vec![2.0f32; 16], (1, 16), &device)?),
        )?);
        let logits = Tensor::from_vec(vec![0.1f32, 0.2, 0.9, -0.4], (1, 4), &device)?;
        let tokens = vec![10u32, 20, 30, 40];
        let image_keys = vec![ImageSpanKey {
            start: 1,
            len: 2,
            digest: [7u8; 32],
        }];
        let cache = vec![Some(LayerCache::Inkling2(Box::new(inkling)))];

        save_snapshot_async(
            dir.clone(),
            1024 * 1024 * 1024,
            "scope-a".into(),
            true,
            tokens.clone(),
            image_keys.clone(),
            &cache,
            Some(&dspark),
            &logits,
            4,
            4,
            4,
            0,
            true,
        )?;
        wait_in_flight();

        let query_tokens = vec![10u32, 20, 30, 40, 50, 60];
        let restored =
            load_best_snapshot(&dir, "scope-a", &query_tokens, &image_keys, true, &device)?
                .expect("disk snapshot should match prefix");
        assert_eq!(restored.tokens, tokens);
        assert_eq!(restored.image_keys, image_keys);
        assert_eq!(restored.next_position, 4);
        assert!(matches!(
            restored.speculative_cache,
            Some(SpeculativeCache::DSpark(_))
        ));
        assert_eq!(
            restored.logits.flatten_all()?.to_vec1::<f32>()?,
            vec![0.1f32, 0.2, 0.9, -0.4]
        );

        let bad_images = vec![ImageSpanKey {
            start: 1,
            len: 2,
            digest: [9u8; 32],
        }];
        assert!(
            load_best_snapshot(&dir, "scope-a", &query_tokens, &bad_images, true, &device)?
                .is_none()
        );

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn round_trips_hybrid_attention_deltanet_mamba2_and_dflash2_on_disk() -> Result<()> {
        let dir = std::env::temp_dir().join(format!(
            "dinf_disk_kv_hybrid_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let device = Device::Cpu;
        let k = Tensor::from_vec(
            vec![half::f16::from_f32(0.5); 1 * 2 * 4 * 8],
            (1, 2, 4, 8),
            &device,
        )?;
        let v = Tensor::from_vec(
            vec![half::f16::from_f32(-1.0); 1 * 2 * 4 * 8],
            (1, 2, 4, 8),
            &device,
        )?;
        let cache = vec![
            Some(LayerCache::Attention(KvCache::F16 {
                k: k.clone(),
                v: v.clone(),
            })),
            Some(LayerCache::DeltaNet(DeltaNetCache {
                conv_state: Tensor::from_vec(vec![1.0f32; 8 * 4], (8, 4), &device)?,
                recurrent_state: Tensor::from_vec(vec![0.25f32; 2 * 4 * 4], (2, 4, 4), &device)?,
                speculative_state: None,
            })),
            Some(LayerCache::Mamba2(Mamba2Cache {
                conv_state: Tensor::from_vec(vec![-0.5f32; 8 * 4], (8, 4), &device)?,
                ssm_state: Tensor::from_vec(vec![0.125f32; 2 * 4 * 4], (2, 4, 4), &device)?,
            })),
        ];
        let dflash =
            SpeculativeCache::DFlash2(DFlash2Cache::import_tensors(4, vec![Some((k, v))])?);
        let logits = Tensor::from_vec(vec![0.3f32, -0.2, 0.7], (1, 3), &device)?;
        let tokens = vec![1u32, 2, 3, 4];

        save_snapshot_async(
            dir.clone(),
            1024 * 1024 * 1024,
            "scope-hybrid".into(),
            true,
            tokens.clone(),
            vec![],
            &cache,
            Some(&dflash),
            &logits,
            4,
            4,
            4,
            0,
            false,
        )?;
        wait_in_flight();

        let restored =
            load_best_snapshot(&dir, "scope-hybrid", &[1, 2, 3, 4, 5], &[], true, &device)?
                .expect("hybrid snapshot should restore");
        assert_eq!(restored.cache.len(), 3);
        assert!(matches!(restored.cache[0], Some(LayerCache::Attention(_))));
        assert!(matches!(restored.cache[1], Some(LayerCache::DeltaNet(_))));
        assert!(matches!(restored.cache[2], Some(LayerCache::Mamba2(_))));
        assert!(matches!(
            restored.speculative_cache,
            Some(SpeculativeCache::DFlash2(_))
        ));

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn automatic_limit_scales_with_drive_size_and_free_space() {
        let gib = 1024 * 1024 * 1024u64;
        assert_eq!(
            automatic_disk_kv_cache_limit_bytes(256 * gib, 150 * gib),
            12 * gib
        );
        assert_eq!(
            automatic_disk_kv_cache_limit_bytes(512 * gib, 300 * gib),
            25 * gib
        );
        assert_eq!(
            automatic_disk_kv_cache_limit_bytes(1024 * gib, 600 * gib),
            51 * gib
        );
        assert_eq!(
            automatic_disk_kv_cache_limit_bytes(4096 * gib, 2000 * gib),
            64 * gib
        );
        assert_eq!(
            automatic_disk_kv_cache_limit_bytes(2048 * gib, 16 * gib),
            4 * gib
        );
        assert_eq!(automatic_disk_kv_cache_limit_bytes(2048 * gib, 4 * gib), 0);
    }
}
