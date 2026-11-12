//! Whole-tile Maya attention for prompt tiles: one dispatch computes every
//! row's sink softmax over its raw window and selected compressed rows.
//!
//! Row `t` of a tile at absolute position `base + t` attends to the raw rows
//! `max(0, p - 127)..=p` (taken from an extended window holding the
//! `raw_prev` rows before the tile followed by the tile's own rows) and then
//! to its selected compressed rows, matching the row path's key order. Keys
//! double as values. Each row reads only its own keys, so results do not
//! depend on how a prompt is split into tiles. Output is BF16-rounded, as the
//! row path's `heads` returns it.
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};

/// Most selected compressed rows per query, matching `select_scores`.
pub(super) const MAX_SELECTED: usize = 512;
/// Raw window length, matching `DeepSeek41Cache::append_raw`.
pub(super) const RAW_WINDOW: usize = 128;
const HEADS: usize = 64;
const WIDTH: usize = 512;

/// Per-row selection for one owner within a tile: `ids` holds every row's
/// selected compressed-row indices back to back (F32, as `ops::topk` returns
/// them); `rows[2 * t]` is row `t`'s offset into `ids` and `rows[2 * t + 1]`
/// its count.
#[derive(Clone)]
pub(super) struct TileSelection {
    pub ids: Option<Tensor>,
    pub rows: Vec<u32>,
}

/// `q` `[T * 64, 512]` (roped), `raw` `[raw_prev + T, 512]`, `compressed`
/// `[N, 512]` holding every row's visible prefix, `sinks` `[64]`. Returns
/// BF16 heads `[T * 64, 512]` before the inverse RoPE.
pub(super) fn forward(
    q: &Tensor,
    raw: &Tensor,
    raw_prev: usize,
    compressed: Option<&Tensor>,
    selection: Option<&TileSelection>,
    sinks: &Tensor,
) -> Result<Tensor> {
    let tokens = raw
        .dim(0)?
        .checked_sub(raw_prev)
        .context("tile attention raw window shorter than its prefix")?;
    ensure!(
        tokens > 0
            && raw_prev < RAW_WINDOW
            && q.dims() == [tokens * HEADS, WIDTH]
            && raw.dims()[1] == WIDTH
            && sinks.dims() == [HEADS],
        "invalid tile attention geometry"
    );
    if let Some(selection) = selection {
        ensure!(
            selection.rows.len() == 2 * tokens
                && compressed.is_some_and(|c| c.rank() == 2 && c.dims()[1] == WIDTH)
                && selection
                    .rows
                    .chunks_exact(2)
                    .all(|r| r[1] as usize <= MAX_SELECTED),
            "invalid tile attention selection"
        );
        let total: u32 = selection.rows.chunks_exact(2).map(|r| r[1]).sum();
        ensure!(
            selection.ids.as_ref().map_or(0, |i| i.elem_count()) >= total as usize,
            "tile attention selection ids are shorter than their counts"
        );
    }
    #[cfg(target_os = "macos")]
    if q.device().is_metal() {
        return metal::forward(q, raw, raw_prev, tokens, compressed, selection, sinks);
    }
    reference(q, raw, raw_prev, tokens, compressed, selection, sinks)
}

/// Plain F32 implementation of the same math, for CPU devices and tests.
fn reference(
    q: &Tensor,
    raw: &Tensor,
    raw_prev: usize,
    tokens: usize,
    compressed: Option<&Tensor>,
    selection: Option<&TileSelection>,
    sinks: &Tensor,
) -> Result<Tensor> {
    let device = q.device().clone();
    let q = q.flatten_all()?.to_vec1::<f32>()?;
    let raw = raw.flatten_all()?.to_vec1::<f32>()?;
    let compressed = compressed
        .map(|c| c.flatten_all()?.to_vec1::<f32>())
        .transpose()?;
    let ids = selection
        .and_then(|s| s.ids.as_ref())
        .map(|i| i.flatten_all()?.to_vec1::<f32>())
        .transpose()?
        .unwrap_or_default();
    let sinks = sinks.to_vec1::<f32>()?;
    let scale = (1.0 / (WIDTH as f64).sqrt()) as f32;
    let mut out = vec![0f32; tokens * HEADS * WIDTH];
    for t in 0..tokens {
        let end = raw_prev + t + 1;
        let start = end.saturating_sub(RAW_WINDOW);
        let mut keys: Vec<&[f32]> = (start..end)
            .map(|r| &raw[r * WIDTH..(r + 1) * WIDTH])
            .collect();
        if let Some(selection) = selection {
            let (offset, count) = (selection.rows[2 * t], selection.rows[2 * t + 1]);
            let compressed = compressed.as_ref().context("missing compressed rows")?;
            for &id in &ids[offset as usize..(offset + count) as usize] {
                let id = id as usize;
                keys.push(&compressed[id * WIDTH..(id + 1) * WIDTH]);
            }
        }
        for h in 0..HEADS {
            let query = &q[(t * HEADS + h) * WIDTH..(t * HEADS + h + 1) * WIDTH];
            let scores: Vec<f32> = keys
                .iter()
                .map(|k| query.iter().zip(*k).map(|(a, b)| a * b).sum::<f32>() * scale)
                .collect();
            let max = scores.iter().fold(sinks[h], |m, &s| m.max(s));
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum = exps.iter().sum::<f32>() + (sinks[h] - max).exp();
            let row = &mut out[(t * HEADS + h) * WIDTH..(t * HEADS + h + 1) * WIDTH];
            for (e, k) in exps.iter().zip(&keys) {
                for (o, v) in row.iter_mut().zip(*k) {
                    *o += e * v;
                }
            }
            for o in row.iter_mut() {
                *o = super::math::bf16_scalar(*o / sum);
            }
        }
    }
    Ok(Tensor::from_vec(out, (tokens * HEADS, WIDTH), &device)?)
}

#[cfg(target_os = "macos")]
mod metal {
    use super::*;
    use candle_core::{MetalStorage, Shape, Storage, backend::BackendStorage};
    use candle_metal_kernels::metal::{Buffer, ComputePipeline};
    use objc2_metal::{
        MTLCompileOptions, MTLComputePipelineState, MTLMathFloatingPointFunctions, MTLMathMode,
        MTLSize,
    };
    use std::collections::HashMap;
    use std::sync::{LazyLock, Mutex};

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct TileArgs { uint raw_prev; uint raw_window; float scale; };
static inline float bf16r(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}
// One threadgroup per (four heads, row). 8 SIMD groups score keys; then
// each thread accumulates two output columns for the four heads.
kernel void dinf_v41_tile_attention(
        device const float *q [[buffer(0)]],
        device const float *raw [[buffer(1)]],
        device const float *comp [[buffer(2)]],
        device const float *ids [[buffer(3)]],
        device const uint *sel [[buffer(4)]],
        device const float *sinks [[buffer(5)]],
        device float *out [[buffer(6)]],
        constant TileArgs &a [[buffer(7)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint H = 4, D = 512, MAXK = 640;
    threadgroup float qs[H * D];
    threadgroup float s[H * MAXK];
    threadgroup float inv[H];
    const uint t = group.y, h0 = group.x * H;
    for (uint i = tid; i < H * D; i += 256) qs[i] = q[(ulong(t) * 64 + h0) * D + i];
    const uint end = a.raw_prev + t + 1;
    const uint start = end > a.raw_window ? end - a.raw_window : 0u;
    const uint nraw = end - start;
    const uint offset = sel[2 * t], count = sel[2 * t + 1];
    const uint n = nraw + count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = sg; j < n; j += 8) {
        device const float *k = j < nraw ? raw + ulong(start + j) * D
                                         : comp + ulong(uint(ids[offset + j - nraw])) * D;
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        for (uint i = lane; i < D; i += 32) {
            const float kv = k[i];
            d0 = fma(qs[i], kv, d0);
            d1 = fma(qs[D + i], kv, d1);
            d2 = fma(qs[2 * D + i], kv, d2);
            d3 = fma(qs[3 * D + i], kv, d3);
        }
        d0 = simd_sum(d0); d1 = simd_sum(d1); d2 = simd_sum(d2); d3 = simd_sum(d3);
        if (lane == 0) {
            s[j] = d0 * a.scale;
            s[MAXK + j] = d1 * a.scale;
            s[2 * MAXK + j] = d2 * a.scale;
            s[3 * MAXK + j] = d3 * a.scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < H) {
        threadgroup float *row = s + sg * MAXK;
        const float sink = sinks[h0 + sg];
        float m = sink;
        for (uint j = lane; j < n; j += 32) m = max(m, row[j]);
        m = simd_max(m);
        float sum = 0.0f;
        for (uint j = lane; j < n; j += 32) {
            const float e = exp(row[j] - m);
            row[j] = e;
            sum += e;
        }
        sum = simd_sum(sum) + exp(sink - m);
        if (lane == 0) inv[sg] = 1.0f / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[H][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
    for (uint j = 0; j < n; ++j) {
        device const float *k = j < nraw ? raw + ulong(start + j) * D
                                         : comp + ulong(uint(ids[offset + j - nraw])) * D;
        const float v0 = k[tid], v1 = k[tid + 256];
        for (uint h = 0; h < H; ++h) {
            const float p = s[h * MAXK + j];
            acc[h][0] = fma(p, v0, acc[h][0]);
            acc[h][1] = fma(p, v1, acc[h][1]);
        }
    }
    for (uint h = 0; h < H; ++h) {
        device float *o = out + (ulong(t) * 64 + h0 + h) * D;
        o[tid] = bf16r(acc[h][0] * inv[h]);
        o[tid + 256] = bf16r(acc[h][1] * inv[h]);
    }
}
"#;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TileArgs {
        raw_prev: u32,
        raw_window: u32,
        scale: f32,
    }

    type Cache = HashMap<candle_core::metal_backend::DeviceId, ComputePipeline>;
    static PIPELINES: LazyLock<Mutex<Cache>> = LazyLock::new(|| Mutex::new(HashMap::new()));

    fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
        let mut cache = PIPELINES
            .lock()
            .map_err(|_| anyhow::anyhow!("tile attention pipeline cache poisoned"))?;
        if let Some(p) = cache.get(&device.id()) {
            return Ok(p.clone());
        }
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        let library = device
            .metal_device()
            .new_library_with_source(SOURCE, Some(&options))
            .map_err(|e| anyhow::anyhow!("tile attention Metal compile: {e}"))?;
        let function = library
            .get_function("dinf_v41_tile_attention", None)
            .map_err(|e| anyhow::anyhow!("tile attention function: {e}"))?;
        let p = device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| anyhow::anyhow!("tile attention pipeline: {e}"))?;
        ensure!(
            p.as_ref().threadExecutionWidth() == 32 && p.max_total_threads_per_threadgroup() >= 256,
            "tile attention needs 32-lane SIMD groups and 256-thread groups"
        );
        cache.insert(device.id(), p.clone());
        Ok(p)
    }

    fn operand(t: &Tensor, device: &candle_core::MetalDevice) -> Result<(Buffer, usize)> {
        ensure!(
            t.dtype() == DType::F32 && t.is_contiguous(),
            "tile attention operands must be contiguous F32"
        );
        let (storage, layout) = t.storage_and_layout();
        let Storage::Metal(s) = &*storage else {
            anyhow::bail!("tile attention operand is not on Metal")
        };
        ensure!(
            s.device().id() == device.id(),
            "tile attention device mismatch"
        );
        let offset = layout.start_offset() * 4;
        ensure!(
            offset + t.elem_count() * 4 <= s.buffer().length(),
            "tile attention operand exceeds its buffer"
        );
        Ok((s.buffer().clone(), offset))
    }

    pub(super) fn forward(
        q: &Tensor,
        raw: &Tensor,
        raw_prev: usize,
        tokens: usize,
        compressed: Option<&Tensor>,
        selection: Option<&TileSelection>,
        sinks: &Tensor,
    ) -> Result<Tensor> {
        let Device::Metal(device) = q.device() else {
            anyhow::bail!("tile attention requires Metal")
        };
        let p = pipeline(device)?;
        let rows = selection
            .map(|s| s.rows.clone())
            .unwrap_or_else(|| vec![0; 2 * tokens]);
        let q = operand(q, device)?;
        let raw = operand(raw, device)?;
        let sinks = operand(sinks, device)?;
        // Unused bindings point at the raw window; zero counts never read them.
        let comp = compressed.map(|c| operand(c, device)).transpose()?;
        let ids = selection
            .and_then(|s| s.ids.as_ref())
            .map(|i| operand(i, device))
            .transpose()?;
        let rows = device.new_buffer_with_data(&rows)?;
        let elements = tokens * HEADS * WIDTH;
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-v41-tile-attention")
            .build()?;
        let args = TileArgs {
            raw_prev: raw_prev as u32,
            raw_window: RAW_WINDOW as u32,
            scale: (1.0 / (WIDTH as f64).sqrt()) as f32,
        };
        let encoder = device.command_encoder()?;
        let e = encoder.as_ref();
        e.set_compute_pipeline_state(&p);
        e.set_input_buffer(0, Some(&q.0), q.1);
        e.set_input_buffer(1, Some(&raw.0), raw.1);
        let comp = comp.as_ref().unwrap_or(&raw);
        e.set_input_buffer(2, Some(&comp.0), comp.1);
        let ids = ids.as_ref().unwrap_or(&raw);
        e.set_input_buffer(3, Some(&ids.0), ids.1);
        e.set_input_buffer(4, Some(&*rows), 0);
        e.set_input_buffer(5, Some(&sinks.0), sinks.1);
        e.set_output_buffer(6, Some(&output), 0);
        e.set_bytes(7, &args);
        e.dispatch_thread_groups(
            MTLSize {
                width: HEADS / 4,
                height: tokens,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(Tensor::from((
            Storage::Metal(MetalStorage::new(
                output,
                device.clone(),
                elements,
                DType::F32,
            )),
            Shape::from((tokens * HEADS, WIDTH)),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 7919 + seed * 104_729) % 2003) as f32 / 1001.0 - 1.0) * scale)
            .collect()
    }

    #[test]
    fn metal_tile_attention_matches_reference() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let cpu = Device::Cpu;
        for (raw_prev, tokens, selected) in
            [(0usize, 5usize, false), (127, 9, true), (40, 130, true)]
        {
            let q = values(tokens * HEADS * WIDTH, 1, 0.3);
            let raw = values((raw_prev + tokens) * WIDTH, 2, 1.0);
            let compressed = values(700 * WIDTH, 3, 1.0);
            let sinks = values(HEADS, 4, 2.0);
            let mut rows = Vec::new();
            let mut ids = Vec::new();
            for t in 0..tokens {
                let count = if selected {
                    (t * 37) % (MAX_SELECTED + 1)
                } else {
                    0
                };
                rows.extend([ids.len() as u32, count as u32]);
                ids.extend((0..count).map(|i| ((i * 13 + t * 7) % 700) as f32));
            }
            let run = |d: &Device| -> Result<Vec<f32>> {
                let selection = selected.then(|| -> Result<TileSelection> {
                    Ok(TileSelection {
                        ids: Some(Tensor::new(ids.as_slice(), d)?),
                        rows: rows.clone(),
                    })
                });
                let selection = selection.transpose()?;
                let compressed = Tensor::from_vec(compressed.clone(), (700, WIDTH), d)?;
                Ok(forward(
                    &Tensor::from_vec(q.clone(), (tokens * HEADS, WIDTH), d)?,
                    &Tensor::from_vec(raw.clone(), (raw_prev + tokens, WIDTH), d)?,
                    raw_prev,
                    selected.then_some(&compressed),
                    selection.as_ref(),
                    &Tensor::new(sinks.as_slice(), d)?,
                )?
                .flatten_all()?
                .to_vec1::<f32>()?)
            };
            let (expected, actual) = (run(&cpu)?, run(&device)?);
            // Both outputs are BF16; allow one BF16 step from F32 sum order.
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                anyhow::ensure!(
                    (a - b).abs() <= b.abs() * 1.0 / 128.0 + 1e-6,
                    "raw_prev={raw_prev} tokens={tokens} element {i}: {a} vs {b}"
                );
            }
            // Rows are independent of how many share a dispatch.
            if tokens > 3 && !selected {
                let part = forward(
                    &Tensor::from_vec(
                        q[..3 * HEADS * WIDTH].to_vec(),
                        (3 * HEADS, WIDTH),
                        &device,
                    )?,
                    &Tensor::from_vec(
                        raw[..(raw_prev + 3) * WIDTH].to_vec(),
                        (raw_prev + 3, WIDTH),
                        &device,
                    )?,
                    raw_prev,
                    None,
                    None,
                    &Tensor::new(sinks.as_slice(), &device)?,
                )?
                .flatten_all()?
                .to_vec1::<f32>()?;
                anyhow::ensure!(
                    part == actual[..part.len()],
                    "tile rows depend on tile size"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn tile_rope_matches_single_position_calls() -> Result<()> {
        let devices = std::iter::once(Device::Cpu).chain(crate::test_metal_device());
        for device in devices {
            let rope = super::super::math::Rope::for_layer(20)?;
            let (positions, heads) = (7usize, 4usize);
            let x = Tensor::from_vec(
                values(positions * heads * 512, 5, 3.0),
                (positions * heads, 512),
                &device,
            )?;
            for (step, inverse) in [(1u32, false), (2, false), (1, true)] {
                let tiled = rope
                    .apply_rows(&x, 1000, heads, step, inverse, inverse)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                for p in 0..positions {
                    let single = rope
                        .apply(
                            &x.narrow(0, p * heads, heads)?,
                            1000 + p as u32 * step,
                            inverse,
                            inverse,
                        )?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    anyhow::ensure!(
                        single == tiled[p * heads * 512..(p + 1) * heads * 512],
                        "position {p} step {step} inverse {inverse} differs"
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod row_tests {
    use super::super::math;
    use anyhow::Result;
    use candle_core::{Device, Tensor};

    fn values(n: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 7919 + seed * 104_729) % 2003) as f32 / 1001.0 - 1.0) * scale)
            .collect()
    }

    /// Tile HC/Engram ops match the single-row ops row by row (bit for bit
    /// except the HC mix, whose dot order differs from Candle's GEMV).
    #[test]
    fn tile_hc_ops_match_single_rows() -> Result<()> {
        let devices = std::iter::once(Device::Cpu).chain(crate::test_metal_device());
        for device in devices {
            let (t, w) = (5usize, 256usize);
            let residual = Tensor::from_vec(values(t * 4 * w, 1, 2.0), (t, 4, w), &device)?;
            let fc = Tensor::from_vec(values(24 * 4 * w, 2, 0.05), (24, 4 * w), &device)?;
            let scale = Tensor::new(&[0.7f32, 1.1, 0.9], &device)?;
            let base = Tensor::from_vec(values(24, 3, 0.5), 24, &device)?;
            let splits = math::hc_split_rows(&residual, &fc, &scale, &base)?;
            let block = Tensor::from_vec(values(t * w, 4, 1.0), (t, w), &device)?;
            let expanded = math::hc_expand_rows(&block, &residual, &splits)?;
            let collapsed = math::hc_collapse_rows(&residual, &splits, 24)?;
            let kv = Tensor::from_vec(values(t * 5 * w, 5, 1.0), (t, 5, w), &device)?;
            let weight = Tensor::from_vec(values(4 * w, 6, 1.0), (4, w), &device)?;
            let gated = math::engram_add_rows(&residual, &kv, &weight, 1e-6)?;
            for r in 0..t {
                let row = residual.get(r)?.contiguous()?;
                let single = math::hc_split(&row, &fc, &scale, &base)?;
                let a = single.backing().to_vec1::<f32>()?;
                let b = splits.get(r)?.to_vec1::<f32>()?;
                for (x, y) in a.iter().zip(&b) {
                    anyhow::ensure!(
                        (x - y).abs() <= 1e-4 * (1.0 + x.abs()),
                        "split row {r}: {x} vs {y}"
                    );
                }
                let split = math::HcSplit::from_backing(splits.get(r)?.contiguous()?)?;
                let one = math::hc_expand(&block.get(r)?.contiguous()?, &row, &split)?;
                anyhow::ensure!(
                    one.flatten_all()?.to_vec1::<f32>()?
                        == expanded.get(r)?.flatten_all()?.to_vec1::<f32>()?,
                    "expand row {r}"
                );
                let one = math::hc_collapse(&row, split.pre())?;
                anyhow::ensure!(
                    one.to_vec1::<f32>()? == collapsed.get(r)?.to_vec1::<f32>()?,
                    "collapse row {r}"
                );
                let one = math::engram_add(
                    &row,
                    &kv.get(r)?.contiguous()?,
                    &weight,
                    &Tensor::ones((4, w), candle_core::DType::F32, &device)?,
                    1e-6,
                )?;
                anyhow::ensure!(
                    one.flatten_all()?.to_vec1::<f32>()?
                        == gated.get(r)?.flatten_all()?.to_vec1::<f32>()?,
                    "engram row {r}"
                );
            }
            // Rows of a split are unchanged by the tile size.
            let part =
                math::hc_split_rows(&residual.narrow(0, 1, 2)?.contiguous()?, &fc, &scale, &base)?;
            anyhow::ensure!(
                part.flatten_all()?.to_vec1::<f32>()?
                    == splits.narrow(0, 1, 2)?.flatten_all()?.to_vec1::<f32>()?,
                "split rows depend on tile size"
            );
        }
        Ok(())
    }
}
