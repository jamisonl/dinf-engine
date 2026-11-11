//! Bounded native cache selection/gather and selected Q8 embedding rows.
//! Derived from ds4 metal/argsort.metal; see adjacent NOTICE and LICENSE-MIT.
use candle_core::{CpuStorage, CustomOp1, DType, Layout, Result, Shape, Tensor};

#[cfg(test)]
thread_local! { pub(super) static INDEX_MANY_ENCODERS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[derive(Clone)]
enum Operation {
    Embedding {
        token: usize,
        width: usize,
    },
    EmbeddingQ5 {
        #[cfg(target_os = "macos")]
        token: usize,
        #[cfg(target_os = "macos")]
        width: usize,
    },
    Topk {
        keep: usize,
    },
    Gather {
        chunks: Vec<Tensor>,
        width: usize,
        rows: usize,
    },
    Blocks,
    Filter {
        mask: Tensor,
    },
    Mask {
        blocks: usize,
    },
}

struct Op {
    operation: Operation,
    shape: Shape,
}

pub fn embedding(bytes: &Tensor, token: usize, width: usize) -> Result<Tensor> {
    bytes.apply_op1_no_bwd(&Op {
        operation: Operation::Embedding { token, width },
        shape: Shape::from(vec![1, width]),
    })
}

pub fn embedding_q5(bytes: &Tensor, token: usize, width: usize) -> Result<Tensor> {
    if bytes.dtype() != DType::U8
        || !bytes.is_contiguous()
        || !width.is_multiple_of(256)
        || width == 0
        || token
            .checked_add(1)
            .and_then(|n| n.checked_mul(width / 256 * 176))
            .is_none_or(|end| end > bytes.elem_count())
    {
        candle_core::bail!("invalid packed Q5_K embedding row");
    }
    bytes.apply_op1_no_bwd(&Op {
        operation: Operation::EmbeddingQ5 {
            #[cfg(target_os = "macos")]
            token,
            #[cfg(target_os = "macos")]
            width,
        },
        shape: Shape::from(vec![1, width]),
    })
}

#[cfg(all(test, target_os = "macos"))]
#[test]
fn coder_q5_embedding_matches_cpu_decoder_with_offset_storage() -> anyhow::Result<()> {
    use candle_core::Device;
    use candle_core::quantized::{GgmlDType, QTensor};
    let device = Device::new_metal(0)?;
    let values = (0..4 * 5120)
        .map(|i| ((i * 37 % 251) as f32 - 125.) / 31.)
        .collect::<Vec<_>>();
    let source = Tensor::from_vec(values, (4, 5120), &Device::Cpu)?;
    let quant = QTensor::quantize(&source, GgmlDType::Q5K)?;
    let expected = quant.dequantize(&Device::Cpu)?.to_vec2::<f32>()?;
    let packed = quant.data()?.to_vec();
    let row_bytes = 5120 / 256 * 176;
    let view = Tensor::from_vec(packed.clone(), packed.len(), &device)?.narrow(
        0,
        row_bytes,
        packed.len() - row_bytes,
    )?;
    for row in [0, 1, 2] {
        let actual = embedding_q5(&view, row, 5120)?.to_vec2::<f32>()?;
        for (i, (&got, &want)) in actual[0].iter().zip(&expected[row + 1]).enumerate() {
            assert!(
                (got - want).abs() <= 5e-6 + want.abs() * 1e-5,
                "row={row} element={i}: {got} != {want}"
            );
        }
    }
    assert!(embedding_q5(&view, 3, 5120).is_err());
    device.synchronize()?;
    Ok(())
}

pub fn topk(scores: &Tensor, keep: usize) -> Result<Tensor> {
    let n = scores.elem_count();
    if scores.dtype() != DType::F32
        || !scores.is_contiguous()
        || keep == 0
        || keep > n
        || n > 1_048_576
    {
        candle_core::bail!("invalid Maya bounded topk");
    }
    scores.apply_op1_no_bwd(&Op {
        operation: Operation::Topk { keep },
        shape: Shape::from(vec![keep]),
    })
}

pub fn gather(ids: &Tensor, chunks: &[Tensor], width: usize, rows: usize) -> Result<Tensor> {
    validate_gather(ids, chunks, width, rows)?;
    ids.apply_op1_no_bwd(&Op {
        operation: Operation::Gather {
            chunks: chunks.to_vec(),
            width,
            rows,
        },
        shape: Shape::from(vec![ids.elem_count(), width]),
    })
}

#[cfg(target_os = "macos")]
const GATHER_PAGES: usize = 24;

// Validate the ENTIRE source list before allocating output or encoding anything.
// No selected-ID readback: the shader checks the causal frontier before loading.
fn validate_gather(ids: &Tensor, chunks: &[Tensor], width: usize, rows: usize) -> Result<()> {
    if chunks.is_empty()
        || ids.dtype() != DType::F32
        || !ids.is_contiguous()
        || ids.elem_count() == 0
        || ids.elem_count() > u32::MAX as usize
        || width == 0
        || width > u32::MAX as usize
        || rows == 0
        || rows > u32::MAX as usize
        || ids
            .elem_count()
            .checked_mul(width)
            .and_then(|n| n.checked_mul(4))
            .is_none()
        || rows
            .checked_mul(width)
            .and_then(|n| n.checked_mul(4))
            .is_none()
    {
        candle_core::bail!("invalid Maya chunk gather");
    }
    fn bounds(t: &Tensor) -> Result<()> {
        let (s, l) = t.storage_and_layout();
        let end = l
            .start_offset()
            .checked_add(t.elem_count())
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| candle_core::Error::Msg("Maya gather offset overflow".into()))?;
        let bytes = match &*s {
            candle_core::Storage::Cpu(s) => s
                .as_slice::<f32>()?
                .len()
                .checked_mul(4)
                .ok_or_else(|| candle_core::Error::Msg("Maya gather storage overflow".into()))?,
            #[cfg(target_os = "macos")]
            candle_core::Storage::Metal(s) => s.buffer().length(),
            _ => candle_core::bail!("unsupported Maya gather device"),
        };
        if end > bytes {
            candle_core::bail!("Maya gather view exceeds storage");
        }
        Ok(())
    }
    bounds(ids)?;
    let mut total = 0usize;
    for chunk in chunks {
        if chunk.dtype() != DType::F32
            || !chunk.is_contiguous()
            || !ids.device().same_device(chunk.device())
            || chunk.rank() != 2
            || chunk.dims()[1] != width
            || chunk.dims()[0] == 0
        {
            candle_core::bail!("invalid Maya gather page");
        }
        total = total
            .checked_add(chunk.dims()[0])
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or_else(|| candle_core::Error::Msg("Maya gather row overflow".into()))?;
        bounds(chunk)?;
    }
    if total != rows {
        candle_core::bail!("Maya gather page rows differ from prefix");
    }
    Ok(())
}

/// Fused candidate scorer: Q[T,32,128], W[T,32], immutable K pages[N,128].
/// Caller owns pages through the normal router/error fence. No full-prefix
/// concatenation or head-by-prefix intermediate. Invisible keys are not read.
pub(crate) fn index_scores_many(
    q: &Tensor,
    weights: &Tensor,
    pages: &[Tensor],
    visible: &[usize],
    pins: &mut Vec<Tensor>,
) -> Result<Tensor> {
    let t = visible.len();
    let n = pages.iter().try_fold(0usize, |sum, p| {
        sum.checked_add(p.dims().first().copied().unwrap_or(0))
            .ok_or_else(|| candle_core::Error::Msg("Maya index row overflow".into()))
    })?;
    if !(1..=super::profile::ATTENTION_V2_MAX_ROWS).contains(&t)
        || q.dims() != [t, 32, 128]
        || weights.dims() != [t, 32]
        || n == 0
        || n > 1_048_576
        || visible.iter().any(|&v| v > n)
        || t.checked_mul(n).and_then(|x| x.checked_mul(4)).is_none()
    {
        candle_core::bail!("invalid Maya multi-query index geometry");
    }
    for page in pages {
        if page.rank() != 2 || page.dims()[1] != 128 || page.dims()[0] == 0 {
            candle_core::bail!("invalid Maya multi-query index page");
        }
    }
    for tensor in std::iter::once(q)
        .chain(std::iter::once(weights))
        .chain(pages)
    {
        if tensor.dtype() != DType::F32
            || !tensor.is_contiguous()
            || !q.device().same_device(tensor.device())
        {
            candle_core::bail!("invalid Maya multi-query index storage");
        }
        let (storage, layout) = tensor.storage_and_layout();
        let end = layout
            .start_offset()
            .checked_add(tensor.elem_count())
            .and_then(|x| x.checked_mul(4))
            .ok_or_else(|| candle_core::Error::Msg("Maya index span overflow".into()))?;
        let bytes = match &*storage {
            candle_core::Storage::Cpu(s) => s.as_slice::<f32>()?.len() * 4,
            #[cfg(target_os = "macos")]
            candle_core::Storage::Metal(s) => s.buffer().length(),
            _ => candle_core::bail!("unsupported Maya multi-query index device"),
        };
        if end > bytes {
            candle_core::bail!("Maya index view exceeds backing");
        }
    }
    pins.push(q.clone());
    pins.push(weights.clone());
    pins.extend(pages.iter().cloned());
    #[cfg(target_os = "macos")]
    if q.device().is_metal() {
        let output = super::cache::private_f32((t, n).into(), q.device())
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        pins.push(output.clone()); // Own output BEFORE the first scoring enqueue.
        super::cache::require_bounded_loan(&output)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        metal::index_scores_into(q, weights, pages, visible, &output)?;
        return Ok(output);
    }
    // Independent small-fixture implementation; never used by native stages.
    let q_data = q.flatten_all()?.to_vec1::<f32>()?;
    let weight_data = weights.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![f32::NEG_INFINITY; t * n];
    let mut base = 0;
    for page in pages {
        let count = page.dims()[0];
        let keys = page.flatten_all()?.to_vec1::<f32>()?;
        for r in 0..t {
            for k in 0..count.min(visible[r].saturating_sub(base)) {
                let mut score = 0f32;
                for h in 0..32 {
                    let mut dot = 0f32;
                    for c in 0..128 {
                        dot += q_data[(r * 32 + h) * 128 + c] * keys[k * 128 + c];
                    }
                    score += dot.max(0.) * weight_data[r * 32 + h];
                }
                out[r * n + base + k] = score / 64.;
            }
        }
        base += count;
    }
    let output = Tensor::from_vec(out, (t, n), q.device())?;
    pins.push(output.clone());
    Ok(output)
}

#[cfg(test)]
#[path = "gather_tests.rs"]
mod gather_tests;

pub fn blocks(scores: &Tensor) -> Result<Tensor> {
    scores.apply_op1_no_bwd(&Op {
        operation: Operation::Blocks,
        shape: Shape::from(vec![scores.elem_count().div_ceil(8)]),
    })
}

pub fn mask(ids: &Tensor, blocks: usize) -> Result<Tensor> {
    ids.apply_op1_no_bwd(&Op {
        operation: Operation::Mask { blocks },
        shape: Shape::from(vec![blocks]),
    })
}

pub fn filter(scores: &Tensor, mask: &Tensor) -> Result<Tensor> {
    scores.apply_op1_no_bwd(&Op {
        operation: Operation::Filter { mask: mask.clone() },
        shape: scores.shape().clone(),
    })
}

fn slice<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    Ok(&s.as_slice::<f32>()?[l.start_offset()..l.start_offset() + l.shape().elem_count()])
}

// Source small rows use strict-comparison bitonic runs, then left-first merges.
// The source radix selector switches wide rows to (score desc, id asc).
fn select(scores: &[f32], keep: usize) -> Vec<f32> {
    let n = scores.len();
    let wide = n >= 16384;
    let run = n.next_power_of_two().min(1024);
    let mut runs = Vec::new();
    for base in (0..n).step_by(run) {
        let mut ids: Vec<usize> = (base..base + run).collect();
        let better = |a: usize, b: usize| {
            a < n && (b >= n || scores[a] > scores[b] || (wide && scores[a] == scores[b] && a < b))
        };
        let mut k = 2;
        while k <= run {
            let mut j = k / 2;
            while j > 0 {
                for col in 0..run {
                    let other = col ^ j;
                    if other > col {
                        let (a, b) = (ids[col], ids[other]);
                        let swap = if col & k == 0 {
                            a >= n || better(b, a)
                        } else {
                            b >= n || better(a, b)
                        };
                        if swap {
                            ids.swap(col, other);
                        }
                    }
                }
                j /= 2;
            }
            k *= 2;
        }
        ids.retain(|&i| i < n);
        ids.truncate(keep);
        runs.push(ids);
    }
    while runs.len() > 1 {
        runs = runs
            .chunks(2)
            .map(|pair| {
                if pair.len() == 1 {
                    return pair[0].clone();
                }
                let (a, b) = (&pair[0], &pair[1]);
                let (mut i, mut j) = (0, 0);
                let mut out = Vec::with_capacity(keep);
                while out.len() < keep && (i < a.len() || j < b.len()) {
                    let left = j == b.len()
                        || (i < a.len()
                            && (scores[a[i]] > scores[b[j]]
                                || (scores[a[i]] == scores[b[j]] && (!wide || a[i] < b[j]))));
                    if left {
                        out.push(a[i]);
                        i += 1;
                    } else {
                        out.push(b[j]);
                        j += 1;
                    }
                }
                out
            })
            .collect();
    }
    runs.pop().unwrap().into_iter().map(|i| i as f32).collect()
}

impl CustomOp1 for Op {
    fn name(&self) -> &'static str {
        "dinf-maya-cache-ops"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let out = match &self.operation {
            Operation::EmbeddingQ5 { .. } => candle_core::bail!("Q5_K embedding requires Metal"),
            Operation::Embedding { token, width } => {
                let b = s.as_slice::<u8>()?;
                let base = l.start_offset() + token * (width / 32) * 34;
                (0..*width)
                    .map(|i| {
                        let block = base + i / 32 * 34;
                        let scale =
                            half::f16::from_bits(u16::from_le_bytes([b[block], b[block + 1]]))
                                .to_f32();
                        scale * (b[block + 2 + i % 32] as i8) as f32
                    })
                    .collect()
            }
            Operation::Topk { keep } => select(slice(s, l)?, *keep),
            Operation::Gather {
                chunks,
                width,
                rows,
            } => {
                let ids = slice(s, l)?;
                let mut all = Vec::with_capacity(rows * width);
                for chunk in chunks {
                    all.extend(chunk.flatten_all()?.to_vec1::<f32>()?);
                }
                let mut out = Vec::with_capacity(ids.len() * width);
                for &id in ids {
                    if id < 0.0 || id as usize >= *rows {
                        candle_core::bail!("Maya gather index out of range");
                    }
                    out.extend_from_slice(&all[id as usize * width..(id as usize + 1) * width]);
                }
                out
            }
            Operation::Blocks => {
                let a = slice(s, l)?;
                let mut out: Vec<f32> = a
                    .chunks(8)
                    .map(|c| c.iter().copied().fold(f32::NEG_INFINITY, f32::max))
                    .collect();
                *out.last_mut().unwrap() = f32::INFINITY;
                out
            }
            Operation::Mask { blocks } => {
                let mut out = vec![f32::NEG_INFINITY; *blocks];
                for &id in slice(s, l)? {
                    out[id as usize] = 0.0;
                }
                out
            }
            Operation::Filter { mask } => {
                let mask = mask.to_vec1::<f32>()?;
                slice(s, l)?
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        if mask[i / 8] == 0.0 {
                            v
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect()
            }
        };
        Ok((CpuStorage::F32(out), self.shape.clone()))
    }

    #[cfg(target_os = "macos")]
    fn metal_fwd(
        &self,
        s: &candle_core::MetalStorage,
        l: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::forward(self, s, l)
    }
}

#[cfg(target_os = "macos")]
mod metal {
    use super::{GATHER_PAGES, Op, Operation};
    use candle_core::backend::BackendStorage;
    use candle_core::{DType, Layout, MetalStorage, Result, Shape, Storage, Tensor};
    use candle_metal_kernels::metal::{ComputePipeline, Library};
    use objc2_metal::{
        MTLCompileOptions, MTLComputePipelineState, MTLLanguageVersion, MTLMathMode, MTLResource,
        MTLSize,
    };
    use std::collections::HashMap;
    use std::sync::{LazyLock, Mutex};

    #[derive(Default)]
    struct Pipelines {
        library: Option<Library>,
        kernels: HashMap<&'static str, ComputePipeline>,
    }
    static PIPELINES: LazyLock<Mutex<HashMap<candle_core::metal_backend::DeviceId, Pipelines>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    fn pipeline(d: &candle_core::MetalDevice, name: &'static str) -> Result<ComputePipeline> {
        let mut cache = PIPELINES
            .lock()
            .map_err(|_| candle_core::Error::Msg("Maya cache pipeline lock poisoned".into()))?;
        let p = cache.entry(d.id()).or_default();
        if let Some(k) = p.kernels.get(name) {
            return Ok(k.clone());
        }
        if p.library.is_none() {
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version4_0);
            options.setMathMode(MTLMathMode::Safe);
            #[cfg(not(test))]
            let source = include_str!("ops.metal");
            #[cfg(test)]
            let source = concat!(
                include_str!("ops.metal"),
                r#"
kernel void maya_gather_legacy(device const float *ids [[buffer(0)]], device const float *chunk [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint id = uint(ids[row]);
    if (id < a.base || id >= a.base + a.count) return;
    for (uint i = lane; i < a.width; i += 256) out[ulong(row) * a.width + i] = chunk[ulong(id - a.base) * a.width + i];
}
"#
            );
            let source = format!(
                "{}\n{}\n{}",
                candle_metal_kernels::source::QUANTIZED,
                source,
                r#"
kernel void coder_embedding_q5(device const uchar *bytes [[buffer(0)]],
        device float *out [[buffer(2)]], constant Args &a [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= a.width / 16) return;
    device const block_q5_K *row = (device const block_q5_K *)(bytes + ulong(a.base) * (a.width / 256) * 176);
    float4x4 values;
    dequantize_q5_K(row + gid / 16, short(gid % 16), values);
    for (uint i = 0; i < 16; ++i) out[gid * 16 + i] = values[i / 4][i % 4];
}
"#
            );
            p.library = Some(
                d.metal_device()
                    .new_library_with_source(&source, Some(&options))
                    .map_err(candle_core::Error::wrap)?,
            );
        }
        let f = p
            .library
            .as_ref()
            .unwrap()
            .get_function(name, None)
            .map_err(candle_core::Error::wrap)?;
        let k = d
            .metal_device()
            .new_compute_pipeline_state_with_function(&f)
            .map_err(candle_core::Error::wrap)?;
        p.kernels.insert(name, k.clone());
        Ok(k)
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Args {
        n: u32,
        keep: u32,
        run: u32,
        stride: u32,
        width: u32,
        base: u32,
        count: u32,
        wide: u32,
    }

    fn tensor_storage(t: &Tensor) -> Result<(std::sync::RwLockReadGuard<'_, Storage>, usize)> {
        if !t.is_contiguous() {
            candle_core::bail!("Maya cache input must be contiguous");
        }
        let (s, l) = t.storage_and_layout();
        Ok((s, l.start_offset() * t.dtype().size_in_bytes()))
    }

    pub(super) fn index_scores_into(
        q: &Tensor,
        weights: &Tensor,
        pages: &[Tensor],
        visible: &[usize],
        output: &Tensor,
    ) -> Result<()> {
        // Retain all storage guards and resolve EVERY fallible resource before
        // obtaining the one encoder. After first dispatch there is no `?`,
        // allocation, pipeline lookup, storage conversion or fallback.
        let (qs, qo) = tensor_storage(q)?;
        let (ws, wo) = tensor_storage(weights)?;
        let (os, oo) = tensor_storage(output)?;
        let Storage::Metal(qs) = &*qs else {
            candle_core::bail!("Maya index Q device");
        };
        let Storage::Metal(ws) = &*ws else {
            candle_core::bail!("Maya index weights device");
        };
        let Storage::Metal(os) = &*os else {
            candle_core::bail!("Maya index output device");
        };
        let d = qs.device();
        for storage in [qs, ws, os] {
            if storage.device().id() != d.id()
                || !std::ptr::eq(
                    storage.buffer().as_ref().device().as_ref(),
                    d.metal_device().as_ref(),
                )
            {
                candle_core::bail!("Maya index actual buffer device mismatch");
            }
        }
        let p = pipeline(d, "maya_index_many")?;
        if p.max_total_threads_per_threadgroup() < 32 || p.as_ref().threadExecutionWidth() != 32 {
            candle_core::bail!("Maya index requires SIMD width32");
        }
        if output.dtype() != DType::F32
            || !q.device().same_device(output.device())
            || output.dims()
                != [
                    visible.len(),
                    pages.iter().map(|p| p.dims()[0]).sum::<usize>(),
                ]
            || oo
                .checked_add(output.elem_count() * 4)
                .is_none_or(|end| end > os.buffer().length())
        {
            candle_core::bail!("Maya index output span");
        }
        let sources = pages
            .iter()
            .map(tensor_storage)
            .collect::<Result<Vec<_>>>()?;
        let mut buffers = Vec::with_capacity(sources.len());
        for (storage, offset) in &sources {
            let Storage::Metal(storage) = &**storage else {
                candle_core::bail!("Maya index page device");
            };
            if storage.device().id() != d.id()
                || !std::ptr::eq(
                    storage.buffer().as_ref().device().as_ref(),
                    d.metal_device().as_ref(),
                )
            {
                candle_core::bail!("Maya index actual page device mismatch");
            }
            buffers.push((storage.buffer(), *offset));
        }
        let mut bounds = [0u32; 32];
        for (bound, &v) in bounds.iter_mut().zip(visible) {
            *bound = v as u32;
        }
        let mut a = Args {
            n: visible.len() as u32,
            width: output.dims()[1] as u32,
            ..Default::default()
        };
        let enc = d.command_encoder()?;
        #[cfg(test)]
        super::INDEX_MANY_ENCODERS.with(|v| v.set(v.get() + 1));
        let enc = enc.as_ref();
        enc.set_compute_pipeline_state(&p);
        enc.set_input_buffer(0, Some(qs.buffer()), qo);
        enc.set_input_buffer(1, Some(ws.buffer()), wo);
        enc.set_output_buffer(3, Some(os.buffer()), oo);
        enc.set_bytes(5, &bounds);
        for ((buffer, offset), page) in buffers.iter().zip(pages) {
            a.count = page.dims()[0] as u32;
            enc.set_input_buffer(2, Some(buffer), *offset);
            enc.set_bytes(4, &a);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: a.count as usize,
                    height: visible.len(),
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            a.base += a.count;
        }
        Ok(())
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Span {
        base: u32,
        count: u32,
    }

    fn gather_into(
        s: &MetalStorage,
        l: &Layout,
        chunks: &[Tensor],
        width: usize,
        rows: usize,
        out: &candle_metal_kernels::metal::Buffer,
    ) -> Result<()> {
        let d = s.device();
        let p = if chunks.len() > 1 {
            Some(pipeline(d, "maya_gather_pages")?)
        } else {
            None
        };
        let single = if chunks.len() % GATHER_PAGES == 1 {
            Some(pipeline(d, "maya_gather_page")?)
        } else {
            None
        };
        for p in p.iter().chain(single.iter()) {
            if p.max_total_threads_per_threadgroup() < 256 {
                candle_core::bail!("Maya gather requires 256 Metal threads");
            }
        }
        let mut base = 0u32;
        let mut a = Args {
            n: l.shape().elem_count() as u32,
            width: width as u32,
            keep: rows as u32,
            ..Default::default()
        };
        for batch in chunks.chunks(GATHER_PAGES) {
            if batch.len() == 1 {
                let chunk = &batch[0];
                a.base = base;
                a.count = chunk.dims()[0] as u32;
                base += a.count;
                let (storage, layout) = chunk.storage_and_layout();
                let Storage::Metal(storage) = &*storage else {
                    unreachable!("preflight validated Metal page");
                };
                let enc = d.command_encoder()?;
                let enc = enc.as_ref();
                enc.set_compute_pipeline_state(single.as_ref().unwrap());
                enc.set_input_buffer(0, Some(s.buffer()), l.start_offset() * 4);
                enc.set_input_buffer(1, Some(storage.buffer()), layout.start_offset() * 4);
                enc.set_output_buffer(2, Some(out), 0);
                enc.set_bytes(3, &a);
                enc.dispatch_thread_groups(
                    MTLSize {
                        width: a.n as usize,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                continue;
            }
            let mut spans = [Span::default(); GATHER_PAGES];
            // All geometry/storage checks have already succeeded, before enqueue.
            for (span, chunk) in spans.iter_mut().zip(batch) {
                *span = Span {
                    base,
                    count: chunk.dims()[0] as u32,
                };
                base += span.count;
            }
            a.count = batch.len() as u32;
            let enc = d.command_encoder()?;
            let enc = enc.as_ref();
            enc.set_compute_pipeline_state(p.as_ref().unwrap());
            enc.set_input_buffer(0, Some(s.buffer()), l.start_offset() * 4);
            enc.set_bytes(1, &spans);
            enc.set_output_buffer(2, Some(out), 0);
            enc.set_bytes(3, &a);
            for page in 0..GATHER_PAGES {
                // Valid filler, count=0: no unset resources, no dereference.
                let chunk = batch.get(page).unwrap_or(&batch[0]);
                let (storage, layout) = chunk.storage_and_layout();
                let Storage::Metal(storage) = &*storage else {
                    unreachable!("preflight validated Metal pages");
                };
                enc.set_input_buffer(4 + page, Some(storage.buffer()), layout.start_offset() * 4);
            }
            enc.dispatch_thread_groups(
                MTLSize {
                    width: a.n as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Ok(())
    }

    // Same input/output resources for both benchmark arms. The reference kernel
    // and per-page dispatch loop are frozen above and exist only in test builds.
    #[cfg(test)]
    pub(super) fn test_gather_into(
        ids: &Tensor,
        chunks: &[Tensor],
        width: usize,
        rows: usize,
        output: &Tensor,
        legacy: bool,
    ) -> Result<()> {
        super::validate_gather(ids, chunks, width, rows)?;
        let (s, l) = ids.storage_and_layout();
        let Storage::Metal(s) = &*s else {
            candle_core::bail!("Metal fixture required");
        };
        let (o, _) = output.storage_and_layout();
        let Storage::Metal(o) = &*o else {
            candle_core::bail!("Metal output required");
        };
        if !legacy {
            return gather_into(s, l, chunks, width, rows, o.buffer());
        }
        let d = s.device();
        let p = pipeline(d, "maya_gather_legacy")?;
        let mut a = Args {
            n: ids.elem_count() as u32,
            width: width as u32,
            ..Default::default()
        };
        let mut base = 0;
        for chunk in chunks {
            a.base = base;
            a.count = chunk.dim(0)? as u32;
            let (cs, co) = tensor_storage(chunk)?;
            let Storage::Metal(cs) = &*cs else {
                unreachable!();
            };
            let enc = d.command_encoder()?;
            let enc = enc.as_ref();
            enc.set_compute_pipeline_state(&p);
            enc.set_input_buffer(0, Some(s.buffer()), l.start_offset() * 4);
            enc.set_input_buffer(1, Some(cs.buffer()), co);
            enc.set_output_buffer(2, Some(o.buffer()), 0);
            enc.set_bytes(3, &a);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: ids.elem_count(),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            base += a.count;
        }
        Ok(())
    }

    pub(super) fn forward(op: &Op, s: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
        let d = s.device();
        if !l.is_contiguous() {
            candle_core::bail!("Maya cache requires contiguous input");
        }
        let elements = op.shape.elem_count();
        let out = d
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("maya-cache-op")
            .build()?;
        let input_offset = l.start_offset() * s.dtype().size_in_bytes();
        let n = l.shape().elem_count();
        let encode = |name: &'static str,
                      a: &Args,
                      input: &candle_metal_kernels::metal::Buffer,
                      input_off: usize,
                      second: Option<(&candle_metal_kernels::metal::Buffer, usize)>,
                      output: &candle_metal_kernels::metal::Buffer,
                      groups: usize,
                      threads: usize|
         -> Result<()> {
            let p = pipeline(d, name)?;
            if p.max_total_threads_per_threadgroup() < threads {
                candle_core::bail!("Maya selection requires {threads} Metal threads");
            }
            let enc = d.command_encoder()?;
            let enc = enc.as_ref();
            enc.set_compute_pipeline_state(&p);
            enc.set_input_buffer(0, Some(input), input_off);
            if let Some((b, off)) = second {
                enc.set_input_buffer(1, Some(b), off);
            }
            enc.set_output_buffer(2, Some(output), 0);
            enc.set_bytes(3, a);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: groups,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        };
        let mut a = Args {
            n: n as u32,
            ..Default::default()
        };
        match &op.operation {
            Operation::EmbeddingQ5 { token, width } => {
                a.base = *token as u32;
                a.width = *width as u32;
                encode(
                    "coder_embedding_q5",
                    &a,
                    s.buffer(),
                    input_offset,
                    None,
                    &out,
                    (width / 16).div_ceil(256),
                    256,
                )?;
            }
            Operation::Embedding { token, width } => {
                a.base = *token as u32;
                a.width = *width as u32;
                encode(
                    "maya_embedding",
                    &a,
                    s.buffer(),
                    input_offset,
                    None,
                    &out,
                    width.div_ceil(256),
                    256,
                )?;
            }
            Operation::Blocks => encode(
                "maya_blocks",
                &a,
                s.buffer(),
                input_offset,
                None,
                &out,
                elements.div_ceil(256),
                256,
            )?,
            Operation::Mask { blocks } => {
                a.count = *blocks as u32;
                encode(
                    "maya_mask",
                    &a,
                    s.buffer(),
                    input_offset,
                    None,
                    &out,
                    elements.div_ceil(256),
                    256,
                )?;
                encode(
                    "maya_mask_scatter",
                    &a,
                    s.buffer(),
                    input_offset,
                    None,
                    &out,
                    n.div_ceil(256),
                    256,
                )?;
            }
            Operation::Filter { mask } => {
                let (ms, mo) = tensor_storage(mask)?;
                let Storage::Metal(ms) = &*ms else {
                    candle_core::bail!("Maya mask device mismatch");
                };
                encode(
                    "maya_filter",
                    &a,
                    s.buffer(),
                    input_offset,
                    Some((ms.buffer(), mo)),
                    &out,
                    elements.div_ceil(256),
                    256,
                )?;
            }
            Operation::Gather {
                chunks,
                width,
                rows,
            } => {
                gather_into(s, l, chunks, *width, *rows, &out)?;
            }
            Operation::Topk { keep } => {
                let run = n.next_power_of_two().min(1024);
                let runs = n.div_ceil(run);
                let stride = run.min(*keep);
                let scratch_len = runs * stride;
                let mut current = if runs == 1 {
                    out.clone()
                } else {
                    d.new_buffer_builder()
                        .with_size_for(scratch_len, DType::F32)
                        .with_label("maya-topk-runs")
                        .build()?
                };
                a.run = run as u32;
                a.keep = *keep as u32;
                a.stride = stride as u32;
                a.wide = u32::from(n >= 16384);
                encode(
                    "maya_sort_runs",
                    &a,
                    s.buffer(),
                    input_offset,
                    None,
                    &current,
                    runs,
                    run,
                )?;
                let mut count = runs;
                let mut span = run;
                let mut current_stride = stride;
                while count > 1 {
                    let next_stride = (2 * current_stride).min(*keep);
                    let next_count = count.div_ceil(2);
                    let next = if next_count == 1 {
                        out.clone()
                    } else {
                        d.new_buffer_builder()
                            .with_size_for(next_count * next_stride, DType::F32)
                            .with_label("maya-topk-merge")
                            .build()?
                    };
                    a.run = span as u32;
                    a.stride = current_stride as u32;
                    a.count = next_stride as u32;
                    encode(
                        "maya_merge",
                        &a,
                        s.buffer(),
                        input_offset,
                        Some((&current, 0)),
                        &next,
                        (next_count * next_stride).div_ceil(256),
                        256,
                    )?;
                    current = next;
                    count = next_count;
                    current_stride = next_stride;
                    span *= 2;
                }
            }
        }
        Ok((
            MetalStorage::new(out, d.clone(), elements, DType::F32),
            op.shape.clone(),
        ))
    }
}
