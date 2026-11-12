//! Native llama.cpp-style prefill. Pinned core/complete MIT notice in adjacent
//! shader, revision de5122d879d00d2895c36be7e0cee94105aeeabd. No decode cutover.
//! Indexed dispatches bind <=12 checked expert/kind banks; grid.z selects the
//! inline record, not a copied packed layer. Only validated GPU route/map pairs
//! supply counts/canonical IDs. Zero/tail groups exit uniformly before barriers.
//! Component qualification uses distinct18x3 banks, counts33/31/1, and the same
//! half precision bounds for M5 tensor and portable branches. Qualified M5 default.
/*
MIT License
Copyright (c) 2023-2026 The ggml authors
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
use super::*;
use candle_core::backend::BackendStorage;
use candle_core::{DType, Device, MetalStorage, Shape, Storage};
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLGPUFamily, MTLLanguageVersion,
    MTLMathFloatingPointFunctions, MTLMathMode, MTLResource, MTLSize,
};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

// One production default, shared by release and tests; no environment switch.
pub(crate) const QUALIFIED_M5_PREFILL_DEFAULT: bool = true;
thread_local! {
    static ACTIVE: std::cell::Cell<bool> = const {std::cell::Cell::new(false)};
    static OVERRIDE: std::cell::Cell<Option<bool>> = const {std::cell::Cell::new(None)};
    static COUNTS: std::cell::Cell<[u64;3]> = const {std::cell::Cell::new([0;3])};
}
#[cfg(test)]
thread_local! {static PORTABLE_TEST: std::cell::Cell<bool> = const {std::cell::Cell::new(false)};}
pub(crate) fn active() -> bool {
    ACTIVE.with(|v| v.get())
}
pub(crate) fn requested() -> bool {
    OVERRIDE
        .with(|v| v.get())
        .unwrap_or(QUALIFIED_M5_PREFILL_DEFAULT)
}
#[cfg(test)]
pub(crate) fn counts() -> [u64; 3] {
    COUNTS.with(|v| v.get())
}
fn count(i: usize) {
    COUNTS.with(|v| {
        let mut c = v.get();
        c[i] += 1;
        v.set(c);
    });
}
pub(crate) struct Scope {
    previous: bool,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Scope {
    pub(crate) fn enter(device: &Device, prefill: bool) -> Self {
        let enabled = prefill
            && requested()
            && device
                .as_metal_device()
                .ok()
                .is_some_and(|d| qualified_device(d) && pipelines(d).is_ok());
        Self {
            previous: ACTIVE.with(|v| v.replace(enabled)),
            _thread: std::marker::PhantomData,
        }
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|v| v.set(self.previous));
    }
}
#[cfg(test)]
pub(crate) struct TestScope {
    previous: Option<bool>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
#[cfg(test)]
impl TestScope {
    // None=no override, Some(false)=explicit accepted OLD, Some(true)=candidate.
    pub(crate) fn new(value: Option<bool>) -> Self {
        Self {
            previous: OVERRIDE.with(|v| v.replace(value)),
            _thread: std::marker::PhantomData,
        }
    }
}
#[cfg(test)]
impl Drop for TestScope {
    fn drop(&mut self) {
        OVERRIDE.with(|v| v.set(self.previous));
    }
}
#[derive(Clone)]
struct Pipelines {
    iq1: ComputePipeline,
    q8: ComputePipeline,
    route: ComputePipeline,
    map: ComputePipeline,
    combine: ComputePipeline,
    /// Dense Coder formats: Q5_K, IQ1_M, IQ2_XXS, IQ2_XS.
    packed: [ComputePipeline; 4],
    /// Expert-indexed variants of `packed`, same order.
    experts: [ComputePipeline; 4],
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Records the selected backend for test evidence.")
    )]
    tensor: bool,
}
type PipelineCache =
    HashMap<(candle_core::metal_backend::DeviceId, bool), std::result::Result<Pipelines, String>>;
static CACHE: LazyLock<Mutex<PipelineCache>> = LazyLock::new(|| Mutex::new(HashMap::new()));
// Metal 4 requires macOS 26 (Darwin 25). Check before constructing options or
// sending newer selectors; even explicit requests cannot bypass this guard.
static METAL4_OS: LazyLock<bool> = LazyLock::new(|| {
    let mut release = [0u8; 256];
    let mut len = release.len();
    let status = unsafe {
        libc::sysctlbyname(
            c"kern.osrelease".as_ptr(),
            release.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    status == 0
        && std::str::from_utf8(&release[..len.min(release.len())])
            .ok()
            .and_then(|s| s.split('.').next()?.parse::<u32>().ok())
            .is_some_and(|major| major >= 25)
});
fn qualified_device(device: &candle_core::MetalDevice) -> bool {
    *METAL4_OS
        && device
            .metal_device()
            .as_ref()
            .supportsFamily(MTLGPUFamily::Apple10)
        && device.metal_device().as_ref().maxThreadgroupMemoryLength() >= 14336
}
fn pipelines(device: &candle_core::MetalDevice) -> Result<Pipelines> {
    anyhow::ensure!(
        qualified_device(device),
        "prefill requires qualified Apple10/Metal4 device"
    );
    let prefer_tensor = true;
    #[cfg(test)]
    let prefer_tensor = prefer_tensor && !PORTABLE_TEST.with(|v| v.get());
    if prefer_tensor && let Ok(p) = pipeline_variant(device, true) {
        return Ok(p);
    }
    pipeline_variant(device, false)
}
fn pipeline_variant(device: &candle_core::MetalDevice, tensor: bool) -> Result<Pipelines> {
    anyhow::ensure!(
        qualified_device(device),
        "prefill requires qualified Apple10/Metal4 device"
    );
    let mut cache = CACHE
        .lock()
        .map_err(|_| anyhow::anyhow!("prefill cache poisoned"))?;
    let result = cache.entry((device.id(), tensor)).or_insert_with(|| {
        (|| -> Result<Pipelines> {
            let d = device.metal_device();
            anyhow::ensure!(
                d.as_ref().supportsFamily(MTLGPUFamily::Apple7)
                    && d.as_ref().maxThreadgroupMemoryLength() >= 14336,
                "prefill SIMD matrix unsupported"
            );
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version4_0);
            options.setMathMode(MTLMathMode::Safe);
            options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            let source = format!(
                "{}\n{}\n{}",
                if tensor {
                    "#define DINF_PREFILL_TENSOR 1"
                } else {
                    ""
                },
                super::metal::iq1_reuse_base_msl(),
                include_str!("deepseek4_prefill_mm.metal")
            );
            let lib = d
                .new_library_with_source(&source, Some(&options))
                .map_err(|e| anyhow::anyhow!("prefill Metal compile: {e}"))?;
            let make = |name: &str, threads: usize| -> Result<ComputePipeline> {
                let f = lib
                    .get_function(name, None)
                    .map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
                let p = d
                    .new_compute_pipeline_state_with_function(&f)
                    .map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
                anyhow::ensure!(
                    p.as_ref().threadExecutionWidth() == 32
                        && p.max_total_threads_per_threadgroup() >= threads
                        && p.as_ref().staticThreadgroupMemoryLength()
                            <= d.as_ref().maxThreadgroupMemoryLength(),
                    "{name}: unsupported geometry"
                );
                Ok(p)
            };
            Ok(Pipelines {
                iq1: make("dinf_prefill_iq1_mm", 128)?,
                q8: make("dinf_prefill_q8_mm", 128)?,
                route: make("dinf_prefill_route", 512)?,
                map: make("dinf_prefill_map", 128)?,
                combine: make("dinf_prefill_combine", 128)?,
                packed: [
                    make("dinf_prefill_q5k_dense", 128)?,
                    make("dinf_prefill_iq1m_dense", 128)?,
                    make("dinf_prefill_iq2xxs_dense", 128)?,
                    make("dinf_prefill_iq2xs_dense", 128)?,
                ],
                experts: [
                    make("dinf_prefill_q5k_experts", 128)?,
                    make("dinf_prefill_iq1m_experts", 128)?,
                    make("dinf_prefill_iq2xxs_experts", 128)?,
                    make("dinf_prefill_iq2xs_experts", 128)?,
                ],
                tensor,
            })
        })()
        .map_err(|e| format!("{e:#}"))
    });
    result.clone().map_err(anyhow::Error::msg)
}
fn actual(buffer: &Buffer, d: &candle_core::MetalDevice) -> bool {
    std::ptr::eq(buffer.as_ref().device().as_ref(), d.metal_device().as_ref())
}
fn operand(
    t: &Tensor,
    d: &candle_core::MetalDevice,
    dtype: DType,
    elements: usize,
) -> Result<(Buffer, usize)> {
    let (s, l) = t.storage_and_layout();
    let Storage::Metal(s) = &*s else {
        anyhow::bail!("prefill operand requires Metal")
    };
    anyhow::ensure!(
        t.dtype() == dtype
            && s.dtype() == dtype
            && s.device().id() == d.id()
            && actual(s.buffer(), d),
        "prefill operand device/dtype mismatch"
    );
    let offset = grouped_iq1_span(l, dtype, elements, s.buffer().length())?;
    anyhow::ensure!(
        dtype != DType::U8 || offset % 2 == 0,
        "prefill unaligned packed weight"
    );
    Ok((s.buffer().clone(), offset))
}
fn private(d: &candle_core::MetalDevice, shape: (usize, usize), dtype: DType) -> Result<Tensor> {
    let elements = shape
        .0
        .checked_mul(shape.1)
        .context("prefill output overflow")?;
    let b = d.new_private_buffer(elements, dtype, "dinf-llama-prefill")?;
    anyhow::ensure!(
        actual(&b, d) && b.length() >= elements * dtype.size_in_bytes(),
        "prefill private output span/device"
    );
    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(b, d.clone(), elements, dtype)),
        Shape::from(shape),
    )))
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Args {
    k: u32,
    m: u32,
    n: u32,
    row_bytes: u32,
    mode: u32,
    expert: u32,
    kind: u32,
    stride: u32,
}
// Output registration occurs once per whole phase; re-registering for every
// expert would impose buffer-wide WAW ordering on disjoint canonical rows.
fn encode(
    e: &candle_metal_kernels::metal::ComputeCommandEncoder,
    p: &ComputePipeline,
    w: &(Buffer, usize),
    x: &(Buffer, usize),
    map: &(Buffer, usize),
    a: Args,
    n: usize,
) {
    e.set_compute_pipeline_state(p);
    e.set_input_buffer(0, Some(&w.0), w.1);
    e.set_input_buffer(1, Some(&x.0), x.1);
    e.set_input_buffer(3, Some(&map.0), map.1);
    e.set_bytes(4, &a);
    e.set_input_buffer(5, Some(&w.0), w.1);
    e.dispatch_thread_groups(
        MTLSize {
            width: (a.m as usize).div_ceil(64),
            height: n.div_ceil(32),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
}
pub(crate) fn dense(weights: &Tensor, input: &Tensor, k: usize, m: usize) -> Result<Tensor> {
    let d = input.device().as_metal_device()?;
    let p = pipelines(d)?;
    anyhow::ensure!(
        input.rank() == 2
            && input.dims()[1] == k
            && (2..=8192).contains(&input.dims()[0])
            && k > 0
            && k.is_multiple_of(32)
            && m > 0
            && m <= 131072
            && k <= 65536,
        "prefill Q8 geometry"
    );
    let n = input.dims()[0];
    let x = operand(input, d, DType::F32, n * k)?;
    let w = operand(weights, d, DType::U8, m * (k / 32 * 34))?;
    let out = private(d, (n, m), DType::F32)?;
    let y = operand(&out, d, DType::F32, n * m)?;
    let a = Args {
        k: k as u32,
        m: m as u32,
        n: n as u32,
        row_bytes: (k / 32 * 34) as u32,
        mode: 0,
        expert: 0,
        kind: 0,
        stride: 0,
    };
    let e = d.command_encoder()?;
    e.as_ref().set_output_buffer(2, Some(&y.0), y.1);
    encode(e.as_ref(), &p.q8, &w, &x, &w, a, n);
    count(1);
    Ok(out)
}
/// Tiled prompt matmul for the packed V4.1 Coder formats. Returns `None`
/// before enqueueing anything when the device, format or geometry is not
/// supported, so callers keep their row path. Rows never interact: each
/// output element sums K in one fixed order for any row count.
pub(crate) fn dense_packed(
    weights: &Tensor,
    input: &Tensor,
    dtype: DeepSeek4DType,
    k: usize,
    m: usize,
) -> Result<Option<Tensor>> {
    let Some(index) = packed_index(dtype) else {
        return Ok(None);
    };
    if !packed_requested()
        || !input.device().is_metal()
        || input.rank() != 2
        || input.dtype() != DType::F32
        || !input.is_contiguous()
        || input.dims()[1] != k
        || !(1..=8192).contains(&input.dims()[0])
        || k == 0
        || !k.is_multiple_of(256)
        || k > 65536
        || m == 0
        || m > 131072
    {
        return Ok(None);
    }
    let d = input.device().as_metal_device()?;
    if !qualified_device(d) {
        return Ok(None);
    }
    let Ok(p) = pipelines(d) else {
        return Ok(None);
    };
    let row_bytes = weight_byte_len(dtype, 1, k)?;
    let n = input.dims()[0];
    let x = operand(input, d, DType::F32, n * k)?;
    let w = operand(weights, d, DType::U8, m * row_bytes)?;
    let out = private(d, (n, m), DType::F32)?;
    let y = operand(&out, d, DType::F32, n * m)?;
    let a = Args {
        k: k as u32,
        m: m as u32,
        n: n as u32,
        row_bytes: row_bytes as u32,
        mode: 0,
        expert: 0,
        kind: 0,
        stride: 0,
    };
    let e = d.command_encoder()?;
    e.as_ref().set_output_buffer(2, Some(&y.0), y.1);
    encode(e.as_ref(), &p.packed[index], &w, &x, &w, a, n);
    Ok(Some(out))
}
/// Whether `dense_packed` and `experts_packed` can run on this device.
pub(crate) fn packed_available(device: &Device) -> bool {
    packed_requested()
        && device
            .as_metal_device()
            .ok()
            .is_some_and(|d| qualified_device(d) && pipelines(d).is_ok())
}
fn packed_index(dtype: DeepSeek4DType) -> Option<usize> {
    match dtype {
        DeepSeek4DType::Q5K => Some(0),
        DeepSeek4DType::IQ1M => Some(1),
        DeepSeek4DType::IQ2XXS => Some(2),
        DeepSeek4DType::IQ2XS => Some(3),
        _ => None,
    }
}
/// One dispatch for several packed experts sharing a format and shape. Group
/// `g` multiplies `weights[g]` by the `input` rows listed in `rows[g]`; output
/// rows are the groups' rows concatenated in order. Each output row equals
/// `dense_packed` of the same input row. Returns `None` before enqueueing
/// anything when unsupported. The caller keeps every weight alive until the
/// GPU finishes: the kernel reads them through a GPU address table, which the
/// command encoder does not retain. Weights are CPU-written shared buffers in
/// the device residency set, so no GPU write precedes these reads.
pub(crate) fn experts_packed(
    weights: &[&Tensor],
    rows: &[Vec<u32>],
    input: &Tensor,
    dtype: DeepSeek4DType,
    k: usize,
    m: usize,
) -> Result<Option<Tensor>> {
    let Some(index) = packed_index(dtype) else {
        return Ok(None);
    };
    let total: usize = rows.iter().map(Vec::len).sum();
    if !packed_requested()
        || weights.is_empty()
        || weights.len() != rows.len()
        || weights.len() > 65535
        || rows.iter().any(Vec::is_empty)
        || total > 65536
        || !input.device().is_metal()
        || input.rank() != 2
        || input.dtype() != DType::F32
        || !input.is_contiguous()
        || input.dims()[1] != k
        || rows
            .iter()
            .flatten()
            .any(|&r| r as usize >= input.dims()[0])
        || k == 0
        || !k.is_multiple_of(256)
        || k > 65536
        || m == 0
        || m > 131072
    {
        return Ok(None);
    }
    let d = input.device().as_metal_device()?;
    if !qualified_device(d) {
        return Ok(None);
    }
    let Ok(p) = pipelines(d) else {
        return Ok(None);
    };
    let row_bytes = weight_byte_len(dtype, 1, k)?;
    let x = operand(input, d, DType::F32, input.dims()[0] * k)?;
    let mut addresses = Vec::with_capacity(weights.len());
    for w in weights {
        let (buffer, offset) = operand(w, d, DType::U8, m * row_bytes)?;
        use objc2_metal::MTLBuffer as _;
        addresses.push(buffer.as_ref().gpuAddress() + offset as u64);
    }
    let mut groups = Vec::with_capacity(2 * rows.len());
    let mut flat = Vec::with_capacity(total);
    for r in rows {
        groups.extend([flat.len() as u32, r.len() as u32]);
        flat.extend_from_slice(r);
    }
    let longest = rows.iter().map(Vec::len).max().unwrap_or(0);
    let pointers = d.new_buffer_with_data(&addresses)?;
    let groups = d.new_buffer_with_data(&groups)?;
    let flat = d.new_buffer_with_data(&flat)?;
    let out = private(d, (total, m), DType::F32)?;
    let y = operand(&out, d, DType::F32, total * m)?;
    let a = Args {
        k: k as u32,
        m: m as u32,
        n: total as u32,
        row_bytes: row_bytes as u32,
        mode: 0,
        expert: 0,
        kind: 0,
        stride: 0,
    };
    let e = d.command_encoder()?;
    let e = e.as_ref();
    e.set_compute_pipeline_state(&p.experts[index]);
    e.set_input_buffer(0, Some(&*pointers), 0);
    e.set_input_buffer(1, Some(&x.0), x.1);
    e.set_output_buffer(2, Some(&y.0), y.1);
    e.set_input_buffer(3, Some(&*flat), 0);
    e.set_bytes(4, &a);
    e.set_input_buffer(5, Some(&*groups), 0);
    e.dispatch_thread_groups(
        MTLSize {
            width: m.div_ceil(64),
            height: longest.div_ceil(32),
            depth: weights.len(),
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(Some(out))
}
/// `DINF_DEEPSEEK41_PACKED_PREFILL_MM=0` keeps the Coder's row kernels.
fn packed_requested() -> bool {
    static REQUESTED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var_os("DINF_DEEPSEEK41_PACKED_PREFILL_MM").is_none_or(|v| v != "0")
    });
    #[cfg(test)]
    if PACKED_TEST_DISABLED.with(|v| v.get()) {
        return false;
    }
    *REQUESTED
}
#[cfg(test)]
thread_local! {
    static PACKED_TEST_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
/// Test-only: run `f` with the Coder's row kernels instead of tiled matmuls.
#[cfg(test)]
pub(crate) fn with_packed_disabled<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            PACKED_TEST_DISABLED.with(|v| v.set(self.0));
        }
    }
    let _restore = Restore(PACKED_TEST_DISABLED.with(|v| v.replace(true)));
    f()
}
pub(crate) struct Routes {
    compact: Tensor,
    map: Tensor,
}
/// Only route() can construct the GPU map; validation binds its compact CPU
/// selections and counts to that exact map without reading the whole map back.
pub(crate) struct ValidatedRoutes {
    compact: Tensor,
    map: Tensor,
    selections: Vec<crate::architectures::deepseek41::math::RouterSelection>,
    counts: [usize; 384],
}
impl Routes {
    pub(crate) fn tensors(&self) -> [&Tensor; 2] {
        [&self.compact, &self.map]
    }
    pub(crate) fn validate(&self) -> Result<ValidatedRoutes> {
        let records = self.compact.to_vec2::<u32>()?;
        let mut selections = Vec::with_capacity(records.len());
        let mut counts = [0; 384];
        for row in records {
            anyhow::ensure!(
                row.len() == 13 && row[12] as i32 == 0,
                "native Maya router error status"
            );
            let indices: [usize; 6] = std::array::from_fn(|s| row[s] as usize);
            let weights: [f32; 6] = std::array::from_fn(|s| f32::from_bits(row[6 + s]));
            for s in 0..6 {
                anyhow::ensure!(
                    indices[s] < 384 && !indices[..s].contains(&indices[s]),
                    "native Maya router invalid/duplicate expert"
                );
                anyhow::ensure!(
                    weights[s].is_finite(),
                    "native Maya router nonfinite weight"
                );
                counts[indices[s]] += 1;
            }
            selections
                .push(crate::architectures::deepseek41::math::RouterSelection { indices, weights });
        }
        Ok(ValidatedRoutes {
            compact: self.compact.clone(),
            map: self.map.clone(),
            selections,
            counts,
        })
    }
}
impl ValidatedRoutes {
    pub(crate) fn tensors(&self) -> [&Tensor; 2] {
        [&self.compact, &self.map]
    }
    pub(crate) fn selections(&self) -> &[crate::architectures::deepseek41::math::RouterSelection] {
        &self.selections
    }
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Binding {
    expert: u32,
    kind: u32,
    count: u32,
    pad: u32,
}
pub(crate) fn route(logits: &Tensor, bias: &[f32]) -> Result<Routes> {
    let d = logits.device().as_metal_device()?;
    let p = pipelines(d)?;
    anyhow::ensure!(
        logits.rank() == 2
            && logits.dims()[1] == 384
            && (2..=128).contains(&logits.dims()[0])
            && bias.len() == 384,
        "prefill router geometry"
    );
    let bias: &[f32; 384] = bias.try_into()?;
    let n = logits.dims()[0];
    let x = operand(logits, d, DType::F32, n * 384)?;
    let compact = private(d, (n, 13), DType::U32)?;
    let map = private(d, (384, 1 + 6 * n), DType::U32)?;
    let c = operand(&compact, d, DType::U32, n * 13)?;
    let ids = operand(&map, d, DType::U32, 384 * (1 + 6 * n))?;
    let e = d.command_encoder()?;
    let e = e.as_ref();
    e.set_compute_pipeline_state(&p.route);
    e.set_input_buffer(0, Some(&x.0), x.1);
    e.set_bytes(1, bias);
    e.set_output_buffer(2, Some(&c.0), c.1);
    e.dispatch_thread_groups(
        MTLSize {
            width: n,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 512,
            height: 1,
            depth: 1,
        },
    );
    e.set_compute_pipeline_state(&p.map);
    e.set_input_buffer(0, Some(&c.0), c.1);
    e.set_output_buffer(1, Some(&ids.0), ids.1);
    e.set_bytes(2, &(n as u32));
    e.dispatch_thread_groups(
        MTLSize {
            width: 3,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    count(2);
    Ok(Routes { compact, map })
}
pub(crate) fn indexed<'a>(
    input: &Tensor,
    routes: &ValidatedRoutes,
    down: bool,
    mut select: impl FnMut(usize, usize) -> Result<&'a DeepSeek4QMatMul>,
) -> Result<Tensor> {
    let t = routes.selections.len();
    anyhow::ensure!((2..=128).contains(&t), "prefill indexed token count");
    let (k, m, input_rows, out_rows) = if down {
        (2304, 5120, 6 * t, 6 * t)
    } else {
        (5120, 2304, t, 12 * t)
    };
    anyhow::ensure!(
        input.dims() == [input_rows, k],
        "prefill indexed input shape"
    );
    let d = input.device().as_metal_device()?;
    let p = pipelines(d)?;
    let x = operand(input, d, DType::F32, input_rows * k)?;
    let ids = operand(&routes.map, d, DType::U32, 384 * (1 + 6 * t))?;
    let binding_count = routes.counts.iter().filter(|&&n| n > 0).count() * if down { 1 } else { 2 };
    let mut prepared = Vec::with_capacity(binding_count);
    // Validate and retain every selected packed owner/binding before any enqueue.
    for (expert, &n) in routes.counts.iter().enumerate().filter(|(_, n)| **n > 0) {
        for kind in if down { 2..3 } else { 0..2 } {
            let w = select(expert, kind)?;
            anyhow::ensure!(
                w.weights.dtype() == DeepSeek4DType::IQ1S
                    && w.weights.shape() == [m, k]
                    && w.weights.byte_len() == m * (k / 256 * 50),
                "prefill exact IQ1 expert required"
            );
            let b = operand(w.weights.bytes(), d, DType::U8, m * (k / 256 * 50))?;
            prepared.push((w.weights.clone(), b, expert, kind, n));
        }
    }
    let out = private(d, (out_rows, m), DType::F32)?;
    let y = operand(&out, d, DType::F32, out_rows * m)?;
    let e = d.command_encoder()?;
    e.as_ref().set_output_buffer(2, Some(&y.0), y.1);
    for packet in prepared.chunks(12) {
        let mut records = [Binding::default(); 12];
        for slot in 0..12 {
            let (_, w, expert, kind, n) = &packet[slot.min(packet.len() - 1)];
            records[slot] = Binding {
                expert: *expert as u32,
                kind: *kind as u32,
                count: *n as u32,
                pad: 0,
            };
            e.as_ref()
                .set_input_buffer(if slot == 0 { 0 } else { 5 + slot }, Some(&w.0), w.1);
        }
        let a = Args {
            k: k as u32,
            m: m as u32,
            n: t as u32,
            row_bytes: (k / 256 * 50) as u32,
            mode: if down { 2 } else { 1 },
            expert: 0,
            kind: 0,
            stride: (1 + 6 * t) as u32,
        };
        e.as_ref().set_compute_pipeline_state(&p.iq1);
        e.as_ref().set_input_buffer(1, Some(&x.0), x.1);
        e.as_ref().set_input_buffer(3, Some(&ids.0), ids.1);
        e.as_ref().set_bytes(4, &a);
        e.as_ref().set_bytes(5, &records);
        let maximum = packet.iter().map(|p| p.4).max().unwrap();
        e.as_ref().dispatch_thread_groups(
            MTLSize {
                width: m.div_ceil(64),
                height: maximum.div_ceil(32),
                depth: packet.len(),
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        count(0);
    }
    Ok(out)
}
pub(crate) fn combine(down: &Tensor, compact: &Tensor) -> Result<Tensor> {
    let d = down.device().as_metal_device()?;
    let p = pipelines(d)?;
    anyhow::ensure!(
        compact.rank() == 2 && compact.dims()[1] == 13,
        "prefill combine compact shape"
    );
    let t = compact.dims()[0];
    anyhow::ensure!(down.dims() == [6 * t, 5120], "prefill combine down shape");
    let x = operand(down, d, DType::F32, 6 * t * 5120)?;
    let c = operand(compact, d, DType::U32, 13 * t)?;
    let out = private(d, (t, 5120), DType::F32)?;
    let y = operand(&out, d, DType::F32, t * 5120)?;
    let e = d.command_encoder()?;
    let e = e.as_ref();
    e.set_compute_pipeline_state(&p.combine);
    e.set_input_buffer(0, Some(&x.0), x.1);
    e.set_input_buffer(1, Some(&c.0), c.1);
    e.set_output_buffer(2, Some(&y.0), y.1);
    e.dispatch_thread_groups(
        MTLSize {
            width: 40,
            height: t,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(out)
}
/// Additional retained/source envelope, not borrowed from resident/KV/copy budgets.
pub(crate) fn carrier_bytes(t: usize) -> usize {
    // Prior/current input, router logits, compact records, maps, shared gate/up/
    // mid/down, canonical gate-up/activation/down and pooled source replacements.
    2 * (t * (5120 * 4 + 384 * 4 + 13 * 4)
        + 384 * (1 + 6 * t) * 4
        + super::iq1_packets::carrier_bytes(t).expect("bounded rows"))
        + t * 512 * 1024
}
#[cfg(test)]
#[path = "deepseek4_prefill_mm_tests.rs"]
mod tests;
