//! Internal, default-off Q8_0 F32 matrix candidate. No decoded weight carrier.
//! Layout is adapted from upstream MIT kernel_mul_mv_mma_gen; see the shader
//! for source attribution and full license. Finite operands are the numerical
//! qualification domain, not an excuse to clamp NaN/Inf or overflowing dots.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use candle_core::backend::BackendStorage;
use candle_core::{
    CpuStorage, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Storage, Tensor,
};
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLGPUFamily, MTLLanguageVersion,
    MTLResource, MTLSize,
};

pub(super) const THREADGROUP_SCRATCH_BYTES: usize = 8192;
const DYNAMIC_SCRATCH_BYTES: usize = 0;
const THREADS: usize = 128;
const MAX_TOKENS: usize = 512;

#[cfg(test)]
thread_local! {
    // Encoding evidence, not a timing proxy. Increment only after MMA dispatch.
    static DISPATCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Clone, Copy, Debug)]
struct Geometry {
    k: u32,
    m: u32,
    tokens: u32,
    row_bytes: u32,
    weight_bytes: usize,
    input_elements: usize,
    output_elements: usize,
}

impl Geometry {
    fn new(k: usize, m: usize, dims: &[usize]) -> Option<Self> {
        // Explicit 2D API only: neither flatten rank 3 nor infer grouped rows.
        // K=32 mod 64 is zero-masked in the final chunk; Q8 blocks stay intact.
        if dims.len() != 2
            || dims[1] != k
            || !(2..=MAX_TOKENS).contains(&dims[0])
            || k == 0
            || !k.is_multiple_of(32)
            || k > 65_536
            || m == 0
            || m > 65_536
        {
            return None;
        }
        let input_elements = dims[0].checked_mul(k)?;
        let output_elements = dims[0].checked_mul(m)?;
        u32::try_from(input_elements).ok()?;
        u32::try_from(output_elements).ok()?;
        let row_bytes = k.checked_div(32)?.checked_mul(34)?;
        Some(Self {
            k: u32::try_from(k).ok()?,
            m: u32::try_from(m).ok()?,
            tokens: u32::try_from(dims[0]).ok()?,
            row_bytes: u32::try_from(row_bytes).ok()?,
            weight_bytes: row_bytes.checked_mul(m)?,
            input_elements,
            output_elements,
        })
    }
}

fn checked_span(layout: &Layout, dtype: DType, elements: usize, bytes: usize) -> Result<usize> {
    if !layout.is_contiguous() || layout.shape().elem_count() != elements {
        candle_core::bail!("Q8 MMA requires the exact contiguous operand span");
    }
    let size = dtype.size_in_bytes();
    let start = layout
        .start_offset()
        .checked_mul(size)
        .ok_or_else(|| candle_core::Error::Msg("Q8 MMA offset overflow".into()))?;
    let end = elements
        .checked_mul(size)
        .and_then(|len| start.checked_add(len))
        .ok_or_else(|| candle_core::Error::Msg("Q8 MMA span overflow".into()))?;
    if end > bytes || (dtype == DType::U8 && start % 2 != 0) {
        candle_core::bail!("Q8 MMA operand is out of bounds or Q8 block is unaligned");
    }
    Ok(start)
}

type Cache = HashMap<candle_core::metal_backend::DeviceId, Option<ComputePipeline>>;
static PIPELINES: LazyLock<Mutex<Cache>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn supported_pipeline(device: &candle_core::MetalDevice) -> Option<ComputePipeline> {
    let mut cache = PIPELINES.lock().ok()?;
    cache
        .entry(device.id())
        .or_insert_with(|| {
            let metal = device.metal_device();
            if !(metal.as_ref().supportsFamily(MTLGPUFamily::Apple7)
                || metal.as_ref().supportsFamily(MTLGPUFamily::Mac2))
                || metal.as_ref().maxThreadgroupMemoryLength() < THREADGROUP_SCRATCH_BYTES
            {
                return None;
            }
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version4_0);
            // This legacy setter selects both safe arithmetic and precise functions.
            // Their separate setters require macOS 15; older systems must reach
            // the compile rejection/fallback without sending a missing selector.
            #[allow(deprecated)]
            options.setFastMathEnabled(false);
            let library = metal
                .new_library_with_source(include_str!("deepseek4_q8_mma.metal"), Some(&options))
                .ok()?;
            let function = library
                .get_function("dinf_dsv41_q8_f32_mma_nt4_rt2_nsg4", None)
                .ok()?;
            let pipeline = metal
                .new_compute_pipeline_state_with_function(&function)
                .ok()?;
            #[cfg(test)]
            eprintln!(
                "Q8 MMA pipeline SIMD={} threads={} static={} dynamic={} cap={}",
                pipeline.as_ref().threadExecutionWidth(),
                pipeline.max_total_threads_per_threadgroup(),
                pipeline.as_ref().staticThreadgroupMemoryLength(),
                DYNAMIC_SCRATCH_BYTES,
                metal.as_ref().maxThreadgroupMemoryLength()
            );
            if pipeline.as_ref().threadExecutionWidth() != 32
                || pipeline.max_total_threads_per_threadgroup() < THREADS
                || pipeline.as_ref().staticThreadgroupMemoryLength() != THREADGROUP_SCRATCH_BYTES
                || pipeline.as_ref().staticThreadgroupMemoryLength() + DYNAMIC_SCRATCH_BYTES
                    > metal.as_ref().maxThreadgroupMemoryLength()
            {
                return None;
            }
            Some(pipeline)
        })
        .clone()
}

fn actual_device(buffer: &Buffer, device: &candle_core::MetalDevice) -> bool {
    std::ptr::eq(
        buffer.as_ref().device().as_ref(),
        device.metal_device().as_ref(),
    )
}

struct MatrixOp {
    geometry: Geometry,
    pipeline: ComputePipeline,
}

impl CustomOp2 for MatrixOp {
    fn name(&self) -> &'static str {
        "dinf-dsv41-q8-f32-mma-nt4-rt2-nsg4"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Q8 MMA requires Metal storage")
    }

    fn metal_fwd(
        &self,
        weights: &MetalStorage,
        wl: &Layout,
        input: &MetalStorage,
        il: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        // Revalidate actual storage at the encoding boundary. Any error here is
        // propagated, NEVER caught for partial fallback after candidate selection.
        let g = self.geometry;
        if weights.dtype() != DType::U8
            || input.dtype() != DType::F32
            || weights.device().id() != input.device().id()
            || !actual_device(weights.buffer(), input.device())
            || !actual_device(input.buffer(), input.device())
            || il.dims() != [g.tokens as usize, g.k as usize]
        {
            candle_core::bail!("Q8 MMA storage type, device or shape changed");
        }
        let wo = checked_span(wl, DType::U8, g.weight_bytes, weights.buffer().length())?;
        let io = checked_span(il, DType::F32, g.input_elements, input.buffer().length())?;
        let device = input.device();
        // The admitted matrix output must not inherit an oversized pool loan.
        // Allocate its sole carrier directly, rather than copying/rejecting later.
        let output =
            device.new_private_buffer(g.output_elements, DType::F32, "dinf-dsv41-q8-f32-mma")?;
        if !actual_device(&output, device) || output.length() < g.output_elements * 4 {
            candle_core::bail!("Q8 MMA output storage mismatch");
        }
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_input_buffer(0, Some(weights.buffer()), wo);
        encoder.set_input_buffer(1, Some(input.buffer()), io);
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &g.k);
        encoder.set_bytes(4, &g.m);
        encoder.set_bytes(5, &g.tokens);
        encoder.set_bytes(6, &g.row_bytes);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (g.m as usize).div_ceil(32),
                height: (g.tokens as usize).div_ceil(16),
                depth: 1,
            },
            MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        #[cfg(test)]
        DISPATCHES.with(|count| count.set(count.get() + 1));
        Ok((
            MetalStorage::new(output, device.clone(), g.output_elements, DType::F32),
            Shape::from((g.tokens as usize, g.m as usize)),
        ))
    }
}

/// None means unsupported BEFORE any candidate allocation/encoder/enqueue. The
/// caller chooses the old route in that case. Compilation is cached separately
/// from all old kernels, including failed/unsupported candidate pipelines.
/// Existing Tensor owners/layouts are used as-is; no contiguous copies. Caller
/// must retain input/output/RawTensor owners through its existing completion
/// fence, including error paths, as with the existing native packed matmul.
pub(super) fn try_forward(
    weights: &Tensor,
    input: &Tensor,
    k: usize,
    m: usize,
) -> Result<Option<Tensor>> {
    let Some(geometry) = Geometry::new(k, m, input.dims()) else {
        return Ok(None);
    };
    if weights.dtype() != DType::U8
        || input.dtype() != DType::F32
        || !weights.is_contiguous()
        || !input.is_contiguous()
        || !weights.device().same_device(input.device())
        || !input.device().is_metal()
    {
        return Ok(None);
    }
    let pipeline = {
        let (ws, wl) = weights.storage_and_layout();
        let (xs, xl) = input.storage_and_layout();
        let (Storage::Metal(ws), Storage::Metal(xs)) = (&*ws, &*xs) else {
            return Ok(None);
        };
        if ws.dtype() != DType::U8
            || xs.dtype() != DType::F32
            || ws.device().id() != xs.device().id()
            || !actual_device(ws.buffer(), xs.device())
            || !actual_device(xs.buffer(), xs.device())
            || checked_span(wl, DType::U8, geometry.weight_bytes, ws.buffer().length()).is_err()
            || checked_span(
                xl,
                DType::F32,
                geometry.input_elements,
                xs.buffer().length(),
            )
            .is_err()
        {
            return Ok(None);
        }
        let Some(pipeline) = supported_pipeline(xs.device()) else {
            return Ok(None);
        };
        pipeline
    };
    // No fallible candidate operation below this boundary is retried via scalar.
    weights
        .apply_op2_no_bwd(input, &MatrixOp { geometry, pipeline })
        .map(Some)
}

#[cfg(test)]
#[path = "deepseek4_q8_mma_tests.rs"]
mod tests;
