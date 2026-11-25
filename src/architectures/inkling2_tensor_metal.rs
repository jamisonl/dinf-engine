//! Dense prompt products on Metal 4 tensor operations, with both operands
//! read from device memory as tensors: no weights staged through threadgroup
//! memory, no barriers.
//!
//! F32 by F32, strict: the router and the loop gates' down projections
//! multiply a chunk's rows by a matrix with only 64 or 128 rows of its own.
//! Candle's GEMM covers so skinny a result with a few dozen threadgroups.
//!
//! Half by BF16: the dense projections of the official checkpoint. The
//! matrix units take BF16 operands as stored, so the weights are exact and
//! the kernel is the multiply alone. The products are those of the half
//! activations and the BF16 weights, summed in F32.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::metal_backend::DeviceId;
use candle_core::{DType, MetalStorage, Result, Shape, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder, ComputePipeline};
use objc2_metal::{MTLCompileOptions, MTLLanguageVersion, MTLSize};

/// Which product a kernel computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kind {
    /// F32 rows times F32 weights (transposed), strict.
    F32,
    /// F32 rows times F32 weights (non-transposed), strict.
    F32Nt,
    /// Half rows times half weights (transposed), half output (F32 accumulation).
    F16,
    /// Half rows times half weights (transposed), F32 output (F32 accumulation).
    F16F32,
    /// Half rows times half weights (non-transposed), half output.
    F16Nt,
    /// Half rows times BF16 weights (transposed).
    Bf16,
}

/// Rows of the left operand, rows of the weight, inner columns per step and
/// simdgroups of a tile.
#[derive(Clone, Copy)]
struct Geometry {
    rows: usize,
    out: usize,
    k: usize,
    simdgroups: usize,
}

const F32_GEOMETRY: Geometry = Geometry {
    rows: 32,
    out: 32,
    k: 0,
    simdgroups: 4,
};
const F32_NT_GEOMETRY: Geometry = Geometry {
    rows: 32,
    out: 64,
    k: 0,
    simdgroups: 4,
};
const F16_GEOMETRY: Geometry = Geometry {
    rows: 32,
    out: 32,
    k: 64,
    simdgroups: 2,
};
const F16_NT_GEOMETRY: Geometry = Geometry {
    rows: 64,
    out: 64,
    k: 0,
    simdgroups: 2,
};
const BF16_GEOMETRY: Geometry = Geometry {
    rows: 64,
    out: 64,
    k: 64,
    simdgroups: 2,
};

const MATMUL_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

struct TensorMatmulParams { uint rows; uint out; uint k; uint pad; };

// output[b, r, o] = sum_k x[b, r, k] * weight[b, o, k] (TR=1) or weight[b, k, o] (TR=0).
kernel void {NAME}(
    device {LEFT} *x [[buffer(0)]],
    device {RIGHT} *weight [[buffer(1)]],
    device {OUT_T} *output [[buffer(2)]],
    constant TensorMatmulParams &p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]]) {
    constexpr int tile_rows = {ROWS};
    constexpr int tile_out = {OUT};
    constexpr int tile_k = {K};
    const int row_start = int(group.x) * tile_rows;
    const int out_start = int(group.y) * tile_out;
    const ulong batch = group.z;
    auto x_t = tensor(
        x + batch * p.rows * p.k,
        dextents<int32_t, 2>(int(p.k), int(p.rows)),
        array<int, 2>({1, int(p.k)}));
#if {TR}
    auto w_t = tensor(
        weight + batch * p.out * p.k,
        dextents<int32_t, 2>(int(p.k), int(p.out)),
        array<int, 2>({1, int(p.k)}));
#else
    auto w_t = tensor(
        weight + batch * p.k * p.out,
        dextents<int32_t, 2>(int(p.out), int(p.k)),
        array<int, 2>({1, int(p.out)}));
#endif
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            tile_rows, tile_out, {KDESC}, false, {TR_BOOL}, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<{SG}>> matmul;
    auto acc = matmul.get_destination_cooperative_tensor<
        decltype(x_t), decltype(w_t), {ACC_T}>();
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        acc[i] = {ACC_ZERO};
    }
#if {KLOOP}
    for (int kc = 0; kc < int(p.k); kc += tile_k) {
        auto x_slice = x_t.slice(kc, row_start);
#if {TR}
        auto w_slice = w_t.slice(kc, out_start);
#else
        auto w_slice = w_t.slice(out_start, kc);
#endif
        matmul.run(x_slice, w_slice, acc);
    }
#else
    auto x_slice = x_t.slice(0, row_start);
#if {TR}
    auto w_slice = w_t.slice(0, out_start);
#else
    auto w_slice = w_t.slice(out_start, 0);
#endif
    matmul.run(x_slice, w_slice, acc);
#endif
#if {CAST_OUT}
    device {OUT_T} *out_base = output + batch * p.rows * p.out;
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) { continue; }
        const auto at = acc.get_multidimensional_index(i);
        const int col = out_start + int(at[0]);
        const int row = row_start + int(at[1]);
        if (col < int(p.out) && row < int(p.rows)) {
            out_base[ulong(row) * p.out + col] = {OUT_T}(acc[i]);
        }
    }
#else
    auto o_t = tensor(
        output + batch * p.rows * p.out,
        dextents<int32_t, 2>(int(p.out), int(p.rows)),
        array<int, 2>({1, int(p.out)}));
    acc.store(o_t.slice(out_start, row_start));
#endif
}
"#;

#[repr(C)]
struct Params {
    rows: u32,
    out: u32,
    k: u32,
    pad: u32,
}

static PIPELINES: OnceLock<Mutex<HashMap<(DeviceId, Kind), ComputePipeline>>> = OnceLock::new();

fn geometry(kind: Kind) -> Geometry {
    match kind {
        Kind::F32 => F32_GEOMETRY,
        Kind::F32Nt => F32_NT_GEOMETRY,
        Kind::F16 | Kind::F16F32 => F16_GEOMETRY,
        Kind::F16Nt => F16_NT_GEOMETRY,
        Kind::Bf16 => BF16_GEOMETRY,
    }
}

fn output_dtype(kind: Kind) -> DType {
    match kind {
        Kind::F16 | Kind::F16Nt => DType::F16,
        Kind::F32 | Kind::F32Nt | Kind::F16F32 | Kind::Bf16 => DType::F32,
    }
}

fn pipeline(device: &candle_core::MetalDevice, kind: Kind) -> Result<ComputePipeline> {
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(pipeline) = pipelines.lock().unwrap().get(&(device.id(), kind)) {
        return Ok(pipeline.clone());
    }
    let geometry = geometry(kind);
    let (name, left, right, out_t, acc_t, acc_zero, tr, cast_out) = match kind {
        Kind::F32 => (
            "inkling_f32_matmul_t",
            "float",
            "float",
            "float",
            "float",
            "0.0f",
            true,
            false,
        ),
        Kind::F32Nt => (
            "inkling_f32_matmul_nt",
            "float",
            "float",
            "float",
            "float",
            "0.0f",
            false,
            false,
        ),
        Kind::F16 => (
            "inkling_f16_matmul_t",
            "half",
            "half",
            "half",
            "float",
            "0.0f",
            true,
            true,
        ),
        Kind::F16F32 => (
            "inkling_f16_matmul_t_f32",
            "half",
            "half",
            "float",
            "float",
            "0.0f",
            true,
            false,
        ),
        Kind::F16Nt => (
            "inkling_f16_matmul_nt",
            "half",
            "half",
            "half",
            "half",
            "0.0h",
            false,
            false,
        ),
        Kind::Bf16 => (
            "inkling_bf16_matmul_t",
            "half",
            "bfloat",
            "float",
            "float",
            "0.0f",
            true,
            false,
        ),
    };
    let source = MATMUL_MSL
        .replace("{NAME}", name)
        .replace("{LEFT}", left)
        .replace("{RIGHT}", right)
        .replace("{OUT_T}", out_t)
        .replace("{ACC_T}", acc_t)
        .replace("{ACC_ZERO}", acc_zero)
        .replace("{TR}", if tr { "1" } else { "0" })
        .replace("{TR_BOOL}", if tr { "true" } else { "false" })
        .replace("{CAST_OUT}", if cast_out { "1" } else { "0" })
        .replace("{ROWS}", &geometry.rows.to_string())
        .replace("{OUT}", &geometry.out.to_string())
        .replace("{K}", &geometry.k.max(1).to_string())
        .replace(
            "{KDESC}",
            if geometry.k == 0 {
                "static_cast<int>(dynamic_extent)"
            } else {
                "tile_k"
            },
        )
        .replace("{KLOOP}", if geometry.k == 0 { "0" } else { "1" })
        .replace("{SG}", &geometry.simdgroups.to_string());
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version4_0);
    let library = device
        .metal_device()
        .new_library_with_source(&source, Some(&options))
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(name, None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    pipelines
        .lock()
        .unwrap()
        .insert((device.id(), kind), pipeline.clone());
    Ok(pipeline)
}

fn view(tensor: &Tensor, dtype: DType) -> Result<Option<(Buffer, usize)>> {
    if tensor.dtype() != dtype || !tensor.is_contiguous() {
        return Ok(None);
    }
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        return Ok(None);
    };
    Ok(Some((
        storage.buffer().clone(),
        layout.start_offset() * dtype.size_in_bytes(),
    )))
}

fn enabled(device: &candle_core::MetalDevice) -> bool {
    super::indexed_moe_metal::supports_metal4_indexed_moe(device)
}

/// Compile all kernels for `device`, so the first prompt does not.
pub(super) fn precompile(device: &candle_core::MetalDevice) -> Result<()> {
    if enabled(device) {
        pipeline(device, Kind::F32)?;
        pipeline(device, Kind::F32Nt)?;
        pipeline(device, Kind::F16)?;
        pipeline(device, Kind::F16F32)?;
        pipeline(device, Kind::F16Nt)?;
        pipeline(device, Kind::Bf16)?;
    }
    Ok(())
}

/// `[batches, rows, out]` from `x` `[batches, rows, k]` and `weight`
/// `[batches, out, k]` (or `[batches, k, out]`) bound at these offsets.
#[allow(clippy::too_many_arguments)]
fn run(
    device: &candle_core::MetalDevice,
    kind: Kind,
    x: (&Buffer, usize),
    weight: (&Buffer, usize),
    [batches, rows, k, out]: [usize; 4],
    shape: Shape,
) -> Result<Tensor> {
    let geometry = geometry(kind);
    let out_dtype = output_dtype(kind);
    let narrow = |value: usize| {
        u32::try_from(value)
            .map_err(|_| candle_core::Error::Msg("tensor product is too large".into()))
    };
    let elements = batches * rows * out;
    let output =
        super::inkling2_metal::alloc_scratch_buffer(device, elements * out_dtype.size_in_bytes())?;
    {
        let guard = device.command_encoder()?;
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&pipeline(device, kind)?);
        encoder.set_input_buffer(0, Some(x.0), x.1);
        encoder.set_input_buffer(1, Some(weight.0), weight.1);
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(
            3,
            &Params {
                rows: narrow(rows)?,
                out: narrow(out)?,
                k: narrow(k)?,
                pad: 0,
            },
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: rows.div_ceil(geometry.rows),
                height: out.div_ceil(geometry.out),
                depth: batches,
            },
            MTLSize {
                width: geometry.simdgroups * 32,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(
            output,
            device.clone(),
            elements,
            out_dtype,
        )),
        shape,
    )))
}

/// `x` `[rows, k]` or `[batches, rows, k]` times the transpose of `weight`
/// `[out, k]` or `[batches, out, k]`, both contiguous F32 on a Metal 4
/// device: `[.., rows, out]`. `None` for anything else, or with
/// `DINF_INKLING2_F32_GEMM` set (Candle's GEMM then).
pub(super) fn try_rows_times_transposed(x: &Tensor, weight: &Tensor) -> Result<Option<Tensor>> {
    let candle_core::Device::Metal(device) = x.device() else {
        return Ok(None);
    };
    if !enabled(device) || std::env::var_os("DINF_INKLING2_F32_GEMM").is_some() {
        return Ok(None);
    }
    let (batches, rows, k, out) = match (x.dims(), weight.dims()) {
        (&[rows, k], &[out, weight_k]) if k == weight_k => (1, rows, k, out),
        (&[batches, rows, k], &[weight_batches, out, weight_k])
            if k == weight_k && batches == weight_batches =>
        {
            (batches, rows, k, out)
        }
        _ => return Ok(None),
    };
    let tile_k = geometry(Kind::F32).k.max(1);
    if batches == 0 || rows == 0 || out == 0 || k == 0 || !k.is_multiple_of(tile_k) {
        return Ok(None);
    }
    let (Some((x_buffer, x_offset)), Some((weight_buffer, weight_offset))) =
        (view(x, DType::F32)?, view(weight, DType::F32)?)
    else {
        return Ok(None);
    };
    let shape = if x.rank() == 2 {
        Shape::from((rows, out))
    } else {
        Shape::from((batches, rows, out))
    };
    run(
        device,
        Kind::F32,
        (&x_buffer, x_offset),
        (&weight_buffer, weight_offset),
        [batches, rows, k, out],
        shape,
    )
    .map(Some)
}

/// `x` `[rows, k]` or `[batches, rows, k]` times `weight` `[k, out]` or
/// `[batches, k, out]`, both contiguous F32 on a Metal 4 device: `[.., rows, out]`.
pub(super) fn try_rows_times_nt(x: &Tensor, weight: &Tensor) -> Result<Option<Tensor>> {
    let candle_core::Device::Metal(device) = x.device() else {
        return Ok(None);
    };
    if !enabled(device) || std::env::var_os("DINF_INKLING2_F32_GEMM").is_some() {
        return Ok(None);
    }
    let (batches, rows, k, out) = match (x.dims(), weight.dims()) {
        (&[rows, k], &[weight_k, out]) if k == weight_k => (1, rows, k, out),
        (&[batches, rows, k], &[weight_batches, weight_k, out])
            if k == weight_k && batches == weight_batches =>
        {
            (batches, rows, k, out)
        }
        _ => return Ok(None),
    };
    let tile_k = geometry(Kind::F32Nt).k.max(1);
    if batches == 0 || rows == 0 || out == 0 || k == 0 || !k.is_multiple_of(tile_k) {
        return Ok(None);
    }
    let (Some((x_buffer, x_offset)), Some((weight_buffer, weight_offset))) =
        (view(x, DType::F32)?, view(weight, DType::F32)?)
    else {
        return Ok(None);
    };
    let shape = if x.rank() == 2 {
        Shape::from((rows, out))
    } else {
        Shape::from((batches, rows, out))
    };
    run(
        device,
        Kind::F32Nt,
        (&x_buffer, x_offset),
        (&weight_buffer, weight_offset),
        [batches, rows, k, out],
        shape,
    )
    .map(Some)
}

/// F16 `x` `[rows, k]` or `[batches, rows, k]` times F16 `weight` transposed
/// `[out, k]` or `[batches, out, k]` -> F16 `[.., rows, out]`.
pub(super) fn try_f16_rows_times_transposed(x: &Tensor, weight: &Tensor) -> Result<Option<Tensor>> {
    let candle_core::Device::Metal(device) = x.device() else {
        return Ok(None);
    };
    if !enabled(device) || std::env::var_os("DINF_INKLING2_F32_GEMM").is_some() {
        return Ok(None);
    }
    let (batches, rows, k, out) = match (x.dims(), weight.dims()) {
        (&[rows, k], &[out, weight_k]) if k == weight_k => (1, rows, k, out),
        (&[batches, rows, k], &[weight_batches, out, weight_k])
            if k == weight_k && batches == weight_batches =>
        {
            (batches, rows, k, out)
        }
        _ => return Ok(None),
    };
    let tile_k = geometry(Kind::F16).k.max(1);
    if batches == 0 || rows == 0 || out == 0 || k == 0 || !k.is_multiple_of(tile_k) {
        return Ok(None);
    }
    let (Some((x_buffer, x_offset)), Some((weight_buffer, weight_offset))) =
        (view(x, DType::F16)?, view(weight, DType::F16)?)
    else {
        return Ok(None);
    };
    let shape = if x.rank() == 2 {
        Shape::from((rows, out))
    } else {
        Shape::from((batches, rows, out))
    };
    run(
        device,
        Kind::F16,
        (&x_buffer, x_offset),
        (&weight_buffer, weight_offset),
        [batches, rows, k, out],
        shape,
    )
    .map(Some)
}

/// F16 `x` `[rows, k]` or `[batches, rows, k]` times F16 `weight` transposed
/// `[out, k]` or `[batches, out, k]` -> F32 `[.., rows, out]`.
pub(super) fn try_f16_rows_times_transposed_f32(
    x: &Tensor,
    weight: &Tensor,
) -> Result<Option<Tensor>> {
    let candle_core::Device::Metal(device) = x.device() else {
        return Ok(None);
    };
    if !enabled(device) || std::env::var_os("DINF_INKLING2_F32_GEMM").is_some() {
        return Ok(None);
    }
    let (batches, rows, k, out) = match (x.dims(), weight.dims()) {
        (&[rows, k], &[out, weight_k]) if k == weight_k => (1, rows, k, out),
        (&[batches, rows, k], &[weight_batches, out, weight_k])
            if k == weight_k && batches == weight_batches =>
        {
            (batches, rows, k, out)
        }
        _ => return Ok(None),
    };
    let tile_k = geometry(Kind::F16F32).k.max(1);
    if batches == 0 || rows == 0 || out == 0 || k == 0 || !k.is_multiple_of(tile_k) {
        return Ok(None);
    }
    let (Some((x_buffer, x_offset)), Some((weight_buffer, weight_offset))) =
        (view(x, DType::F16)?, view(weight, DType::F16)?)
    else {
        return Ok(None);
    };
    let shape = if x.rank() == 2 {
        Shape::from((rows, out))
    } else {
        Shape::from((batches, rows, out))
    };
    run(
        device,
        Kind::F16F32,
        (&x_buffer, x_offset),
        (&weight_buffer, weight_offset),
        [batches, rows, k, out],
        shape,
    )
    .map(Some)
}

/// F16 `x` `[rows, k]` or `[batches, rows, k]` times F16 `weight`
/// `[k, out]` or `[batches, k, out]` -> F16 `[.., rows, out]`.
pub(super) fn try_f16_rows_times_nt(x: &Tensor, weight: &Tensor) -> Result<Option<Tensor>> {
    let candle_core::Device::Metal(device) = x.device() else {
        return Ok(None);
    };
    if !enabled(device) || std::env::var_os("DINF_INKLING2_F32_GEMM").is_some() {
        return Ok(None);
    }
    let (batches, rows, k, out) = match (x.dims(), weight.dims()) {
        (&[rows, k], &[weight_k, out]) if k == weight_k => (1, rows, k, out),
        (&[batches, rows, k], &[weight_batches, weight_k, out])
            if k == weight_k && batches == weight_batches =>
        {
            (batches, rows, k, out)
        }
        _ => return Ok(None),
    };
    let tile_k = geometry(Kind::F16Nt).k.max(1);
    if batches == 0 || rows == 0 || out == 0 || k == 0 || !k.is_multiple_of(tile_k) {
        return Ok(None);
    }
    let (Some((x_buffer, x_offset)), Some((weight_buffer, weight_offset))) =
        (view(x, DType::F16)?, view(weight, DType::F16)?)
    else {
        return Ok(None);
    };
    let shape = if x.rank() == 2 {
        Shape::from((rows, out))
    } else {
        Shape::from((batches, rows, out))
    };
    run(
        device,
        Kind::F16Nt,
        (&x_buffer, x_offset),
        (&weight_buffer, weight_offset),
        [batches, rows, k, out],
        shape,
    )
    .map(Some)
}

/// Half `x` `[rows, k]` (`rows * k` values in `x`) times the transpose of
/// BF16 `weight` `[out, k]`, read from device memory as it is stored:
/// `[rows, out]` F32. `None` off Metal 4, for a weight it does not take, or
/// with `DINF_INKLING2_BF16_TILES` set (the staged tile kernel then).
pub(super) fn try_half_rows_times_bf16_transposed(
    x: &Buffer,
    rows: usize,
    weight: &Tensor,
) -> Result<Option<Tensor>> {
    let candle_core::Device::Metal(device) = weight.device() else {
        return Ok(None);
    };
    if !enabled(device) || std::env::var_os("DINF_INKLING2_BF16_TILES").is_some() {
        return Ok(None);
    }
    let (out, k) = weight.dims2()?;
    let tile_k = geometry(Kind::Bf16).k.max(1);
    if rows == 0 || out == 0 || k == 0 || !k.is_multiple_of(tile_k) {
        return Ok(None);
    }
    let Some((weight_buffer, weight_offset)) = view(weight, DType::BF16)? else {
        return Ok(None);
    };
    run(
        device,
        Kind::Bf16,
        (x, 0),
        (&weight_buffer, weight_offset),
        [1, rows, k, out],
        Shape::from((rows, out)),
    )
    .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn values(count: usize, seed: u32, scale: f32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0 * scale
            })
            .collect()
    }

    /// The F32 product against Candle's on the CPU, with partial tiles on
    /// both sides and a batch.
    #[test]
    fn f32_product_matches_candle() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for (batches, rows, out, k) in [(1usize, 70usize, 33usize, 128usize), (3, 45, 64, 192)] {
            let x = values(batches * rows * k, 1, 1.0);
            let w = values(batches * out * k, 2, 0.5);
            let (x_cpu, w_cpu) = (
                Tensor::from_vec(x, (batches, rows, k), &Device::Cpu)?,
                Tensor::from_vec(w, (batches, out, k), &Device::Cpu)?,
            );
            let expected = x_cpu
                .matmul(&w_cpu.transpose(1, 2)?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let (x_gpu, w_gpu) = if batches == 1 {
                (
                    x_cpu.squeeze(0)?.to_device(&device)?,
                    w_cpu.squeeze(0)?.to_device(&device)?,
                )
            } else {
                (x_cpu.to_device(&device)?, w_cpu.to_device(&device)?)
            };
            let Some(got) = try_rows_times_transposed(&x_gpu, &w_gpu)? else {
                // No Metal 4 tensor operations on this device.
                return Ok(());
            };
            let got = got.flatten_all()?.to_vec1::<f32>()?;
            let scale = expected.iter().fold(1f32, |m, v| m.max(v.abs()));
            let error = expected
                .iter()
                .zip(&got)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                error <= 1e-5 * scale,
                "{batches} x [{rows}, {k}] by [{out}, {k}]: error {error} (scale {scale})"
            );
        }
        Ok(())
    }

    /// The BF16 product multiplies the half activations by the weights as
    /// stored: against the same products summed in f64.
    #[test]
    fn bf16_product_uses_exact_weights_and_half_activations() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (rows, out, k) = (150usize, 70usize, 256usize);
        let x = values(rows * k, 3, 4.0);
        // Weights over a wide range of magnitudes, rounded to BF16.
        let w: Vec<f32> = values(out * k, 4, 1.0)
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let scaled = v * 2f32.powi(-((i % 23) as i32));
                f32::from_bits(scaled.to_bits() & 0xffff_0000)
            })
            .collect();
        let x_cpu = Tensor::from_vec(x.clone(), (rows, k), &Device::Cpu)?;
        let half_x = x_cpu
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let w_bf16 = Tensor::from_vec(w.clone(), (out, k), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_device(&device)?;
        let x_half = x_cpu.to_dtype(DType::F16)?.to_device(&device)?;
        let Some((half_buffer, 0)) = view(&x_half, DType::F16)? else {
            panic!("half activations are not a Metal buffer")
        };
        let Some(got) = try_half_rows_times_bf16_transposed(&half_buffer, rows, &w_bf16)? else {
            return Ok(());
        };
        let got = got.flatten_all()?.to_vec1::<f32>()?;
        let mut worst = 0f64;
        for r in 0..rows {
            for o in 0..out {
                let (mut sum, mut magnitude) = (0f64, 0f64);
                for i in 0..k {
                    let product = half_x[r * k + i] as f64 * w[o * k + i] as f64;
                    sum += product;
                    magnitude += product.abs();
                }
                let error = (got[r * out + o] as f64 - sum).abs() / magnitude.max(1e-30);
                worst = worst.max(error);
            }
        }
        // F32 accumulation of exact products; rounding either operand to
        // eight bits would leave about 1e-4 here.
        assert!(worst < 2e-6, "relative error {worst}");
        Ok(())
    }
}
