//! Few-row quantized matrix products (2..16 input rows) for Apple GPUs
//! without per-core neural accelerators (M1-M4).
//!
//! Draft verification multiplies every weight by a handful of rows. The
//! Metal 4 tensor kernels and the mat-vec kernels both re-read weights per
//! row on these GPUs, so verifying four tokens of Qwen3.8 27B cost about
//! 3x one token and drafting lost to plain decoding. These kernels
//! dequantize each weight once for all rows on 8x8 simdgroup matrices; see
//! `few_row_mma.metal` for the source and its license. They are used for
//! Q4_K from three rows and for Q6_K and Q8_0 from two. Q4_K uses a block
//! kernel derived from a Q5_K one. `DINF_DISABLE_FEW_ROW_MMA=1`
//! turns them off.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::quantized::GgmlDType;
use candle_core::{Layout, MetalStorage, Result};
use candle_metal_kernels::metal::{Buffer, ComputePipeline, ConstantValues, Library, Value};
use objc2_metal::{MTLCompileOptions, MTLDevice, MTLGPUFamily, MTLSize};

/// Most input rows these kernels take.
pub(super) const MAX_ROWS: usize = 16;
/// Rows per 8x8 tile.
const TILE: usize = 8;
const NT_MAX: usize = 4;
const SMEM_MAX: usize = 16384;
const MIN_THREADGROUPS: usize = 128;

/// Mirrors `dinf_few_row_mma_args` in the shader.
#[repr(C)]
#[derive(Clone, Copy)]
struct Args {
    ne00: i32,
    ne01: i32,
    nb01: u64,
    ne11: i32,
    nb11: u64,
    ne0: i32,
    nsg: i32,
}

/// The fewest rows at which these kernels beat the alternatives, per type.
fn min_rows(dtype: GgmlDType) -> Option<usize> {
    match dtype {
        GgmlDType::Q4K => Some(3),
        GgmlDType::Q6K | GgmlDType::Q8_0 => Some(2),
        _ => None,
    }
}

fn type_name(dtype: GgmlDType) -> &'static str {
    match dtype {
        GgmlDType::Q4K => "q4_K",
        GgmlDType::Q6K => "q6_K",
        _ => "q8_0",
    }
}

/// Whether `rows` rows of `dtype` with input width `in_dim` take this path.
pub(super) fn applies(dtype: GgmlDType, in_dim: usize, rows: usize) -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    let Some(min) = min_rows(dtype) else {
        return false;
    };
    (min..=MAX_ROWS).contains(&rows)
        && in_dim.is_multiple_of(64)
        && in_dim <= i32::MAX as usize
        && !*DISABLED.get_or_init(|| {
            std::env::var_os("DINF_DISABLE_FEW_ROW_MMA").is_some_and(|value| value == "1")
        })
}

type LibraryCache = HashMap<candle_core::metal_backend::DeviceId, Option<Library>>;
type PipelineCache =
    HashMap<(candle_core::metal_backend::DeviceId, String), Option<ComputePipeline>>;

static LIBRARIES: LazyLock<Mutex<LibraryCache>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static PIPELINES: LazyLock<Mutex<PipelineCache>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn pipeline(
    device: &candle_core::MetalDevice,
    name: &str,
    nsg: usize,
    in_dim: usize,
) -> Option<ComputePipeline> {
    specialized(
        device,
        name,
        format!("nsg{nsg}_k{in_dim}"),
        vec![
            (0, Value::U16(u16::try_from(nsg).ok()?)),
            (1, Value::USize(in_dim)),
        ],
    )
}

/// The pipeline for kernel `name` with function `constants`, compiled once
/// per device and `key`.
fn specialized(
    device: &candle_core::MetalDevice,
    name: &str,
    key: String,
    constants: Vec<(usize, Value)>,
) -> Option<ComputePipeline> {
    let key = (device.id(), format!("{name}_{key}"));
    if let Some(cached) = PIPELINES.lock().ok()?.get(&key) {
        return cached.clone();
    }
    let metal = device.metal_device();
    let library = LIBRARIES
        .lock()
        .ok()?
        .entry(device.id())
        .or_insert_with(|| {
            if !metal.as_ref().supportsFamily(MTLGPUFamily::Apple7) {
                return None;
            }
            // Metal's default compile options.
            let options = MTLCompileOptions::new();
            match metal.new_library_with_source(include_str!("few_row_mma.metal"), Some(&options)) {
                Ok(library) => Some(library),
                Err(error) => {
                    tracing::warn!("few-row MMA kernels did not compile: {error}");
                    #[cfg(test)]
                    eprintln!("few-row MMA kernels did not compile: {error}");
                    None
                }
            }
        })
        .clone();
    let constants = ConstantValues::new(constants);
    let pipeline = library.and_then(|library| {
        let function = library.get_function(name, Some(&constants)).ok()?;
        metal
            .new_compute_pipeline_state_with_function(&function)
            .ok()
    });
    PIPELINES.lock().ok()?.insert(key, pipeline.clone());
    pipeline
}

fn halve_to_limit(mut n: usize, limit: usize) -> usize {
    while n > 1 && n > limit {
        n /= 2;
    }
    n
}

/// Encode `output[rows, out_dim] = input[rows, in_dim] x weights^T` for GGML
/// `dtype` weights stored row-major (`out_dim` rows of `in_dim` values).
/// Returns `false` before encoding anything when the shape or device does not
/// fit, so the caller can take another path.
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch(
    dtype: GgmlDType,
    in_dim: usize,
    out_dim: usize,
    rows: usize,
    weights: &MetalStorage,
    input: &MetalStorage,
    input_layout: &Layout,
    output: &Buffer,
) -> Result<bool> {
    if !applies(dtype, in_dim, rows) || out_dim == 0 || out_dim > i32::MAX as usize {
        return Ok(false);
    }
    let input_offset = input_layout.start_offset() * std::mem::size_of::<f32>();
    if !input_offset.is_multiple_of(16) {
        return Ok(false);
    }
    let rt = if rows > TILE { 2 } else { 1 };
    // Q8_0 up to eight rows and Q4_K use block kernels; the rest dequantize
    // through the generic kernel.
    let q8_blk = dtype == GgmlDType::Q8_0 && rt == 1;
    let q4k_blk = dtype == GgmlDType::Q4K;
    let k_step = if q8_blk { 32 } else { 64 };
    let mut nsg = if out_dim <= 64 {
        32
    } else if out_dim <= 6144 {
        16
    } else {
        8
    };
    nsg = halve_to_limit(nsg, in_dim / k_step);
    let tile_bytes = TILE * TILE * std::mem::size_of::<f32>();
    let nt_smem = SMEM_MAX / (nsg * rt * tile_bytes);
    let nt_rows = out_dim / (MIN_THREADGROUPS * TILE);
    let nt = halve_to_limit(NT_MAX, nt_smem.min(nt_rows));
    let name = if q8_blk || q4k_blk {
        format!("dinf_few_row_mma_{}_blk_nt{nt}_rt{rt}", type_name(dtype))
    } else {
        format!("dinf_few_row_mma_{}_nt{nt}_rt{rt}", type_name(dtype))
    };
    let device = input.device();
    let pipeline = loop {
        let Some(pipeline) = pipeline(device, &name, nsg, in_dim) else {
            return Ok(false);
        };
        if nsg == 1 || pipeline.max_total_threads_per_threadgroup() >= nsg * 32 {
            break pipeline;
        }
        nsg /= 2;
    };
    let args = Args {
        ne00: in_dim as i32,
        ne01: out_dim as i32,
        nb01: (in_dim / dtype.block_size() * dtype.type_size()) as u64,
        ne11: rows as i32,
        nb11: (in_dim * std::mem::size_of::<f32>()) as u64,
        ne0: out_dim as i32,
        nsg: nsg as i32,
    };
    let encoder = device.command_encoder()?;
    let encoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_bytes(0, &args);
    encoder.set_input_buffer(1, Some(weights.buffer()), 0);
    encoder.set_input_buffer(2, Some(input.buffer()), input_offset);
    encoder.set_output_buffer(3, Some(output), 0);
    encoder.set_threadgroup_memory_length(0, nsg * nt * rt * tile_bytes);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: out_dim.div_ceil(TILE * nt),
            height: rows.div_ceil(TILE * rt),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: nsg,
            depth: 1,
        },
    );
    Ok(true)
}

/// Mirrors `dinf_mm_args` in the shader.
#[repr(C)]
#[derive(Clone, Copy)]
struct MmArgs {
    ne00: i32,
    nb01: u64,
    nb11: u64,
    ne0: i32,
    ne1: i32,
}

/// Whether a prompt batch of `rows` rows takes [`dispatch_mm`].
pub(super) fn mm_applies(dtype: GgmlDType, rows: usize) -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    matches!(dtype, GgmlDType::Q4K | GgmlDType::Q6K | GgmlDType::Q8_0)
        && rows > MAX_ROWS
        && !*DISABLED.get_or_init(|| {
            std::env::var_os("DINF_DISABLE_LLAMA_MM").is_some_and(|value| value == "1")
        })
}

/// Encode a prompt-sized product (17+ rows) with the steel-layout kernel in
/// `few_row_mma.metal`: on-the-fly dequantization in MLX's threadgroup
/// layout, 64x64 tiles, f16 input. On Qwen3.8 27B's projections it is
/// faster than Candle's kernel. Same layout and contract as [`dispatch`];
/// `DINF_DISABLE_LLAMA_MM=1` keeps Candle's kernel.
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_mm(
    dtype: GgmlDType,
    in_dim: usize,
    out_dim: usize,
    rows: usize,
    weights: &MetalStorage,
    input: &MetalStorage,
    input_layout: &Layout,
    output: &Buffer,
) -> Result<bool> {
    if !mm_applies(dtype, rows)
        || !in_dim.is_multiple_of(dtype.block_size())
        || in_dim > i32::MAX as usize
        || out_dim > i32::MAX as usize
        || rows > i32::MAX as usize
    {
        return Ok(false);
    }
    let input_offset = input_layout.start_offset() * std::mem::size_of::<f32>();
    if !input_offset.is_multiple_of(16) || input.dtype() != candle_core::DType::F32 {
        return Ok(false);
    }
    let device = input.device();
    // Q4_K/Q6_K/Q8_0 rows are whole blocks, so K is a multiple of 32.
    let bm = 64;
    let bc_out = !out_dim.is_multiple_of(64) || !rows.is_multiple_of(bm);
    let (name, constants, key) = (
        format!("dinf_mms_{}", type_name(dtype)),
        vec![(11, Value::Bool(bc_out))],
        format!("bco{}", bc_out as u8),
    );
    let Some(pipeline) = specialized(device, &name, key, constants) else {
        return Ok(false);
    };
    let Some(cast) = specialized(device, "dinf_cast_f32_f16", String::new(), vec![]) else {
        return Ok(false);
    };
    let elements = rows * in_dim;
    let encoder = device.command_encoder()?;
    let encoder = encoder.as_ref();
    // The kernel reads its input as f16 (see the shader).
    let half_input = device
        .new_buffer_builder()
        .with_size_for(elements, candle_core::DType::F16)
        .with_label("dinf-mm-f16-input")
        .build()?;
    let n4 = u32::try_from(elements / 4)
        .map_err(|_| candle_core::Error::Msg("prompt GEMM input too large".into()))?;
    encoder.set_compute_pipeline_state(&cast);
    encoder.set_input_buffer(0, Some(input.buffer()), input_offset);
    encoder.set_output_buffer(1, Some(&half_input), 0);
    encoder.set_bytes(2, &n4);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: (n4 as usize).div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    let args = MmArgs {
        ne00: in_dim as i32,
        nb01: (in_dim / dtype.block_size() * dtype.type_size()) as u64,
        nb11: (in_dim * std::mem::size_of::<half::f16>()) as u64,
        ne0: out_dim as i32,
        ne1: rows as i32,
    };
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_bytes(0, &args);
    encoder.set_input_buffer(1, Some(weights.buffer()), 0);
    encoder.set_input_buffer(2, Some(&half_input), 0);
    encoder.set_output_buffer(3, Some(output), 0);
    // input and weight tiles: (bm + 64) rows of 40 halves
    encoder.set_threadgroup_memory_length(0, (bm + 64) * 40 * 2);
    let grid = MTLSize {
        width: out_dim.div_ceil(64),
        height: rows.div_ceil(bm),
        depth: 1,
    };
    encoder.dispatch_thread_groups(
        grid,
        MTLSize {
            width: bm * 2,
            height: 1,
            depth: 1,
        },
    );
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::QTensor;
    use candle_core::{CpuStorage, CustomOp2, DType, Device, Shape, Tensor};

    struct FewRow {
        dtype: GgmlDType,
        in_dim: usize,
        out_dim: usize,
    }

    impl CustomOp2 for FewRow {
        fn name(&self) -> &'static str {
            "few-row-mma-test"
        }

        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("Metal only")
        }

        fn metal_fwd(
            &self,
            weights: &MetalStorage,
            _: &Layout,
            input: &MetalStorage,
            input_layout: &Layout,
        ) -> Result<(MetalStorage, Shape)> {
            let rows = input_layout.dims()[0];
            let device = input.device();
            let output = device
                .new_buffer_builder()
                .with_size_for(rows * self.out_dim, DType::F32)
                .build()?;
            let dispatch = if rows > MAX_ROWS {
                dispatch_mm
            } else {
                dispatch
            };
            if !dispatch(
                self.dtype,
                self.in_dim,
                self.out_dim,
                rows,
                weights,
                input,
                input_layout,
                &output,
            )? {
                candle_core::bail!("few-row MMA did not dispatch");
            }
            Ok((
                MetalStorage::new(output, device.clone(), rows * self.out_dim, DType::F32),
                Shape::from((rows, self.out_dim)),
            ))
        }
    }

    fn metal() -> Option<Device> {
        match Device::new_metal(0) {
            Ok(device) => Some(device),
            Err(_) if std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_none() => None,
            Err(error) => panic!("Metal required: {error}"),
        }
    }

    #[test]
    fn few_row_mma_matches_dequantized_reference() -> anyhow::Result<()> {
        let Some(device) = metal() else {
            return Ok(());
        };
        for dtype in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            for (in_dim, out_dim) in [(512usize, 37usize), (1024, 1027), (2048, 4096)] {
                let weights: Vec<f32> = (0..in_dim * out_dim)
                    .map(|i| (((i * 2654435761usize) >> 7) % 2001) as f32 / 1000.0 - 1.0)
                    .collect();
                let weights = Tensor::from_vec(weights, (out_dim, in_dim), &Device::Cpu)?;
                let quantized = QTensor::quantize(&weights, dtype)?;
                let dequantized = quantized.dequantize(&Device::Cpu)?;
                let raw = quantized.data()?.into_owned();
                let raw = Tensor::from_raw_buffer(&raw, DType::U8, &[raw.len()], &device)?;
                for rows in [2usize, 3, 4, 7, 8, 9, 13, 16, 17, 32, 33, 64, 100] {
                    if !applies(dtype, in_dim, rows) && !mm_applies(dtype, rows) {
                        continue;
                    }
                    let input: Vec<f32> = (0..rows * in_dim)
                        .map(|i| (((i * 40503) >> 3) % 997) as f32 / 500.0 - 1.0)
                        .collect();
                    let input = Tensor::from_vec(input, (rows, in_dim), &Device::Cpu)?;
                    let expected = input.matmul(&dequantized.t()?)?.to_vec2::<f32>()?;
                    let actual = raw
                        .apply_op2_no_bwd(
                            &input.to_device(&device)?,
                            &FewRow {
                                dtype,
                                in_dim,
                                out_dim,
                            },
                        )?
                        .to_vec2::<f32>()?;
                    if rows > MAX_ROWS {
                        // Half operand tiles, as upstream: check relative RMSE.
                        let (mut err, mut norm) = (0f64, 0f64);
                        for (e_row, a_row) in expected.iter().zip(&actual) {
                            for (e, a) in e_row.iter().zip(a_row) {
                                err += f64::from(e - a).powi(2);
                                norm += f64::from(*e).powi(2);
                            }
                        }
                        let relative = (err / norm).sqrt();
                        assert!(
                            relative < 2e-3,
                            "{dtype:?} {in_dim}x{out_dim} rows={rows}: relative RMSE {relative}"
                        );
                        continue;
                    }
                    for (r, (e_row, a_row)) in expected.iter().zip(&actual).enumerate() {
                        for (c, (e, a)) in e_row.iter().zip(a_row).enumerate() {
                            let tolerance = 2e-3 * (1.0 + e.abs()) * (in_dim as f32 / 512.0).sqrt();
                            assert!(
                                (e - a).abs() <= tolerance,
                                "{dtype:?} {in_dim}x{out_dim} rows={rows} [{r},{c}]: {a} vs {e}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Per-call time of the few-row kernels on Qwen3.8 27B shapes, against
    /// the time to read the weights once.
    /// `cargo test --release -p dinf-engine --lib bench_few_row_mma -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_few_row_mma() -> anyhow::Result<()> {
        let Some(device) = metal() else {
            return Ok(());
        };
        for (dtype, in_dim, out_dim) in [
            (GgmlDType::Q4K, 5120usize, 34816usize),
            (GgmlDType::Q4K, 17408, 5120),
            (GgmlDType::Q8_0, 5120, 16384),
            (GgmlDType::Q6K, 6144, 5120),
        ] {
            let weights = Tensor::randn(0f32, 1.0, (out_dim, in_dim), &Device::Cpu)?;
            let raw = QTensor::quantize(&weights, dtype)?.data()?.into_owned();
            let bytes = raw.len();
            let raw = Tensor::from_raw_buffer(&raw, DType::U8, &[bytes], &device)?;
            for rows in [2usize, 3, 4, 7, 8, 16] {
                if !applies(dtype, in_dim, rows) {
                    continue;
                }
                let input = Tensor::randn(0f32, 1.0, (rows, in_dim), &device)?;
                let op = FewRow {
                    dtype,
                    in_dim,
                    out_dim,
                };
                for _ in 0..3 {
                    drop(raw.apply_op2_no_bwd(&input, &op)?);
                }
                device.synchronize()?;
                let started = std::time::Instant::now();
                for _ in 0..30 {
                    drop(raw.apply_op2_no_bwd(&input, &op)?);
                }
                device.synchronize()?;
                let ms = started.elapsed().as_secs_f64() * 1e3 / 30.0;
                eprintln!(
                    "{dtype:?} {out_dim}x{in_dim} rows={rows}: {ms:.3} ms, {:.0} GB/s weights",
                    bytes as f64 / ms / 1e6
                );
            }
        }
        Ok(())
    }

    /// Prompt GEMM against Candle's quantized matmul on Qwen3.8 27B shapes.
    /// `cargo test --release -p dinf-engine --lib bench_llama_mm -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_llama_mm() -> anyhow::Result<()> {
        let Some(device) = metal() else {
            return Ok(());
        };
        let shapes = match std::env::var("DINF_BENCH_MM_SHAPE") {
            // e.g. q8_0:5120:96
            Ok(shape) => {
                let parts: Vec<_> = shape.split(':').collect();
                let dtype = match parts[0] {
                    "q4_K" => GgmlDType::Q4K,
                    "q6_K" => GgmlDType::Q6K,
                    _ => GgmlDType::Q8_0,
                };
                vec![(dtype, parts[1].parse()?, parts[2].parse()?)]
            }
            Err(_) => vec![
                (GgmlDType::Q4K, 5120usize, 34816usize),
                (GgmlDType::Q4K, 17408, 5120),
                (GgmlDType::Q8_0, 5120, 16384),
                (GgmlDType::Q6K, 6144, 5120),
            ],
        };
        for (dtype, in_dim, out_dim) in shapes {
            let weights = Tensor::randn(0f32, 1.0, (out_dim, in_dim), &Device::Cpu)?;
            let quantized = QTensor::quantize(&weights, dtype)?;
            let raw = quantized.data()?.into_owned();
            let raw = Tensor::from_raw_buffer(&raw, DType::U8, &[raw.len()], &device)?;
            let candle = candle_core::quantized::QMatMul::from_qtensor(QTensor::quantize_onto(
                &weights, dtype, &device,
            )?)?;
            let rows = 512;
            let input = Tensor::randn(0f32, 1.0, (rows, in_dim), &device)?;
            let op = FewRow {
                dtype,
                in_dim,
                out_dim,
            };
            let mut ours = Vec::new();
            let mut theirs = Vec::new();
            for _ in 0..5 {
                for (port, times) in [(true, &mut ours), (false, &mut theirs)] {
                    let run = || -> anyhow::Result<()> {
                        if port {
                            drop(raw.apply_op2_no_bwd(&input, &op)?);
                        } else {
                            use candle_core::Module;
                            drop(candle.forward(&input)?);
                        }
                        Ok(())
                    };
                    run()?;
                    device.synchronize()?;
                    let started = std::time::Instant::now();
                    for _ in 0..5 {
                        run()?;
                    }
                    device.synchronize()?;
                    times.push(started.elapsed().as_secs_f64() * 1e3 / 5.0);
                }
            }
            // Alternative: dequantize to f16 once, then Candle's f16 GEMM
            // (MLX steel kernels). Times the two parts separately.
            let qdev = QTensor::quantize_onto(&weights, dtype, &device)?;
            let input16 = input.to_dtype(DType::F16)?;
            let w16 = qdev.dequantize_f16(&device)?;
            let (mut deq, mut gemm) = (Vec::new(), Vec::new());
            for _ in 0..5 {
                device.synchronize()?;
                let started = std::time::Instant::now();
                for _ in 0..5 {
                    drop(qdev.dequantize_f16(&device)?);
                }
                device.synchronize()?;
                deq.push(started.elapsed().as_secs_f64() * 1e3 / 5.0);
                let started = std::time::Instant::now();
                for _ in 0..5 {
                    drop(input16.matmul(&w16.t()?)?);
                }
                device.synchronize()?;
                gemm.push(started.elapsed().as_secs_f64() * 1e3 / 5.0);
            }
            let best = |v: &[f64]| v.iter().copied().fold(f64::MAX, f64::min);
            let tflops = |ms: f64| 2.0 * (rows * in_dim * out_dim) as f64 / ms / 1e9;
            eprintln!(
                "{dtype:?} {out_dim}x{in_dim} rows={rows}: port {:.2} ms ({:.2} TFLOPS), candle {:.2} ms, f16 dequant {:.2} + steel gemm {:.2} ms ({:.2} TFLOPS)",
                best(&ours),
                tflops(best(&ours)),
                best(&theirs),
                best(&deq),
                best(&gemm),
                tflops(best(&gemm))
            );
        }
        Ok(())
    }
}
