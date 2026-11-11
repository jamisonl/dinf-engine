//! Synthetic public Q8 data only. No model files or proof-artifact rewrites.
//! Criteria are declared here BEFORE any timing hook: finite candidate/scalar,
//! decoded FP64 dot oracle, gamma(n)*L1 per-output bounds, plus global RMS error.
//! Bit identity is deliberately NOT the matrix contract; old exact gates remain
//! on their original paths with the new thread-local selector OFF by default.

use super::super::{
    DeepSeek4DType, DeepSeek4QMatMul, Q8_PREFILL_MATRIX_ENABLED, RawTensor,
    with_q8_prefill_matrix_enabled,
};
use super::{
    DISPATCHES, Geometry, THREADGROUP_SCRATCH_BYTES, checked_span, supported_pipeline, try_forward,
};
use anyhow::Result;
use candle_core::{DType, Device, Layout, Shape, Tensor};
use std::sync::Arc;

const ABS_FLOOR: f64 = 2.0e-5;
const MATRIX_RMS_REL: f64 = 2.0e-5;
const SCALAR_RMS_REL: f64 = 4.0e-5;

fn dispatches() -> usize {
    DISPATCHES.with(|v| v.get())
}
thread_local! { static NUMERIC_RECORDS: std::cell::RefCell<Vec<serde_json::Value>> = const { std::cell::RefCell::new(Vec::new()) }; }

#[test]
fn q8_mma_geometry_and_span_boundaries() -> Result<()> {
    for k in [32, 96, 1280, 5120, 6144, 8192, 65536] {
        for t in [2, 3, 4, 8, 16, 25, 31, 32, 128, 512] {
            let g = Geometry::new(k, 33, &[t, k]).expect("valid bounded 2D shape");
            assert_eq!(g.input_elements, t * k);
            assert_eq!(g.output_elements, t * 33);
            assert_eq!(g.weight_bytes, 33 * k / 32 * 34);
        }
    }
    for (k, m, dims) in [
        (0, 33, vec![2, 0]),
        (31, 33, vec![2, 31]),
        (32, 0, vec![2, 32]),
        (32, 65537, vec![2, 32]),
        (65568, 33, vec![2, 65568]),
        (32, 33, vec![1, 32]),
        (32, 33, vec![513, 32]),
        (32, 33, vec![0, 32]),
        (32, 33, vec![2, 31]),
        (32, 33, vec![1, 2, 32]),
        (usize::MAX, 33, vec![2, usize::MAX]),
    ] {
        assert!(Geometry::new(k, m, &dims).is_none(), "{dims:?}");
    }
    let input = Layout::new(Shape::from((3, 96)), vec![96, 1], 3);
    assert_eq!(checked_span(&input, DType::F32, 288, 291 * 4)?, 12);
    assert!(checked_span(&input, DType::F32, 288, 291 * 4 - 1).is_err());
    let packed = Layout::new(Shape::from(102), vec![1], 34);
    assert_eq!(checked_span(&packed, DType::U8, 102, 136)?, 34);
    assert!(checked_span(&packed, DType::U8, 101, 136).is_err());
    let odd = Layout::new(Shape::from(102), vec![1], 1);
    assert!(checked_span(&odd, DType::U8, 102, 103).is_err());
    let overflow = Layout::new(Shape::from((2, 32)), vec![32, 1], usize::MAX);
    assert!(checked_span(&overflow, DType::F32, 64, usize::MAX).is_err());
    let strided = Layout::new(Shape::from((2, 32)), vec![1, 2], 0);
    assert!(checked_span(&strided, DType::F32, 64, 256).is_err());
    Ok(())
}

#[test]
fn q8_mma_selector_restores_nested_and_unwound_scopes() {
    assert!(!Q8_PREFILL_MATRIX_ENABLED.with(|v| v.get()));
    with_q8_prefill_matrix_enabled(true, || {
        assert!(Q8_PREFILL_MATRIX_ENABLED.with(|v| v.get()));
        with_q8_prefill_matrix_enabled(false, || {
            assert!(!Q8_PREFILL_MATRIX_ENABLED.with(|v| v.get()))
        });
        assert!(Q8_PREFILL_MATRIX_ENABLED.with(|v| v.get()));
        let panic = std::panic::catch_unwind(|| {
            with_q8_prefill_matrix_enabled(false, || panic!("selector unwind"))
        });
        assert!(panic.is_err());
        assert!(Q8_PREFILL_MATRIX_ENABLED.with(|v| v.get()));
    });
    assert!(!Q8_PREFILL_MATRIX_ENABLED.with(|v| v.get()));
}

struct Fixture {
    matrix: DeepSeek4QMatMul,
    input: Tensor,
    packed: Vec<u8>,
    host: Vec<f32>,
    k: usize,
    m: usize,
    t: usize,
}

#[allow(
    clippy::excessive_precision,
    reason = "Exact binary fractions define quantization boundary fixtures."
)]
fn fixture(device: &Device, k: usize, m: usize, t: usize, offset: bool) -> Result<Fixture> {
    // Nonunit, signed, max finite half, minimum half subnormal, zero. The
    // maximum product 65504 * -128 must NOT overflow through a half temporary.
    let scales = [
        0x355bu16, 0x3c01, 0x7bff, 0xfbff, 0x0001, 0x03ff, 0x0400, 0xbc01, 0,
    ];
    let row_bytes = k / 32 * 34;
    let mut packed = Vec::with_capacity(m * row_bytes);
    for row in 0..m {
        for block in 0..k / 32 {
            packed.extend_from_slice(&scales[(row * 3 + block * 7) % scales.len()].to_le_bytes());
            for j in 0..32 {
                // Row zero has equal neighboring weights. Alternating inputs
                // below force cancellation rather than only positive sums.
                let jj = if row == 0 { j / 2 } else { j };
                let q = ((row * 19 + block * 11 + jj * 17) % 256) as u8;
                packed.push(q);
            }
        }
    }
    let wp = if offset { 34 } else { 0 };
    let mut backing = vec![0xff; wp];
    backing.extend_from_slice(&packed);
    let len = backing.len();
    let weights = Tensor::from_vec(backing, len, device)?.narrow(0, wp, packed.len())?;
    assert_eq!(weights.storage_and_layout().1.start_offset(), wp);
    let matrix = DeepSeek4QMatMul::new_v41_on_device(
        Arc::new(RawTensor::new(vec![m, k], DeepSeek4DType::Q8_0, weights)?),
        device,
    )?;
    let host = (0..t * k)
        .map(|i| {
            let token = i / k;
            let j = i % k;
            let magnitude = (((j / 2) * 13 + token * 7) % 31) as f32 / 16.0 + 0.03125;
            let signed = if j.is_multiple_of(2) {
                magnitude
            } else {
                -magnitude
            };
            // Exact cancellation for token 0, near cancellation for token 1,
            // varied signs/magnitudes for later tokens. Distinguishes row order.
            if token == 0 {
                signed
            } else if token == 1 {
                signed
                    + if j.is_multiple_of(2) {
                        0.0001220703125
                    } else {
                        0.0
                    }
            } else {
                signed + ((j * 5 + token * 11) % 17) as f32 / 32.0
            }
        })
        .collect::<Vec<_>>();
    let ip = if offset { 3 } else { 0 };
    let mut backing = vec![f32::NAN; ip];
    backing.extend_from_slice(&host);
    let len = backing.len();
    let input = Tensor::from_vec(backing, len, device)?
        .narrow(0, ip, host.len())?
        .reshape((t, k))?;
    assert_eq!(input.storage_and_layout().1.start_offset(), ip);
    Ok(Fixture {
        matrix,
        input,
        packed,
        host,
        k,
        m,
        t,
    })
}

// Independent scalar decode/indexing of the on-disk Q8 block format; no Metal
// lane mapping, shader decoder or backend reduction helper is reused here.
fn oracle(f: &Fixture, token: usize, row: usize) -> (f64, f64) {
    let mut dot = 0.0;
    let mut l1 = 0.0;
    for block in 0..f.k / 32 {
        let p = (row * (f.k / 32) + block) * 34;
        let d = half::f16::from_bits(u16::from_le_bytes([f.packed[p], f.packed[p + 1]])).to_f64();
        for j in 0..32 {
            let q = f.packed[p + 2 + j] as i8;
            let product = d * f64::from(q) * f64::from(f.host[token * f.k + block * 32 + j]);
            dot += product;
            l1 += product.abs();
        }
    }
    (dot, l1)
}

fn gamma(n: usize) -> f64 {
    let e = n as f64 * f64::from(f32::EPSILON);
    e / (1.0 - e)
}

fn check_numeric(f: &Fixture, actual: &[f32], scalar: &[f32], sampled: bool) {
    assert_eq!(actual.len(), f.t * f.m);
    assert_eq!(scalar.len(), actual.len());
    // Strict finite behavior across ALL decoded output rows, even in full-shape
    // sampled-oracle gates. No NaN-equals-NaN, skipped finite checks, or BF16 mask.
    assert!(
        actual.iter().all(|v| v.is_finite()),
        "MMA generated nonfinite output"
    );
    assert!(
        scalar.iter().all(|v| v.is_finite()),
        "scalar generated nonfinite output"
    );
    let mut signal2 = 0.0;
    let mut error2 = 0.0;
    let mut scalar_error2 = 0.0;
    let mut max_abs = 0.0f64;
    let mut n = 0;
    for token in 0..f.t {
        for row in 0..f.m {
            if sampled && !(row < 34 || row + 2 >= f.m || row % 1024 == 0) {
                continue;
            }
            let (reference, l1) = oracle(f, token, row);
            assert!(reference.is_finite() && l1.is_finite());
            let i = token * f.m + row;
            let a = f64::from(actual[i]);
            let s = f64::from(scalar[i]);
            // Each SIMD group processes at most ceil(K/4) terms; add 8 for
            // cross-group reduction/rounding. Scalar bound is conservative.
            let matrix_bound = ABS_FLOOR + gamma(f.k.div_ceil(4) + 8) * l1;
            let scalar_bound = ABS_FLOOR + gamma(f.k + 32) * l1;
            assert!(
                (a - reference).abs() <= matrix_bound,
                "MMA vs FP64 K={} M={} T={} row={row} token={token}: a={a} ref={reference} L1={l1} bound={matrix_bound}",
                f.k,
                f.m,
                f.t
            );
            assert!(
                (s - reference).abs() <= scalar_bound,
                "scalar vs FP64 row={row} token={token}"
            );
            assert!(
                (a - s).abs() <= matrix_bound + scalar_bound,
                "MMA vs scalar row={row} token={token}"
            );
            signal2 += reference * reference;
            error2 += (a - reference).powi(2);
            max_abs = max_abs.max((a - reference).abs());
            scalar_error2 += (s - reference).powi(2);
            n += 1;
        }
    }
    let floor = ABS_FLOOR * (n as f64).sqrt();
    assert!(
        error2.sqrt() <= MATRIX_RMS_REL * signal2.sqrt() + floor,
        "MMA RMS error={} signal={} count={n}",
        error2.sqrt(),
        signal2.sqrt()
    );
    assert!(
        scalar_error2.sqrt() <= SCALAR_RMS_REL * signal2.sqrt() + floor,
        "scalar RMS error={} signal={} count={n}",
        scalar_error2.sqrt(),
        signal2.sqrt()
    );
    NUMERIC_RECORDS.with(|v| {
        v.borrow_mut().push(serde_json::json!({
        "K":f.k,"M":f.m,"T":f.t,"compared":n,"max_abs":max_abs,
        "rms":(error2/n as f64).sqrt(),"scalar_rms":(scalar_error2/n as f64).sqrt(),
        "signal_l2":signal2.sqrt(),"all_finite":true}))
    });
}

fn candidate(f: &Fixture, attention: bool) -> Result<Tensor> {
    with_q8_prefill_matrix_enabled(true, || {
        if attention {
            f.matrix.forward_attention_prefill(&f.input)
        } else {
            f.matrix.forward_prefill_matrix(&f.input)
        }
    })
}

fn qualify(f: &Fixture, attention: bool, sampled: bool) -> Result<()> {
    let before = dispatches();
    let actual = candidate(f, attention)?;
    // Retain all owners through a fence even if scalar encoding returns error.
    let scalar = with_q8_prefill_matrix_enabled(false, || f.matrix.forward(&f.input));
    let fence = f.input.device().synchronize();
    fence?;
    let scalar = scalar?;
    assert_eq!(
        dispatches() - before,
        1,
        "candidate must actually select MMA, not time fallback"
    );
    assert_eq!(actual.dims(), &[f.t, f.m]);
    assert_eq!(actual.dtype(), DType::F32);
    assert!(actual.is_contiguous());
    let a = actual.flatten_all()?.to_vec1::<f32>()?;
    let s = scalar.flatten_all()?.to_vec1::<f32>()?;
    check_numeric(f, &a, &s, sampled);
    Ok(())
}

#[test]
fn q8_mma_fp64_scalar_f32_offsets_tails_and_extreme_scales() -> Result<()> {
    let device = Device::new_metal(0)?;
    assert!(
        supported_pipeline(device.as_metal_device()?).is_some(),
        "MMA pipeline must compile; no silent skip"
    );
    for (k, m) in [(32, 1), (96, 33), (256, 65)] {
        for t in [2, 3, 4, 8, 16, 25, 31, 32, 128, 512] {
            for offset in [false, true] {
                qualify(&fixture(&device, k, m, t, offset)?, false, false)?;
            }
        }
    }
    // Real native K widths, bounded M. Includes K=6144 for Engram, K=8192
    // outputB, K=1280 Q_B/indexer, K=5120 QA/KV/compressor/shared projections.
    for (k, t) in [(1280, 25), (5120, 31), (6144, 32), (8192, 128)] {
        qualify(&fixture(&device, k, 33, t, true)?, false, false)?;
    }
    let pipeline = supported_pipeline(device.as_metal_device()?).unwrap();
    use objc2_metal::{MTLComputePipelineState, MTLDevice};
    let records = NUMERIC_RECORDS.with(|v| v.borrow().clone());
    println!(
        "Q8 MMA numeric evidence: {}",
        serde_json::to_string_pretty(&serde_json::json!({"records":records,
            "static_scratch_bytes":pipeline.as_ref().staticThreadgroupMemoryLength(),
            "dynamic_scratch_bytes":0,"device_cap_bytes":device.as_metal_device()?.metal_device().as_ref().maxThreadgroupMemoryLength()}))?
    );
    Ok(())
}

#[test]
fn q8_mma_f32_input_range_stays_finite_without_half_staging() -> Result<()> {
    let device = Device::new_metal(0)?;
    let mut f = fixture(&device, 96, 33, 3, true)?;
    // 2^70 is far outside finite half range, but even max Q8 weights and
    // the entire dot fit comfortably in F32. Lane-varying offsets prevent
    // exact cancellation from hiding accidental half staging.
    for token in 0..f.t {
        for j in 0..f.k {
            let large = 2.0f32.powi(70) * (1.0 + ((j * 3 + token) % 7) as f32 / 8.0);
            f.host[token * f.k + j] = if j % 2 == 0 { large } else { -large * 0.75 };
        }
    }
    let mut backing = vec![f32::NAN; 3];
    backing.extend_from_slice(&f.host);
    let len = backing.len();
    f.input = Tensor::from_vec(backing, len, &device)?
        .narrow(0, 3, f.host.len())?
        .reshape((f.t, f.k))?;
    qualify(&f, false, false)
}

#[test]
fn q8_mma_overflow_redo_preserves_finite_scalar_cancellation() -> Result<()> {
    let d = Device::new_metal(0)?;
    for (k, m, t) in [(32, 1, 2), (96, 33, 3), (1280, 7, 17)] {
        let mut bytes = Vec::new();
        for _ in 0..m * (k / 32) {
            bytes.extend_from_slice(&0x7bffu16.to_le_bytes());
            bytes.extend_from_slice(&[1u8; 32]);
        }
        let len = bytes.len();
        let weights = Tensor::from_vec(bytes, len, &d)?;
        let matrix = DeepSeek4QMatMul::new_v41_on_device(
            Arc::new(RawTensor::new(vec![m, k], DeepSeek4DType::Q8_0, weights)?),
            &d,
        )?;
        let x = Tensor::from_vec(
            (0..t * k)
                .map(|i| if i % 2 == 0 { 1e34f32 } else { -1e34f32 })
                .collect(),
            (t, k),
            &d,
        )?;
        let scalar = matrix.forward(&x)?;
        let before = dispatches();
        let output = with_q8_prefill_matrix_enabled(true, || matrix.forward_prefill_matrix(&x))?;
        d.synchronize()?;
        assert_eq!(dispatches(), before + 1);
        let expected = scalar.flatten_all()?.to_vec1::<f32>()?;
        let actual = output.flatten_all()?.to_vec1::<f32>()?;
        assert!(expected.iter().all(|&v| v == 0.0 && v.is_finite()));
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits())
        );
    }
    Ok(())
}

#[test]
fn q8_mma_negative_preflight_never_enqueues() -> Result<()> {
    let device = Device::new_metal(0)?;
    let weights = Tensor::zeros(34 * 33, DType::U8, &device)?;
    let input = Tensor::zeros((2, 32), DType::F32, &device)?;
    let before = dispatches();
    for bad in [
        Tensor::zeros((1, 32), DType::F32, &device)?,
        Tensor::zeros((513, 32), DType::F32, &device)?,
        Tensor::zeros((2, 31), DType::F32, &device)?,
        Tensor::zeros((1, 2, 32), DType::F32, &device)?,
        Tensor::zeros((2, 32), DType::F16, &device)?,
        Tensor::zeros((32, 2), DType::F32, &device)?.t()?,
        Tensor::zeros((2, 32), DType::F32, &Device::Cpu)?,
    ] {
        assert!(try_forward(&weights, &bad, 32, 33)?.is_none());
    }
    for bad in [
        Tensor::zeros(34 * 33, DType::F32, &device)?,
        Tensor::zeros(34 * 33 - 1, DType::U8, &device)?,
        Tensor::zeros(34 * 33 + 1, DType::U8, &device)?,
        Tensor::zeros(34 * 33 + 1, DType::U8, &device)?.narrow(0, 1, 34 * 33)?,
        Tensor::zeros((34 * 33, 2), DType::U8, &device)?.narrow(1, 0, 1)?,
        Tensor::zeros(34 * 33, DType::U8, &Device::Cpu)?,
    ] {
        assert!(try_forward(&bad, &input, 32, 33)?.is_none());
    }
    assert_eq!(
        dispatches(),
        before,
        "unsupported inputs must not enqueue an MMA prefix"
    );
    let f = fixture(&device, 96, 33, 3, true)?;
    let baseline =
        with_q8_prefill_matrix_enabled(false, || f.matrix.forward_prefill_matrix(&f.input))?;
    let ordinary = with_q8_prefill_matrix_enabled(true, || f.matrix.forward(&f.input))?;
    device.synchronize()?;
    assert_eq!(
        dispatches(),
        before,
        "OFF and ordinary forward must not dispatch candidate"
    );
    assert_eq!(
        baseline.flatten_all()?.to_vec1::<f32>()?,
        ordinary.flatten_all()?.to_vec1::<f32>()?
    );
    // Explicit 2D API must not promote grouped/rank-3 inputs even in a scope.
    let rank3 = f.input.reshape((1, 3, 96))?;
    let out = with_q8_prefill_matrix_enabled(true, || f.matrix.forward_prefill_matrix(&rank3))?;
    device.synchronize()?;
    assert_eq!(out.dims(), &[1, 3, 33]);
    assert_eq!(dispatches(), before);
    Ok(())
}

#[test]
#[ignore = "full native synthetic matrices; root runs after source writers exit, no model files"]
fn q8_mma_native_shapes_selection_and_numeric_gate() -> Result<()> {
    let device = Device::new_metal(0)?;
    for (k, m) in [(1280, 32768), (8192, 5120), (5120, 1280), (5120, 512)] {
        for t in [2, 3, 4, 8, 16, 25, 31, 32, 128] {
            qualify(&fixture(&device, k, m, t, true)?, true, true)?;
        }
    }
    // Explicit API only: not an expansion of the attention whitelist.
    for (k, m) in [(1280, 4096), (6144, 25600), (5120, 512)] {
        qualify(&fixture(&device, k, m, 25, true)?, false, true)?;
    }
    Ok(())
}

fn timed_batch(f: &Fixture, enabled: bool, iterations: usize) -> Result<f64> {
    f.input.device().synchronize()?;
    let start = std::time::Instant::now();
    let mut outputs = Vec::with_capacity(iterations);
    let result = with_q8_prefill_matrix_enabled(enabled, || -> Result<()> {
        for _ in 0..iterations {
            outputs.push(f.matrix.forward_attention_prefill(&f.input)?);
        }
        Ok(())
    });
    // On both success and failure, retain outputs and original inputs/weights
    // through completion. The timed region is identical for both selectors.
    let fence = f.input.device().synchronize();
    let elapsed = start.elapsed().as_secs_f64();
    fence?;
    result?;
    std::hint::black_box(&outputs);
    Ok(elapsed)
}

#[test]
#[ignore = "matched ABBA component timing only; numeric gate runs first; not a model speed claim"]
fn q8_mma_matched_component_timing_hook() -> Result<()> {
    let device = Device::new_metal(0)?;
    for (k, m) in [(1280, 32768), (8192, 5120), (5120, 1280), (5120, 512)] {
        for t in [2, 8, 16, 25, 31, 32, 128, 512] {
            let f = fixture(&device, k, m, t, true)?;
            qualify(&f, true, true)?;
            timed_batch(&f, false, 2)?;
            let before = dispatches();
            timed_batch(&f, true, 2)?;
            assert_eq!(dispatches() - before, 2, "warmup must select MMA");
            for enabled in [false, true, true, false] {
                let before = dispatches();
                let seconds = timed_batch(&f, enabled, 8)?;
                let selected = dispatches() - before;
                assert_eq!(selected, if enabled { 8 } else { 0 });
                eprintln!(
                    "Q8_MMA_COMPONENT {}",
                    serde_json::json!({
                        "k": k, "m": m, "tokens": t, "matrix_enabled": enabled,
                        "matrix_dispatches": selected, "iterations": 8, "seconds": seconds,
                        "transient_threadgroup_bytes": THREADGROUP_SCRATCH_BYTES,
                        "retained_carrier_delta_bytes": 0,
                        "baseline": "existing forward_attention_prefill selector OFF",
                        "scope": "synthetic component only; not whole-model gain or quality"
                    })
                );
            }
        }
    }
    Ok(())
}

#[test]
fn q8_mma_malformed_storage_and_same_gpu_distinct_candle_device_reject() -> Result<()> {
    let d = Device::new_metal(0)?;
    let other = Device::new_metal(0)?;
    let weights = Tensor::zeros(34, DType::U8, &d)?;
    let input = Tensor::zeros((2, 32), DType::F32, &d)?;
    let short = d
        .as_metal_device()?
        .new_private_buffer(1, DType::F32, "q8-short-fixture")?;
    let malformed = Tensor::from((
        candle_core::Storage::Metal(candle_core::MetalStorage::new(
            short,
            d.as_metal_device()?.clone(),
            64,
            DType::F32,
        )),
        Shape::from((2, 32)),
    ));
    let before = dispatches();
    assert!(try_forward(&weights, &malformed, 32, 1)?.is_none());
    let other_input = Tensor::zeros((2, 32), DType::F32, &other)?;
    let other_weights = Tensor::zeros(34, DType::U8, &other)?;
    assert!(try_forward(&weights, &other_input, 32, 1)?.is_none());
    assert!(try_forward(&other_weights, &input, 32, 1)?.is_none());
    assert_eq!(dispatches(), before);
    // Both logical devices refer to the SAME physical GPU; no foreign GPU claim.
    d.synchronize()?;
    other.synchronize()?;
    Ok(())
}
