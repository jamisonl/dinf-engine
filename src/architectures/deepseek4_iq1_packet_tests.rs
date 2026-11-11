//! Synthetic fixtures and negative tests for the IQ1 packet kernels.
use super::*;
use candle_core::Device;
use serde_json::{Value, json};

thread_local! {
    // Enabled only by malformed-plan tests; ordinary forwards leave it unset.
    pub(super) static ENCODER_PROBE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

impl Plan {
    pub(crate) fn diagnostics(&self) -> Value {
        let p: Vec<usize> = self.packets.iter().map(Vec::len).collect();
        let gu: Vec<usize> = p.iter().map(|v| 2 * v).collect();
        let down = p.clone();
        json!({"rows":self.routes.len(),"P1":p[0],"P2":p[1],"P3":p[2],"P4":p[3],
            "occurrences":6*self.routes.len(),"singleton_fraction":p[0] as f64/(6*self.routes.len()) as f64,
            "gate_up_weight_packets":gu,"down_weight_packets":down,
            "gate_up_groups_by_arity":gu.iter().map(|v|v.div_ceil(GROUP)).collect::<Vec<_>>(),
            "down_groups_by_arity":down.iter().map(|v|v.div_ceil(GROUP)).collect::<Vec<_>>()})
    }
}

/// Print a compact summary of a test run (visible with `--nocapture`).
fn report(name: &str, value: &Value) -> Result<()> {
    println!("{name}: {value}");
    Ok(())
}

#[test]
fn iq1_packet_128_cpu_coverage_and_caps() -> Result<()> {
    for t in 1..=MAX_ROWS {
        for fanout in [1, 2, 3, 4, 5, 7, 33, 192] {
            let routes: Vec<[usize; 6]> = (0..t)
                .map(|r| std::array::from_fn(|s| ((6 * r + s) / fanout) % 384))
                .collect();
            let p = Plan::new(&routes)?;
            for phase in [Phase::GateUp, Phase::Down] {
                let (inputs, _, outputs, _) = phase.geometry(t);
                let mut seen = vec![0; outputs];
                for bucket in &p.packets {
                    for packet in bucket {
                        for &kind in phase.kinds() {
                            for &o in &packet.occurrences {
                                let (src, dst) = phase.rows(o, kind);
                                assert!(src < inputs);
                                seen[dst] += 1;
                            }
                        }
                    }
                }
                assert!(seen.iter().all(|&n| n == 1));
            }
            let mut bad = p.clone();
            bad.packets
                .iter_mut()
                .find(|b| !b.is_empty())
                .unwrap()
                .pop();
            assert!(bad.validate().is_err());
        }
    }
    assert!(Plan::new(&[[0; 6]; 129]).is_err());
    assert!(carrier_bytes(129).is_err());
    assert_eq!(carrier_bytes(128)?, 64 * 1024 * 1024);
    Ok(())
}

#[test]
#[ignore = "slow; exhaustive CPU packet planning and malformed-plan guards"]
fn iq1_packet_cpu_plan_and_spans() -> Result<()> {
    let mut cases = 0;
    for t in 1..=32 {
        for fanout in [1, 2, 3, 4, 5, 7, 33, 192] {
            let routes: Vec<[usize; 6]> = (0..t)
                .map(|token| std::array::from_fn(|s| (6 * token + s) / fanout))
                .collect();
            let p = Plan::new(&routes)?;
            for phase in [Phase::GateUp, Phase::Down] {
                let (inputs, _, outputs, _) = phase.geometry(t);
                let mut seen = vec![0; outputs];
                for bucket in &p.packets {
                    for packet in bucket {
                        for &kind in phase.kinds() {
                            for &o in &packet.occurrences {
                                let (src, dst) = phase.rows(o, kind);
                                assert!(src < inputs);
                                seen[dst] += 1;
                            }
                        }
                    }
                }
                assert!(seen.iter().all(|&v| v == 1));
            }
            let mut bad = p.clone();
            bad.packets
                .iter_mut()
                .find(|b| !b.is_empty())
                .unwrap()
                .pop();
            assert!(bad.validate().is_err());
            let mut bad = p.clone();
            let packet = bad
                .packets
                .iter_mut()
                .find(|b| !b.is_empty())
                .unwrap()
                .first_mut()
                .unwrap();
            packet.occurrences[0].token = usize::MAX;
            assert!(bad.validate().is_err());
            let mut bad = p.clone();
            let packet = bad
                .packets
                .iter_mut()
                .find(|b| !b.is_empty())
                .unwrap()
                .first_mut()
                .unwrap();
            packet.expert = 383;
            assert!(bad.validate().is_err());
            cases += 1;
        }
    }
    assert!(Plan::new(&[]).is_err());
    assert!(Plan::new(&[[0; 6]; 129]).is_err());
    assert!(Plan::new(&[[384; 6]]).is_err());
    assert!(carrier_bytes(0).is_err());
    assert!(carrier_bytes(usize::MAX).is_err());
    for t in [25, 26, 32] {
        assert_eq!(carrier_bytes(t)?, 16 * 1024 * 1024);
    }
    for phase in [Phase::GateUp, Phase::Down] {
        let (r, k, _, _) = phase.geometry(32);
        let layout = |offset| Layout::new(Shape::from((r, k)), vec![k, 1], offset);
        assert_eq!(
            grouped_iq1_span(&layout(3), DType::F32, r * k, (r * k + 3) * 4)?,
            12
        );
        assert!(grouped_iq1_span(&layout(3), DType::F32, r * k, (r * k + 3) * 4 - 1).is_err());
        assert!(grouped_iq1_span(&layout(usize::MAX), DType::F32, r * k, usize::MAX).is_err());
        assert!(
            grouped_iq1_span(
                &Layout::new(Shape::from((r, k)), vec![k * 2, 2], 0),
                DType::F32,
                r * k,
                usize::MAX
            )
            .is_err()
        );
    }
    for offset in [1, 3, usize::MAX] {
        let l = Layout::new(Shape::from(PACKED), vec![1], offset);
        assert!(grouped_iq1_span(&l, DType::U8, PACKED, usize::MAX).is_err());
    }
    report(
        "iq1-packet-cpu-plan",
        &json!({"passed":true,"planning_cases":cases,"T":"1..32",
        "canonical_T25_T26_T32_bytes":16777216,"encoder_dispatches":0}),
    )
}

#[allow(
    clippy::excessive_precision,
    reason = "Exact binary fractions define quantization boundary fixtures."
)]
pub(crate) fn fixture_weights(
    device: &Device,
    out: usize,
    k: usize,
    slot: usize,
) -> Result<DeepSeek4QMatMul> {
    let prefix = 50 * (slot + 1);
    let mut bytes = vec![0xff; prefix];
    let scales = [
        half::f16::from_bits(1).to_f32(),
        0.00006103515625,
        0.03125,
        0.375,
        1.0009765625,
        3.,
        31.,
        256.,
        2048.,
        16384.,
        65504.,
        -65504.,
    ];
    for row in 0..out {
        for block in 0..k / 256 {
            let mut packed = [0u8; 50];
            packed[..2].copy_from_slice(
                &half::f16::from_f32(scales[(slot + row / 17 + block) % scales.len()])
                    .to_le_bytes(),
            );
            for (j, b) in packed[2..].iter_mut().enumerate() {
                *b = ((slot * 53 + row * 17 + block * 31 + j * 7) % 256) as u8;
            }
            bytes.extend_from_slice(&packed);
        }
    }
    bytes.extend_from_slice(&[0xff; 50]);
    let len = bytes.len();
    let view = Tensor::from_vec(bytes, len, device)?.narrow(0, prefix, PACKED)?;
    DeepSeek4QMatMul::new(Arc::new(RawTensor::new(
        vec![out, k],
        DeepSeek4DType::IQ1S,
        view,
    )?))
}
fn fixture_input(device: &Device, rows: usize, k: usize, case: usize) -> Result<Tensor> {
    let mut values = vec![999f32; 3];
    let boundary = [
        0.,
        -0.,
        1.,
        -1.,
        1.0039063,
        -1.0039063,
        1e-20,
        -1e-20,
        1e20,
        -1e20,
        65504.,
        -65504.,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        7.,
        -7.,
    ];
    for r in 0..rows {
        for i in 0..k {
            // Repeated input rows remain separate writes, including aliased weights.
            let token = r % 5;
            values.push(match case {
                0 => {
                    if i == 17 + token * 113 {
                        1. + token as f32 / 8.
                    } else {
                        0.
                    }
                }
                1 => (((i * 37 + token * 71) % 257) as f32 - 128.) / (token + 7) as f32,
                2 => {
                    (if (i + token) % 2 == 0 { 1. } else { -1. })
                        * (if i % 7 == 0 { 1e12 } else { 1e-12 })
                        * (token + 1) as f32
                }
                _ => boundary[(i + token * 7) % boundary.len()],
            });
        }
    }
    values.extend_from_slice(&[999.; 3]);
    let len = values.len();
    Ok(Tensor::from_vec(values, len, device)?
        .narrow(0, 3, rows * k)?
        .reshape((rows, k))?)
}
fn finite_bits(t: &Tensor) -> Result<Vec<u32>> {
    t.flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|v| {
            anyhow::ensure!(v.is_finite(), "nonfinite packet output");
            Ok(v.to_bits())
        })
        .collect()
}
struct Terminal<'a> {
    device: &'a Device,
    done: bool,
}
impl Terminal<'_> {
    fn finish(&mut self) -> Result<()> {
        self.device.synchronize()?;
        self.done = true;
        Ok(())
    }
}
impl Drop for Terminal<'_> {
    fn drop(&mut self) {
        if !self.done {
            let _ = self.device.synchronize();
        }
    }
}
#[test]
#[ignore = "requires a Metal device; indexed N=1..4 finite-bit and scatter coverage"]
fn iq1_packet_metal_indexed_finite_bits() -> Result<()> {
    let _scalar_scope = MatrixTestScope::new(false);
    anyhow::ensure!(
        DeepSeek4QMatMul::iq1_simd_enabled(),
        "scalar SIMD oracle required"
    );
    let device = Device::new_metal(0)?;
    let gate = (0..4)
        .map(|i| fixture_weights(&device, 2304, 5120, i))
        .collect::<Result<Vec<_>>>()?;
    let up = (0..4)
        .map(|i| fixture_weights(&device, 2304, 5120, i + 4))
        .collect::<Result<Vec<_>>>()?;
    let down = (0..4)
        .map(|i| fixture_weights(&device, 5120, 2304, i + 8))
        .collect::<Result<Vec<_>>>()?;
    let mut owners = Vec::<Tensor>::new();
    let mut terminal = Terminal {
        device: &device,
        done: false,
    };
    let mut compared = 0usize;
    let mut reports = Vec::new();
    let mut arities = [false; 4];
    for t in 1..=32 {
        for case in 0..4 {
            // Runs of 1,2,3,4,5,9 occurrences exercise remainders and >4 splits.
            let runs = [1, 2, 3, 4, 5, 9];
            let mut ids = Vec::new();
            while ids.len() < 6 * t {
                for (expert, &n) in runs.iter().enumerate() {
                    ids.extend(std::iter::repeat_n(expert, n));
                }
            }
            let routes: Vec<[usize; 6]> = (0..t)
                .map(|r| std::array::from_fn(|s| ids[6 * r + s]))
                .collect();
            let plan = Plan::new(&routes)?;
            for (i, b) in plan.packets.iter().enumerate() {
                arities[i] |= !b.is_empty();
            }
            let select = |expert: usize, kind| -> Result<&DeepSeek4QMatMul> {
                Ok(match kind {
                    0 => &gate[expert % 4],
                    1 => &up[expert % 4],
                    2 => &down[expert % 4],
                    _ => unreachable!(),
                })
            };
            for phase in [Phase::GateUp, Phase::Down] {
                let (input_rows, k, output_rows, out) = phase.geometry(t);
                let x = fixture_input(&device, input_rows, k, case)?;
                owners.push(x.clone());
                let actual = forward(&x, &plan, phase, select)?;
                owners.push(actual.clone());
                let mut expected = Vec::with_capacity(output_rows);
                for (token, route) in routes.iter().enumerate() {
                    for (slot, &expert) in route.iter().enumerate() {
                        for &kind in phase.kinds() {
                            let (src, _) = phase.rows(Occurrence { token, slot }, kind);
                            let y = select(expert, kind)?.forward(&x.narrow(0, src, 1)?)?;
                            owners.push(y.clone());
                            expected.push(y);
                        }
                    }
                }
                // Allocation pressure while every queued operand/result is owned.
                let pressure = Tensor::full(-987f32, (output_rows, out), &device)?;
                owners.push(pressure);
                let bad = Tensor::zeros((input_rows, k + 1), DType::F32, &device)?;
                owners.push(bad.clone());
                assert!(forward(&bad, &plan, phase, select).is_err());
                let canonical = Tensor::cat(&expected, 0)?;
                owners.push(canonical.clone());
                let bits = finite_bits(&actual)?;
                let oracle = finite_bits(&canonical)?;
                assert_eq!(actual.dims(), [output_rows, out]);
                assert_eq!(bits.len(), output_rows * out);
                // Full canonical placement equality: every destination, not aggregate sums.
                assert_eq!(bits, oracle, "T{t},case{case},phase{phase:?}");
                compared += bits.len();
                owners.clear();
            }
            if case == 0 {
                reports.push(plan.diagnostics());
            }
        }
    }
    assert_eq!(arities, [true; 4]);
    terminal.finish()?;
    report(
        "iq1-packet-metal-finite",
        &json!({"passed":true,"finite_raw_f32_comparisons":compared,
        "T":"1..32","arities":[1,2,3,4],"cases":["basis","dense","cancellation","wide-scale"],
        "nonzero_input_offset_bytes":12,"weight_offsets":"50-byte aligned, all nonzero",
        "aliased_weights":true,"repeated_inputs_distinct_destinations":true,"canonical_complete_writes":true,
        "same_size_allocation_pressure":true,"injected_shape_errors":true,"plans":reports,
        "cleanup":{"owned_children":0,"terminal_fence":true}}),
    )
}

#[test]
#[ignore = "requires a Metal device; malformed metadata/storage must reject before encoding"]
fn iq1_packet_metal_preflight_rejections() -> Result<()> {
    let _scalar_scope = MatrixTestScope::new(false);
    packet_preflight_rejections("iq1-packet-metal-guards")
}
fn packet_preflight_rejections(report_name: &str) -> Result<()> {
    let d = Device::new_metal(0)?;
    let matrix = fixture_weights(&d, 2304, 5120, 0)?;
    let other_device = Device::new_metal(0)?;
    let foreign = fixture_weights(&other_device, 2304, 5120, 1)?;
    let mut owners = Vec::new();
    let mut terminal = Terminal {
        device: &d,
        done: false,
    };
    struct ProbeRestore(Option<usize>);
    impl Drop for ProbeRestore {
        fn drop(&mut self) {
            ENCODER_PROBE.with(|v| v.set(self.0));
        }
    }
    let _probe = ProbeRestore(ENCODER_PROBE.with(|v| v.replace(Some(0))));
    let plan = Plan::new(&[[0, 1, 1, 2, 2, 2]])?;
    let x = fixture_input(&d, 1, 5120, 1)?;
    owners.push(x.clone());
    let down_x = fixture_input(&d, 6, 2304, 1)?;
    owners.push(down_x.clone());
    let make = || {
        let mut buckets: [Vec<MatrixPacket>; 4] = std::array::from_fn(|_| Vec::new());
        for (n, bucket) in plan.packets.iter().enumerate() {
            for packet in bucket {
                for &kind in Phase::GateUp.kinds() {
                    let mut indices = Indices::default();
                    for (j, &o) in packet.occurrences.iter().enumerate() {
                        let (src, dst) = Phase::GateUp.rows(o, kind);
                        indices.source[j] = src as u32;
                        indices.destination[j] = dst as u32;
                    }
                    buckets[n].push(MatrixPacket {
                        weights: matrix.weights.clone(),
                        indices,
                    });
                }
            }
        }
        PacketOp {
            plan: plan.clone(),
            phase: Phase::GateUp,
            buckets,
        }
    };
    let mut rejected = 0;
    for case in 0..8 {
        let mut op = make();
        match case {
            0 => {
                op.buckets[2].pop();
            }
            1 => op.buckets[2][0].indices.source[0] = u32::MAX,
            2 => op.buckets[2][0].indices.destination[0] = u32::MAX,
            3 => op.buckets[2][0].indices.destination[1] = op.buckets[2][0].indices.destination[0],
            4 => op.buckets[0][0].indices.source[1] = 1,
            5 => op.phase = Phase::Down,
            6 => op.plan.packets[2][0].expert = 383,
            7 => op.buckets[2][0].weights = foreign.weights.clone(),
            _ => unreachable!(),
        }
        let input = if case == 5 { &down_x } else { &x };
        let error = input
            .apply_op1_no_bwd(&op)
            .expect_err(&format!("malformed case{case}"));
        if case == 5 {
            assert!(
                error.to_string().contains("packet stale phase/count"),
                "{error}"
            );
        }
        assert_eq!(ENCODER_PROBE.with(|v| v.get()), Some(0));
        rejected += 1;
    }
    for (name, dtype, bytes, message) in [
        (
            "wrong IQ1 dtype",
            DeepSeek4DType::IQ2XS,
            matrix.weights.bytes().clone(),
            "packet stale weight geometry",
        ),
        (
            "short packed length",
            DeepSeek4DType::IQ1S,
            matrix.weights.bytes().narrow(0, 0, PACKED - 1)?,
            "packet stale weight geometry",
        ),
        (
            "odd packed offset",
            DeepSeek4DType::IQ1S,
            Tensor::zeros(PACKED + 1, DType::U8, &d)?.narrow(0, 1, PACKED)?,
            "grouped IQ1 packed offset must be two-byte aligned",
        ),
        (
            "noncontiguous packed vector",
            DeepSeek4DType::IQ1S,
            Tensor::zeros(1, DType::U8, &d)?.broadcast_as(PACKED)?,
            "grouped IQ1 storage must have an exact contiguous layout",
        ),
    ] {
        owners.push(bytes.clone());
        let mut op = make();
        op.buckets[2][0].weights = Arc::new(RawTensor::new(vec![2304, 5120], dtype, bytes)?);
        let error = x.apply_op1_no_bwd(&op).expect_err(name);
        assert!(error.to_string().contains(message), "{name}: {error}");
        assert_eq!(ENCODER_PROBE.with(|v| v.get()), Some(0), "{name}");
        rejected += 1;
    }
    let strided = Tensor::zeros((5120, 2), DType::F32, &d)?
        .narrow(1, 0, 1)?
        .transpose(0, 1)?;
    assert_eq!(strided.dims(), x.dims());
    assert!(!strided.is_contiguous());
    owners.push(strided.clone());
    let error = strided
        .apply_op1_no_bwd(&make())
        .expect_err("correctly shaped noncontiguous F32 input");
    assert!(
        error
            .to_string()
            .contains("grouped IQ1 storage must have an exact contiguous layout"),
        "{error}"
    );
    assert_eq!(ENCODER_PROBE.with(|v| v.get()), Some(0));
    rejected += 1;
    for bad in [
        x.to_dtype(DType::F16)?,
        x.to_dtype(DType::BF16)?,
        x.flatten_all()?,
        Tensor::zeros((1, 5119), DType::F32, &d)?,
        Tensor::zeros((1, 10240), DType::F32, &d)?
            .narrow(1, 0, 5120)?
            .transpose(0, 1)?,
    ] {
        owners.push(bad.clone());
        assert!(forward(&bad, &plan, Phase::GateUp, |_, _| Ok(&matrix)).is_err());
        assert_eq!(ENCODER_PROBE.with(|v| v.get()), Some(0));
        rejected += 1;
    }
    assert!(forward(&x, &plan, Phase::GateUp, |_, _| Ok(&foreign)).is_err());
    rejected += 1;
    let wrong_shape = DeepSeek4QMatMul::new(Arc::new(RawTensor::new(
        vec![5120, 2304],
        DeepSeek4DType::IQ1S,
        matrix.weights.bytes().clone(),
    )?))?;
    assert!(forward(&x, &plan, Phase::GateUp, |_, _| Ok(&wrong_shape)).is_err());
    rejected += 1;
    assert_eq!(ENCODER_PROBE.with(|v| v.get()), Some(0));
    ENCODER_PROBE.with(|v| v.set(None));
    let actual = forward(&x, &plan, Phase::GateUp, |_, _| Ok(&matrix))?;
    owners.push(actual.clone());
    let pressure = Tensor::full(123f32, (12, 2304), &d)?;
    owners.push(pressure);
    terminal.finish()?;
    assert_eq!(finite_bits(&actual)?.len(), 12 * 2304);
    report(
        report_name,
        &json!({"passed":true,"rejections":rejected,
        "invalid_encoder_acquisitions":0,"invalid_dispatches":0,"guard_categories":[
            "late-group count","source index","destination index","duplicate destination","inactive metadata",
            "mixed phase/count with native Down input","stale expert","Candle/storage device","F16/BF16 input","rank","native width","native weight shape",
            "wrong IQ1 dtype","short packed length","odd packed offset","noncontiguous packed vector","correctly shaped noncontiguous F32 input"],
        "queued_owners_through_error_fence":true,"terminal_fence":true}),
    )
}

// Independent CPU decode: consume only the public IQ1 codebook data, not the
// GPU dequantizer. Decode each 8-value grid directly in FP64 from packed bytes.
fn iq1_codebook() -> Result<Vec<u32>> {
    let table = candle_metal_kernels::source::QUANTIZED
        .split_once("GGML_TABLE_BEGIN(uint32_t, iq1s_grid_gpu, NGRID_IQ1S)")
        .context("IQ1 public codebook")?
        .1
        .split("GGML_TABLE_END")
        .next()
        .unwrap();
    let values = table
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            Ok(u32::from_str_radix(
                s.strip_prefix("0x").context("IQ1 hex grid")?,
                16,
            )?)
        })
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(values.len() == 2048, "IQ1 codebook size");
    Ok(values)
}
fn decode_iq1_fp64(bytes: &[u8], grid: &[u32]) -> Vec<f64> {
    assert_eq!(bytes.len() % 50, 0);
    let mut values = Vec::with_capacity(bytes.len() / 50 * 256);
    for b in bytes.chunks_exact(50) {
        let d = half::f16::from_le_bytes([b[0], b[1]]).to_f64();
        for j in 0..8 {
            let h = u16::from_le_bytes([b[34 + 2 * j], b[35 + 2 * j]]);
            let scale = d * f64::from(2 * ((h >> 12) & 7) + 1);
            let delta = if h & 0x8000 == 0 { 0.125 } else { -0.125 };
            for q in 0..4 {
                let index = usize::from(b[2 + 4 * j + q]) | usize::from((h >> (3 * q)) & 7) << 8;
                let word = grid[index];
                for v in 0..8 {
                    let nibble = (word >> (8 * (v % 4) + 4 * (v / 4))) & 15;
                    values.push(scale * (f64::from(nibble) - 1. + delta));
                }
            }
        }
    }
    values
}
fn fp64_dots(
    w: &[f64],
    x: &[f32],
    k: usize,
    out: usize,
    indices: Indices,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut dots = Vec::with_capacity(n * out);
    let mut magnitude = Vec::with_capacity(n * out);
    for c in 0..n {
        for row in 0..out {
            let input = &x[indices.source[c] as usize * k..][..k];
            let (mut dot, mut l1) = (0., 0.);
            for (&a, &b) in w[row * k..][..k].iter().zip(input) {
                let p = a * f64::from(b);
                dot += p;
                l1 += p.abs();
            }
            dots.push(dot);
            magnitude.push(l1);
        }
    }
    (dots, magnitude)
}
// Bounds are on every decoded dot, including near-zero cancellation; no relative
// error division by a cancelling dot. Report raw absolute errors as well.
// max |error|/max(dot-L1,1e-30) <= 2e-5
// RMS(error)/RMS(dot-L1) and L2(error)/L2(dot-L1) <= 5e-6.
// Additionally L2(error)/L2(reference) <= 2e-4 when reference is well-conditioned.
fn bounded_error(actual: &[f32], reference: &[f64], l1: &[f64]) -> Result<Value> {
    anyhow::ensure!(
        actual.len() == reference.len() && actual.len() == l1.len() && !actual.is_empty(),
        "numeric geometry"
    );
    let (mut max_abs, mut max_conditioned, mut error2, mut ref2, mut l12) =
        (0f64, 0f64, 0f64, 0f64, 0f64);
    for ((&a, &r), &m) in actual.iter().zip(reference).zip(l1) {
        anyhow::ensure!(
            a.is_finite() && r.is_finite() && m.is_finite(),
            "nonfinite component result"
        );
        let e = (f64::from(a) - r).abs();
        max_abs = max_abs.max(e);
        max_conditioned = max_conditioned.max(e / m.max(1e-30));
        error2 += e * e;
        ref2 += r * r;
        l12 += m * m;
    }
    let conditioned_norm = (error2 / l12.max(1e-60)).sqrt();
    let relative_norm = (error2 / ref2.max(1e-60)).sqrt();
    anyhow::ensure!(
        max_conditioned <= 2e-5 && conditioned_norm <= 5e-6,
        "IQ1 conditioned error max={max_conditioned} norm={conditioned_norm}"
    );
    if ref2 > 1e-6 * l12 {
        anyhow::ensure!(relative_norm <= 2e-4, "IQ1 relative norm {relative_norm}");
    }
    Ok(
        json!({"max_absolute":max_abs,"rms_absolute":(error2/actual.len() as f64).sqrt(),
        "max_dot_l1_normalized":max_conditioned,"rms_l1_normalized":conditioned_norm,
        "norm_l1_normalized":conditioned_norm,"relative_norm":relative_norm,
        "relative_norm_gate_applied":ref2>1e-6*l12}),
    )
}

#[test]
#[ignore = "slow; CPU FP64 IQ1 oracle and cancellation error bounds"]
fn iq1_packet_matrix_cpu_oracle() -> Result<()> {
    let grid = iq1_codebook()?;
    // Grid index 1 has first value +1, then seven -1 before the delta.
    let mut packed = [0u8; 50];
    packed[..2].copy_from_slice(&half::f16::from_f32(0.375).to_le_bytes());
    packed[2] = 1;
    packed[34..36].copy_from_slice(&0xb000u16.to_le_bytes());
    let values = decode_iq1_fp64(&packed, &grid);
    assert_eq!(
        &values[..8],
        &[
            2.296875, -2.953125, -2.953125, -2.953125, -2.953125, -2.953125, -2.953125, -2.953125
        ]
    );
    assert_eq!(values[32], -0.328125); // Next subgroup has independent scale/sign.
    let good = bounded_error(&[1.000001, -0.000001], &[1., 0.], &[1., 100.])?;
    assert!(bounded_error(&[1.1, 0.], &[1., 0.], &[1., 100.]).is_err());
    report(
        "iq1-packet-matrix-cpu-oracle",
        &json!({"passed":true,"bounds":good,
        "oracle":"direct FP64 decode using public IQ1 codebook; FP64 dot and dot-L1"}),
    )
}

// Small-shape harness exercises output tails, all indexed sources/destinations,
// and exact redzones without relaxing production PacketOp's native preflight.
struct MatrixTile {
    weights: Tensor,
    output: Tensor,
    n: usize,
    k: usize,
    out: usize,
    indices: Indices,
    matrix: bool,
}
impl CustomOp1 for MatrixTile {
    fn name(&self) -> &'static str {
        "dinf-test-iq1-matrix-tile"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("matrix tile requires Metal")
    }
    fn metal_fwd(
        &self,
        x: &MetalStorage,
        xl: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let wrap = candle_core::Error::wrap;
        let (ws, wl) = self.weights.storage_and_layout();
        let (os, ol) = self.output.storage_and_layout();
        let (Storage::Metal(w), Storage::Metal(o)) = (&*ws, &*os) else {
            candle_core::bail!("Metal tile fixture")
        };
        let wo = grouped_iq1_span(
            wl,
            DType::U8,
            self.out * self.k / 256 * 50,
            w.buffer().length(),
        )
        .map_err(wrap)?;
        let xo = grouped_iq1_span(xl, DType::F32, 6 * self.k, x.buffer().length()).map_err(wrap)?;
        let p = pipeline(x.device(), self.n, self.matrix)?;
        let geometry = [
            self.k as u32,
            self.out as u32,
            (self.k / 256 * 50) as u32,
            0,
        ];
        let mut indices = [Indices::default(); GROUP];
        indices[0] = self.indices;
        let encoder = x.device().command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&p);
        encoder.set_input_buffer(0, Some(x.buffer()), xo);
        // One full untouched row on either side of six canonical rows.
        encoder.set_output_buffer(1, Some(o.buffer()), (ol.start_offset() + self.out) * 4);
        encoder.set_bytes(2, &indices);
        encoder.set_bytes(3, &geometry);
        for i in 0..GROUP {
            encoder.set_input_buffer(4 + i, Some(w.buffer()), wo);
        }
        if self.matrix {
            encoder.set_threadgroup_memory_length(0, MMA_SCRATCH_BYTES);
        }
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.out.div_ceil(if self.matrix { 32 } else { 4 }),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(
                o.buffer().clone().into(),
                x.device().clone(),
                8 * self.out,
                DType::F32,
            ),
            Shape::from((8, self.out)),
        ))
    }
}
#[allow(
    clippy::too_many_arguments,
    reason = "Keeps the kernel's independent spans, indices and dispatch mode explicit."
)]
fn tile_values(
    device: &Device,
    bytes: &Tensor,
    x: &Tensor,
    n: usize,
    k: usize,
    out: usize,
    indices: Indices,
    matrix: bool,
) -> Result<Vec<f32>> {
    let output = Tensor::full(123456.75f32, (8, out), device)?;
    let mut owners = vec![bytes.clone(), x.clone(), output.clone()];
    let mut terminal = Terminal {
        device,
        done: false,
    };
    let result = x.apply_op1_no_bwd(&MatrixTile {
        weights: bytes.clone(),
        output,
        n,
        k,
        out,
        indices,
        matrix,
    })?;
    owners.push(result.clone());
    let values = result.flatten_all()?.to_vec1::<f32>()?;
    terminal.finish()?;
    let mut touched = vec![false; 8 * out];
    let mut selected = Vec::with_capacity(n * out);
    for c in 0..n {
        let start = (1 + indices.destination[c] as usize) * out;
        touched[start..start + out].fill(true);
        selected.extend_from_slice(&values[start..start + out]);
    }
    for (i, &v) in values.iter().enumerate() {
        if !touched[i] {
            assert_eq!(v.to_bits(), 123456.75f32.to_bits(), "redzone {i}");
        }
    }
    Ok(selected)
}

#[test]
#[ignore = "requires a Metal device; F32 MMA vs FP64 oracle and scalar F32, N=2..4 tails/scatter/overflow"]
#[allow(
    clippy::excessive_precision,
    reason = "Exact binary fractions define quantization boundary fixtures."
)]
fn iq1_packet_matrix_metal_tiles() -> Result<()> {
    let d = Device::new_metal(0)?;
    let grid = iq1_codebook()?;
    let mut owners = Vec::<Tensor>::new();
    let mut terminal = Terminal {
        device: &d,
        done: false,
    };
    let indices = Indices {
        source: [5, 1, 4, 0],
        destination: [3, 0, 5, 2],
    };
    let mut records = Vec::new();
    for n in 2..=4 {
        for out in [1, 7, 31, 32, 33, 65] {
            for case in 0..4 {
                let k = if case == 3 { 1024 } else { 768 };
                let mut raw = Vec::with_capacity(out * k / 256 * 50);
                for row in 0..out {
                    for block in 0..k / 256 {
                        let mut b = [0u8; 50];
                        let scale = if case == 3 {
                            1.
                        } else {
                            [0.375, 1.0009765625, 65504., -65504.][(row + block) % 4]
                        };
                        b[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
                        if case != 3 {
                            for (i, v) in b[2..].iter_mut().enumerate() {
                                *v = ((row * 17 + block * 31 + i * 7) % 256) as u8;
                            }
                        }
                        raw.extend_from_slice(&b);
                    }
                }
                let decoded = decode_iq1_fp64(&raw, &grid);
                let mut padded = vec![0xff; 50];
                padded.extend_from_slice(&raw);
                padded.extend_from_slice(&[0xff; 50]);
                let len = padded.len();
                let weights = Tensor::from_vec(padded, len, &d)?.narrow(0, 50, raw.len())?;
                let x = if case == 3 {
                    let mut v = vec![999f32; 3];
                    for _ in 0..6 {
                        for i in 0..k {
                            v.push(if i / 16 < 32 { 8e36 } else { -8e36 });
                        }
                    }
                    let len = v.len();
                    Tensor::from_vec(v, len, &d)?
                        .narrow(0, 3, 6 * k)?
                        .reshape((6, k))?
                } else {
                    fixture_input(&d, 6, k, case)?
                };
                owners.extend([weights.clone(), x.clone()]);
                let cpu_x = x.flatten_all()?.to_vec1::<f32>()?;
                let (oracle, l1) = fp64_dots(&decoded, &cpu_x, k, out, indices, n);
                let scalar = tile_values(&d, &weights, &x, n, k, out, indices, false)?;
                // Strict pipeline call: compile failures do NOT silently qualify scalar.
                let matrix = tile_values(&d, &weights, &x, n, k, out, indices, true)?;
                let matrix_fp64 = bounded_error(&matrix, &oracle, &l1)?;
                let scalar_fp64 = bounded_error(&scalar, &oracle, &l1)?;
                let matrix_scalar = bounded_error(
                    &matrix,
                    &scalar.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                    &l1,
                )?;
                if case == 3 {
                    // Scalar lane sums remain finite while a 64-wide MMA slice overflows.
                    // Before-store tile recompute must reproduce the deterministic scalar result.
                    assert_eq!(
                        matrix.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "overflow guard"
                    );
                }
                records.push(
                    json!({"n":n,"out":out,"k":k,"case":case,"matrix_fp64":matrix_fp64,
            "scalar_fp64":scalar_fp64,"matrix_scalar":matrix_scalar,"exact_redzones":true}),
                );
                owners.clear();
            }
        }
    }
    terminal.finish()?;
    report(
        "iq1-packet-matrix-tiles",
        &json!({"passed":true,"scratch_bytes":MMA_SCRATCH_BYTES,"records":records,
        "input_offset_bytes":12,"weight_offset_bytes":50,"output_offset":"one full row","guard_case":3}),
    )
}

#[test]
#[ignore = "requires a Metal device; malformed matrix wrapper must reject before encoding"]
fn iq1_packet_matrix_metal_preflight_rejections() -> Result<()> {
    let scope = MatrixTestScope::new(true);
    packet_preflight_rejections("iq1-packet-matrix-guards")?;
    let counts = scope.dispatch_counts();
    anyhow::ensure!(
        counts[0] > 0 && counts[2] == 0,
        "candidate not exercised: {counts:?}"
    );
    Ok(())
}

#[test]
#[ignore = "requires a Metal device; packet matrix vs FP64/scalar indexed and capability fallback"]
fn iq1_packet_matrix_metal_native() -> Result<()> {
    let d = Device::new_metal(0)?;
    let grid = iq1_codebook()?;
    let mut owners = Vec::<Tensor>::new();
    let mut terminal = Terminal {
        device: &d,
        done: false,
    };
    let mut records = Vec::new();
    // T9 splits repeated experts into P4/P4/P1; gate/up crosses a 12-weight
    // binding boundary. Distinct experts intentionally alias the same lease.
    for (case, t) in [2, 3, 4, 9].into_iter().enumerate() {
        let routes = vec![[0, 1, 2, 3, 4, 5]; t];
        let plan = Plan::new(&routes)?;
        for phase in [Phase::GateUp, Phase::Down] {
            let (input_rows, k, output_rows, out) = phase.geometry(t);
            let matrix = fixture_weights(&d, out, k, case)?;
            let decoded = decode_iq1_fp64(&matrix.weights.to_bytes()?, &grid);
            owners.push(matrix.weights.bytes().clone());
            let x = fixture_input(&d, input_rows, k, case)?;
            owners.push(x.clone());
            let host_x = x.flatten_all()?.to_vec1::<f32>()?;
            let scalar = {
                let scope = MatrixTestScope::new(false);
                let y = forward(&x, &plan, phase, |_, _| Ok(&matrix))?;
                assert_eq!(scope.dispatch_counts()[0], 0);
                y
            };
            owners.push(scalar.clone());
            let (actual, counts) = {
                let scope = MatrixTestScope::new(true);
                let y = forward(&x, &plan, phase, |_, _| Ok(&matrix))?;
                (y, scope.dispatch_counts())
            };
            owners.push(actual.clone());
            anyhow::ensure!(
                counts[0] > 0 && counts[2] == 0,
                "matrix not selected: {counts:?}"
            );
            let actual = actual.flatten_all()?.to_vec1::<f32>()?;
            let scalar = scalar.flatten_all()?.to_vec1::<f32>()?;
            let mut oracle = vec![0.; output_rows * out];
            let mut l1 = vec![0.; output_rows * out];
            for (token, route) in routes.iter().enumerate() {
                for (slot, _) in route.iter().enumerate() {
                    for &kind in phase.kinds() {
                        let (src, dst) = phase.rows(Occurrence { token, slot }, kind);
                        let indices = Indices {
                            source: [src as u32, 0, 0, 0],
                            destination: [0; 4],
                        };
                        let (dots, mag) = fp64_dots(&decoded, &host_x, k, out, indices, 1);
                        oracle[dst * out..][..out].copy_from_slice(&dots);
                        l1[dst * out..][..out].copy_from_slice(&mag);
                    }
                }
            }
            let matrix_fp64 = bounded_error(&actual, &oracle, &l1)?;
            let scalar_fp64 = bounded_error(&scalar, &oracle, &l1)?;
            let matrix_scalar = bounded_error(
                &actual,
                &scalar.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                &l1,
            )?;
            // A forced unsupported device takes the original kernel before any
            // enqueue, and counters must not label it matrix-selected.
            let fallback_counts = if t == 2 {
                let scope = MatrixTestScope::unsupported();
                let y = forward(&x, &plan, phase, |_, _| Ok(&matrix))?;
                owners.push(y.clone());
                assert_eq!(
                    finite_bits(&y)?,
                    scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                let counts = scope.dispatch_counts();
                assert_eq!(counts[0], 0);
                assert!(counts[1] > 0 && counts[2] == counts[1]);
                Some(counts)
            } else {
                None
            };
            records.push(
                json!({"t":t,"phase":format!("{phase:?}"),"case":case,"plan":plan.diagnostics(),
                "dispatch_counts":counts,"forced_unsupported_counts":fallback_counts,
                "matrix_fp64":matrix_fp64,"scalar_fp64":scalar_fp64,"matrix_scalar":matrix_scalar}),
            );
            owners.clear();
        }
    }
    terminal.finish()?;
    report(
        "iq1-packet-matrix-native",
        &json!({"passed":true,"records":records,
        "canonical_every_destination":true,"nonzero_offsets":true,"aliased_leases":true,
        "bounds":{"max_dot_l1_normalized":2e-5,"rms_l1_normalized":5e-6,"norm_l1_normalized":5e-6,
            "relative_norm_if_reference_l2_exceeds_1e_minus3_times_dot_l1_l2":2e-4},
        "scope":"component only"}),
    )
}
