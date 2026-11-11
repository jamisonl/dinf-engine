//! Component tests for the native prefill matmul kernels. They run real Metal
//! kernels against synthetic fixtures and FP64 oracles; no model is loaded.
use super::*;
use serde_json::json;
#[path = "deepseek4_prefill_mm_calibration.rs"]
mod calibration;
fn errors(actual: &[f32], oracle: &[f64]) -> Result<serde_json::Value> {
    let mut se = 0f64;
    let mut energy = 0f64;
    let mut maximum = 0f64;
    for (&a, &b) in actual.iter().zip(oracle) {
        anyhow::ensure!(
            a.is_finite() && b.is_finite(),
            "nonfinite finite-domain result"
        );
        let e = a as f64 - b;
        se += e * e;
        energy += b * b;
        maximum = maximum.max(e.abs());
    }
    let nmse = se / energy;
    let relative_l2 = nmse.sqrt();
    let normalized_max = maximum / (energy / oracle.len() as f64).sqrt();
    eprintln!(
        "half MM NMSE={nmse:e} relative_L2={relative_l2:e} max_abs={maximum:e} normalized_max={normalized_max:e}"
    );
    // Fixed before execution: half operand rounding, F32 accumulation. No
    // per-element relative denominator at cancellation zeros or output clamps.
    anyhow::ensure!(
        nmse <= 2e-6 && normalized_max <= 0.015,
        "half MM precision gate"
    );
    Ok(
        json!({"nmse":nmse,"relative_l2":relative_l2,"max_abs":maximum,"normalized_max":normalized_max,"nmse_limit":2e-6,"normalized_max_limit":0.015}),
    )
}
fn fixture(device: &Device, out: usize, k: usize, slot: usize) -> Result<DeepSeek4QMatMul> {
    let prefix = 50 * (slot + 1);
    let mut bytes = vec![0xff; prefix];
    for r in 0..out {
        for b in 0..k / 256 {
            bytes.extend_from_slice(
                &half::f16::from_f32(0.00390625 * (1 + (r + b + slot) % 7) as f32).to_le_bytes(),
            );
            for j in 0..48 {
                bytes.push(((r * 17 + b * 31 + j * 7 + slot * 53) % 256) as u8);
            }
        }
    }
    let len = bytes.len();
    let view = Tensor::from_vec(bytes, len, device)?.narrow(0, prefix, out * (k / 256 * 50))?;
    DeepSeek4QMatMul::new(Arc::new(RawTensor::new(
        vec![out, k],
        DeepSeek4DType::IQ1S,
        view,
    )?))
}
#[test]
#[ignore = "requires a Metal device; synthetic fixtures, no model"]
fn native_llama_prefill_components() -> Result<()> {
    let d = Device::new_metal(0)?;
    let selected = pipelines(d.as_metal_device()?)?;
    let tensor_supported = d
        .as_metal_device()?
        .metal_device()
        .as_ref()
        .supportsFamily(MTLGPUFamily::Apple10);
    if tensor_supported && !PORTABLE_TEST.with(|v| v.get()) {
        // A tensor compiler failure must be visible to this gate, not masked by
        // the production-safe portable fallback.
        pipeline_variant(d.as_metal_device()?, true)?;
        assert!(selected.tensor);
    }
    let mut dense_results = Vec::new();
    let mut upstream_calibration = Vec::new();
    for (n, m, k) in [(32, 64, 256), (33, 65, 288), (2, 7, 32)] {
        let mut packed = vec![0u8; 2];
        let mut decoded = Vec::new();
        for r in 0..m {
            for b in 0..k / 32 {
                let scale = half::f16::from_f32((1 + (r + b) % 7) as f32 / 128.);
                packed.extend_from_slice(&scale.to_le_bytes());
                for j in 0..32 {
                    let q = ((r * 17 + b * 31 + j * 7) % 255) as i32 - 127;
                    packed.push(q as i8 as u8);
                    decoded.push(q as f64 * scale.to_f32() as f64);
                }
            }
        }
        let len = packed.len();
        let w = Tensor::from_vec(packed, len, &d)?.narrow(0, 2, m * k / 32 * 34)?;
        let input: Vec<f32> = (0..n * k)
            .map(|i| ((i * 37 % 257) as f32 - 128.) / 137.)
            .collect();
        let mut offset = vec![99f32; 3];
        offset.extend_from_slice(&input);
        let len = offset.len();
        let x = Tensor::from_vec(offset, len, &d)?
            .narrow(0, 3, n * k)?
            .reshape((n, k))?;
        let actual = dense(&w, &x, k, m)?.flatten_all()?.to_vec1::<f32>()?;
        let oracle = (0..n * m)
            .map(|i| {
                (0..k)
                    .map(|j| input[i / m * k + j] as f64 * decoded[i % m * k + j])
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        dense_results.push(json!({"shape":[n,m,k],"errors":errors(&actual,&oracle)?}));
        let pinned = calibration::upstream_mm(&w, &x, k, m, false)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        upstream_calibration
            .push(json!({"type":"Q8","shape":[n,m,k],"errors":errors(&pinned,&oracle)?}));
        assert!(dense(&w, &x.t()?, k, m).is_err());
    }
    let md = d.as_metal_device()?;
    let tail_weight = fixture(&d, 65, 256, 0)?;
    let mut tail_input = vec![99f32; 3];
    tail_input.extend((0..33 * 256).map(|i| ((i * 37 % 257) as f32 - 128.) / 137.));
    let len = tail_input.len();
    let tail_input = Tensor::from_vec(tail_input, len, &d)?
        .narrow(0, 3, 33 * 256)?
        .reshape((33, 256))?;
    let tx = operand(&tail_input, md, DType::F32, 33 * 256)?;
    let tw = operand(tail_weight.weights.bytes(), md, DType::U8, 65 * 50)?;
    let tail_output = private(md, (33, 65), DType::F32)?;
    let ty = operand(&tail_output, md, DType::F32, 33 * 65)?;
    {
        let e = md.command_encoder()?;
        e.as_ref().set_output_buffer(2, Some(&ty.0), ty.1);
        for slot in 6..17 {
            e.as_ref().set_input_buffer(slot, Some(&tw.0), tw.1);
        }
        encode(
            e.as_ref(),
            &selected.iq1,
            &tw,
            &tx,
            &tw,
            Args {
                k: 256,
                m: 65,
                n: 33,
                row_bytes: 50,
                mode: 0,
                expert: 0,
                kind: 0,
                stride: 0,
            },
            33,
        );
    }
    let tail_actual = tail_output.flatten_all()?.to_vec1::<f32>()?;
    let tail_old = tail_weight
        .forward(&tail_input)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    let iq1_channel_tail = errors(&tail_actual, &tail_old)?;
    let pinned = calibration::upstream_mm(tail_weight.weights.bytes(), &tail_input, 256, 65, true)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    upstream_calibration
        .push(json!({"type":"IQ1","shape":[33,65,256],"errors":errors(&pinned,&tail_old)?}));
    let prior_counts = counts();
    assert!(operand(&tail_input, md, DType::F32, 33 * 256 + 1).is_err());
    assert!(
        operand(
            &Tensor::zeros((33, 256), DType::F32, &Device::Cpu)?,
            md,
            DType::F32,
            33 * 256
        )
        .is_err()
    );
    let other = Device::new_metal(0)?;
    assert!(operand(&tail_input, other.as_metal_device()?, DType::F32, 33 * 256).is_err());
    let bad = Tensor::zeros(51, DType::U8, &d)?.narrow(0, 1, 50)?;
    assert!(operand(&bad, md, DType::U8, 50).is_err());
    assert_eq!(counts(), prior_counts);
    // Same input/bias for CPU oracle and GPU sorting, including exact ties,
    // underflow, extreme finite logits and negative biased scores.
    let t = 33;
    let bias: Vec<f32> = (0..384).map(|i| if i < 6 { 10. } else { -10. }).collect();
    let values: Vec<f32> = (0..t * 384)
        .map(|i| match i / 384 {
            0 => 0.,
            1 => -100.,
            2 => 1e30,
            _ => ((i * 17 % 101) as f32 - 50.) / 13.,
        })
        .collect();
    let logits = Tensor::from_vec(values.clone(), (t, 384), &d)?;
    let gpu = route(&logits, &bias)?;
    let records = gpu.compact.flatten_all()?.to_vec1::<u32>()?;
    let mut route_weight_max = 0f32;
    let mut routes = Vec::new();
    for r in 0..t {
        let old = crate::architectures::deepseek41::math::router_top6(
            &values[r * 384..(r + 1) * 384],
            Some(&bias),
        )?;
        let row = &records[r * 13..(r + 1) * 13];
        assert_eq!(row[12], 0);
        for s in 0..6 {
            assert_eq!(row[s] as usize, old.indices[s]);
            route_weight_max =
                route_weight_max.max((f32::from_bits(row[6 + s]) - old.weights[s]).abs());
        }
        routes.push(old.indices);
    }
    assert!(route_weight_max <= 2e-6);
    let map = gpu.map.flatten_all()?.to_vec1::<u32>()?;
    for expert in 0..384 {
        let expected: Vec<u32> = routes
            .iter()
            .enumerate()
            .flat_map(|(r, ids)| {
                ids.iter()
                    .enumerate()
                    .filter_map(move |(s, &id)| (id == expert).then_some((6 * r + s) as u32))
            })
            .collect();
        let base = expert * (1 + 6 * t);
        assert_eq!(map[base] as usize, expected.len());
        assert_eq!(
            &map[base + 1..base + 1 + expected.len()],
            expected.as_slice()
        );
    }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut v = vec![0f32; 2 * 384];
        v[17] = bad;
        let r = route(&Tensor::from_vec(v, (2, 384), &d)?, &bias)?;
        assert_eq!(r.compact.to_vec2::<u32>()?[0][12], 1);
        assert!(r.validate().is_err());
    }
    let mut bad_bias = bias.clone();
    bad_bias[83] = f32::INFINITY;
    let invalid = route(&logits, &bad_bias)?;
    assert_eq!(invalid.compact.to_vec2::<u32>()?[0][12], 1);
    assert!(invalid.validate().is_err());
    // 18 distinct experts, three distinct banks each: counts33 (full32+tail1),
    // 31 and1. Both gate/up36 and down18 bindings cross the12-buffer boundary.
    let t = 65;
    let values = (0..t * 384)
        .map(|i| {
            let row = i / 384;
            let base = if row < 33 {
                0
            } else if row < 64 {
                6
            } else {
                12
            };
            if (base..base + 6).contains(&(i % 384)) {
                10.
            } else {
                -10.
            }
        })
        .collect::<Vec<f32>>();
    let gpu = route(&Tensor::from_vec(values, (t, 384), &d)?, &vec![0.; 384])?;
    let records = gpu.compact.flatten_all()?.to_vec1::<u32>()?;
    for (column, value) in [
        (0, 384),
        (1, records[0]),
        (6, f32::NAN.to_bits()),
        (12, u32::MAX),
    ] {
        let mut invalid = records.clone();
        invalid[column] = value;
        let invalid = Routes {
            compact: Tensor::from_vec(invalid, (t, 13), &d)?,
            map: gpu.map.clone(),
        };
        assert!(invalid.validate().is_err());
    }
    let validated = gpu.validate()?;
    assert_eq!(
        &validated.counts[..18],
        &[
            33, 33, 33, 33, 33, 33, 31, 31, 31, 31, 31, 31, 1, 1, 1, 1, 1, 1
        ]
    );
    let routes = validated
        .selections()
        .iter()
        .map(|r| r.indices)
        .collect::<Vec<_>>();
    let weights = (0..18 * 3)
        .map(|slot| {
            fixture(
                &d,
                if slot % 3 == 2 { 5120 } else { 2304 },
                if slot % 3 == 2 { 2304 } else { 5120 },
                slot,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let select = |expert: usize, kind: usize| -> Result<&DeepSeek4QMatMul> {
        Ok(&weights[3 * expert + kind])
    };
    let mut indexed_results = Vec::new();
    for down in [false, true] {
        let (rows, k) = if down { (6 * t, 2304) } else { (t, 5120) };
        let input: Vec<f32> = (0..rows * k)
            .map(|i| ((i * 37 % 257) as f32 - 128.) / 137.)
            .collect();
        let mut offset = vec![999f32; 3];
        offset.extend_from_slice(&input);
        let len = offset.len();
        let x = Tensor::from_vec(offset, len, &d)?
            .narrow(0, 3, rows * k)?
            .reshape((rows, k))?;
        let before = counts()[0];
        assert!(
            indexed(&x, &validated, down, |expert, kind| {
                if expert == 17 {
                    Ok(&tail_weight)
                } else {
                    select(expert, kind)
                }
            })
            .is_err()
        );
        assert_eq!(
            counts()[0],
            before,
            "late bank rejection must precede all enqueue"
        );
        let actual = indexed(&x, &validated, down, select)?;
        assert_eq!(counts()[0] - before, if down { 2 } else { 3 });
        let plan = super::super::iq1_packets::Plan::new(&routes)?;
        let old = super::super::iq1_packets::forward(
            &x,
            &plan,
            if down {
                super::super::iq1_packets::Phase::Down
            } else {
                super::super::iq1_packets::Phase::GateUp
            },
            select,
        )?;
        let a = actual.flatten_all()?.to_vec1::<f32>()?;
        let o = old
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>();
        indexed_results.push(json!({"down":down,"errors":errors(&a,&o)?,"distinct_experts":18,"distinct_banks":54,"counts":[33,31,1],"dispatches":counts()[0]-before}));
        if down {
            let c = combine(&actual, &gpu.compact)?.to_vec2::<f32>()?;
            for row in [0, 32, 33, 63, 64] {
                for col in [0, 17, 5119] {
                    let mut expected = 0f32;
                    for s in 0..6 {
                        let weighted = a[(6 * row + s) * 5120 + col]
                            .mul_add(f32::from_bits(records[13 * row + 6 + s]), 0.);
                        expected += weighted;
                    }
                    assert_eq!(c[row][col].to_bits(), expected.to_bits());
                }
            }
        }
    }
    // Conversion policy is explicit: finite F32 beyond F16 range produces a
    // nonfinite result, never saturation/recompute; NaN/Inf are not hidden.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&half::f16::ONE.to_le_bytes());
    bytes.extend_from_slice(&[1u8; 32]);
    let w = Tensor::from_vec(bytes, 34, &d)?;
    for value in [70000f32, f32::INFINITY, f32::NAN] {
        let x = Tensor::full(value, (2, 32), &d)?;
        assert!(
            dense(&w, &x, 32, 1)?
                .to_vec2::<f32>()?
                .iter()
                .flatten()
                .all(|v| !v.is_finite())
        );
    }
    let mut overflow = half::f16::MAX.to_le_bytes().to_vec();
    overflow.extend_from_slice(&[127u8; 32]);
    let overflow = Tensor::from_vec(overflow, 34, &d)?;
    assert!(
        dense(&overflow, &Tensor::ones((2, 32), DType::F32, &d)?, 32, 1)?
            .to_vec2::<f32>()?
            .iter()
            .flatten()
            .all(|v| !v.is_finite())
    );
    let before = counts();
    {
        let _request = TestScope::new(Some(true));
        let _scope = Scope::enter(&d, true);
        assert!(active());
    }
    assert!(!active());
    {
        let _old = TestScope::new(Some(false));
        let _scope = Scope::enter(&d, true);
        assert!(!active());
    }
    {
        let _request = TestScope::new(Some(true));
        let _scope = Scope::enter(&Device::Cpu, true);
        assert!(!active());
    }
    assert_eq!(requested(), QUALIFIED_M5_PREFILL_DEFAULT);
    {
        let _default = TestScope::new(None);
        let _scope = Scope::enter(&d, true);
        assert!(active());
    }
    {
        let _request = TestScope::new(Some(true));
        let _scope = Scope::enter(&d, false);
        assert!(!active());
    }
    let policy = crate::architectures::deepseek41::native_prefill_policy_checks(&d)?;
    let admission = {
        let _request = TestScope::new(Some(true));
        let _scope = Scope::enter(&d, true);
        let plans=[0,64852,1_048_448].into_iter().map(|start| {
            let p=crate::architectures::deepseek41::profile::prefill_tile_plan(128,start,128)?;
            assert_eq!(p.requested,128);assert_eq!(p.end,start+128);
            assert!(p.scratch_bound_bytes<=crate::architectures::deepseek41::FIXED_CACHE_BYTES);
            Ok(json!({"start":start,"requested":p.requested,"effective":p.effective,"fixed_bound":p.scratch_bound_bytes}))
        }).collect::<Result<Vec<_>>>()?;
        json!({"plans":plans,"additional_candidate_carrier_bytes_at128":carrier_bytes(128),
            "production_default_requested":crate::architectures::deepseek41::profile::DEFAULT_PREFILL_TILE})
    };
    d.synchronize()?;
    let report = json!({"passed":true,"geometry":"64x32 K32 half/F32","tensor_selected":selected.tensor,"tensor_device_supported":tensor_supported,"dense":dense_results,"indexed":indexed_results,"iq1_33x65_channel_tail":iq1_channel_tail,"admission":admission,"negative_guards":["shape","span","noncontiguous","unaligned","CPU storage","different device ID","unsupported whole-policy fallback"],"same_input_router_max_weight_error":route_weight_max,"dispatch_counts_iq1_dense_route":before,"nonfinite_policy":"GPU error flag for router; IEEE overflow/NaN propagation for half operands; no clamp","precision_bounds":"fixed before execution NMSE2e-6/normalized_max0.015; calibrated against actual pinned portable mul_mm body and dequantizers on same fixtures, Q8 FP64 and IQ1 old F32 oracles","upstream_calibration":upstream_calibration,"model_loaded":false});
    let mut report = report;
    report["whole_candidate_policy"] = policy;
    report["provenance_negative_guards"] = json!([
        "ID384",
        "duplicateID",
        "nonfiniteweight",
        "signedstatus-1",
        "latebankshape-before-any-enqueue"
    ]);
    eprintln!("{report}");
    Ok(())
}

#[test]
#[ignore = "requires a Metal device; forces the portable SIMD-group pipeline"]
fn native_llama_prefill_components_portable() -> Result<()> {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            PORTABLE_TEST.with(|v| v.set(self.0));
        }
    }
    let _restore = Restore(PORTABLE_TEST.with(|v| v.replace(true)));
    native_llama_prefill_components()
}
