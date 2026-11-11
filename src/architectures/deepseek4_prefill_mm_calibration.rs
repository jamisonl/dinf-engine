//! Numerical calibration against the pinned upstream portable mul_mm kernel.
//! The pinned public sources ship under fixtures/llama-prefill-oracle.
//! No inference backend or model is loaded. Full upstream MIT notice is retained
//! in deepseek4_prefill_mm.rs/.metal and deepseek41/NOTICE.
use super::*;
fn pinned_sources() -> Result<(&'static str, &'static str)> {
    use sha2::{Digest, Sha256};
    let core = include_str!("../../fixtures/llama-prefill-oracle/mul_mm.metal");
    let dequant = include_str!("../../fixtures/llama-prefill-oracle/dequantize.h");
    for (name, source, expected) in [
        (
            "mul_mm.metal",
            core,
            "c3653dbd858e6d68e2db73aa4e37fe914f389fe346f36e1c3b19e9be4bcf307f",
        ),
        (
            "dequantize.h",
            dequant,
            "b64da60146cad11863bae09873e542dce76897c787849d47abaa2a5837d5bf8f",
        ),
    ] {
        anyhow::ensure!(
            format!("{:x}", Sha256::digest(source.as_bytes())) == expected,
            "pinned upstream calibration source changed: {name}"
        );
    }
    Ok((core, dequant))
}
#[test]
fn bundled_upstream_calibration_sources_match_pin() -> Result<()> {
    pinned_sources()?;
    Ok(())
}
pub(super) fn upstream_mm(
    weights: &Tensor,
    input: &Tensor,
    k: usize,
    m: usize,
    iq1: bool,
) -> Result<Tensor> {
    let (core, dequant) = pinned_sources()?;
    let body = core
        .lines()
        .skip(149)
        .take(214)
        .collect::<Vec<_>>()
        .join("\n");
    let q8 = dequant
        .lines()
        .skip(324)
        .take(13)
        .collect::<Vec<_>>()
        .join("\n")
        .replace("dequantize_q8_0", "dinf_pinned_q8");
    let iq = dequant
        .lines()
        .skip(625)
        .take(20)
        .collect::<Vec<_>>()
        .join("\n")
        .replace("dequantize_iq1_s", "dinf_pinned_iq1");
    let block = if iq1 { "block_iq1_s" } else { "block_q8_0" };
    let nl = if iq1 { 16 } else { 2 };
    let deq = if iq1 {
        "dinf_pinned_iq1"
    } else {
        "dinf_pinned_q8"
    };
    let source = format!(
        "{}\n#define FOR_UNROLL(x) for(x)\n\
        struct ggml_metal_kargs_mul_mm {{ulong ne00,ne0,ne1,nb01,nb02,nb03,nb10,nb11,nb12,nb13;}};\n\
        constant bool FC_mul_mm_bc_inp=true,FC_mul_mm_bc_out=true;\n\
        constant short FC_mul_mm_ne12=1,FC_mul_mm_ne13=1,FC_mul_mm_r2=1,FC_mul_mm_r3=1;\n\
        {q8}\n{iq}\n{body}\n\
        typedef decltype(kernel_mul_mm<half,half4x4,simdgroup_half8x8,half,half2x4,simdgroup_half8x8,{block},{nl},{deq},float,float4x4,float,float2x4>) pinned_fn;\n\
        template [[host_name(\"dinf_pinned_reference\")]] kernel pinned_fn kernel_mul_mm<half,half4x4,simdgroup_half8x8,half,half2x4,simdgroup_half8x8,{block},{nl},{deq},float,float4x4,float,float2x4>;",
        super::super::super::metal::iq1_reuse_base_msl()
    );
    let d = input.device().as_metal_device()?;
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version4_0);
    options.setMathMode(MTLMathMode::Safe);
    options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
    let library = d
        .metal_device()
        .new_library_with_source(&source, Some(&options))
        .map_err(|e| anyhow::anyhow!("upstream calibration compile: {e}"))?;
    let function = library
        .get_function("dinf_pinned_reference", None)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let pipeline = d
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let n = input.dims()[0];
    let row_bytes = if iq1 { k / 256 * 50 } else { k / 32 * 34 };
    let w = operand(weights, d, DType::U8, m * row_bytes)?;
    let x = operand(input, d, DType::F32, n * k)?;
    let output = private(d, (n, m), DType::F32)?;
    let y = operand(&output, d, DType::F32, n * m)?;
    let args = [
        k as u64,
        m as u64,
        n as u64,
        row_bytes as u64,
        0,
        0,
        4,
        (k * 4) as u64,
        0,
        0,
    ];
    let e = d.command_encoder()?;
    let e = e.as_ref();
    e.set_compute_pipeline_state(&pipeline);
    e.set_bytes(0, &args);
    e.set_input_buffer(1, Some(&w.0), w.1);
    e.set_input_buffer(2, Some(&x.0), x.1);
    e.set_output_buffer(3, Some(&y.0), y.1);
    e.set_threadgroup_memory_length(0, 8192);
    e.dispatch_thread_groups(
        MTLSize {
            width: n.div_ceil(32),
            height: m.div_ceil(64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(output)
}
