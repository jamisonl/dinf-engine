//! Token-at-a-time Maya-XS V4.1 math, not model dispatch or a V4 alias.
//!
//! All tensors are contiguous F32 carriers (including BF16-rounded results).
//! CPU execution is a scalar oracle; Metal execution uses native CustomOps,
//! never a CPU roundtrip. Cache quantization returns dequantized F32, so callers
//! must charge four bytes per cached element, not the simulated FP8/FP4 size.
//! Sources: audreyt/ds4 commit 0dff39f91721bcf3472813c6d2bebf9353df2a1a,
//! metal/dsv41.metal, metal/dsv4_kv.metal, metal/dsv4_hc.metal,
//! metal/norm.metal:47-81 and ds4_metal.m:5643-5649,6927,6954,48914-48938.
//! RMS follows float4/two-stage reductions; plain HC flatnorm stays F32
//! (dsv4_hc.metal:1231-1250). HC uses stable tanh gates, Engram exp gates.
//! See adjacent NOTICE and LICENSE-MIT.

use candle_core::{CpuStorage, CustomOp3, DType, Layout, Result, Shape, Tensor};

/// Pinned Maya-XS RMS epsilon, including the HC projection's plain RMS.
pub const RMS_EPS: f32 = 1.0e-20;
const HC_EPS: f32 = 1.0e-6;

// ds4_metal.m:5643-5649 clamps the power-of-two candidate back to width/4:
// width 1280 therefore uses 320 threads, not 512. Tiny/nonmultiple fixtures
// retain a full SIMD group and a masked scalar tail.
fn rms_threads(width: usize) -> usize {
    (width / 4).clamp(32, 1024).div_ceil(32) * 32
}

fn sum_simd(mut lanes: [f32; 32]) -> f32 {
    for stride in [16, 8, 4, 2, 1] {
        for lane in 0..stride {
            lanes[lane] += lanes[lane + stride];
        }
    }
    lanes[0]
}

// CPU oracle for norm.metal:47-66's float4/threadgroup layout. Hardware
// dot/sqrt may differ at an F32 ulp; the source uses 1.0f / sqrt(mean + eps).
fn rms_sum(row: &[f32]) -> f32 {
    let threads = rms_threads(row.len());
    let mut partial = [0.0f32; 1024];
    for (thread, sum) in partial[..threads].iter_mut().enumerate() {
        for chunk in (thread..row.len() / 4).step_by(threads) {
            let x = &row[4 * chunk..4 * chunk + 4];
            *sum += ((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2]) + x[3] * x[3];
        }
        if thread < row.len() % 4 {
            let x = row[row.len() / 4 * 4 + thread];
            *sum += x * x;
        }
    }
    let mut groups = [0.0f32; 32];
    for (i, group) in partial[..threads].chunks_exact(32).enumerate() {
        groups[i] = sum_simd(group.try_into().unwrap());
    }
    sum_simd(groups)
}

/// Nearest-even BF16 rounding with an F32 carrier, including signed zero.
/// Port of dsv41_bf16; nonfinite exponents are not incremented.
pub fn bf16_scalar(x: f32) -> f32 {
    let mut bits = x.to_bits();
    if bits & 0x7f80_0000 != 0x7f80_0000 {
        bits = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    }
    f32::from_bits(bits & 0xffff_0000)
}

/// Quantize then dequantize the *whole* row, including its rotary tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CacheFormat {
    /// 32-value E4M3FN blocks with power-of-two E8M0 scales.
    RawKvFp8E8M0 = 1,
    /// 32-value E2M1 blocks with power-of-two E8M0 scales.
    IndexQkFp4E8M0 = 2,
    /// 16-value E2M1 blocks with nearest-even E4M3FN scales.
    CompressedKvFp4E4M3 = 3,
}

impl CacheFormat {
    #[cfg(test)]
    fn block(self) -> usize {
        if self == Self::CompressedKvFp4E4M3 {
            16
        } else {
            32
        }
    }
}

fn checked(x: &Tensor) -> Result<()> {
    if x.dtype() != DType::F32 || !x.is_contiguous() || x.elem_count() == 0 {
        candle_core::bail!("V4.1 math requires nonempty contiguous F32 tensors");
    }
    if x.elem_count() > u32::MAX as usize {
        candle_core::bail!("V4.1 math tensor exceeds u32 indexing");
    }
    if !x.device().is_cpu() && !x.device().is_metal() {
        candle_core::bail!("V4.1 math supports CPU oracle or native Metal only");
    }
    Ok(())
}

fn same_device(a: &Tensor, b: &Tensor) -> Result<()> {
    checked(b)?;
    if !a.device().same_device(b.device()) {
        candle_core::bail!("V4.1 math operands must share a device");
    }
    Ok(())
}

fn rows_width(x: &Tensor) -> Result<(usize, usize)> {
    checked(x)?;
    let width = *x
        .dims()
        .last()
        .ok_or_else(|| candle_core::Error::Msg("V4.1 math requires a row dimension".into()))?;
    Ok((x.elem_count() / width, width))
}

fn shape(x: &Tensor, dims: &[usize]) -> Result<()> {
    checked(x)?;
    if x.dims() != dims {
        candle_core::bail!("V4.1 math expected shape {dims:?}, got {:?}", x.dims());
    }
    Ok(())
}

/// BF16 nearest-even, retaining shape and F32 dtype; never converts through F16.
pub fn bf16(x: &Tensor) -> Result<Tensor> {
    checked(x)?;
    Op::new(Kind::Bf16, x.elem_count(), 1, x.shape().clone()).apply(x, x, x)
}

/// Weighted RMS over the last dimension, then BF16 rounding. Weight is `[width]`.
/// Epsilon must be positive and finite; Maya-XS uses RMS_EPS, not V4's epsilon.
pub fn rms_norm(x: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor> {
    let (rows, width) = rows_width(x)?;
    shape(weight, &[width])?;
    same_device(x, weight)?;
    validate_epsilon(epsilon)?;
    let mut op = Op::new(Kind::Rms, width, rows, x.shape().clone());
    op.args.eps = epsilon;
    op.args.mode = 1;
    op.apply(x, weight, x)
}

fn validate_epsilon(epsilon: f32) -> Result<()> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        candle_core::bail!("V4.1 RMS epsilon must be positive and finite");
    }
    Ok(())
}

/// Block-local BF16-input QDQ. Partial final blocks and arbitrary leading row
/// dimensions are supported; a block never crosses a row boundary.
pub fn quantize_dequantize(x: &Tensor, format: CacheFormat) -> Result<Tensor> {
    let (rows, width) = rows_width(x)?;
    let mut op = Op::new(Kind::Quantize, width, rows, x.shape().clone());
    op.args.mode = format as u32;
    op.apply(x, x, x)
}

/// Layer-selected frequency family shared by Q, every KV/index cache, and
/// inverse output heads. Only 32 frequencies are stored, never a position table.
#[derive(Clone, Debug)]
pub struct Rope {
    frequencies: [f32; 32],
}

impl Rope {
    /// Layers 0/1 use theta 10000; layers 2..39 use theta 160000 and YaRN16.
    pub fn for_layer(layer: usize) -> Result<Self> {
        if layer >= 40 {
            candle_core::bail!("Maya-XS RoPE layer must be in 0..40");
        }
        let compressed = layer >= 2;
        let base: f32 = if compressed { 160_000.0 } else { 10_000.0 };
        // Exact powf/reciprocal and double-precision correction bounds from
        // ds4_metal.m:48914-48938. No magnitude scaling is applied in V4.1.
        let correction = |rotations: f64| {
            64.0 * (65_536.0 / (rotations * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * f64::from(base).ln())
        };
        let low = correction(32.0).floor() as f32;
        let high = correction(1.0).ceil() as f32;
        let frequencies = std::array::from_fn(|i| {
            let f = 1.0 / base.powf(i as f32 / 32.0);
            if compressed {
                let smooth = 1.0 - ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                (f / 16.0) * (1.0 - smooth) + f * smooth
            } else {
                f
            }
        });
        Ok(Self { frequencies })
    }

    /// Rotate adjacent pairs in the final 64 columns of [...,128] or [...,512]
    /// at one position, BF16-rounding rotated outputs. Non-tail values are
    /// unchanged unless round_all is true (needed before inverse head RoPE).
    pub fn apply(
        &self,
        x: &Tensor,
        position: u32,
        inverse: bool,
        round_all: bool,
    ) -> Result<Tensor> {
        let (rows, width) = rows_width(x)?;
        if !matches!(width, 128 | 512) || position >= 1_048_576 {
            candle_core::bail!("Maya-XS RoPE requires width 128/512 and position < 1048576");
        }
        let mut op = Op::new(Kind::Rope, width, rows, x.shape().clone());
        op.args.position = position;
        op.args.inverse = u32::from(inverse);
        op.args.round_all = u32::from(round_all);
        op.args.frequencies = self.frequencies;
        op.apply(x, x, x)
    }

    /// `apply` for a tile: row `r` of `x` uses position
    /// `first + (r / rows_per_position) * step`, with the same bits a
    /// single-position call gives for that row.
    pub fn apply_rows(
        &self,
        x: &Tensor,
        first: u32,
        rows_per_position: usize,
        step: u32,
        inverse: bool,
        round_all: bool,
    ) -> Result<Tensor> {
        let (rows, width) = rows_width(x)?;
        let last = (rows.saturating_sub(1) / rows_per_position.max(1)) as u64 * u64::from(step)
            + u64::from(first);
        if !matches!(width, 128 | 512)
            || rows_per_position == 0
            || !rows.is_multiple_of(rows_per_position)
            || last >= 1_048_576
        {
            candle_core::bail!(
                "Maya-XS tile RoPE requires width 128/512, whole positions and positions < 1048576"
            );
        }
        let mut op = Op::new(Kind::Rope, width, rows, x.shape().clone());
        op.args.position = first;
        op.args.inverse = u32::from(inverse);
        op.args.round_all = u32::from(round_all);
        op.args.frequencies = self.frequencies;
        op.args.rows_per_position = rows_per_position as u32;
        op.args.position_step = step;
        op.apply(x, x, x)
    }
}

/// Current sublayer gates. Keep pre for the *next* sublayer; collapse must use
/// previous pre, initialized to `[1,0,0,0]`. comb is `[source stream,destination]`.
#[derive(Clone, Debug)]
pub struct HcSplit {
    pre: Tensor,
    post: Tensor,
    comb: Tensor,
    packed: Tensor,
}

impl HcSplit {
    /// Exact packed owner, including the storage shared by all three views.
    pub(super) fn backing(&self) -> &Tensor {
        &self.packed
    }

    /// Rebuild views after an exact owner copy; no HC calculation or rounding.
    pub(super) fn from_backing(packed: Tensor) -> Result<Self> {
        shape(&packed, &[24])?;
        Ok(Self {
            pre: packed.narrow(0, 0, 4)?,
            post: packed.narrow(0, 4, 4)?,
            comb: packed.narrow(0, 8, 16)?.reshape((4, 4))?,
            packed,
        })
    }

    /// Current pre `[4]`, to retain for the next sublayer's collapse.
    pub fn pre(&self) -> &Tensor {
        &self.pre
    }

    /// Current post gates `[4]`, used by hc_expand.
    pub fn post(&self) -> &Tensor {
        &self.post
    }

    /// Current combination matrix `[source,destination]`, used by hc_expand.
    pub fn comb(&self) -> &Tensor {
        &self.comb
    }
}

/// Current gates from RMSnorm(flat residual,1e-20) @ fc.T, without rounding
/// the plain RMS intermediate. residual `[4,width]`, fc `[24,4*width]`, scale `[3]`,
/// base `[24]`. No learned HC head and no current-pre collapse are performed.
pub fn hc_split(residual: &Tensor, fc: &Tensor, scale: &Tensor, base: &Tensor) -> Result<HcSplit> {
    let (streams, width) = residual.dims2()?;
    checked(residual)?;
    if streams != 4 {
        candle_core::bail!("HC requires four residual streams");
    }
    shape(fc, &[24, 4 * width])?;
    shape(scale, &[3])?;
    shape(base, &[24])?;
    for x in [fc, scale, base] {
        same_device(residual, x)?;
    }
    let flat = residual.reshape((1, 4 * width))?;
    let mut norm = Op::new(Kind::Rms, 4 * width, 1, flat.shape().clone());
    norm.args.eps = RMS_EPS;
    let normalized = norm.apply(&flat, &flat, &flat)?;
    let mix = normalized.matmul(&fc.t()?)?.reshape(24)?;
    let packed = Op::new(Kind::HcSplit, 24, 1, Shape::from(vec![24])).apply(&mix, scale, base)?;
    Ok(HcSplit {
        pre: packed.narrow(0, 0, 4)?,
        post: packed.narrow(0, 4, 4)?,
        comb: packed.narrow(0, 8, 16)?.reshape((4, 4))?,
        packed,
    })
}

/// Collapse `[4,width]` using PREVIOUS pre `[4]`, then BF16-round to `[width]`.
pub fn hc_collapse(residual: &Tensor, previous_pre: &Tensor) -> Result<Tensor> {
    let (streams, width) = residual.dims2()?;
    checked(residual)?;
    if streams != 4 {
        candle_core::bail!("HC requires four residual streams");
    }
    shape(previous_pre, &[4])?;
    same_device(residual, previous_pre)?;
    Op::new(Kind::Collapse, width, 1, Shape::from(vec![width])).apply(
        residual,
        previous_pre,
        residual,
    )
}

/// Expand block `[width]` using current post/comb into residual `[4,width]`, BF16
/// rounding only the final sum. The supplied block must already have the
/// sublayer's required BF16 rounding, as in ds41_hc_expand (no block_add path).
pub fn hc_expand(block: &Tensor, residual: &Tensor, current: &HcSplit) -> Result<Tensor> {
    let (streams, width) = residual.dims2()?;
    checked(residual)?;
    if streams != 4 {
        candle_core::bail!("HC requires four residual streams");
    }
    shape(block, &[width])?;
    same_device(residual, block)?;
    same_device(residual, &current.packed)?;
    Op::new(Kind::Expand, width, 1, residual.shape().clone()).apply(
        block,
        residual,
        &current.packed,
    )
}

/// Engram's four independent weighted-RMS gates and shared value add.
/// residual/qnorm/knorm `[4,width]`, projected_kv `[5,width]`. Keys/values and final
/// adds are BF16-rounded; residuals and norm weights are not rounded first.
/// Masked tokens must bypass this call. qnorm*knorm remains on the same device.
pub fn engram_add(
    residual: &Tensor,
    projected_kv: &Tensor,
    qnorm: &Tensor,
    knorm: &Tensor,
    epsilon: f32,
) -> Result<Tensor> {
    let (streams, width) = residual.dims2()?;
    checked(residual)?;
    if streams != 4 {
        candle_core::bail!("Engram gate requires four residual streams");
    }
    shape(projected_kv, &[5, width])?;
    shape(qnorm, &[4, width])?;
    shape(knorm, &[4, width])?;
    for x in [projected_kv, qnorm, knorm] {
        same_device(residual, x)?;
    }
    validate_epsilon(epsilon)?;
    let weight = qnorm.mul(knorm)?;
    let mut op = Op::new(Kind::Engram, width, 4, residual.shape().clone());
    op.args.eps = epsilon;
    op.apply(residual, projected_kv, &weight)
}

/// Tile HC split: residuals `[T,4,width]` to packed splits `[T,24]` (pre,
/// post, comb per row, as `HcSplit::backing`). Each row uses the same RMS and
/// split arithmetic as `hc_split`; the 24-way mix is one fixed-order dot per
/// output, independent of the row count.
pub fn hc_split_rows(
    residual: &Tensor,
    fc: &Tensor,
    scale: &Tensor,
    base: &Tensor,
) -> Result<Tensor> {
    let (rows, streams, width) = residual.dims3()?;
    checked(residual)?;
    if streams != 4 || rows == 0 {
        candle_core::bail!("HC tile requires rows of four residual streams");
    }
    shape(fc, &[24, 4 * width])?;
    shape(scale, &[3])?;
    shape(base, &[24])?;
    for x in [fc, scale, base] {
        same_device(residual, x)?;
    }
    let flat = residual.reshape((rows, 4 * width))?;
    let mut norm = Op::new(Kind::Rms, 4 * width, rows, flat.shape().clone());
    norm.args.eps = RMS_EPS;
    let normalized = norm.apply(&flat, &flat, &flat)?;
    let mix = Op::new(Kind::HcMix, 4 * width, rows, Shape::from((rows, 24))).apply(
        &normalized,
        fc,
        fc,
    )?;
    Op::new(Kind::HcSplit, 24, rows, Shape::from((rows, 24))).apply(&mix, scale, base)
}

/// Tile collapse: residuals `[T,4,width]` with each row's pre gates taken
/// from `pre` at row stride `stride` (4 for `[T,4]`, 24 for packed splits).
pub fn hc_collapse_rows(residual: &Tensor, pre: &Tensor, stride: usize) -> Result<Tensor> {
    let (rows, streams, width) = residual.dims3()?;
    checked(residual)?;
    if streams != 4 || !(stride == 4 || stride == 24) || pre.elem_count() != rows * stride {
        candle_core::bail!("HC tile collapse geometry");
    }
    same_device(residual, pre)?;
    let mut op = Op::new(Kind::Collapse, width, rows, Shape::from((rows, width)));
    op.args.mode = stride as u32;
    op.apply(residual, pre, residual)
}

/// Tile expand: blocks `[T,width]` into residuals `[T,4,width]` with packed
/// splits `[T,24]`, row by row as `hc_expand`.
pub fn hc_expand_rows(block: &Tensor, residual: &Tensor, splits: &Tensor) -> Result<Tensor> {
    let (rows, streams, width) = residual.dims3()?;
    checked(residual)?;
    if streams != 4 {
        candle_core::bail!("HC tile requires four residual streams");
    }
    shape(block, &[rows, width])?;
    shape(splits, &[rows, 24])?;
    same_device(residual, block)?;
    same_device(residual, splits)?;
    Op::new(Kind::Expand, width, rows, residual.shape().clone()).apply(block, residual, splits)
}

/// Tile Engram gate: residuals `[T,4,width]`, projected kv `[T,5,width]`,
/// `weight = qnorm * knorm` `[4,width]`, row by row as `engram_add`.
pub fn engram_add_rows(
    residual: &Tensor,
    projected_kv: &Tensor,
    weight: &Tensor,
    epsilon: f32,
) -> Result<Tensor> {
    let (rows, streams, width) = residual.dims3()?;
    checked(residual)?;
    if streams != 4 {
        candle_core::bail!("Engram tile requires four residual streams");
    }
    shape(projected_kv, &[rows, 5, width])?;
    shape(weight, &[4, width])?;
    same_device(residual, projected_kv)?;
    same_device(residual, weight)?;
    validate_epsilon(epsilon)?;
    let mut op = Op::new(Kind::Engram, width, 4 * rows, residual.shape().clone());
    op.args.eps = epsilon;
    op.apply(residual, projected_kv, weight)
}

/// Six selected expert IDs and unbiased, normalized routing weights.
#[derive(Clone, Debug, PartialEq)]
pub struct RouterSelection {
    /// Descending biased score, with source bitonic-network tie ordering.
    pub indices: [usize; 6],
    /// Unbiased probability / max(sum,2^-14), then multiplied by 1.5.
    pub weights: [f32; 6],
}

/// One already-synchronized logits row (6..=1024 experts). Rejects all nonfinite
/// logits/bias/scores. Source dsv41.metal:47-107 uses strict-comparison bitonic
/// sort, NOT stable index ordering for ties; reproduce that network, including
/// padding-last. Weights use unbiased sqrtsoftplus, not the biased score. The
/// F32 quotient is rounded before multiplying by 1.5. No [-20,20] V4 clamp.
pub fn router_top6(logits: &[f32], bias: Option<&[f32]>) -> Result<RouterSelection> {
    let n = logits.len();
    if !(6..=1024).contains(&n) || bias.is_some_and(|b| b.len() != n) {
        candle_core::bail!("V4.1 router requires 6..=1024 logits and matching bias");
    }
    let mut probabilities = [0.0f32; 1024];
    let mut scores = [0.0f32; 1024];
    for (i, &x) in logits.iter().enumerate() {
        let b = bias.map_or(0.0, |b| b[i]);
        if !x.is_finite() || !b.is_finite() {
            candle_core::bail!("V4.1 router rejects nonfinite logits/bias");
        }
        let sp = if x > 20.0 {
            x
        } else {
            let ex = x.exp();
            if ex < 0.03125 {
                let em = ex;
                em * (1.0 - em * (0.5 - em * (1.0 / 3.0 - 0.25 * em)))
            } else {
                (1.0 + ex).ln()
            }
        };
        let p = sp.sqrt();
        if !(p + b).is_finite() {
            candle_core::bail!("V4.1 router biased score overflow");
        }
        probabilities[i] = p;
        scores[i] = p + b;
    }
    let ntg = n.next_power_of_two();
    let mut indices: [usize; 1024] = std::array::from_fn(|i| i);
    let mut k = 2;
    while k <= ntg {
        let mut j = k / 2;
        while j > 0 {
            for col in 0..ntg {
                let other = col ^ j;
                if other <= col {
                    continue;
                }
                let (a, b) = (indices[col], indices[other]);
                let swap = if col & k == 0 {
                    a >= n || (b < n && scores[a] < scores[b])
                } else {
                    b >= n || (a < n && scores[a] > scores[b])
                };
                if swap {
                    indices.swap(col, other);
                }
            }
            j /= 2;
        }
        k *= 2;
    }
    let indices = std::array::from_fn(|i| indices[i]);
    // SIMD tree reduction with inactive lanes zero. Keep each F32 addition.
    let mut lanes = [0.0f32; 32];
    for i in 0..6 {
        lanes[i] = probabilities[indices[i]];
    }
    for stride in [16, 8, 4, 2, 1] {
        for i in 0..stride {
            lanes[i] += lanes[i + stride];
        }
    }
    let total = lanes[0].max(6.103_515_6e-5);
    let weights = std::array::from_fn(|i| {
        let quotient = probabilities[indices[i]] / total;
        quotient * 1.5
    });
    Ok(RouterSelection { indices, weights })
}

// Quantization helpers port the finite E4M3FN/E2M1 nearest-even encodings from
// metal/dsv4_kv.metal:58-106, not IEEE infinity-bearing FP8 or F16 emulation.
fn e4m3_value(code: u32) -> f32 {
    let exponent = code >> 3;
    let mantissa = code & 7;
    if exponent == 0 {
        mantissa as f32 * 0.001_953_125
    } else {
        (1.0 + mantissa as f32 * 0.125) * f32::from_bits((exponent + 120) << 23)
    }
}

fn nearest_e4m3(x: f32) -> f32 {
    let x = x.abs().min(448.0);
    let (mut lo, mut hi) = (0u32, 126);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if e4m3_value(mid) <= x {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if lo < 126 {
        let a = x - e4m3_value(lo);
        let b = (x - e4m3_value(lo + 1)).abs();
        if b < a || (a == b && lo & 1 != 0) {
            lo += 1;
        }
    }
    e4m3_value(lo)
}

fn nearest_e2m1(x: f32) -> f32 {
    const VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    const MID: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
    let mut code = 0;
    for (i, &mid) in MID.iter().enumerate() {
        if x > mid || (x == mid && i & 1 != 0) {
            code = i + 1;
        }
    }
    VALUES[code]
}

fn pow2_ceil(x: f32) -> f32 {
    let bits = x.to_bits();
    let carry = if bits & 0x007f_ffff != 0 {
        0x0080_0000
    } else {
        0
    };
    f32::from_bits((bits & 0x7f80_0000) + carry)
}

fn quant_scale(amax: f32, mode: u32) -> f32 {
    match mode {
        1 => pow2_ceil(amax.max(1.0e-4) * (1.0 / 448.0)),
        2 => pow2_ceil(amax.max(7.052_966e-38) * (1.0 / 6.0)),
        _ => nearest_e4m3(amax.max(0.011_718_75) / 6.0),
    }
}

#[derive(Clone, Copy)]
enum Kind {
    HcMix,
    Bf16,
    Rms,
    Quantize,
    Rope,
    HcSplit,
    Collapse,
    Expand,
    Engram,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Args {
    width: u32,
    rows: u32,
    mode: u32,
    position: u32,
    inverse: u32,
    round_all: u32,
    eps: f32,
    frequencies: [f32; 32],
    rows_per_position: u32,
    position_step: u32,
}

struct Op {
    kind: Kind,
    args: Args,
    output: Shape,
}

impl Op {
    fn new(kind: Kind, width: usize, rows: usize, output: Shape) -> Self {
        Self {
            kind,
            args: Args {
                width: width as u32,
                rows: rows as u32,
                mode: 0,
                position: 0,
                inverse: 0,
                round_all: 0,
                eps: 0.0,
                frequencies: [0.0; 32],
                rows_per_position: 0,
                position_step: 0,
            },
            output,
        }
    }

    fn apply(self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        a.apply_op3_no_bwd(b, c, &self)
    }
}

fn cpu_slice<'a>(storage: &'a CpuStorage, layout: &Layout) -> Result<&'a [f32]> {
    if !layout.is_contiguous() {
        candle_core::bail!("V4.1 math requires contiguous storage");
    }
    let slice = storage.as_slice::<f32>()?;
    slice
        .get(layout.start_offset()..layout.start_offset() + layout.shape().elem_count())
        .ok_or_else(|| candle_core::Error::Msg("V4.1 CPU storage span out of bounds".into()))
}

impl CustomOp3 for Op {
    fn name(&self) -> &'static str {
        "dinf-deepseek41-math"
    }

    fn cpu_fwd(
        &self,
        a: &CpuStorage,
        al: &Layout,
        b: &CpuStorage,
        bl: &Layout,
        c: &CpuStorage,
        cl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let a = cpu_slice(a, al)?;
        let b = cpu_slice(b, bl)?;
        let c = cpu_slice(c, cl)?;
        let width = self.args.width as usize;
        let rows = self.args.rows as usize;
        let mut y = vec![0.0; self.output.elem_count()];
        match self.kind {
            Kind::Bf16 => {
                for (y, &x) in y.iter_mut().zip(a) {
                    *y = bf16_scalar(x);
                }
            }
            Kind::Rms => {
                for row in 0..rows {
                    let offset = row * width;
                    let sum = rms_sum(&a[offset..offset + width]);
                    let scale = 1.0 / (sum / width as f32 + self.args.eps).sqrt();
                    for i in 0..width {
                        let v = a[offset + i] * scale;
                        y[offset + i] = if self.args.mode == 1 {
                            bf16_scalar(v * b[i])
                        } else {
                            v
                        };
                    }
                }
            }
            Kind::Quantize => {
                let block = if self.args.mode == 3 { 16 } else { 32 };
                for row in 0..rows {
                    for start in (0..width).step_by(block) {
                        let end = (start + block).min(width);
                        let offset = row * width;
                        let amax = a[offset + start..offset + end]
                            .iter()
                            .map(|&x| bf16_scalar(x).abs())
                            .fold(0.0, f32::max);
                        let scale = quant_scale(amax, self.args.mode);
                        for i in start..end {
                            let value = bf16_scalar(a[offset + i]);
                            let normalized = value.abs() / scale;
                            let q = if self.args.mode == 1 {
                                nearest_e4m3(normalized)
                            } else {
                                nearest_e2m1(normalized)
                            };
                            y[offset + i] = bf16_scalar(q.copysign(value) * scale);
                        }
                    }
                }
            }
            Kind::Rope => {
                y.copy_from_slice(a);
                if self.args.round_all != 0 {
                    for v in &mut y {
                        *v = bf16_scalar(*v);
                    }
                }
                for row in 0..rows {
                    let position = self.args.position
                        + (row as u32)
                            .checked_div(self.args.rows_per_position)
                            .unwrap_or(0)
                            * self.args.position_step;
                    for pair in 0..32 {
                        let theta = position as f32 * self.args.frequencies[pair];
                        let cos = theta.cos();
                        let sin = if self.args.inverse != 0 {
                            -theta.sin()
                        } else {
                            theta.sin()
                        };
                        let i = row * width + width - 64 + 2 * pair;
                        let (re, im) = (y[i], y[i + 1]);
                        // Source kernel's contraction: multiply the cross term
                        // first, then a single FMA, then BF16 output rounding.
                        y[i] = bf16_scalar(re.mul_add(cos, -(im * sin)));
                        y[i + 1] = bf16_scalar(im.mul_add(cos, re * sin));
                    }
                }
            }
            Kind::HcSplit => {
                for row in 0..rows {
                    hc_split_scalar(&a[24 * row..], b, c, &mut y[24 * row..24 * row + 24]);
                }
            }
            Kind::HcMix => {
                for row in 0..rows {
                    for o in 0..24 {
                        let mut sum = 0.0f32;
                        for i in 0..width {
                            sum = a[row * width + i].mul_add(b[o * width + i], sum);
                        }
                        y[24 * row + o] = sum;
                    }
                }
            }
            Kind::Collapse => {
                let stride = if self.args.mode == 0 {
                    4
                } else {
                    self.args.mode as usize
                };
                for row in 0..rows {
                    let r = &a[row * 4 * width..];
                    for i in 0..width {
                        let mut sum = 0.0;
                        for stream in 0..4 {
                            sum = r[stream * width + i].mul_add(b[row * stride + stream], sum);
                        }
                        y[row * width + i] = bf16_scalar(sum);
                    }
                }
            }
            Kind::Expand => {
                for row in 0..rows {
                    let (r, sp) = (row * 4 * width, &c[24 * row..]);
                    for dest in 0..4 {
                        for i in 0..width {
                            let mut sum = a[row * width + i] * sp[4 + dest];
                            for src in 0..4 {
                                sum = sp[8 + src * 4 + dest].mul_add(b[r + src * width + i], sum);
                            }
                            y[r + dest * width + i] = bf16_scalar(sum);
                        }
                    }
                }
            }
            Kind::Engram => {
                for group in 0..rows {
                    let (row, stream) = (group / 4, group % 4);
                    let offset = group * width;
                    let b = &b[row * 5 * width..];
                    let (mut h2, mut k2, mut dot) = (0.0, 0.0, 0.0);
                    for i in 0..width {
                        let h = a[offset + i];
                        let k = bf16_scalar(b[stream * width + i]);
                        h2 += h * h;
                        k2 += k * k;
                        dot += (h * c[stream * width + i]) * k;
                    }
                    dot = dot
                        * (1.0 / (h2 / width as f32 + self.args.eps).sqrt())
                        * (1.0 / (k2 / width as f32 + self.args.eps).sqrt())
                        * (1.0 / (width as f32).sqrt());
                    let gate = 1.0 / (1.0 + (-dot.abs().max(1.0e-6).sqrt().copysign(dot)).exp());
                    for i in 0..width {
                        y[offset + i] =
                            bf16_scalar(gate.mul_add(bf16_scalar(b[4 * width + i]), a[offset + i]));
                    }
                }
            }
        }
        Ok((CpuStorage::F32(y), self.output.clone()))
    }

    #[cfg(target_os = "macos")]
    fn metal_fwd(
        &self,
        a: &candle_core::MetalStorage,
        al: &Layout,
        b: &candle_core::MetalStorage,
        bl: &Layout,
        c: &candle_core::MetalStorage,
        cl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::forward(self, [(a, al), (b, bl), (c, cl)])
    }
}

// metal/dsv4_hc.metal:141-202. First row softmax adds eps AFTER division;
// columns add eps in their denominator, then 19 alternating row/column steps.
// Gates: :100-104; DS4_METAL_HC_STABLE defaults ON (ds4_metal.m:6927,6954).
// Fused V41 also uses these helpers (:513,530,1404,1407), unlike V4/Engram.
fn hc_split_scalar(mix: &[f32], scale: &[f32], base: &[f32], out: &mut [f32]) {
    for i in 0..4 {
        let pre_z = mix[i].mul_add(scale[0], base[i]);
        let post_z = mix[4 + i].mul_add(scale[1], base[4 + i]);
        out[i] = 0.5 * (0.5 * pre_z).tanh() + 0.5 + HC_EPS;
        out[4 + i] = 1.0 + (0.5 * post_z).tanh();
    }
    let mut comb = [0.0f32; 16];
    for row in 0..4 {
        let offset = row * 4;
        for col in 0..4 {
            let i = 8 + offset + col;
            comb[offset + col] = mix[i].mul_add(scale[2], base[i]);
        }
        let peak = comb[offset..offset + 4]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        for v in &mut comb[offset..offset + 4] {
            *v = (*v - peak).exp();
        }
        let sum: f32 = comb[offset..offset + 4].iter().sum();
        let inv = 1.0 / sum;
        for v in &mut comb[offset..offset + 4] {
            *v = *v * inv + HC_EPS;
        }
    }
    for iteration in 0..20 {
        if iteration != 0 {
            for row in comb.chunks_exact_mut(4) {
                let sum: f32 = row.iter().sum();
                let inv = 1.0 / (sum + HC_EPS);
                for v in row {
                    *v *= inv;
                }
            }
        }
        for col in 0..4 {
            let sum = comb[col] + comb[4 + col] + comb[8 + col] + comb[12 + col];
            let inv = 1.0 / (sum + HC_EPS);
            for row in 0..4 {
                comb[row * 4 + col] *= inv;
            }
        }
    }
    out[8..24].copy_from_slice(&comb);
}

#[cfg(target_os = "macos")]
mod metal {
    use std::collections::HashMap;
    use std::sync::{LazyLock, Mutex};

    use candle_core::backend::BackendStorage;
    use candle_core::{DType, Layout, MetalStorage, Result, Shape};
    use candle_metal_kernels::metal::{ComputePipeline, Library};
    use objc2_metal::{
        MTLCompileOptions, MTLComputePipelineState, MTLLanguageVersion, MTLMathMode, MTLSize,
    };

    use super::{Kind, Op};

    #[derive(Default)]
    struct Pipelines {
        library: Option<Library>,
        kernels: HashMap<&'static str, ComputePipeline>,
    }

    static PIPELINES: LazyLock<Mutex<HashMap<candle_core::metal_backend::DeviceId, Pipelines>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    fn pipeline(device: &candle_core::MetalDevice, name: &'static str) -> Result<ComputePipeline> {
        let mut cache = PIPELINES
            .lock()
            .map_err(|_| candle_core::Error::Msg("V4.1 Metal pipeline lock poisoned".into()))?;
        let entry = cache.entry(device.id()).or_default();
        if let Some(pipeline) = entry.kernels.get(name) {
            return Ok(pipeline.clone());
        }
        if entry.library.is_none() {
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version4_0);
            // Explicit FMA sites in math.metal, not compiler reassociation.
            options.setMathMode(MTLMathMode::Safe);
            entry.library = Some(
                device
                    .metal_device()
                    .new_library_with_source(include_str!("math.metal"), Some(&options))
                    .map_err(candle_core::Error::wrap)?,
            );
        }
        let function = entry
            .library
            .as_ref()
            .unwrap()
            .get_function(name, None)
            .map_err(candle_core::Error::wrap)?;
        let pipeline = device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(candle_core::Error::wrap)?;
        entry.kernels.insert(name, pipeline.clone());
        Ok(pipeline)
    }

    pub(super) fn forward(
        op: &Op,
        inputs: [(&MetalStorage, &Layout); 3],
    ) -> Result<(MetalStorage, Shape)> {
        let device = inputs[0].0.device();
        let elements = op.output.elem_count();
        // Candidate owners cannot accept arbitrarily oversized pool loans.
        // Select the sole bounded carrier before encoding, never a repair copy.
        let output = if super::super::super::deepseek4_qmatmul::bounded_prefill_output(device) {
            device.new_private_buffer(elements, DType::F32, "dinf-v41-math")?
        } else {
            device
                .new_buffer_builder()
                .with_size_for(elements, DType::F32)
                .with_label("dinf-v41-math")
                .build()?
        };
        let name = match op.kind {
            Kind::Bf16 => "v41_bf16",
            Kind::Rms => "v41_rms",
            Kind::Quantize => "v41_quantize",
            Kind::Rope => "v41_rope",
            Kind::HcSplit => "v41_hc_split",
            Kind::Collapse => "v41_collapse",
            Kind::Expand => "v41_expand",
            Kind::Engram => "v41_engram",
            Kind::HcMix => "v41_hc_mix",
        };
        let pipeline = pipeline(device, name)?;
        if pipeline.as_ref().threadExecutionWidth() != 32
            || pipeline.max_total_threads_per_threadgroup() < 32
        {
            candle_core::bail!("V4.1 math requires 32-lane Metal SIMD groups");
        }
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        for (slot, (storage, layout)) in inputs.iter().enumerate() {
            if storage.dtype() != DType::F32
                || !layout.is_contiguous()
                || storage.device().id() != device.id()
            {
                candle_core::bail!("V4.1 Metal operands require contiguous F32 on one device");
            }
            let offset = layout
                .start_offset()
                .checked_mul(4)
                .ok_or_else(|| candle_core::Error::Msg("V4.1 Metal offset overflow".into()))?;
            let end = layout
                .shape()
                .elem_count()
                .checked_mul(4)
                .and_then(|n| offset.checked_add(n))
                .ok_or_else(|| candle_core::Error::Msg("V4.1 Metal span overflow".into()))?;
            if end > storage.buffer().length() {
                candle_core::bail!("V4.1 Metal span outside buffer");
            }
            encoder.set_input_buffer(slot, Some(storage.buffer()), offset);
        }
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &op.args);
        let width = op.args.width as usize;
        let rows = op.args.rows as usize;
        let groups = match op.kind {
            Kind::Rms | Kind::Rope | Kind::Engram | Kind::HcSplit | Kind::HcMix => rows,
            Kind::Quantize => rows * width.div_ceil(if op.args.mode == 3 { 16 } else { 32 }),
            Kind::Expand => (rows * width).div_ceil(32),
            _ => elements.div_ceil(32),
        };
        let threads = match op.kind {
            Kind::Rms => super::rms_threads(width),
            Kind::HcMix => 256,
            _ => 32,
        };
        if threads > pipeline.max_total_threads_per_threadgroup() {
            candle_core::bail!("V4.1 RMS pipeline cannot support source thread count {threads}");
        }
        encoder.dispatch_thread_groups(
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
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            op.output.clone(),
        ))
    }
}

#[cfg(test)]
#[path = "math_tests.rs"]
mod tests;
