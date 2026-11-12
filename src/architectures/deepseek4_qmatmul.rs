//! Packed matmul for DeepSeek V4/V4.1 GGUF tensors, including IQ1_S.
//!
//! IQ1_S decodes directly to F32. The V4.1 Q8_0 path multiplies an eight-value
//! F32 dot by the F32-promoted block scale rather than decoding to half weights;
//! default constructors keep the Q8_0 half decode used by V4.
//! A test-only matrix selector decodes Q8 weights directly to F32 SIMD-group
//! operands; see `deepseek4_q8_mma` for its bounded 2D contract.

use std::sync::Arc;

#[cfg(target_os = "macos")]
use anyhow::Context;
use anyhow::Result;
use candle_core::Tensor;

use crate::deepseek4_gguf::{DeepSeek4DType, RawTensor};

#[cfg(target_os = "macos")]
#[path = "deepseek4_q8_mma.rs"]
mod q8_mma;

#[cfg(target_os = "macos")]
#[path = "deepseek4_prefill_mm.rs"]
pub(crate) mod prefill_mm;

thread_local! {
    static PACKED_REUSE4_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static Q8_PREFILL_MATRIX_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Keep the mixed-quant selector local to one Coder forward and restore it on errors.
pub(crate) struct PackedReuse4Scope(bool, std::marker::PhantomData<std::rc::Rc<()>>);
impl PackedReuse4Scope {
    pub(crate) fn enter(enabled: bool) -> Self {
        Self(
            PACKED_REUSE4_ENABLED.with(|v| v.replace(enabled)),
            std::marker::PhantomData,
        )
    }
}
impl Drop for PackedReuse4Scope {
    fn drop(&mut self) {
        PACKED_REUSE4_ENABLED.with(|v| v.set(self.0));
    }
}
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn with_packed_reuse4_enabled<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    let _restore = PackedReuse4Scope::enter(enabled);
    f()
}

/// Test-only selector for the Q8_0 SIMD-group matrix prefill path; there is no
/// environment or user-facing flag and release builds never select it.
/// Restores the previous selector on return or unwind, including nested scopes.
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn with_q8_prefill_matrix_enabled<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            Q8_PREFILL_MATRIX_ENABLED.with(|v| v.set(self.0));
        }
    }
    let _restore = Restore(Q8_PREFILL_MATRIX_ENABLED.with(|v| v.replace(enabled)));
    f()
}

#[cfg(target_os = "macos")]
thread_local! {
    static BOUNDED_PREFILL_DEVICE: std::cell::Cell<Option<candle_core::metal_backend::DeviceId>> =
        const { std::cell::Cell::new(None) };
}

/// Allocation policy for strict admitted tiled producers, not a math selector.
/// Device-local, nested and unwind-safe; ordinary row work stays outside.
#[cfg(target_os = "macos")]
pub(crate) struct BoundedPrefillScope {
    previous: Option<candle_core::metal_backend::DeviceId>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
#[cfg(target_os = "macos")]
impl BoundedPrefillScope {
    pub(crate) fn enter(device: &candle_core::Device, v2: bool) -> Result<Self> {
        let id = if device.is_metal() && (v2 || Q8_PREFILL_MATRIX_ENABLED.with(|v| v.get())) {
            Some(device.as_metal_device()?.id())
        } else {
            None
        };
        Ok(Self {
            previous: BOUNDED_PREFILL_DEVICE.with(|v| v.replace(id)),
            _thread: std::marker::PhantomData,
        })
    }
}
#[cfg(target_os = "macos")]
impl Drop for BoundedPrefillScope {
    fn drop(&mut self) {
        BOUNDED_PREFILL_DEVICE.with(|v| v.set(self.previous));
    }
}
#[cfg(target_os = "macos")]
pub(crate) fn bounded_prefill_output(device: &candle_core::MetalDevice) -> bool {
    BOUNDED_PREFILL_DEVICE.with(|v| v.get() == Some(device.id()))
}

/// A V4 weight matrix. The matrix stays in GGUF byte form until the fused
/// Metal kernel consumes it.
pub struct DeepSeek4QMatMul {
    weights: Arc<RawTensor>,
    dense_weights: Option<Tensor>,
    out_dim: usize,
    in_dim: usize,
    v41_q8: bool,
}

impl DeepSeek4QMatMul {
    /// Validate and retain a rank-2 raw GGUF weight matrix.
    ///
    /// The weight bytes remain in their packed representation until `forward`
    /// is called, unless the dense device constructor is used.
    pub fn new(weights: Arc<RawTensor>) -> Result<Self> {
        anyhow::ensure!(
            weights.shape().len() == 2,
            "DeepSeek V4 matmul needs a rank-2 weight, got {:?}",
            weights.shape()
        );
        let [out_dim, in_dim] = [weights.shape()[0], weights.shape()[1]];
        anyhow::ensure!(
            out_dim > 0 && in_dim > 0,
            "DeepSeek V4 matmul dimensions are empty"
        );
        anyhow::ensure!(
            weights.byte_len() == weight_byte_len(weights.dtype(), out_dim, in_dim)?,
            "DeepSeek V4 packed weight length does not match its shape"
        );
        Ok(Self {
            weights,
            dense_weights: None,
            out_dim,
            in_dim,
            v41_q8: false,
        })
    }

    /// Build a matrix and keep dense weights on the target device.
    ///
    /// Router and hyper-connection weights in V4 are small BF16/F32 tensors.
    /// Keeping their decoded form avoids a device-to-host read on every token.
    pub fn new_on_device(weights: Arc<RawTensor>, device: &candle_core::Device) -> Result<Self> {
        let mut result = Self::new(weights)?;
        if is_dense(result.dtype()) {
            result.dense_weights = Some(dense_f32(&result.weights, device)?);
        }
        Ok(result)
    }

    /// Opt into Maya V4.1 Q8_0 decode arithmetic for every independent row.
    /// Returns F32; the caller owns BF16 input/output rounding boundaries.
    /// Other dtypes retain `new_on_device` behavior. No dense Q8 materialization.
    ///
    /// Source: audreyt/ds4 0dff39f91721bcf3472813c6d2bebf9353df2a1a,
    /// metal/dense.metal:87-125,285-368; ds4_metal.m:5403-5417,19714-19716.
    pub fn new_v41_on_device(
        weights: Arc<RawTensor>,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let mut result = Self::new_on_device(weights, device)?;
        result.v41_q8 = result.dtype() == DeepSeek4DType::Q8_0;
        Ok(result)
    }

    /// Return the packed or dense GGUF dtype of this matrix.
    pub fn dtype(&self) -> DeepSeek4DType {
        self.weights.dtype()
    }

    /// Return the matrix output width.
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Return the matrix input width.
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Multiply an f32 tensor of rank two or greater by this matrix.
    ///
    /// Returns an error for non-f32 input, rank below two, or a mismatched
    /// final dimension. Packed weights use the available fused backend.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_impl(input, false)
    }

    /// Batched attention-prefill projections (QA, KV, QB and output-B shapes).
    /// V4.1 Q8_0 inputs of 2..=32 rows use the `rows_seq4` kernel; other
    /// shapes and formats fall back to the ordinary scalar/decode paths.
    pub(crate) fn forward_attention_prefill(&self, input: &Tensor) -> Result<Tensor> {
        #[cfg(target_os = "macos")]
        if matches!(
            (self.in_dim, self.out_dim),
            (1280, 32768) | (8192, 5120) | (5120, 1280) | (5120, 512)
        ) && let Some(output) = self.try_prefill_matrix(input)?
        {
            return Ok(output);
        }
        #[cfg(target_os = "macos")]
        if self.v41_q8
            && input.device().is_metal()
            && input.rank() == 2
            && input.dtype() == candle_core::DType::F32
            && input.is_contiguous()
            && input.dim(1)? == self.in_dim
            && (2..=32).contains(&input.dim(0)?)
            && matches!(
                (self.in_dim, self.out_dim),
                (1280, 32768) | (8192, 5120) | (5120, 1280) | (5120, 512)
            )
        {
            return Ok(metal::forward_with_mode(
                self.weights.bytes(),
                input,
                self.dtype(),
                self.out_dim,
                self.in_dim,
                metal::KernelKind::V41Q8RowsSeq4,
                false,
            )?);
        }
        if matches!(
            self.dtype(),
            DeepSeek4DType::Q5K
                | DeepSeek4DType::IQ1M
                | DeepSeek4DType::IQ2XXS
                | DeepSeek4DType::IQ2XS
        ) {
            self.forward_prefill_rows(input)
        } else {
            self.forward(input)
        }
    }

    /// Joined 2D projection used by batched compressor/indexer, Engram and
    /// shared-FFN prefill. When native prefill is active, V4.1 Q8_0 inputs use
    /// the native prefill matmul. The optional SIMD-group matrix path requires
    /// V4.1 Q8_0, T=2..=512, 0<K,M<=65536, K divisible by 32, contiguous F32
    /// input and packed U8 weights on one Metal device with checked spans and
    /// Q8 alignment. Unsupported geometry, device or pipeline falls back to the
    /// ordinary forward before anything is enqueued; encoding or allocation
    /// errors are propagated with no partial fallback. Rank-3/grouped inputs
    /// never use the matrix path.
    ///
    /// The matrix path adds no retained carrier and uses 8192 transient
    /// threadgroup bytes per workgroup (128 threads, ceil(M/32) x ceil(T/16)
    /// groups). Callers keep input/output/weight owners alive until their GPU
    /// fence. Finite half scales decode exactly to F32; NaN/Inf are not clamped.
    /// Its F32 reduction order differs from the scalar path.
    pub(crate) fn forward_prefill_matrix(&self, input: &Tensor) -> Result<Tensor> {
        #[cfg(target_os = "macos")]
        if prefill_mm::active()
            && self.v41_q8
            && self.dtype() == DeepSeek4DType::Q8_0
            && input.rank() == 2
            && input.dims()[0] > 1
        {
            return prefill_mm::dense(self.weights.bytes(), input, self.in_dim, self.out_dim);
        }
        #[cfg(target_os = "macos")]
        if let Some(output) = self.try_prefill_matrix(input)? {
            return Ok(output);
        }
        #[cfg(target_os = "macos")]
        if input.rank() == 2
            && input.device().is_metal()
            && is_dense(self.dtype())
            && bounded_prefill_output(input.device().as_metal_device()?)
        {
            // Same Candle MLX GEMM, with its destination allocated directly.
            // A pooled destination could exceed the bounded tile owner charge.
            let weight = self
                .dense_weights
                .as_ref()
                .context("missing resident dense matrix")?;
            return dense_prefill_bounded(input, weight, self.in_dim, self.out_dim);
        }
        if matches!(
            self.dtype(),
            DeepSeek4DType::Q5K
                | DeepSeek4DType::IQ1M
                | DeepSeek4DType::IQ2XXS
                | DeepSeek4DType::IQ2XS
        ) {
            self.forward_prefill_rows(input)
        } else {
            self.forward(input)
        }
    }

    #[cfg(target_os = "macos")]
    fn try_prefill_matrix(&self, input: &Tensor) -> Result<Option<Tensor>> {
        if !self.v41_q8
            || self.dtype() != DeepSeek4DType::Q8_0
            || !Q8_PREFILL_MATRIX_ENABLED.with(|v| v.get())
        {
            return Ok(None);
        }
        Ok(q8_mma::try_forward(
            self.weights.bytes(),
            input,
            self.in_dim,
            self.out_dim,
        )?)
    }

    /// Project prompt rows of a packed Coder format. On qualified Metal 4
    /// devices this is one tiled matmul whose rows do not depend on the batch
    /// size; elsewhere, or for other formats, it keeps the decode-row kernels.
    pub(crate) fn forward_prefill_rows(&self, input: &Tensor) -> Result<Tensor> {
        #[cfg(target_os = "macos")]
        if let Some(output) = prefill_mm::dense_packed(
            self.weights.bytes(),
            input,
            self.dtype(),
            self.in_dim,
            self.out_dim,
        )? {
            return Ok(output);
        }
        self.forward_decode_rows(input)
    }

    /// Project several packed experts' prompt rows in one dispatch: matrix
    /// `g` multiplies the `input` rows listed in `rows[g]`, and output rows are
    /// the groups' rows in order. Each row matches `forward_prefill_rows`.
    /// Returns `None` when the matrices differ in format or shape, or the
    /// tiled path is unavailable; nothing is enqueued then.
    pub(crate) fn forward_experts_prefill(
        matrices: &[&Self],
        rows: &[Vec<u32>],
        input: &Tensor,
    ) -> Result<Option<Tensor>> {
        let Some(first) = matrices.first() else {
            return Ok(None);
        };
        if matrices.iter().any(|m| {
            m.dtype() != first.dtype() || m.in_dim != first.in_dim || m.out_dim != first.out_dim
        }) {
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        {
            let weights = matrices
                .iter()
                .map(|m| m.weights.bytes())
                .collect::<Vec<_>>();
            prefill_mm::experts_packed(
                &weights,
                rows,
                input,
                first.dtype(),
                first.in_dim,
                first.out_dim,
            )
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (rows, input);
            Ok(None)
        }
    }

    /// Grouped prompt projection for packed formats: `input` is
    /// `[tokens * groups, in_dim]` where row `t * groups + g` feeds group `g`,
    /// which owns `out_dim / groups` consecutive weight rows. Returns
    /// `[tokens, out_dim]`, each row matching `forward_grouped_packed` up to
    /// the tiled kernel's half operands. `None` when unavailable.
    pub(crate) fn forward_grouped_prefill(
        &self,
        input: &Tensor,
        groups: usize,
        tokens: usize,
    ) -> Result<Option<Tensor>> {
        if self.v41_q8
            || groups == 0
            || tokens == 0
            || !self.out_dim.is_multiple_of(groups)
            || input.dims() != [tokens * groups, self.in_dim]
        {
            return Ok(None);
        }
        let rows = self.out_dim / groups;
        let matrices = (0..groups)
            .map(|g| -> Result<Self> {
                Ok(Self {
                    weights: Arc::new(self.weights.slice_rows(g * rows, rows)?),
                    dense_weights: None,
                    out_dim: rows,
                    in_dim: self.in_dim,
                    v41_q8: false,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let lists = (0..groups)
            .map(|g| (0..tokens).map(|t| (t * groups + g) as u32).collect())
            .collect::<Vec<_>>();
        let refs = matrices.iter().collect::<Vec<_>>();
        let Some(output) = Self::forward_experts_prefill(&refs, &lists, input)? else {
            return Ok(None);
        };
        // Group-major [groups, tokens, rows] to [tokens, groups * rows].
        Ok(Some(
            output
                .reshape((groups, tokens, rows))?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((tokens, self.out_dim))?,
        ))
    }

    /// Project independent decode rows with the single-token reduction order.
    /// Coder prefill may reuse decoded weights while keeping this exact sum order.
    pub fn forward_decode_rows(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_impl(input, true)
    }

    /// Project one input row per group using that group's consecutive weight
    /// rows. V4.1 Q8_0 uses its dedicated grouped kernel; other formats use
    /// packed views.
    pub(crate) fn forward_grouped_packed(&self, input: &Tensor, groups: usize) -> Result<Tensor> {
        if self.v41_q8 {
            return self.forward_grouped_v41(input, groups);
        }
        anyhow::ensure!(
            groups > 0
                && groups <= 8
                && self.out_dim.is_multiple_of(groups)
                && input.dims() == [groups, self.in_dim]
                && input.dtype() == candle_core::DType::F32
                && input.is_contiguous()
                && input.device().same_device(self.weights.bytes().device()),
            "invalid grouped packed projection"
        );
        let rows = self.out_dim / groups;
        let mut output = Vec::with_capacity(groups);
        for group in 0..groups {
            let first = group * rows;
            let matrix = Self {
                weights: Arc::new(self.weights.slice_rows(first, rows)?),
                dense_weights: self
                    .dense_weights
                    .as_ref()
                    .map(|w| w.narrow(0, first, rows))
                    .transpose()?,
                out_dim: rows,
                in_dim: self.in_dim,
                v41_q8: false,
            };
            output.push(matrix.forward(&input.narrow(0, group, 1)?)?);
        }
        Ok(Tensor::cat(&output, 1)?)
    }

    /// Project `[groups, in_dim]` into contiguous F32 `[1, out_dim]`.
    ///
    /// Opt-in V4.1 Q8_0 only, on one Metal device with contiguous F32 input
    /// and packed weights. Each group owns `out_dim / groups` consecutive
    /// output rows; that count must be even and at most 65,536 so pairs never
    /// cross groups and each row retains `forward`'s NSG4 reduction exactly.
    /// No BF16 rounding is performed here. Ordinary `forward` is unchanged.
    pub fn forward_grouped_v41(&self, input: &Tensor, groups: usize) -> Result<Tensor> {
        anyhow::ensure!(
            self.v41_q8 && self.dtype() == DeepSeek4DType::Q8_0,
            "V4.1 grouped matmul requires opt-in V4.1 Q8_0 weights"
        );
        let _geometry = GroupedV41Geometry::new(self.out_dim, self.in_dim, input.dims(), groups)?;
        anyhow::ensure!(
            input.dtype() == candle_core::DType::F32 && input.is_contiguous(),
            "V4.1 grouped matmul requires contiguous F32 input"
        );
        let weights = self.weights.bytes();
        anyhow::ensure!(
            weights.dtype() == candle_core::DType::U8 && weights.is_contiguous(),
            "V4.1 grouped matmul requires contiguous packed U8 weights"
        );
        anyhow::ensure!(
            input.device().same_device(weights.device()),
            "V4.1 grouped weights and input must share a device"
        );
        #[cfg(target_os = "macos")]
        if input.device().is_metal() {
            return Ok(metal::forward_grouped_v41(weights, input, _geometry)?);
        }
        anyhow::bail!("V4.1 grouped Q8 matmul needs the Metal backend")
    }

    pub(crate) fn iq1_simd_enabled() -> bool {
        #[cfg(target_os = "macos")]
        {
            let configured = std::env::var_os("DINF_DEEPSEEK4_SIMD");
            metal::simd_enabled(configured.as_deref(), cfg!(target_arch = "aarch64"))
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    /// Opt-in native selected-expert gate/up matvec, with one Metal dispatch.
    ///
    /// Accepts 1..=12 independent IQ1_S `[2304,5120]` matrices and the same
    /// contiguous F32 `[1,5120]` input. Returns contiguous F32 `[count,2304]`
    /// in slice order: pass expert0 gate/up, expert1 gate/up, ... expert5 gate/up.
    /// `output.narrow(0, slot, 1)` is a checked `[1,2304]` view, without a copy.
    /// Contiguous nonzero input/packed-weight offsets and repeated weights work.
    /// Packed offsets must be two-byte aligned for block_iq1_s's half/u16 fields.
    ///
    /// SIMD must be enabled by the same policy as `forward`; native integration
    /// uses separate `forward` calls when disabled.
    /// No CPU fallback, BF16 boundary, activation, readback or fence is added.
    ///
    /// Owners are retained during custom-op encoding only. The caller MUST keep
    /// input, output and selected ExpertTriplet/RawTensor owners alive through
    /// the existing whole-forward GPU fence, including error paths. Metal Buffer
    /// retention does not pin Candle's outer pooled Arc against reuse. Arbitrary
    /// early owner drop is NOT supported. Native MoeLeases retains these owners
    /// until the next successful router readback or final/error fence.
    pub(crate) fn forward_grouped_iq1(input: &Tensor, matrices: &[&Self]) -> Result<Tensor> {
        let geometry = GroupedIq1Geometry::new(matrices.len(), input.dims())?;
        anyhow::ensure!(
            input.dtype() == candle_core::DType::F32 && input.is_contiguous(),
            "grouped IQ1 requires contiguous F32 input"
        );
        for matrix in matrices {
            geometry.validate_weight(
                matrix.weights.shape(),
                matrix.dtype(),
                matrix.weights.byte_len(),
            )?;
            let bytes = matrix.weights.bytes();
            anyhow::ensure!(
                bytes.dtype() == candle_core::DType::U8 && bytes.is_contiguous(),
                "grouped IQ1 requires contiguous packed U8 weights"
            );
            anyhow::ensure!(
                input.device().same_device(bytes.device()),
                "grouped IQ1 input and weights must share one Metal device"
            );
        }
        #[cfg(target_os = "macos")]
        if input.device().is_metal() {
            return Ok(metal::forward_grouped_iq1(input, matrices, geometry)?);
        }
        anyhow::bail!("grouped IQ1 requires the Metal backend")
    }

    /// Test-only postprocessing: six IQ1_S [5120,2304] matrices, each
    /// consuming its OWN F32 [6,2304] activation row. Completion leases are
    /// the caller's responsibility, exactly as for grouped gate/up.
    #[cfg(test)]
    pub(crate) fn forward_grouped_iq1_down(input: &Tensor, matrices: &[&Self]) -> Result<Tensor> {
        let geometry = GroupedIq1Geometry::down(matrices.len(), input.dims())?;
        anyhow::ensure!(
            input.dtype() == candle_core::DType::F32 && input.is_contiguous(),
            "grouped IQ1 down requires contiguous F32 [6,2304]"
        );
        for matrix in matrices {
            geometry.validate_weight(
                matrix.weights.shape(),
                matrix.dtype(),
                matrix.weights.byte_len(),
            )?;
            let bytes = matrix.weights.bytes();
            anyhow::ensure!(
                bytes.dtype() == candle_core::DType::U8 && bytes.is_contiguous(),
                "grouped IQ1 down requires contiguous packed U8 weights"
            );
            anyhow::ensure!(
                input.device().same_device(bytes.device()),
                "grouped IQ1 down input and weights must share one Metal device"
            );
        }
        #[cfg(target_os = "macos")]
        if input.device().is_metal() {
            return Ok(metal::forward_grouped_iq1(input, matrices, geometry)?);
        }
        anyhow::bail!("grouped IQ1 down requires the Metal backend")
    }

    fn forward_impl(&self, input: &Tensor, decode_rows: bool) -> Result<Tensor> {
        anyhow::ensure!(
            input.dtype() == candle_core::DType::F32,
            "DeepSeek V4 packed matmul needs an f32 input"
        );
        anyhow::ensure!(
            input.rank() >= 2,
            "DeepSeek V4 packed matmul needs an input of rank two or greater"
        );
        anyhow::ensure!(
            input.dim(candle_core::D::Minus1)? == self.in_dim,
            "DeepSeek V4 matmul input width {} does not match weight width {}",
            input.dim(candle_core::D::Minus1)?,
            self.in_dim
        );

        if is_dense(self.dtype()) {
            let input = input.contiguous()?;
            let weight = match self.dense_weights.as_ref() {
                Some(weight) => weight.clone(),
                None => dense_f32(&self.weights, input.device())?,
            };
            if decode_rows && input.elem_count() > self.in_dim && input.device().is_metal() {
                // Router scores choose experts. Keep their single-row
                // reduction order even when packed projections batch on the
                // GPU: a batch of one-row products against a stride-0
                // broadcast weight runs the same GEMV per row in one dispatch.
                let rows = input.elem_count() / self.in_dim;
                let flat = input.reshape((rows, 1, self.in_dim))?;
                let weight =
                    weight
                        .t()?
                        .unsqueeze(0)?
                        .broadcast_as((rows, self.in_dim, self.out_dim))?;
                let mut shape = input.dims().to_vec();
                *shape.last_mut().unwrap() = self.out_dim;
                return Ok(flat.matmul(&weight)?.reshape(shape)?);
            }
            if decode_rows && input.elem_count() > self.in_dim {
                // CPU's batched GEMM does not keep single-row bits; stay per row.
                let rows = input.elem_count() / self.in_dim;
                let flat = input.reshape((rows, self.in_dim))?;
                let outputs = (0..rows)
                    .map(|row| flat.narrow(0, row, 1)?.matmul(&weight.t()?))
                    .collect::<candle_core::Result<Vec<_>>>()?;
                let mut shape = input.dims().to_vec();
                *shape.last_mut().unwrap() = self.out_dim;
                return Ok(Tensor::cat(&outputs, 0)?.reshape(shape)?);
            }
            if input.rank() == 2 {
                return Ok(input.matmul(&weight.t()?)?);
            }
            let mut output_shape = input.dims().to_vec();
            *output_shape.last_mut().unwrap() = self.out_dim;
            let rows = input.elem_count() / self.in_dim;
            return Ok(input
                .reshape((rows, self.in_dim))?
                .matmul(&weight.t()?)?
                .reshape(output_shape)?);
        }

        #[cfg(target_os = "macos")]
        if prefill_mm::active()
            && self.v41_q8
            && self.dtype() == DeepSeek4DType::Q8_0
            && input.rank() == 2
            && input.dims()[0] > 1
        {
            return prefill_mm::dense(self.weights.bytes(), input, self.in_dim, self.out_dim);
        }
        #[cfg(target_os = "macos")]
        if input.device().is_metal() {
            if self.v41_q8 {
                let kind = if self.out_dim > 65_536 {
                    metal::KernelKind::V41Q8Nsg8
                } else {
                    metal::KernelKind::V41Q8Nsg4
                };
                return Ok(metal::forward_with_mode(
                    self.weights.bytes(),
                    input,
                    self.dtype(),
                    self.out_dim,
                    self.in_dim,
                    kind,
                    false,
                )?);
            }
            return Ok(metal::forward(
                self.weights.bytes(),
                input,
                self.dtype(),
                self.out_dim,
                self.in_dim,
                decode_rows,
            )?);
        }

        anyhow::bail!(
            "DeepSeek V4 packed {:?} matmul needs the Metal backend",
            self.dtype()
        )
    }
}

fn weight_byte_len(dtype: DeepSeek4DType, out_dim: usize, in_dim: usize) -> Result<usize> {
    anyhow::ensure!(
        out_dim > 0 && in_dim > 0,
        "DeepSeek V4 matmul dimensions are empty"
    );
    // Each row must contain whole blocks, not just the matrix as a whole.
    // Computing bytes per row also avoids overflowing out_dim * in_dim.
    dtype
        .byte_len(in_dim)?
        .checked_mul(out_dim)
        .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 matmul weight byte length overflow"))
}

/// Checked before dispatch/allocation, including on non-Metal builds.
#[cfg_attr(
    not(any(test, target_os = "macos")),
    expect(
        dead_code,
        reason = "CPU calls still validate this geometry; Metal dispatch and tests read its fields."
    )
)]
struct GroupedV41Geometry {
    in_dim: u32,
    out_dim: u32,
    rows_per_group: u32,
    row_bytes: u32,
    weight_bytes: usize,
    input_elements: usize,
}

impl GroupedV41Geometry {
    fn new(out_dim: usize, in_dim: usize, dims: &[usize], groups: usize) -> Result<Self> {
        anyhow::ensure!(groups > 0, "V4.1 grouped matmul groups must be nonzero");
        anyhow::ensure!(
            dims == [groups, in_dim],
            "V4.1 grouped input must have shape [groups, in_dim]"
        );
        anyhow::ensure!(
            out_dim > 0 && out_dim.is_multiple_of(groups),
            "V4.1 grouped output rows must be divisible by groups"
        );
        let rows_per_group = out_dim / groups;
        anyhow::ensure!(
            rows_per_group.is_multiple_of(2) && rows_per_group <= 65_536,
            "V4.1 grouped output rows per group must be even and at most 65536 (NSG4)"
        );
        let weight_bytes = weight_byte_len(DeepSeek4DType::Q8_0, out_dim, in_dim)?;
        let input_elements = groups
            .checked_mul(in_dim)
            .ok_or_else(|| anyhow::anyhow!("V4.1 grouped input element count overflow"))?;
        let checked_u32 = |v| {
            u32::try_from(v)
                .map_err(|_| anyhow::anyhow!("V4.1 grouped matmul exceeds Metal uint indexing"))
        };
        checked_u32(input_elements)?;
        input_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("V4.1 grouped input byte count overflow"))?;
        out_dim
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("V4.1 grouped output byte count overflow"))?;
        Ok(Self {
            in_dim: checked_u32(in_dim)?,
            out_dim: checked_u32(out_dim)?,
            rows_per_group: checked_u32(rows_per_group)?,
            row_bytes: checked_u32(weight_bytes / out_dim)?,
            weight_bytes,
            input_elements,
        })
    }
}

const GROUPED_IQ1_MAX: usize = 12;
const GROUPED_IQ1_IN: usize = 5120;
const GROUPED_IQ1_OUT: usize = 2304;

/// Only the native routed-expert geometry is supported; no general matrices.
#[allow(dead_code)]
struct GroupedIq1Geometry {
    count: usize,
    weight_bytes: usize,
    output_elements: usize,
    in_dim: usize,
    out_dim: usize,
    input_rows: usize,
}

#[allow(dead_code)]
impl GroupedIq1Geometry {
    fn new(count: usize, input_dims: &[usize]) -> Result<Self> {
        anyhow::ensure!(
            (1..=GROUPED_IQ1_MAX).contains(&count),
            "grouped IQ1 count must be in 1..=12"
        );
        anyhow::ensure!(
            input_dims == [1, GROUPED_IQ1_IN],
            "grouped IQ1 input must have shape [1,5120]"
        );
        let weight_bytes = weight_byte_len(DeepSeek4DType::IQ1S, GROUPED_IQ1_OUT, GROUPED_IQ1_IN)?;
        let output_elements = count
            .checked_mul(GROUPED_IQ1_OUT)
            .ok_or_else(|| anyhow::anyhow!("grouped IQ1 output count overflow"))?;
        u32::try_from(output_elements)?;
        output_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("grouped IQ1 output bytes overflow"))?;
        Ok(Self {
            count,
            weight_bytes,
            output_elements,
            in_dim: GROUPED_IQ1_IN,
            out_dim: GROUPED_IQ1_OUT,
            input_rows: 1,
        })
    }

    #[cfg(test)]
    fn down(count: usize, input_dims: &[usize]) -> Result<Self> {
        anyhow::ensure!(
            count == 6 && input_dims == [6, 2304],
            "grouped IQ1 down requires exactly six [6,2304] rows"
        );
        let weight_bytes = weight_byte_len(DeepSeek4DType::IQ1S, 5120, 2304)?;
        let output_elements = count
            .checked_mul(5120)
            .ok_or_else(|| anyhow::anyhow!("grouped IQ1 down output overflow"))?;
        u32::try_from(output_elements)?;
        output_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("grouped IQ1 down output bytes overflow"))?;
        Ok(Self {
            count,
            weight_bytes,
            output_elements,
            in_dim: 2304,
            out_dim: 5120,
            input_rows: 6,
        })
    }

    fn validate_weight(&self, dims: &[usize], dtype: DeepSeek4DType, bytes: usize) -> Result<()> {
        anyhow::ensure!(
            dtype == DeepSeek4DType::IQ1S,
            "grouped IQ1 requires IQ1_S weights"
        );
        anyhow::ensure!(
            dims == [self.out_dim, self.in_dim],
            "grouped IQ1 weights have the wrong native geometry"
        );
        anyhow::ensure!(
            bytes == self.weight_bytes,
            "grouped IQ1 packed span must be exact"
        );
        Ok(())
    }
}

/// Pure checked span validation, also exercised without initializing Metal.
#[allow(dead_code)]
fn grouped_iq1_span(
    layout: &candle_core::Layout,
    dtype: candle_core::DType,
    elements: usize,
    buffer_bytes: usize,
) -> Result<usize> {
    let count = layout
        .dims()
        .iter()
        .try_fold(1usize, |n, &dim| n.checked_mul(dim))
        .ok_or_else(|| anyhow::anyhow!("grouped IQ1 layout element count overflow"))?;
    anyhow::ensure!(
        count == elements && layout.is_contiguous(),
        "grouped IQ1 storage must have an exact contiguous layout"
    );
    let size = dtype.size_in_bytes();
    let offset = layout
        .start_offset()
        .checked_mul(size)
        .ok_or_else(|| anyhow::anyhow!("grouped IQ1 buffer offset overflow"))?;
    let alignment = if dtype == candle_core::DType::U8 {
        2
    } else {
        size
    };
    anyhow::ensure!(
        offset.is_multiple_of(alignment),
        "grouped IQ1 packed offset must be two-byte aligned"
    );
    let end = elements
        .checked_mul(size)
        .and_then(|bytes| offset.checked_add(bytes))
        .ok_or_else(|| anyhow::anyhow!("grouped IQ1 buffer span overflow"))?;
    anyhow::ensure!(end <= buffer_bytes, "grouped IQ1 span outside buffer");
    Ok(offset)
}

fn dense_f32(weights: &RawTensor, device: &candle_core::Device) -> Result<Tensor> {
    // Raw dense decoding already reads host bytes. Cast there before upload:
    // Metal cannot cast I32 to F32, and this avoids a separate GPU cast for
    // half-precision weights. Keep RawTensor::to_dense's original dtype for
    // non-matmul users such as the integer hash-routing tables.
    let dense = weights
        .to_dense(&candle_core::Device::Cpu)?
        .to_dtype(candle_core::DType::F32)?
        .to_device(device)?;
    #[cfg(all(test, target_os = "macos"))]
    crate::deepseek4_gguf::observe_maya_dense_copy(weights, &dense)?;
    Ok(dense)
}

#[cfg(target_os = "macos")]
fn dense_prefill_bounded(input: &Tensor, weight: &Tensor, k: usize, n: usize) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::{DType, MetalStorage, Storage};
    use candle_metal_kernels::{GemmDType, Kernels, call_mlx_gemm};
    use objc2_metal::MTLResource;
    use std::collections::HashMap;
    use std::sync::{LazyLock, Mutex};
    static KERNELS: LazyLock<Mutex<HashMap<candle_core::metal_backend::DeviceId, Kernels>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    anyhow::ensure!(
        input.dtype() == DType::F32
            && weight.dtype() == DType::F32
            && input.is_contiguous()
            && weight.is_contiguous()
            && input.dims() == [input.dim(0)?, k]
            && weight.dims() == [n, k]
            && input.device().same_device(weight.device()),
        "bounded dense prefill shape/device"
    );
    let m = input.dim(0)?;
    anyhow::ensure!(
        [m, n, k].iter().all(|&v| v > 0 && v <= i32::MAX as usize),
        "bounded dense prefill requires positive MLX i32 dimensions"
    );
    let elements = m.checked_mul(n).context("dense output overflow")?;
    let output_bytes = elements
        .checked_mul(4)
        .context("dense output byte overflow")?;
    let input_elements = m.checked_mul(k).context("dense input overflow")?;
    let weight_elements = n.checked_mul(k).context("dense weight overflow")?;
    let (lhs, ll) = input.storage_and_layout();
    let (rhs, rl) = weight.storage_and_layout();
    let (Storage::Metal(lhs), Storage::Metal(rhs)) = (&*lhs, &*rhs) else {
        anyhow::bail!("bounded dense prefill requires Metal");
    };
    let device = lhs.device();
    anyhow::ensure!(
        lhs.dtype() == DType::F32
            && rhs.dtype() == DType::F32
            && rhs.device().id() == device.id()
            && std::ptr::eq(
                lhs.buffer().as_ref().device().as_ref(),
                device.metal_device().as_ref()
            )
            && std::ptr::eq(
                rhs.buffer().as_ref().device().as_ref(),
                device.metal_device().as_ref()
            ),
        "bounded dense actual storage device/dtype mismatch"
    );
    let lhs_offset = grouped_iq1_span(ll, DType::F32, input_elements, lhs.buffer().length())?;
    let rhs_offset = grouped_iq1_span(rl, DType::F32, weight_elements, rhs.buffer().length())?;
    let output = device.new_private_buffer(elements, DType::F32, "dinf-dense-prefill")?;
    let output_buffer: &candle_metal_kernels::metal::Buffer = &output;
    anyhow::ensure!(
        output.length() >= output_bytes
            && std::ptr::eq(
                output_buffer.as_ref().device().as_ref(),
                device.metal_device().as_ref()
            ),
        "bounded dense output span/device mismatch"
    );
    let mut kernels = KERNELS
        .lock()
        .map_err(|_| anyhow::anyhow!("dense kernel cache poisoned"))?;
    let kernels = kernels.entry(device.id()).or_default();
    let encoder = device.command_encoder()?;
    call_mlx_gemm(
        device.metal_device(),
        encoder.as_ref(),
        kernels,
        GemmDType::F32,
        (1, m, n, k),
        ll.stride(),
        lhs_offset,
        lhs.buffer(),
        &[1, k],
        rhs_offset,
        rhs.buffer(),
        &output,
    )?;
    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(
            output,
            device.clone(),
            elements,
            DType::F32,
        )),
        candle_core::Shape::from((m, n)),
    )))
}

fn is_dense(dtype: DeepSeek4DType) -> bool {
    matches!(
        dtype,
        DeepSeek4DType::F32 | DeepSeek4DType::F16 | DeepSeek4DType::BF16 | DeepSeek4DType::I32
    )
}

#[cfg(target_os = "macos")]
mod metal {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use candle_core::backend::BackendStorage;
    use candle_core::{
        CpuStorage, CustomOp1, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Storage,
        Tensor,
    };
    use candle_metal_kernels::metal::{ComputePipeline, Library};
    use objc2_metal::{MTLCompileOptions, MTLComputePipelineState, MTLLanguageVersion, MTLSize};

    use super::DeepSeek4DType;

    pub(super) fn iq1_reuse_base_msl() -> String {
        format!(
            "{}\n{}\n{}",
            candle_metal_kernels::source::QUANTIZED,
            EXTRA_MSL,
            include_str!("deepseek4_q8_rows_seq4.metal")
        )
    }

    const EXTRA_MSL: &str = r#"
using namespace metal;

typedef struct {
    uint8_t e;
    uint8_t qs[16];
} block_mxfp4;

static inline float dsv4_e8m0_to_fp32(uint8_t x) {
    // MXFP4 stores an E8M0 scale and the packed E2M1 values are doubled.
    // This matches GGML_E8M0_TO_FP32_HALF in llama.cpp.
    uint32_t bits = x < 2 ? (0x00200000 << x) : (uint32_t) (x - 1) << 23;
    return as_type<float>(bits);
}

constant float dsv4_mxfp4_values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

template <typename type4x4>
void dsv4_dequantize_mxfp4(
        device const block_mxfp4 * block,
        short il,
        thread type4x4 & values) {
    device const uint8_t * q = block->qs;
    const float scale = dsv4_e8m0_to_fp32(block->e);
    const uint8_t shift = il >= 1 ? 4 : 0;
    for (int i = 0; i < 16; ++i) {
        values[i / 4][i % 4] =
            scale * dsv4_mxfp4_values[(q[i] >> shift) & 0x0f];
    }
}

template<typename block_q>
struct dsv4_decode_scratch {
    typedef half4x4 type;
};

// Native IQ1_S experts decode in F32. A half intermediate both rounds legal
// weights and overflows for finite block scales; keep other formats unchanged.
template<>
struct dsv4_decode_scratch<block_iq1_s> {
    typedef float4x4 type;
};

// Q4_K retains finite decoded F32 weights without a half register boundary.
template<>
struct dsv4_decode_scratch<block_q4_K> {
    typedef float4x4 type;
};

// Q4_K decoder adapted from Candle's MIT-licensed quantized.metal.
// Preserve its layout and arithmetic, but promote before dividing the scale:
// half division by 16 loses finite subnormal scales for the upper nibbles.
template <typename type4x4>
void dsv4_dequantize_q4_K(device const block_q4_K *xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? float(xb->d) : float(xb->d) / 16.0f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il<2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}


template<
    typename block_q,
    short nl,
    void (*dequantize)(device const block_q *, short, thread typename dsv4_decode_scratch<block_q>::type &),
    bool use_simd = false,
    bool decode_rows = false>
kernel void dinf_dsv4_qmatmul(
        device const uchar * weights [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant uint & k_dim [[buffer(3)]],
        constant uint & out_dim [[buffer(4)]],
        constant uint & total_rows [[buffer(5)]],
        constant uint & weight_row_bytes [[buffer(6)]],
        uint gid [[thread_position_in_grid]],
        uint group [[threadgroup_position_in_grid]],
        ushort simd_group [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    if (use_simd) {
        // Four independent outputs per threadgroup. A whole SIMD group
        // takes the same early exit, so simd_sum never has inactive lanes.
        gid = group * 4u + uint(simd_group);
    }
    if (gid >= total_rows * out_dim) {
        return;
    }
    // Adjacent workgroups reuse a weight row across independent requests.
    const uint row = decode_rows ? gid / total_rows : gid % out_dim;
    const uint token = decode_rows ? gid % total_rows : gid / out_dim;
    device const block_q * weight_row =
        (device const block_q *)(weights + size_t(row) * weight_row_bytes);
    const uint blocks = k_dim / (uint(nl) * 16u);
    float sum = 0.0f;
    if (use_simd) {
        // Each chunk decodes 16 adjacent weights. Distributing chunks (rather
        // than whole quant blocks) also fills lanes for narrow expert matrices.
        for (uint chunk = uint(lane); chunk < blocks * uint(nl); chunk += 32u) {
            const uint block_index = chunk / uint(nl);
            const short il = short(chunk % uint(nl));
            typename dsv4_decode_scratch<block_q>::type values;
            dequantize(weight_row + block_index, il, values);
            const uint base = block_index * uint(nl) * 16u + uint(il) * 16u;
            for (uint i = 0; i < 16; ++i) {
                sum += float(values[i / 4][i % 4]) * input[token * k_dim + base + i];
            }
        }
        sum = simd_sum(sum);
    } else {
        // Preserve the original loop and arithmetic as the reference/fallback.
        for (uint block_index = 0; block_index < blocks; ++block_index) {
            for (short il = 0; il < nl; ++il) {
                typename dsv4_decode_scratch<block_q>::type values;
                dequantize(weight_row + block_index, il, values);
                const uint base = block_index * uint(nl) * 16u + uint(il) * 16u;
                for (uint i = 0; i < 16; ++i) {
                    sum += float(values[i / 4][i % 4]) * input[token * k_dim + base + i];
                }
            }
        }
    }
    if (!use_simd || lane == 0) {
        output[decode_rows ? token * out_dim + row : gid] = sum;
    }
}

// One SIMD group owns one weight row and four separate token sums.
// Each sum uses precisely the scalar SIMD lane/chunk order; only the decode
// and weight loads are shared. Tail guards stay uniform across all 32 lanes.
template<typename block_q, short nl,
    void (*dequantize)(device const block_q *, short, thread typename dsv4_decode_scratch<block_q>::type &)>
kernel void dinf_dsv4_reuse4(
        device const uchar * weights [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant uint & k_dim [[buffer(3)]],
        constant uint & out_dim [[buffer(4)]],
        constant uint & total_rows [[buffer(5)]],
        constant uint & weight_row_bytes [[buffer(6)]],
        uint group [[threadgroup_position_in_grid]],
        ushort simd_group [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint row_groups = (out_dim + 3u) / 4u;
    const uint row = (group % row_groups) * 4u + uint(simd_group);
    const uint token = (group / row_groups) * 4u;
    if (row >= out_dim || token >= total_rows) return;
    device const block_q * weight_row =
        (device const block_q *)(weights + size_t(row) * weight_row_bytes);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (uint chunk = uint(lane); chunk < k_dim / 16u; chunk += 32u) {
        const uint block_index = chunk / uint(nl);
        const short il = short(chunk % uint(nl));
        typename dsv4_decode_scratch<block_q>::type values;
        dequantize(weight_row + block_index, il, values);
        const uint base = chunk * 16u;
        for (uint i = 0; i < 16; ++i) {
            const float w = float(values[i / 4][i % 4]);
            s0 += w * input[token * k_dim + base + i];
            if (token + 1u < total_rows) s1 += w * input[(token + 1u) * k_dim + base + i];
            if (token + 2u < total_rows) s2 += w * input[(token + 2u) * k_dim + base + i];
            if (token + 3u < total_rows) s3 += w * input[(token + 3u) * k_dim + base + i];
        }
    }
    s0 = simd_sum(s0); s1 = simd_sum(s1); s2 = simd_sum(s2); s3 = simd_sum(s3);
    if (lane == 0) {
        output[token * out_dim + row] = s0;
        if (token + 1u < total_rows) output[(token + 1u) * out_dim + row] = s1;
        if (token + 2u < total_rows) output[(token + 2u) * out_dim + row] = s2;
        if (token + 3u < total_rows) output[(token + 3u) * out_dim + row] = s3;
    }
}

typedef decltype(dinf_dsv4_reuse4<block_q5_K, 16, dequantize_q5_K>) dsv4_reuse4_fn;
template [[host_name("dinf_dsv4_q5k_reuse4")]] kernel dsv4_reuse4_fn
    dinf_dsv4_reuse4<block_q5_K, 16, dequantize_q5_K>;
template [[host_name("dinf_dsv4_iq1_m_reuse4")]] kernel dsv4_reuse4_fn
    dinf_dsv4_reuse4<block_iq1_m, 16, dequantize_iq1_m>;
template [[host_name("dinf_dsv4_iq2_xxs_reuse4")]] kernel dsv4_reuse4_fn
    dinf_dsv4_reuse4<block_iq2_xxs, 16, dequantize_iq2_xxs>;
template [[host_name("dinf_dsv4_iq2_xs_reuse4")]] kernel dsv4_reuse4_fn
    dinf_dsv4_reuse4<block_iq2_xs, 16, dequantize_iq2_xs>;

typedef decltype(dinf_dsv4_qmatmul<block_iq2_xs, 16, dequantize_iq2_xs>) dsv4_iq2_xs_fn;
typedef decltype(dinf_dsv4_qmatmul<block_iq3_xxs, 16, dequantize_iq3_xxs>) dsv4_iq3_xxs_fn;
typedef decltype(dinf_dsv4_qmatmul<block_iq3_s, 16, dequantize_iq3_s>) dsv4_iq3_s_fn;
typedef decltype(dinf_dsv4_qmatmul<block_iq1_m, 16, dequantize_iq1_m>) dsv4_iq1_m_fn;
// IQ1_S uses the vendored decoder: 16 chunks of 16 values per 50-byte block.
typedef decltype(dinf_dsv4_qmatmul<block_iq1_s, 16, dequantize_iq1_s>) dsv4_iq1_s_fn;
typedef decltype(dinf_dsv4_qmatmul<block_q6_K, 16, dequantize_q6_K>) dsv4_q6k_fn;
typedef decltype(dinf_dsv4_qmatmul<block_q8_0, 2, dequantize_q8_0>) dsv4_q8_0_fn;
typedef decltype(dinf_dsv4_qmatmul<block_mxfp4, 2, dsv4_dequantize_mxfp4>) dsv4_mxfp4_fn;

typedef decltype(dinf_dsv4_qmatmul<block_iq2_xxs, 16, dequantize_iq2_xxs>) dsv4_iq2_xxs_fn;
typedef decltype(dinf_dsv4_qmatmul<block_q5_K, 16, dequantize_q5_K>) dsv4_q5k_fn;
typedef decltype(dinf_dsv4_qmatmul<block_q4_K, 16, dsv4_dequantize_q4_K>) dsv4_q4k_fn;
template [[host_name("dinf_dsv4_q4k")]] kernel dsv4_q4k_fn
    dinf_dsv4_qmatmul<block_q4_K, 16, dsv4_dequantize_q4_K>;
template [[host_name("dinf_dsv4_q4k_simd")]] kernel dsv4_q4k_fn
    dinf_dsv4_qmatmul<block_q4_K, 16, dsv4_dequantize_q4_K, true>;
template [[host_name("dinf_dsv4_q4k_rows")]] kernel dsv4_q4k_fn
    dinf_dsv4_qmatmul<block_q4_K, 16, dsv4_dequantize_q4_K, false, true>;
template [[host_name("dinf_dsv4_q4k_simd_rows")]] kernel dsv4_q4k_fn
    dinf_dsv4_qmatmul<block_q4_K, 16, dsv4_dequantize_q4_K, true, true>;

template [[host_name("dinf_dsv4_iq2_xxs")]] kernel dsv4_iq2_xxs_fn
    dinf_dsv4_qmatmul<block_iq2_xxs, 16, dequantize_iq2_xxs>;
template [[host_name("dinf_dsv4_iq2_xxs_simd")]] kernel dsv4_iq2_xxs_fn
    dinf_dsv4_qmatmul<block_iq2_xxs, 16, dequantize_iq2_xxs, true>;
template [[host_name("dinf_dsv4_iq2_xxs_rows")]] kernel dsv4_iq2_xxs_fn
    dinf_dsv4_qmatmul<block_iq2_xxs, 16, dequantize_iq2_xxs, false, true>;
template [[host_name("dinf_dsv4_iq2_xxs_simd_rows")]] kernel dsv4_iq2_xxs_fn
    dinf_dsv4_qmatmul<block_iq2_xxs, 16, dequantize_iq2_xxs, true, true>;
template [[host_name("dinf_dsv4_q5k")]] kernel dsv4_q5k_fn
    dinf_dsv4_qmatmul<block_q5_K, 16, dequantize_q5_K>;
template [[host_name("dinf_dsv4_q5k_simd")]] kernel dsv4_q5k_fn
    dinf_dsv4_qmatmul<block_q5_K, 16, dequantize_q5_K, true>;
template [[host_name("dinf_dsv4_q5k_rows")]] kernel dsv4_q5k_fn
    dinf_dsv4_qmatmul<block_q5_K, 16, dequantize_q5_K, false, true>;
template [[host_name("dinf_dsv4_q5k_simd_rows")]] kernel dsv4_q5k_fn
    dinf_dsv4_qmatmul<block_q5_K, 16, dequantize_q5_K, true, true>;
template [[host_name("dinf_dsv4_iq2_xs")]] kernel dsv4_iq2_xs_fn
    dinf_dsv4_qmatmul<block_iq2_xs, 16, dequantize_iq2_xs>;
template [[host_name("dinf_dsv4_iq3_xxs")]] kernel dsv4_iq3_xxs_fn
    dinf_dsv4_qmatmul<block_iq3_xxs, 16, dequantize_iq3_xxs>;
template [[host_name("dinf_dsv4_iq3_s")]] kernel dsv4_iq3_s_fn
    dinf_dsv4_qmatmul<block_iq3_s, 16, dequantize_iq3_s>;
template [[host_name("dinf_dsv4_iq1_m")]] kernel dsv4_iq1_m_fn
    dinf_dsv4_qmatmul<block_iq1_m, 16, dequantize_iq1_m>;
template [[host_name("dinf_dsv4_iq1_s")]] kernel dsv4_iq1_s_fn
    dinf_dsv4_qmatmul<block_iq1_s, 16, dequantize_iq1_s>;
template [[host_name("dinf_dsv4_q6k")]] kernel dsv4_q6k_fn
    dinf_dsv4_qmatmul<block_q6_K, 16, dequantize_q6_K>;
template [[host_name("dinf_dsv4_q8_0")]] kernel dsv4_q8_0_fn
    dinf_dsv4_qmatmul<block_q8_0, 2, dequantize_q8_0>;
template [[host_name("dinf_dsv4_mxfp4")]] kernel dsv4_mxfp4_fn
    dinf_dsv4_qmatmul<block_mxfp4, 2, dsv4_dequantize_mxfp4>;

template [[host_name("dinf_dsv4_iq2_xs_simd")]] kernel dsv4_iq2_xs_fn
    dinf_dsv4_qmatmul<block_iq2_xs, 16, dequantize_iq2_xs, true>;
template [[host_name("dinf_dsv4_iq3_xxs_simd")]] kernel dsv4_iq3_xxs_fn
    dinf_dsv4_qmatmul<block_iq3_xxs, 16, dequantize_iq3_xxs, true>;
template [[host_name("dinf_dsv4_iq3_s_simd")]] kernel dsv4_iq3_s_fn
    dinf_dsv4_qmatmul<block_iq3_s, 16, dequantize_iq3_s, true>;
template [[host_name("dinf_dsv4_iq1_m_simd")]] kernel dsv4_iq1_m_fn
    dinf_dsv4_qmatmul<block_iq1_m, 16, dequantize_iq1_m, true>;
template [[host_name("dinf_dsv4_iq1_s_simd")]] kernel dsv4_iq1_s_fn
    dinf_dsv4_qmatmul<block_iq1_s, 16, dequantize_iq1_s, true>;
template [[host_name("dinf_dsv4_q6k_simd")]] kernel dsv4_q6k_fn
    dinf_dsv4_qmatmul<block_q6_K, 16, dequantize_q6_K, true>;
template [[host_name("dinf_dsv4_q8_0_simd")]] kernel dsv4_q8_0_fn
    dinf_dsv4_qmatmul<block_q8_0, 2, dequantize_q8_0, true>;
template [[host_name("dinf_dsv4_mxfp4_simd")]] kernel dsv4_mxfp4_fn
    dinf_dsv4_qmatmul<block_mxfp4, 2, dsv4_dequantize_mxfp4, true>;

template [[host_name("dinf_dsv4_iq2_xs_rows")]] kernel dsv4_iq2_xs_fn
    dinf_dsv4_qmatmul<block_iq2_xs, 16, dequantize_iq2_xs, false, true>;
template [[host_name("dinf_dsv4_iq3_xxs_rows")]] kernel dsv4_iq3_xxs_fn
    dinf_dsv4_qmatmul<block_iq3_xxs, 16, dequantize_iq3_xxs, false, true>;
template [[host_name("dinf_dsv4_iq3_s_rows")]] kernel dsv4_iq3_s_fn
    dinf_dsv4_qmatmul<block_iq3_s, 16, dequantize_iq3_s, false, true>;
template [[host_name("dinf_dsv4_iq1_m_rows")]] kernel dsv4_iq1_m_fn
    dinf_dsv4_qmatmul<block_iq1_m, 16, dequantize_iq1_m, false, true>;
template [[host_name("dinf_dsv4_iq1_s_rows")]] kernel dsv4_iq1_s_fn
    dinf_dsv4_qmatmul<block_iq1_s, 16, dequantize_iq1_s, false, true>;
template [[host_name("dinf_dsv4_q6k_rows")]] kernel dsv4_q6k_fn
    dinf_dsv4_qmatmul<block_q6_K, 16, dequantize_q6_K, false, true>;
template [[host_name("dinf_dsv4_q8_0_rows")]] kernel dsv4_q8_0_fn
    dinf_dsv4_qmatmul<block_q8_0, 2, dequantize_q8_0, false, true>;
template [[host_name("dinf_dsv4_mxfp4_rows")]] kernel dsv4_mxfp4_fn
    dinf_dsv4_qmatmul<block_mxfp4, 2, dsv4_dequantize_mxfp4, false, true>;

template [[host_name("dinf_dsv4_iq2_xs_simd_rows")]] kernel dsv4_iq2_xs_fn
    dinf_dsv4_qmatmul<block_iq2_xs, 16, dequantize_iq2_xs, true, true>;
template [[host_name("dinf_dsv4_iq3_xxs_simd_rows")]] kernel dsv4_iq3_xxs_fn
    dinf_dsv4_qmatmul<block_iq3_xxs, 16, dequantize_iq3_xxs, true, true>;
template [[host_name("dinf_dsv4_iq3_s_simd_rows")]] kernel dsv4_iq3_s_fn
    dinf_dsv4_qmatmul<block_iq3_s, 16, dequantize_iq3_s, true, true>;
template [[host_name("dinf_dsv4_iq1_m_simd_rows")]] kernel dsv4_iq1_m_fn
    dinf_dsv4_qmatmul<block_iq1_m, 16, dequantize_iq1_m, true, true>;
template [[host_name("dinf_dsv4_iq1_s_simd_rows")]] kernel dsv4_iq1_s_fn
    dinf_dsv4_qmatmul<block_iq1_s, 16, dequantize_iq1_s, true, true>;
template [[host_name("dinf_dsv4_q6k_simd_rows")]] kernel dsv4_q6k_fn
    dinf_dsv4_qmatmul<block_q6_K, 16, dequantize_q6_K, true, true>;
template [[host_name("dinf_dsv4_q8_0_simd_rows")]] kernel dsv4_q8_0_fn
    dinf_dsv4_qmatmul<block_q8_0, 2, dequantize_q8_0, true, true>;
template [[host_name("dinf_dsv4_mxfp4_simd_rows")]] kernel dsv4_mxfp4_fn
    dinf_dsv4_qmatmul<block_mxfp4, 2, dsv4_dequantize_mxfp4, true, true>;

// audreyt/ds4 0dff39f91721bcf3472813c6d2bebf9353df2a1a,
// metal/dense.metal:87-125,285-368. Two outputs share one activation load;
// each independent token uses the same decode tree, never half tiled prefill.
template <ushort NSG>
kernel void dinf_dsv41_q8_f32(
        device const uchar *weights [[buffer(0)]],
        device const float *input [[buffer(1)]],
        device float *output [[buffer(2)]],
        constant uint &k_dim [[buffer(3)]],
        constant uint &out_dim [[buffer(4)]],
        constant uint &total_rows [[buffer(5)]],
        constant uint &weight_row_bytes [[buffer(6)]],
        uint group [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[2][32];
    const uint groups_per_token = out_dim / 2u + out_dim % 2u;
    const uint token = group / groups_per_token;
    const uint row0 = (group % groups_per_token) * 2u;
    const uint ix = lane / 4u, il = lane % 4u;
    float sum[2] = {0.0f, 0.0f};
    for (uint ib = uint(sg) * 8u + ix; ib < k_dim / 32u; ib += NSG * 8u) {
        float yl[8];
        for (ushort i = 0; i < 8; ++i)
            yl[i] = input[token * k_dim + ib * 32u + il * 8u + i];
        for (ushort row = 0; row < 2 && row0 + row < out_dim; ++row) {
            device const block_q8_0 *blocks =
                (device const block_q8_0 *)(weights + size_t(row0 + row) * weight_row_bytes);
            device const int8_t *qs = blocks[ib].qs + il * 8u;
            float sumq = 0.0f;
            for (ushort i = 0; i < 8; ++i) sumq += float(qs[i]) * yl[i];
            sum[row] += sumq * float(blocks[ib].d);
        }
    }
    for (ushort row = 0; row < 2; ++row) {
        if (sg == 0) partial[row][lane] = 0.0f;
        sum[row] = simd_sum(sum[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort row = 0; row < 2; ++row)
        if (lane == 0) partial[row][sg] = sum[row];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort row = 0; row < 2 && row0 + row < out_dim; ++row) {
        const float total = simd_sum(partial[row][lane]);
        if (lane == 0 && sg == 0) output[token * out_dim + row0 + row] = total;
    }
}
typedef decltype(dinf_dsv41_q8_f32<4>) dsv41_q8_fn;
template [[host_name("dinf_dsv41_q8_f32_nsg4")]] kernel dsv41_q8_fn dinf_dsv41_q8_f32<4>;
template [[host_name("dinf_dsv41_q8_f32_nsg8")]] kernel dsv41_q8_fn dinf_dsv41_q8_f32<8>;

// Grouped output-A: only activation selection differs from NSG4 above.
// Even rows_per_group ensures the two rows always share one input group.
kernel void dinf_dsv41_q8_f32_grouped_nsg4(
        device const uchar *weights [[buffer(0)]],
        device const float *input [[buffer(1)]],
        device float *output [[buffer(2)]],
        constant uint &k_dim [[buffer(3)]],
        constant uint &out_dim [[buffer(4)]],
        constant uint &rows_per_group [[buffer(5)]],
        constant uint &weight_row_bytes [[buffer(6)]],
        uint group [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[2][32];
    const uint row0 = group * 2u;
    const uint token = row0 / rows_per_group;
    const uint ix = lane / 4u, il = lane % 4u;
    float sum[2] = {0.0f, 0.0f};
    for (uint ib = uint(sg) * 8u + ix; ib < k_dim / 32u; ib += 4u * 8u) {
        float yl[8];
        for (ushort i = 0; i < 8; ++i)
            yl[i] = input[token * k_dim + ib * 32u + il * 8u + i];
        for (ushort row = 0; row < 2 && row0 + row < out_dim; ++row) {
            device const block_q8_0 *blocks =
                (device const block_q8_0 *)(weights + size_t(row0 + row) * weight_row_bytes);
            device const int8_t *qs = blocks[ib].qs + il * 8u;
            float sumq = 0.0f;
            for (ushort i = 0; i < 8; ++i) sumq += float(qs[i]) * yl[i];
            sum[row] += sumq * float(blocks[ib].d);
        }
    }
    for (ushort row = 0; row < 2; ++row) {
        if (sg == 0) partial[row][lane] = 0.0f;
        sum[row] = simd_sum(sum[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort row = 0; row < 2; ++row)
        if (lane == 0) partial[row][sg] = sum[row];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort row = 0; row < 2 && row0 + row < out_dim; ++row) {
        const float total = simd_sum(partial[row][lane]);
        if (lane == 0 && sg == 0) output[row0 + row] = total;
    }
}

// Same IQ1_S F32 scratch, lane/chunk loop and SIMD reduction as the separate
// matvec. Only choose an independent packed buffer and destination span.
template<
    typename block_q,
    short nl, bool per_matrix_input,
    void (*dequantize)(device const block_q *, short, thread typename dsv4_decode_scratch<block_q>::type &)>
kernel void dinf_dsv4_iq1_grouped(
        device const uchar *w0 [[buffer(0)]],
        device const uchar *w1 [[buffer(1)]],
        device const uchar *w2 [[buffer(2)]],
        device const uchar *w3 [[buffer(3)]],
        device const uchar *w4 [[buffer(4)]],
        device const uchar *w5 [[buffer(5)]],
        device const uchar *w6 [[buffer(6)]],
        device const uchar *w7 [[buffer(7)]],
        device const uchar *w8 [[buffer(8)]],
        device const uchar *w9 [[buffer(9)]],
        device const uchar *w10 [[buffer(10)]],
        device const uchar *w11 [[buffer(11)]],
        device const float *input [[buffer(12)]],
        device float *output [[buffer(13)]],
        constant uint4 &geometry [[buffer(14)]],
        uint group [[threadgroup_position_in_grid]],
        ushort simd_group [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint k_dim = geometry.x, out_dim = geometry.y;
    const uint weight_row_bytes = geometry.z;
    const uint gid = group * 4u + uint(simd_group);
    if (gid >= geometry.w * out_dim) return;
    const uint slot = gid / out_dim;
    device const uchar *weights = w0;
    switch (slot) {
        case 1: weights = w1; break;
        case 2: weights = w2; break;
        case 3: weights = w3; break;
        case 4: weights = w4; break;
        case 5: weights = w5; break;
        case 6: weights = w6; break;
        case 7: weights = w7; break;
        case 8: weights = w8; break;
        case 9: weights = w9; break;
        case 10: weights = w10; break;
        case 11: weights = w11; break;
    }
    const uint row = gid % out_dim;
    const uint token = per_matrix_input ? slot : 0;
    device const block_q * weight_row =
        (device const block_q *)(weights + size_t(row) * weight_row_bytes);
    const uint blocks = k_dim / (uint(nl) * 16u);
    float sum = 0.0f;
    for (uint chunk = uint(lane); chunk < blocks * uint(nl); chunk += 32u) {
        const uint block_index = chunk / uint(nl);
        const short il = short(chunk % uint(nl));
        typename dsv4_decode_scratch<block_q>::type values;
        dequantize(weight_row + block_index, il, values);
        const uint base = block_index * uint(nl) * 16u + uint(il) * 16u;
        for (uint i = 0; i < 16; ++i) {
            sum += float(values[i / 4][i % 4]) * input[token * k_dim + base + i];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[gid] = sum;
}
typedef decltype(dinf_dsv4_iq1_grouped<block_iq1_s, 16, false, dequantize_iq1_s>) dsv4_iq1_grouped_fn;
template [[host_name("dinf_dsv4_iq1_s_grouped_simd")]] kernel dsv4_iq1_grouped_fn
    dinf_dsv4_iq1_grouped<block_iq1_s, 16, false, dequantize_iq1_s>;
typedef decltype(dinf_dsv4_iq1_grouped<block_iq1_s, 16, true, dequantize_iq1_s>) dsv4_iq1_down_fn;
template [[host_name("dinf_dsv4_iq1_s_grouped_down_simd")]] kernel dsv4_iq1_down_fn
    dinf_dsv4_iq1_grouped<block_iq1_s, 16, true, dequantize_iq1_s>;
"#;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub(super) enum KernelKind {
        Serial,
        Simd,
        Reuse4,
        V41Q8Nsg4,
        V41Q8Nsg8,
        V41Q8GroupedNsg4,
        V41Q8RowsSeq4,
        GroupedIq1Simd,
        #[cfg(test)]
        GroupedIq1DownSimd,
    }

    #[derive(Default)]
    struct DevicePipelines {
        library: Option<Library>,
        pipelines: HashMap<(DeepSeek4DType, KernelKind, bool), ComputePipeline>,
    }

    static PIPELINES: OnceLock<
        Mutex<HashMap<candle_core::metal_backend::DeviceId, DevicePipelines>>,
    > = OnceLock::new();

    fn kernel_name(dtype: DeepSeek4DType) -> Result<&'static str> {
        Ok(match dtype {
            DeepSeek4DType::IQ2XXS => "dinf_dsv4_iq2_xxs",
            DeepSeek4DType::Q4K => "dinf_dsv4_q4k",
            DeepSeek4DType::Q5K => "dinf_dsv4_q5k",
            DeepSeek4DType::IQ2XS => "dinf_dsv4_iq2_xs",
            DeepSeek4DType::IQ3XXS => "dinf_dsv4_iq3_xxs",
            DeepSeek4DType::IQ3S => "dinf_dsv4_iq3_s",
            DeepSeek4DType::IQ1M => "dinf_dsv4_iq1_m",
            DeepSeek4DType::IQ1S => "dinf_dsv4_iq1_s",
            DeepSeek4DType::Q6K => "dinf_dsv4_q6k",
            DeepSeek4DType::Q8_0 => "dinf_dsv4_q8_0",
            DeepSeek4DType::MXFP4 => "dinf_dsv4_mxfp4",
            dtype => candle_core::bail!("DeepSeek V4 Metal matmul does not support {dtype:?}"),
        })
    }

    fn pipeline(
        device: &candle_core::MetalDevice,
        dtype: DeepSeek4DType,
        kind: KernelKind,
        decode_rows: bool,
    ) -> Result<ComputePipeline> {
        let key = (dtype, kind, decode_rows);
        let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = pipelines
            .lock()
            .map_err(|_| candle_core::Error::Msg("V4 pipeline cache lock poisoned".into()))?;
        let entry = cache.entry(device.id()).or_default();
        if let Some(pipeline) = entry.pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        let name = kernel_name(dtype)?;
        if entry.library.is_none() {
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version4_0);
            let source = format!("{}\n{}", candle_metal_kernels::source::QUANTIZED, EXTRA_MSL);
            let source = format!("{source}\n{}", include_str!("deepseek4_q8_rows_seq4.metal"));
            entry.library = Some(
                device
                    .metal_device()
                    .new_library_with_source(&source, Some(&options))
                    .map_err(candle_core::Error::wrap)?,
            );
        }
        let mut name = match kind {
            KernelKind::Serial => name.to_owned(),
            KernelKind::Simd => format!("{name}_simd"),
            KernelKind::Reuse4 => format!("{name}_reuse4"),
            KernelKind::V41Q8Nsg4 => "dinf_dsv41_q8_f32_nsg4".to_owned(),
            KernelKind::V41Q8Nsg8 => "dinf_dsv41_q8_f32_nsg8".to_owned(),
            KernelKind::V41Q8GroupedNsg4 => "dinf_dsv41_q8_f32_grouped_nsg4".to_owned(),
            KernelKind::V41Q8RowsSeq4 => "dinf_dsv41_q8_f32_rows_seq4".to_owned(),
            KernelKind::GroupedIq1Simd => "dinf_dsv4_iq1_s_grouped_simd".to_owned(),
            #[cfg(test)]
            KernelKind::GroupedIq1DownSimd => "dinf_dsv4_iq1_s_grouped_down_simd".to_owned(),
        };
        if decode_rows && kind != KernelKind::Reuse4 {
            name.push_str("_rows");
        }
        let function = entry
            .library
            .as_ref()
            .unwrap()
            .get_function(&name, None)
            .map_err(candle_core::Error::wrap)?;
        let pipeline = device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(candle_core::Error::wrap)?;
        entry.pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    struct Dsv4QMatMul {
        dtype: DeepSeek4DType,
        out_dim: usize,
        in_dim: usize,
        kind: KernelKind,
        decode_rows: bool,
        grouped: Option<super::GroupedV41Geometry>,
    }

    struct Geometry {
        in_dim: u32,
        out_dim: u32,
        rows: u32,
        row_bytes: u32,
        weight_bytes: usize,
        input_elements: usize,
        output_elements: usize,
    }

    impl Geometry {
        fn new(
            dtype: DeepSeek4DType,
            out_dim: usize,
            in_dim: usize,
            input_dims: &[usize],
        ) -> Result<Self> {
            if input_dims.len() < 2 || input_dims.last() != Some(&in_dim) {
                candle_core::bail!("V4 input shape does not match weight width");
            }
            let weight_bytes =
                super::weight_byte_len(dtype, out_dim, in_dim).map_err(candle_core::Error::wrap)?;
            let rows = input_dims[..input_dims.len() - 1]
                .iter()
                .try_fold(1usize, |count, dim| count.checked_mul(*dim))
                .ok_or_else(|| candle_core::Error::Msg("V4 input row count overflow".into()))?;
            if rows == 0 {
                candle_core::bail!("V4 packed matmul input rows are empty");
            }
            let input_elements = rows
                .checked_mul(in_dim)
                .ok_or_else(|| candle_core::Error::Msg("V4 input element count overflow".into()))?;
            let output_elements = rows.checked_mul(out_dim).ok_or_else(|| {
                candle_core::Error::Msg("V4 output element count overflow".into())
            })?;
            // The kernel uses uint activation/output indices. Check products,
            // not only dimensions, before converting parameters or allocating.
            let checked_u32 = |v| {
                u32::try_from(v).map_err(|_| {
                    candle_core::Error::Msg("V4 matmul exceeds Metal uint indexing".into())
                })
            };
            checked_u32(input_elements)?;
            checked_u32(output_elements)?;
            Ok(Self {
                in_dim: checked_u32(in_dim)?,
                out_dim: checked_u32(out_dim)?,
                rows: checked_u32(rows)?,
                row_bytes: checked_u32(weight_bytes / out_dim)?,
                weight_bytes,
                input_elements,
                output_elements,
            })
        }
    }

    fn buffer_offset(storage: &MetalStorage, layout: &Layout, elements: usize) -> Result<usize> {
        if !layout.is_contiguous() || layout.shape().elem_count() != elements {
            candle_core::bail!("V4 matmul storage layout does not match its shape");
        }
        let size = storage.dtype().size_in_bytes();
        let offset = layout
            .start_offset()
            .checked_mul(size)
            .ok_or_else(|| candle_core::Error::Msg("V4 matmul buffer offset overflow".into()))?;
        let end = elements
            .checked_mul(size)
            .and_then(|bytes| offset.checked_add(bytes))
            .ok_or_else(|| candle_core::Error::Msg("V4 matmul buffer span overflow".into()))?;
        if end > storage.buffer().length() {
            candle_core::bail!("V4 matmul tensor exceeds its Metal buffer");
        }
        Ok(offset)
    }

    impl CustomOp2 for Dsv4QMatMul {
        fn name(&self) -> &'static str {
            "dinf-deepseek4-fused-qmatmul"
        }

        fn cpu_fwd(
            &self,
            _weights: &CpuStorage,
            _weights_layout: &Layout,
            _input: &CpuStorage,
            _input_layout: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("DeepSeek V4 fused matmul is a Metal op")
        }

        fn metal_fwd(
            &self,
            weights: &MetalStorage,
            weights_layout: &Layout,
            input: &MetalStorage,
            input_layout: &Layout,
        ) -> Result<(MetalStorage, Shape)> {
            if weights.dtype() != DType::U8 {
                candle_core::bail!("V4 weights must be u8 bytes");
            }
            if input.dtype() != DType::F32 {
                candle_core::bail!("V4 input must be f32");
            }
            if !weights_layout.is_contiguous() {
                candle_core::bail!("V4 weights must be contiguous");
            }
            if !input_layout.is_contiguous() {
                candle_core::bail!("V4 input must be contiguous");
            }
            if weights.device().id() != input.device().id() {
                candle_core::bail!("V4 weights and input must share a Metal device");
            }
            let geometry = if let Some(grouped) = &self.grouped {
                Geometry {
                    in_dim: grouped.in_dim,
                    out_dim: grouped.out_dim,
                    rows: grouped.rows_per_group,
                    row_bytes: grouped.row_bytes,
                    weight_bytes: grouped.weight_bytes,
                    input_elements: grouped.input_elements,
                    output_elements: grouped.out_dim as usize,
                }
            } else {
                Geometry::new(self.dtype, self.out_dim, self.in_dim, input_layout.dims())?
            };
            if self.kind == KernelKind::V41Q8RowsSeq4
                && (self.dtype != DeepSeek4DType::Q8_0
                    || self.grouped.is_some()
                    || self.decode_rows
                    || input_layout.dims().len() != 2
                    || !(matches!(geometry.in_dim, 2304 | 5120)
                        || matches!(
                            (geometry.in_dim, geometry.out_dim),
                            (1280, 32768) | (8192, 5120)
                        ))
                    || !(2..=32).contains(&geometry.rows)
                    || geometry.out_dim > 65_536)
            {
                candle_core::bail!("unsupported V4.1 exact rows_seq4 geometry");
            }
            if self.kind == KernelKind::Reuse4
                && (!matches!(
                    self.dtype,
                    DeepSeek4DType::Q5K
                        | DeepSeek4DType::IQ1M
                        | DeepSeek4DType::IQ2XXS
                        | DeepSeek4DType::IQ2XS
                ) || self.grouped.is_some()
                    || input_layout.dims().len() != 2
                    || geometry.in_dim < 1024
                    || !(2..=128).contains(&geometry.rows))
            {
                candle_core::bail!("unsupported packed reuse4 geometry");
            }
            let weights_offset = buffer_offset(weights, weights_layout, geometry.weight_bytes)?;
            let input_offset = buffer_offset(input, input_layout, geometry.input_elements)?;
            if self.kind == KernelKind::V41Q8RowsSeq4 && weights_offset % 2 != 0 {
                candle_core::bail!("V4.1 rows_seq4 requires aligned Q8 blocks");
            }

            let output_shape = if self.grouped.is_some() {
                Shape::from((1, self.out_dim))
            } else {
                let mut output_dims = input_layout.dims().to_vec();
                *output_dims.last_mut().unwrap() = self.out_dim;
                Shape::from(output_dims)
            };
            let output_elements = geometry.output_elements;
            let device = input.device();
            // New explicit matrix consumers retain fallback outputs too. Their
            // admitted owner cannot inherit an arbitrarily larger pool bucket.
            let output = if super::bounded_prefill_output(device) {
                device.new_private_buffer(
                    output_elements,
                    DType::F32,
                    "dinf-deepseek4-fused-qmatmul",
                )?
            } else {
                device
                    .new_buffer_builder()
                    .with_size_for(output_elements, DType::F32)
                    .with_label("dinf-deepseek4-fused-qmatmul")
                    .build()?
            };
            let pipeline = pipeline(device, self.dtype, self.kind, self.decode_rows)?;
            let (threads, groups) = match self.kind {
                KernelKind::Serial => {
                    let threads = 256usize.min(pipeline.max_total_threads_per_threadgroup());
                    (threads, output_elements.div_ceil(threads))
                }
                KernelKind::Simd | KernelKind::GroupedIq1Simd => {
                    if pipeline.as_ref().threadExecutionWidth() != 32
                        || pipeline.max_total_threads_per_threadgroup() < 128
                    {
                        candle_core::bail!(
                            "V4 SIMD matmul requires 32-lane SIMD groups and 128-thread groups"
                        );
                    }
                    (128, output_elements.div_ceil(4))
                }
                #[cfg(test)]
                KernelKind::GroupedIq1DownSimd => {
                    candle_core::bail!("grouped IQ1 down requires the checked grouped custom op");
                }
                KernelKind::Reuse4 => {
                    if pipeline.as_ref().threadExecutionWidth() != 32
                        || pipeline.max_total_threads_per_threadgroup() < 128
                    {
                        candle_core::bail!(
                            "packed reuse4 requires 32-lane SIMD and 128-thread groups"
                        );
                    }
                    (
                        128,
                        self.out_dim.div_ceil(4) * (geometry.rows as usize).div_ceil(4),
                    )
                }
                KernelKind::V41Q8RowsSeq4 => {
                    if pipeline.as_ref().threadExecutionWidth() != 32
                        || pipeline.max_total_threads_per_threadgroup() < 128
                    {
                        candle_core::bail!("V4.1 rows_seq4 requires 32-lane SIMD groups");
                    }
                    (
                        128,
                        self.out_dim.div_ceil(4) * (geometry.rows as usize).div_ceil(8),
                    )
                }
                KernelKind::V41Q8Nsg4 | KernelKind::V41Q8Nsg8 | KernelKind::V41Q8GroupedNsg4 => {
                    let nsg = if self.kind == KernelKind::V41Q8Nsg8 {
                        8
                    } else {
                        4
                    };
                    let threads = nsg * 32;
                    if pipeline.as_ref().threadExecutionWidth() != 32
                        || pipeline.max_total_threads_per_threadgroup() < threads
                    {
                        candle_core::bail!("V4.1 Q8 matvec requires 32-lane SIMD groups");
                    }
                    let groups = if self.grouped.is_some() {
                        self.out_dim / 2
                    } else {
                        self.out_dim.div_ceil(2) * geometry.rows as usize
                    };
                    (threads, groups)
                }
            };
            let encoder = device.command_encoder()?;
            let encoder_ref = encoder.as_ref();
            encoder_ref.set_compute_pipeline_state(&pipeline);
            encoder_ref.set_input_buffer(0, Some(weights.buffer()), weights_offset);
            encoder_ref.set_input_buffer(1, Some(input.buffer()), input_offset);
            encoder_ref.set_output_buffer(2, Some(&output), 0);
            encoder_ref.set_bytes(3, &geometry.in_dim);
            encoder_ref.set_bytes(4, &geometry.out_dim);
            encoder_ref.set_bytes(5, &geometry.rows);
            encoder_ref.set_bytes(6, &geometry.row_bytes);
            encoder_ref.dispatch_thread_groups(
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
                MetalStorage::new(output, device.clone(), output_elements, DType::F32),
                output_shape,
            ))
        }
    }

    pub(super) fn forward(
        weights: &Tensor,
        input: &Tensor,
        dtype: DeepSeek4DType,
        out_dim: usize,
        in_dim: usize,
        decode_rows: bool,
    ) -> Result<Tensor> {
        // Full-model output, slice, chunk and reuse checks pass on Apple
        // Silicon. Keep an explicit opt-out and leave other Macs unchanged.
        let configured = std::env::var_os("DINF_DEEPSEEK4_SIMD");
        let enabled = simd_enabled(configured.as_deref(), cfg!(target_arch = "aarch64"));
        let elements = if decode_rows {
            in_dim
        } else {
            input.elem_count()
        };
        let reuse4 = decode_rows
            && enabled
            && super::PACKED_REUSE4_ENABLED.with(|v| v.get())
            && in_dim >= 1024
            && input.dims().len() == 2
            && (2..=128).contains(&input.dims()[0])
            && matches!(
                dtype,
                DeepSeek4DType::Q5K
                    | DeepSeek4DType::IQ1M
                    | DeepSeek4DType::IQ2XXS
                    | DeepSeek4DType::IQ2XS
            );
        let kind = if reuse4 {
            KernelKind::Reuse4
        } else {
            select_kernel(in_dim, elements, enabled)
        };
        forward_with_mode(weights, input, dtype, out_dim, in_dim, kind, decode_rows)
    }

    fn select_kernel(in_dim: usize, input_elements: usize, enabled: bool) -> KernelKind {
        // Large multi-token matrices regressed in the benchmark. Leave
        // multi-row and narrow projections on the original kernel even when opted in.
        if enabled && in_dim >= 1024 && input_elements == in_dim {
            KernelKind::Simd
        } else {
            KernelKind::Serial
        }
    }

    pub(super) fn simd_enabled(configured: Option<&std::ffi::OsStr>, apple_silicon: bool) -> bool {
        configured.map_or(apple_silicon, |value| value == "1")
    }

    #[cfg(test)]
    pub(super) fn forward_with_kernel(
        weights: &Tensor,
        input: &Tensor,
        dtype: DeepSeek4DType,
        out_dim: usize,
        in_dim: usize,
        kind: KernelKind,
    ) -> Result<Tensor> {
        forward_with_mode(weights, input, dtype, out_dim, in_dim, kind, false)
    }

    pub(super) fn forward_with_mode(
        weights: &Tensor,
        input: &Tensor,
        dtype: DeepSeek4DType,
        out_dim: usize,
        in_dim: usize,
        kind: KernelKind,
        decode_rows: bool,
    ) -> Result<Tensor> {
        if kind == KernelKind::V41Q8RowsSeq4 && !input.is_contiguous() {
            candle_core::bail!("V4.1 rows_seq4 requires contiguous row-major input");
        }
        weights.apply_op2_no_bwd(
            &input.contiguous()?,
            &Dsv4QMatMul {
                dtype,
                out_dim,
                in_dim,
                kind,
                decode_rows,
                grouped: None,
            },
        )
    }

    // Bounded Arc clones keep the Tensor storage owners alive while apply_op1
    // encodes. These are not a completion lease; the caller supplies that lease.
    struct GroupedIq1 {
        weights: [Option<std::sync::Arc<crate::deepseek4_gguf::RawTensor>>; super::GROUPED_IQ1_MAX],
        geometry: super::GroupedIq1Geometry,
    }

    impl CustomOp1 for GroupedIq1 {
        fn name(&self) -> &'static str {
            "dinf-deepseek4-grouped-iq1"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("grouped IQ1 requires the Metal backend")
        }

        fn metal_fwd(
            &self,
            input: &MetalStorage,
            layout: &Layout,
        ) -> Result<(MetalStorage, Shape)> {
            let geometry = &self.geometry;
            let device = input.device();
            if input.dtype() != DType::F32
                || layout.dims() != [geometry.input_rows, geometry.in_dim]
            {
                candle_core::bail!("grouped IQ1 input dtype/geometry mismatch");
            }
            let input_offset = super::grouped_iq1_span(
                layout,
                DType::F32,
                geometry
                    .input_rows
                    .checked_mul(geometry.in_dim)
                    .ok_or_else(|| candle_core::Error::Msg("grouped IQ1 input overflow".into()))?,
                input.buffer().length(),
            )
            .map_err(candle_core::Error::wrap)?;
            let guards: [_; super::GROUPED_IQ1_MAX] = std::array::from_fn(|i| {
                self.weights[i]
                    .as_ref()
                    .map(|w| w.bytes().storage_and_layout())
            });
            let mut buffers = [input.buffer(); super::GROUPED_IQ1_MAX];
            let mut offsets = [0; super::GROUPED_IQ1_MAX];
            // Check every actual storage, device and end bound before allocation
            // or encoding. Nonzero contiguous views are bound at their byte offset.
            for i in 0..geometry.count {
                let (storage, layout) = guards[i].as_ref().unwrap();
                let Storage::Metal(storage) = &**storage else {
                    candle_core::bail!("grouped IQ1 requires Metal weight storage");
                };
                if storage.dtype() != DType::U8 || storage.device().id() != device.id() {
                    candle_core::bail!("grouped IQ1 requires U8 weights on the input Metal device");
                }
                offsets[i] = super::grouped_iq1_span(
                    layout,
                    DType::U8,
                    geometry.weight_bytes,
                    storage.buffer().length(),
                )
                .map_err(candle_core::Error::wrap)?;
                buffers[i] = storage.buffer();
            }
            // All twelve shader arguments are bound; unused slots alias slot 0
            // and are unreachable because the checked count bounds the grid.
            for i in geometry.count..super::GROUPED_IQ1_MAX {
                buffers[i] = buffers[0];
                offsets[i] = offsets[0];
            }
            let pipeline = if geometry.input_rows == 1 {
                pipeline(
                    device,
                    DeepSeek4DType::IQ1S,
                    KernelKind::GroupedIq1Simd,
                    false,
                )?
            } else {
                #[cfg(test)]
                {
                    pipeline(
                        device,
                        DeepSeek4DType::IQ1S,
                        KernelKind::GroupedIq1DownSimd,
                        false,
                    )?
                }
                #[cfg(not(test))]
                {
                    candle_core::bail!("grouped IQ1 per-matrix inputs are test-only");
                }
            };
            if pipeline.as_ref().threadExecutionWidth() != 32
                || pipeline.max_total_threads_per_threadgroup() < 128
            {
                candle_core::bail!("grouped IQ1 requires 32-lane SIMD and 128-thread groups");
            }
            let output = device
                .new_buffer_builder()
                .with_size_for(geometry.output_elements, DType::F32)
                .with_label("dinf-deepseek4-grouped-iq1")
                .build()?;
            let metadata = [
                u32::try_from(geometry.in_dim).map_err(candle_core::Error::wrap)?,
                u32::try_from(geometry.out_dim).map_err(candle_core::Error::wrap)?,
                u32::try_from(geometry.weight_bytes / geometry.out_dim)
                    .map_err(candle_core::Error::wrap)?,
                u32::try_from(geometry.count).map_err(candle_core::Error::wrap)?,
            ];
            let encoder = device.command_encoder()?;
            let encoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipeline);
            for i in 0..super::GROUPED_IQ1_MAX {
                encoder.set_input_buffer(i, Some(buffers[i]), offsets[i]);
            }
            encoder.set_input_buffer(12, Some(input.buffer()), input_offset);
            encoder.set_output_buffer(13, Some(&output), 0);
            encoder.set_bytes(14, &metadata);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: geometry.output_elements / 4,
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
                MetalStorage::new(output, device.clone(), geometry.output_elements, DType::F32),
                Shape::from((geometry.count, geometry.out_dim)),
            ))
        }
    }

    pub(super) fn forward_grouped_iq1(
        input: &Tensor,
        matrices: &[&super::DeepSeek4QMatMul],
        geometry: super::GroupedIq1Geometry,
    ) -> Result<Tensor> {
        forward_grouped_iq1_with_mode(
            input,
            matrices,
            geometry,
            super::DeepSeek4QMatMul::iq1_simd_enabled(),
        )
    }

    pub(super) fn forward_grouped_iq1_with_mode(
        input: &Tensor,
        matrices: &[&super::DeepSeek4QMatMul],
        geometry: super::GroupedIq1Geometry,
        enabled: bool,
    ) -> Result<Tensor> {
        if !enabled {
            candle_core::bail!("grouped IQ1 requires SIMD enabled; use separate forward calls");
        }
        input.apply_op1_no_bwd(&GroupedIq1 {
            weights: std::array::from_fn(|i| matrices.get(i).map(|m| m.weights.clone())),
            geometry,
        })
    }

    pub(super) fn forward_grouped_v41(
        weights: &Tensor,
        input: &Tensor,
        geometry: super::GroupedV41Geometry,
    ) -> Result<Tensor> {
        weights.apply_op2_no_bwd(
            input,
            &Dsv4QMatMul {
                dtype: DeepSeek4DType::Q8_0,
                out_dim: geometry.out_dim as usize,
                in_dim: geometry.in_dim as usize,
                kind: KernelKind::V41Q8GroupedNsg4,
                decode_rows: false,
                grouped: Some(geometry),
            },
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn simd_policy_keeps_prefill_narrow_and_disabled_paths_serial() {
            assert_eq!(select_kernel(4096, 4096, false), KernelKind::Serial);
            assert_eq!(select_kernel(512, 512, true), KernelKind::Serial);
            assert_eq!(select_kernel(1024, 1024, true), KernelKind::Simd);
            assert_eq!(select_kernel(4096, 4096, true), KernelKind::Simd);
            assert_eq!(select_kernel(4096, 8192, true), KernelKind::Serial);
        }

        #[test]
        fn simd_defaults_to_apple_silicon_and_honors_explicit_overrides() {
            use std::ffi::OsStr;
            assert!(simd_enabled(None, true));
            assert!(!simd_enabled(None, false));
            for native in [false, true] {
                assert!(simd_enabled(Some(OsStr::new("1")), native));
                for value in ["0", "false", "invalid", ""] {
                    assert!(!simd_enabled(Some(OsStr::new(value)), native));
                }
            }
        }

        #[test]
        fn quant_formats_and_variants_share_the_device_library() -> Result<()> {
            let Some(device) = crate::test_metal_device() else {
                return Ok(());
            };
            let device = device.as_metal_device()?;
            let first = pipeline(device, DeepSeek4DType::Q6K, KernelKind::Serial, false)?;
            pipeline(device, DeepSeek4DType::Q6K, KernelKind::Simd, false)?;
            pipeline(device, DeepSeek4DType::Q8_0, KernelKind::Simd, true)?;
            let again = pipeline(device, DeepSeek4DType::Q6K, KernelKind::Serial, false)?;
            assert!(std::ptr::eq(first.as_ref(), again.as_ref()));
            let cache = PIPELINES.get().unwrap().lock().unwrap();
            let entry = &cache[&device.id()];
            assert!(entry.library.is_some());
            assert_eq!(entry.pipelines.len(), 3);
            Ok(())
        }

        #[test]
        fn rejects_overflow_and_partial_blocks_before_gpu_allocation() {
            for (out, width, dims) in [
                (2, 16, vec![1, 16]),
                (2, 32, vec![0, 32]),
                (2, 32, vec![usize::MAX, 2, 32]),
                (2, 32, vec![u32::MAX as usize, 32]),
                (u32::MAX as usize, 32, vec![2, 32]),
                (usize::MAX, 32, vec![1, 32]),
            ] {
                assert!(Geometry::new(DeepSeek4DType::Q8_0, out, width, &dims).is_err());
            }
            let geometry = Geometry::new(DeepSeek4DType::IQ3XXS, 5, 768, &[2, 3, 768]).unwrap();
            assert_eq!(
                (geometry.rows, geometry.row_bytes, geometry.output_elements),
                (6, 294, 30)
            );
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::deepseek4_gguf::RawTensor;
    use candle_core::{DType, Tensor};

    #[test]
    fn bounded_dense_mlx_offsets_spans_and_scope() -> Result<()> {
        use candle_core::{MetalStorage, Shape, Storage};
        let device = candle_core::Device::new_metal(0)?;
        let metal = device.as_metal_device()?;
        for m in [1, 3, 14, 28] {
            let (n, k) = (17, 32);
            let x = (0..(m + 2) * k)
                .map(|i| ((i * 13 % 101) as f32 - 50.0) / 32.0)
                .collect::<Vec<_>>();
            let w = (0..(n + 1) * k)
                .map(|i| ((i * 7 % 97) as f32 - 48.0) / 16.0)
                .collect::<Vec<_>>();
            let x = Tensor::from_vec(x, (m + 2, k), &device)?.narrow(0, 1, m)?;
            let w = Tensor::from_vec(w, (n + 1, k), &device)?.narrow(0, 1, n)?;
            let expected = x.matmul(&w.t()?)?.flatten_all()?.to_vec1::<f32>()?;
            let output = dense_prefill_bounded(&x, &w, k, n)?;
            let actual = output.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "same MLX GEMM, m={m}"
            );
            let (storage, _) = output.storage_and_layout();
            let Storage::Metal(storage) = &*storage else {
                panic!("Metal output required");
            };
            assert!(
                storage.buffer().length() >= m * n * 4
                    && storage.buffer().length() <= (m * n * 4).next_power_of_two()
            );
        }
        let weight = Tensor::ones((17, 32), DType::F32, &device)?;
        let forged = |m: usize, k: usize| -> Result<Tensor> {
            let buffer = metal.new_private_buffer(1, DType::F32, "bounded-dense-invalid-span")?;
            Ok(Tensor::from((
                Storage::Metal(MetalStorage::new(buffer, metal.clone(), 1, DType::F32)),
                Shape::from((m, k)),
            )))
        };
        assert!(
            dense_prefill_bounded(&forged(2, 32)?, &weight, 32, 17)
                .unwrap_err()
                .to_string()
                .contains("span outside buffer")
        );
        let input = Tensor::ones((2, 32), DType::F32, &device)?;
        let short_weight = forged(17, 32)?;
        assert!(
            dense_prefill_bounded(&input, &short_weight, 32, 17)
                .unwrap_err()
                .to_string()
                .contains("span outside buffer")
        );
        for m in [0, i32::MAX as usize + 1] {
            assert!(
                dense_prefill_bounded(&forged(m, 32)?, &weight, 32, 17)
                    .unwrap_err()
                    .to_string()
                    .contains("MLX i32")
            );
        }
        with_q8_prefill_matrix_enabled(false, || -> Result<()> {
            let _ordinary = BoundedPrefillScope::enter(&device, false)?;
            assert!(!bounded_prefill_output(metal));
            {
                let _tile = BoundedPrefillScope::enter(&device, true)?;
                assert!(bounded_prefill_output(metal));
                {
                    let _nested = BoundedPrefillScope::enter(&device, false)?;
                    assert!(!bounded_prefill_output(metal));
                }
                assert!(bounded_prefill_output(metal));
            }
            assert!(!bounded_prefill_output(metal));
            let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _tile = BoundedPrefillScope::enter(&device, true).unwrap();
                panic!("exercise bounded allocation unwind");
            }));
            assert!(unwind.is_err());
            assert!(!bounded_prefill_output(metal));
            Ok(())
        })?;
        device.synchronize()?;
        Ok(())
    }

    #[test]
    fn fused_metal_matmul_compiles_and_runs_v4_quant_types() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for (dtype, in_dim) in [
            (DeepSeek4DType::IQ2XS, 256),
            (DeepSeek4DType::IQ3XXS, 256),
            (DeepSeek4DType::IQ3S, 256),
            (DeepSeek4DType::IQ1M, 256),
            (DeepSeek4DType::IQ1S, 256),
            (DeepSeek4DType::Q6K, 256),
            (DeepSeek4DType::Q8_0, 32),
            (DeepSeek4DType::MXFP4, 32),
        ] {
            let mut bytes = vec![0u8; dtype.byte_len(in_dim)?];
            if dtype == DeepSeek4DType::Q8_0 {
                bytes[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
                bytes[2..].fill(1);
            } else if dtype == DeepSeek4DType::MXFP4 {
                bytes[0] = 128;
            }
            let raw = Arc::new(RawTensor::new(
                vec![1, in_dim],
                dtype,
                Tensor::from_vec(bytes, dtype.byte_len(in_dim)?, &device)?,
            )?);
            let matmul = DeepSeek4QMatMul::new(raw)?;
            let input = Tensor::ones((2, in_dim), DType::F32, &device)?;
            let output = matmul.forward(&input)?;
            device.synchronize()?;
            let values = output.to_vec2::<f32>()?;
            assert!(values.iter().flatten().all(|value| value.is_finite()));
            if dtype == DeepSeek4DType::Q8_0 {
                assert!((values[0][0] - 32.0).abs() < 1e-3);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "deepseek4_qmatmul_tests.rs"]
mod regression_tests;

#[cfg(test)]
#[path = "deepseek4_routed_post_tests.rs"]
mod routed_post_tests;

#[cfg(target_os = "macos")]
#[path = "deepseek4_iq1_packets.rs"]
pub(crate) mod iq1_packets;
