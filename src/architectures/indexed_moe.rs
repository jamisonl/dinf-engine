#[cfg(any(target_os = "macos", feature = "cuda"))]
use anyhow::Context;
use anyhow::Result;
#[cfg(any(target_os = "macos", feature = "cuda", test))]
use candle_core::quantized::GgmlDType;
#[cfg(any(target_os = "macos", feature = "cuda"))]
use candle_core::quantized::QTensor;
#[cfg(any(target_os = "macos", feature = "cuda", test))]
use candle_core::{D, DType, Device};
use candle_core::{Module, Tensor};

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use super::indexed_moe_epilogue_cuda::{
    normalized_route_weights as cuda_normalized_route_weights, silu_mul as cuda_silu_mul,
    weighted_route_sum as cuda_weighted_route_sum,
};

#[cfg(target_os = "macos")]
use super::indexed_moe_metal::{
    MetalIndexedMoeMatMul, fused_normalized_route_weights, fused_silu_mul, fused_silu_mul_clamped,
    fused_weighted_route_sum,
};

#[cfg(any(target_os = "macos", feature = "cuda", test))]
const ROUTE_WEIGHT_DENOMINATOR_FLOOR: f64 = 6.103_515_625e-5;

#[cfg(any(target_os = "macos", feature = "cuda", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IndexedMoeBackendKind {
    #[cfg(any(target_os = "macos", test))]
    Metal,
    #[cfg(any(feature = "cuda", test))]
    Cuda,
}

#[cfg(any(target_os = "macos", feature = "cuda", test))]
fn backend_supports_dtype(backend: IndexedMoeBackendKind, dtype: GgmlDType) -> bool {
    match backend {
        #[cfg(any(target_os = "macos", test))]
        IndexedMoeBackendKind::Metal => matches!(
            dtype,
            GgmlDType::F32
                | GgmlDType::F16
                | GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        ),
        #[cfg(any(feature = "cuda", test))]
        IndexedMoeBackendKind::Cuda => matches!(
            dtype,
            GgmlDType::Q4_0
                | GgmlDType::Q5_0
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        ),
    }
}

#[cfg(any(target_os = "macos", feature = "cuda", test))]
fn select_backend(
    backend: Option<IndexedMoeBackendKind>,
    dtype: GgmlDType,
) -> Option<IndexedMoeBackendKind> {
    backend.filter(|backend| backend_supports_dtype(*backend, dtype))
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
fn device_backend(device: &Device) -> Option<IndexedMoeBackendKind> {
    #[cfg(target_os = "macos")]
    if device.is_metal() {
        return Some(IndexedMoeBackendKind::Metal);
    }
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        return Some(IndexedMoeBackendKind::Cuda);
    }
    None
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
pub(super) fn indexed_moe_available(device: &Device, dtype: GgmlDType) -> bool {
    select_backend(device_backend(device), dtype).is_some()
}

/// Quantized indexed expert matmul with one backend-neutral contract.
///
/// Inputs are F32 `[tokens, 1|top_k, in_dim]`, IDs are contiguous U32
/// `[tokens, top_k]`, and outputs are F32 `[tokens, top_k, out_dim]`.
/// IDs come only from an arg-sort over exactly `num_experts` columns or the
/// bounded host fallback. Reading them back here would introduce the route
/// synchronization this path exists to avoid.
#[cfg(any(target_os = "macos", feature = "cuda"))]
pub(super) struct IndexedMoeMatMul {
    backend: IndexedMoeBackend,
    device: Device,
    num_experts: usize,
    out_dim: usize,
    in_dim: usize,
    /// Prompts use the half-precision tensor-core CUDA kernel where it
    /// covers the expert format.
    #[cfg(feature = "cuda")]
    cuda_grouped_wmma: bool,
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
enum IndexedMoeBackend {
    #[cfg(target_os = "macos")]
    Metal(MetalIndexedMoeMatMul),
    #[cfg(feature = "cuda")]
    Cuda(Arc<QTensor>),
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
impl IndexedMoeMatMul {
    pub(super) fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub(super) fn from_qtensor(
        packed: &QTensor,
        num_experts: usize,
        out_dim: usize,
        in_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        anyhow::ensure!(
            num_experts > 0 && out_dim > 0 && in_dim > 0,
            "indexed MoE dimensions must be nonzero"
        );
        anyhow::ensure!(
            packed.shape().dims() == [num_experts, out_dim, in_dim],
            "packed expert shape {:?} differs from [{num_experts}, {out_dim}, {in_dim}]",
            packed.shape()
        );
        anyhow::ensure!(
            in_dim.is_multiple_of(packed.dtype().block_size()),
            "packed expert input dimension is not quantization aligned"
        );

        let backend_kind = device_backend(device)
            .context("indexed MoE requires a supported accelerator device")?;
        anyhow::ensure!(
            select_backend(Some(backend_kind), packed.dtype()).is_some(),
            "indexed MoE {:?} does not support {:?}",
            backend_kind,
            packed.dtype()
        );
        anyhow::ensure!(
            packed.device().is_cpu() || packed.device().same_device(device),
            "packed expert source is on an unrelated device"
        );

        let blocks = num_experts
            .checked_mul(out_dim)
            .and_then(|value| value.checked_mul(in_dim))
            .context("packed expert element count overflow")?
            / packed.dtype().block_size();
        let expected_bytes = blocks
            .checked_mul(packed.dtype().type_size())
            .context("packed expert storage size overflow")?;
        anyhow::ensure!(
            packed.data()?.len() == expected_bytes,
            "packed expert storage has an unexpected byte length"
        );

        let backend = match backend_kind {
            #[cfg(target_os = "macos")]
            IndexedMoeBackendKind::Metal => IndexedMoeBackend::Metal(
                MetalIndexedMoeMatMul::from_qtensor(packed, num_experts, out_dim, in_dim, device)?,
            ),
            #[cfg(feature = "cuda")]
            IndexedMoeBackendKind::Cuda => {
                let weights = if packed.device().same_device(device) {
                    Arc::new(candle_core::quantized::ggml_file::qtensor_from_ggml(
                        packed.dtype(),
                        packed.data()?.as_ref(),
                        vec![num_experts, out_dim, in_dim],
                        device,
                    )?)
                } else {
                    let data = packed.data()?;
                    Arc::new(candle_core::quantized::ggml_file::qtensor_from_ggml(
                        packed.dtype(),
                        data.as_ref(),
                        vec![num_experts, out_dim, in_dim],
                        device,
                    )?)
                };
                anyhow::ensure!(
                    weights.device().same_device(device),
                    "indexed MoE CUDA weights were not materialized on the target device"
                );
                super::indexed_moe_epilogue_cuda::precompile(device)?;
                if super::indexed_moe_cuda::supports(weights.dtype()) {
                    super::indexed_moe_cuda::precompile(device)?;
                }
                IndexedMoeBackend::Cuda(weights)
            }
            #[allow(unreachable_patterns)]
            _ => anyhow::bail!("indexed MoE backend is unavailable in this build"),
        };
        Ok(Self {
            backend,
            device: device.clone(),
            num_experts,
            out_dim,
            in_dim,
            #[cfg(feature = "cuda")]
            cuda_grouped_wmma: false,
        })
    }

    /// Run CUDA prompts (more than one token) through the tensor-core grouped
    /// kernel when it covers this expert format. It rounds inputs and weights
    /// to F16; single-token decode is unchanged.
    #[cfg(feature = "cuda")]
    pub(super) fn prefer_cuda_grouped_wmma(&mut self) -> Result<()> {
        if let IndexedMoeBackend::Cuda(weights) = &self.backend
            && (super::indexed_moe_cuda::supports_grouped_wmma(weights.dtype())
                || super::indexed_moe_cuda::supports_f32_decode(weights.dtype()))
        {
            super::indexed_moe_cuda::precompile(&self.device)?;
            self.cuda_grouped_wmma = true;
        }
        Ok(())
    }

    /// Use the native vector order for one admitted Metal request.
    #[cfg(target_os = "macos")]
    pub(super) fn forward_native_metal_vector(
        &self,
        input: &Tensor,
        expert_ids: &Tensor,
    ) -> Result<Tensor> {
        self.forward_impl(input, expert_ids, false, true)
    }

    pub(super) fn forward(&self, input: &Tensor, expert_ids: &Tensor) -> Result<Tensor> {
        self.forward_impl(input, expert_ids, false, false)
    }

    /// Build Metal route tables once for several projections that share the
    /// same expert IDs and expert count (for example gate, up and down).
    #[cfg(target_os = "macos")]
    pub(super) fn metal_route_plan(
        &self,
        expert_ids: &Tensor,
    ) -> Result<Option<super::indexed_moe_metal::MetalIndexedMoeRoutePlan>> {
        match &self.backend {
            IndexedMoeBackend::Metal(metal) => Ok(Some(metal.prepare_route_plan(expert_ids)?)),
            #[allow(unreachable_patterns)]
            _ => Ok(None),
        }
    }

    /// Fused SwiGLU expert projections, `silu(gate) * up`, with `self` as the
    /// gate. `None` when the backend or type has no fused kernel.
    #[cfg(target_os = "macos")]
    pub(super) fn forward_gate_up_silu_with_metal_route_plan(
        &self,
        up: &Self,
        input: &Tensor,
        plan: &super::indexed_moe_metal::MetalIndexedMoeRoutePlan,
    ) -> Result<Option<Tensor>> {
        match (&self.backend, &up.backend) {
            (IndexedMoeBackend::Metal(gate), IndexedMoeBackend::Metal(up)) => {
                Ok(gate.forward_gate_up_silu_with_route_plan(up, input, plan)?)
            }
            #[allow(unreachable_patterns)]
            _ => Ok(None),
        }
    }

    /// Compile the packed GEGLU expert kernel ahead of use (Metal only).
    #[cfg(target_os = "macos")]
    pub(super) fn prepare_geglu(&self) -> Result<()> {
        match &self.backend {
            IndexedMoeBackend::Metal(metal) => Ok(metal.prepare_geglu(&self.device)?),
            #[allow(unreachable_patterns)]
            _ => Ok(()),
        }
    }

    /// Packed `[gate | up]` experts with `gelu(gate) * up` in one Metal 4
    /// dispatch. `None` when the backend or type has no fused kernel.
    #[cfg(target_os = "macos")]
    pub(super) fn forward_geglu_with_metal_route_plan(
        &self,
        input: &Tensor,
        plan: &super::indexed_moe_metal::MetalIndexedMoeRoutePlan,
    ) -> Result<Option<Tensor>> {
        match &self.backend {
            IndexedMoeBackend::Metal(metal) => {
                Ok(metal.forward_geglu_with_route_plan(input, plan)?)
            }
            #[allow(unreachable_patterns)]
            _ => Ok(None),
        }
    }

    /// Sanitize Metal route IDs (u32, or the f32 IDs from the fused top-k
    /// route) once for several projections. `None` off the Metal backend.
    #[cfg(target_os = "macos")]
    pub(super) fn metal_sanitize_ids(&self, expert_ids: &Tensor) -> Result<Option<Tensor>> {
        match &self.backend {
            IndexedMoeBackend::Metal(metal) => Ok(Some(metal.sanitize_ids(expert_ids)?)),
            #[allow(unreachable_patterns)]
            _ => Ok(None),
        }
    }

    /// Projection with IDs from [`Self::metal_sanitize_ids`] for the same
    /// expert count; `native_vector` selects the native vector kernel.
    #[cfg(target_os = "macos")]
    pub(super) fn forward_metal_sanitized(
        &self,
        input: &Tensor,
        expert_ids: &Tensor,
        native_vector: bool,
    ) -> Result<Tensor> {
        match &self.backend {
            IndexedMoeBackend::Metal(metal) if native_vector => {
                Ok(metal.forward_native_vector_sanitized(input, expert_ids)?)
            }
            IndexedMoeBackend::Metal(metal) => Ok(metal.forward_sanitized(input, expert_ids)?),
            #[allow(unreachable_patterns)]
            _ => anyhow::bail!("sanitized Metal IDs require the Metal indexed MoE backend"),
        }
    }

    /// Matrix-kernel projection with a route plan from [`Self::metal_route_plan`].
    #[cfg(target_os = "macos")]
    pub(super) fn forward_with_metal_route_plan(
        &self,
        input: &Tensor,
        plan: &super::indexed_moe_metal::MetalIndexedMoeRoutePlan,
    ) -> Result<Tensor> {
        match &self.backend {
            IndexedMoeBackend::Metal(metal) => Ok(metal.forward_with_route_plan(input, plan)?),
            #[allow(unreachable_patterns)]
            _ => anyhow::bail!("Metal route plans require the Metal indexed MoE backend"),
        }
    }

    /// Use consistent Metal matrix math across short and long prompt chunks.
    pub(super) fn forward_prefill(&self, input: &Tensor, expert_ids: &Tensor) -> Result<Tensor> {
        self.forward_impl(input, expert_ids, true, false)
    }

    fn forward_impl(
        &self,
        input: &Tensor,
        expert_ids: &Tensor,
        _prefill: bool,
        _native_vector: bool,
    ) -> Result<Tensor> {
        let (tokens, input_rows, in_dim) = input.dims3()?;
        let (id_tokens, top_k) = expert_ids.dims2()?;
        anyhow::ensure!(
            tokens > 0
                && top_k > 0
                && top_k <= self.num_experts
                && tokens == id_tokens
                && in_dim == self.in_dim
                && (input_rows == 1 || input_rows == top_k),
            "indexed MoE input or routing geometry differs"
        );
        anyhow::ensure!(input.dtype() == DType::F32, "indexed MoE input must be f32");
        anyhow::ensure!(
            expert_ids.dtype() == DType::U32,
            "indexed MoE expert IDs must be u32"
        );
        anyhow::ensure!(
            input.device().same_device(&self.device)
                && expert_ids.device().same_device(&self.device),
            "indexed MoE input, IDs, and weights must share one device"
        );

        // The GGUF prompt kernel reads its storage from byte zero and does not
        // apply tensor layout offsets. `contiguous()` retains an
        // already-contiguous narrow view, so prompt chunks after the first one
        // would silently reread chunk zero. Copy only offset CUDA views for
        // that kernel; every other kernel consumes its layout offsets.
        #[cfg(feature = "cuda")]
        let grouped_wmma = self.cuda_grouped_wmma && tokens > 1;
        // The same opt-in sends one token through the one-launch F32 kernel
        // for formats whose default decode path takes several launches.
        #[cfg(feature = "cuda")]
        let f32_decode = self.cuda_grouped_wmma
            && tokens == 1
            && matches!(
                &self.backend,
                IndexedMoeBackend::Cuda(weights)
                    if super::indexed_moe_cuda::supports_f32_decode(weights.dtype())
            );
        #[cfg(feature = "cuda")]
        let copy_cuda_views = !grouped_wmma
            && !f32_decode
            && matches!(
                &self.backend,
                IndexedMoeBackend::Cuda(weights)
                    if !super::indexed_moe_cuda::supports(weights.dtype())
                        && (tokens > 1 || weights.dtype() == GgmlDType::Q8_0)
            );
        #[cfg(not(feature = "cuda"))]
        let copy_cuda_views = false;
        let input = accelerator_kernel_input(input, copy_cuda_views)?;
        let expert_ids = accelerator_kernel_input(expert_ids, copy_cuda_views)?;
        let output = match &self.backend {
            #[cfg(target_os = "macos")]
            IndexedMoeBackend::Metal(inner) if _native_vector => {
                inner.forward_native_vector(&input, &expert_ids)?
            }
            #[cfg(target_os = "macos")]
            IndexedMoeBackend::Metal(inner) if _prefill => {
                inner.forward_matrix(&input, &expert_ids)?
            }
            #[cfg(target_os = "macos")]
            IndexedMoeBackend::Metal(inner) => inner.forward(&input, &expert_ids)?,
            #[cfg(feature = "cuda")]
            IndexedMoeBackend::Cuda(weights) => {
                if grouped_wmma {
                    super::indexed_moe_cuda::forward_grouped_wmma(
                        &input,
                        &expert_ids,
                        weights,
                        self.num_experts,
                        self.out_dim,
                        self.in_dim,
                    )?
                } else if f32_decode || super::indexed_moe_cuda::supports(weights.dtype()) {
                    super::indexed_moe_cuda::forward(
                        &input,
                        &expert_ids,
                        weights,
                        self.num_experts,
                        self.out_dim,
                        self.in_dim,
                    )?
                } else if tokens > 1 || weights.dtype() == GgmlDType::Q8_0 {
                    // Grouped WMMA is the prompt path for every supported
                    // CUDA quant. Q8_0 must also use this path for decode:
                    // Candle's indexed Q8_0 kernel advances experts with the
                    // K-quant block size and would read the wrong expert.
                    cuda_gguf_moe_forward(
                        &input,
                        &expert_ids,
                        weights,
                        self.num_experts,
                        self.out_dim,
                        self.in_dim,
                    )?
                } else {
                    weights.indexed_moe_forward(&input, &expert_ids)?
                }
            }
        };
        anyhow::ensure!(
            output.dims3()? == (tokens, top_k, self.out_dim)
                && output.dtype() == DType::F32
                && output.device().same_device(&self.device),
            "indexed MoE backend returned an invalid output"
        );
        Ok(output)
    }

    /// Fuse gate/up indexed MMVQ and SwiGLU for small CUDA K-quant route sets.
    pub(super) fn silu_gate_up(
        &self,
        up: &Self,
        input: &Tensor,
        expert_ids: &Tensor,
    ) -> Result<Option<Tensor>> {
        #[cfg(not(feature = "cuda"))]
        let _ = (input, expert_ids);
        if self.num_experts != up.num_experts
            || self.out_dim != up.out_dim
            || self.in_dim != up.in_dim
            || !self.device.same_device(&up.device)
        {
            return Ok(None);
        }
        match (&self.backend, &up.backend) {
            #[cfg(feature = "cuda")]
            (IndexedMoeBackend::Cuda(gate), IndexedMoeBackend::Cuda(up)) => {
                super::dense_qmatmul_cuda::indexed_silu_gate_up(
                    input,
                    expert_ids,
                    gate,
                    up,
                    self.num_experts,
                    self.out_dim,
                    self.in_dim,
                )
            }
            #[allow(unreachable_patterns)]
            _ => Ok(None),
        }
    }

    /// Apply ReLU-squared while reading Q4_0 expert inputs, avoiding a full
    /// route-sized activation tensor before the down projection.
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    pub(super) fn forward_relu_squared(
        &self,
        input: &Tensor,
        expert_ids: &Tensor,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let IndexedMoeBackend::Cuda(weights) = &self.backend
            && super::indexed_moe_cuda::supports(weights.dtype())
        {
            return super::indexed_moe_cuda::forward_relu_squared(
                input,
                expert_ids,
                weights,
                self.num_experts,
                self.out_dim,
                self.in_dim,
            );
        }
        let relu = input.relu()?;
        let activated = (&relu * &relu)?;
        self.forward(&activated, expert_ids)
    }
}

#[cfg(feature = "cuda")]
fn cuda_gguf_moe_forward(
    input: &Tensor,
    expert_ids: &Tensor,
    weights: &QTensor,
    num_experts: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<Tensor> {
    let (tokens, input_rows, _) = input.dims3()?;
    let (_, top_k) = expert_ids.dims2()?;
    let routes = tokens
        .checked_mul(top_k)
        .context("CUDA GGUF MoE route count overflow")?;
    anyhow::ensure!(
        routes <= i32::MAX as usize && num_experts <= i32::MAX as usize,
        "CUDA GGUF MoE exceeds the kernel index limit"
    );

    // Group routes by expert without leaving the GPU. The kernel writes each
    // result back to its original route index through `sorted_token_ids`.
    let route_experts = expert_ids.flatten_all()?.contiguous()?;
    let sorted_token_ids = route_experts.arg_sort_last_dim(true)?;
    let sorted_experts = route_experts
        .gather(&sorted_token_ids, D::Minus1)?
        .contiguous()?;

    let input = if input_rows == 1 {
        input.reshape((tokens, in_dim))?
    } else {
        input.reshape((routes, in_dim))?
    };
    let route_weights = if input_rows == 1 {
        None
    } else {
        // A route-specific input must not be collapsed back to its token row.
        // Supplying unit weights selects Candle's one-input-row-per-route form
        // without changing the result; callers apply real route weights later.
        Some(Tensor::ones(routes, DType::F32, input.device())?)
    };
    let output = candle_nn::moe::moe_gemm_gguf(
        &input,
        weights,
        &route_weights,
        &sorted_token_ids,
        &sorted_experts,
        top_k,
        tokens > 1,
        DType::F16,
    )?;
    anyhow::ensure!(
        output.dims2()? == (routes, out_dim),
        "CUDA GGUF MoE returned an invalid matrix"
    );
    output.reshape((tokens, top_k, out_dim)).map_err(Into::into)
}

#[cfg(any(target_os = "macos", feature = "cuda", test))]
fn accelerator_kernel_input(tensor: &Tensor, ignores_layout_offset: bool) -> Result<Tensor> {
    if ignores_layout_offset && tensor.layout().start_offset() != 0 {
        Ok(tensor.force_contiguous()?)
    } else {
        Ok(tensor.contiguous()?)
    }
}

/// Gather and normalize selected route probabilities entirely on-device.
#[cfg(any(target_os = "macos", feature = "cuda", test))]
pub(super) fn normalized_route_weights(
    probabilities: &Tensor,
    expert_ids: &Tensor,
) -> Result<Tensor> {
    let (tokens, expert_count) = probabilities.dims2()?;
    let (id_tokens, top_k) = expert_ids.dims2()?;
    anyhow::ensure!(
        tokens > 0 && expert_count > 0 && id_tokens == tokens && top_k > 0 && top_k <= expert_count,
        "indexed MoE route-weight geometry differs"
    );
    anyhow::ensure!(
        probabilities.dtype() == DType::F32 && expert_ids.dtype() == DType::U32,
        "indexed MoE route weights require f32 probabilities and u32 IDs"
    );
    anyhow::ensure!(
        probabilities.device().same_device(expert_ids.device()),
        "indexed MoE route probabilities and IDs must share one device"
    );

    #[cfg(target_os = "macos")]
    if probabilities.device().is_metal() {
        return fused_normalized_route_weights(
            &probabilities.contiguous()?,
            &expert_ids.contiguous()?,
        )
        .map_err(Into::into);
    }

    #[cfg(feature = "cuda")]
    if probabilities.device().is_cuda() {
        return cuda_normalized_route_weights(
            &probabilities.contiguous()?,
            &expert_ids.contiguous()?,
        );
    }

    let selected = probabilities.gather(expert_ids, D::Minus1)?;
    let mut weight_sum = selected.narrow(D::Minus1, 0, 1)?;
    for route in 1..top_k {
        weight_sum = (&weight_sum + selected.narrow(D::Minus1, route, 1)?)?;
    }
    let denominator = weight_sum.clamp(ROUTE_WEIGHT_DENOMINATOR_FLOOR, f32::MAX as f64)?;
    selected.broadcast_div(&denominator).map_err(Into::into)
}

/// Apply the SwiGLU elementwise epilogue without leaving the accelerator.
pub(super) fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    anyhow::ensure!(gate.shape() == up.shape(), "SwiGLU gate/up shapes differ");
    anyhow::ensure!(gate.dtype() == up.dtype(), "SwiGLU gate/up dtypes differ");
    anyhow::ensure!(
        gate.dtype().is_float(),
        "SwiGLU inputs must be floating point"
    );
    anyhow::ensure!(
        gate.device().same_device(up.device()),
        "SwiGLU gate/up tensors must share one device"
    );

    #[cfg(target_os = "macos")]
    if gate.device().is_metal()
        && gate.dtype() == DType::F32
        && gate.is_contiguous()
        && up.is_contiguous()
    {
        return fused_silu_mul(gate, up).map_err(Into::into);
    }
    #[cfg(feature = "cuda")]
    if gate.device().is_cuda() && gate.dtype() == DType::F32 {
        return cuda_silu_mul(&gate.contiguous()?, &up.contiguous()?);
    }
    Ok((candle_nn::Activation::Silu.forward(gate)? * up)?)
}

/// Apply GLM's bounded SwiGLU epilogue. Metal fuses both clamps, SiLU, and
/// multiplication into one dispatch.
pub(super) fn silu_mul_clamped(gate: &Tensor, up: &Tensor, limit: f32) -> Result<Tensor> {
    anyhow::ensure!(gate.shape() == up.shape(), "SwiGLU gate/up shapes differ");
    anyhow::ensure!(gate.dtype() == up.dtype(), "SwiGLU gate/up dtypes differ");
    anyhow::ensure!(
        limit.is_finite() && limit > 0.0,
        "SwiGLU clamp must be positive"
    );
    anyhow::ensure!(
        gate.device().same_device(up.device()),
        "SwiGLU gate/up tensors must share one device"
    );

    #[cfg(target_os = "macos")]
    if gate.device().is_metal()
        && gate.dtype() == DType::F32
        && gate.is_contiguous()
        && up.is_contiguous()
    {
        return fused_silu_mul_clamped(gate, up, limit).map_err(Into::into);
    }

    let gate = gate.clamp(f64::NEG_INFINITY, limit as f64)?;
    let up = up.clamp(-(limit as f64), limit as f64)?;
    Ok((candle_nn::Activation::Silu.forward(&gate)? * up)?)
}

/// Sum `[tokens, top_k, hidden]` route outputs using `[tokens, top_k]` weights.
#[cfg(any(target_os = "macos", feature = "cuda", test))]
pub(super) fn weighted_route_sum(routes: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let (tokens, top_k, hidden_dim) = routes.dims3()?;
    anyhow::ensure!(
        tokens > 0 && top_k > 0 && hidden_dim > 0 && weights.dims2()? == (tokens, top_k),
        "indexed MoE weighted-route geometry differs"
    );
    anyhow::ensure!(
        routes.dtype() == DType::F32 && weights.dtype() == DType::F32,
        "indexed MoE weighted route sum requires f32 tensors"
    );
    anyhow::ensure!(
        routes.device().same_device(weights.device()),
        "indexed MoE routes and weights must share one device"
    );

    #[cfg(target_os = "macos")]
    if routes.device().is_metal() {
        return fused_weighted_route_sum(&routes.contiguous()?, &weights.contiguous()?)
            .map_err(Into::into);
    }

    #[cfg(feature = "cuda")]
    if routes.device().is_cuda() {
        return cuda_weighted_route_sum(&routes.contiguous()?, &weights.contiguous()?);
    }

    let first = routes
        .narrow(1, 0, 1)?
        .squeeze(1)?
        .broadcast_mul(&weights.narrow(1, 0, 1)?)?;
    let mut output = first;
    for route in 1..top_k {
        let weighted = routes
            .narrow(1, route, 1)?
            .squeeze(1)?
            .broadcast_mul(&weights.narrow(1, route, 1)?)?;
        output = (&output + weighted)?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_dtype_selection_is_explicit_and_safe() {
        for dtype in [
            GgmlDType::Q4_0,
            GgmlDType::Q5_0,
            GgmlDType::Q8_0,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            assert_eq!(
                select_backend(Some(IndexedMoeBackendKind::Cuda), dtype),
                Some(IndexedMoeBackendKind::Cuda)
            );
        }
        for dtype in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::BF16,
            GgmlDType::Q4_1,
            GgmlDType::Q5_1,
            GgmlDType::Q8_1,
            GgmlDType::Q8K,
        ] {
            assert_eq!(
                select_backend(Some(IndexedMoeBackendKind::Cuda), dtype),
                None
            );
        }
    }

    #[test]
    fn offset_ignoring_kernels_receive_zero_offset_storage() -> Result<()> {
        let source = Tensor::arange(0f32, 24f32, &Device::Cpu)?.reshape((4, 2, 3))?;
        let view = source.narrow(0, 2, 2)?;
        assert!(view.is_contiguous());
        assert_ne!(view.layout().start_offset(), 0);

        let copied = accelerator_kernel_input(&view, true)?;
        assert_eq!(copied.layout().start_offset(), 0);
        assert_eq!(copied.to_vec3::<f32>()?, view.to_vec3::<f32>()?);
        Ok(())
    }

    #[test]
    fn backend_selection_honors_device_availability() {
        assert_eq!(
            select_backend(Some(IndexedMoeBackendKind::Metal), GgmlDType::Q4_0),
            Some(IndexedMoeBackendKind::Metal)
        );
        assert_eq!(select_backend(None, GgmlDType::Q4K), None);
    }

    #[test]
    fn portable_route_helpers_preserve_selection_order_and_math() -> Result<()> {
        let probabilities = Tensor::new(&[[0.1f32, 0.6, 0.3]], &Device::Cpu)?;
        let expert_ids = Tensor::new(&[[2u32, 1]], &Device::Cpu)?;
        let weights = normalized_route_weights(&probabilities, &expert_ids)?;
        assert_eq!(weights.to_vec2::<f32>()?, vec![vec![1.0 / 3.0, 2.0 / 3.0]]);

        let routes = Tensor::new(&[[[3.0f32, 6.0], [9.0, 12.0]]], &Device::Cpu)?;
        let summed = weighted_route_sum(&routes, &weights)?;
        let values = summed.to_vec2::<f32>()?;
        assert!((values[0][0] - 7.0).abs() < 1e-6);
        assert!((values[0][1] - 10.0).abs() < 1e-6);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn fused_indexed_k_quant_swiglu_matches_separate_routes() -> Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let (num_experts, out_dim, in_dim) = (3, 64, 256);
        let source = |multiplier: usize| -> Result<Tensor> {
            let values = (0..num_experts * out_dim * in_dim)
                .map(|index| ((index * multiplier % 97) as f32 - 48.0) / 29.0)
                .collect::<Vec<_>>();
            Ok(Tensor::from_vec(
                values,
                (num_experts, out_dim, in_dim),
                &Device::Cpu,
            )?)
        };
        let input = Tensor::from_vec(
            (0..in_dim)
                .map(|index| ((index * 11 % 89) as f32 - 44.0) / 31.0)
                .collect::<Vec<_>>(),
            (1, 1, in_dim),
            &device,
        )?;
        let expert_ids = Tensor::new(&[[2u32, 0]], &device)?;

        for dtype in [GgmlDType::Q2K, GgmlDType::Q4K] {
            let gate_packed = QTensor::quantize_onto(&source(17)?, dtype, &Device::Cpu)?;
            let up_packed = QTensor::quantize_onto(&source(29)?, dtype, &Device::Cpu)?;
            let gate = IndexedMoeMatMul::from_qtensor(
                &gate_packed,
                num_experts,
                out_dim,
                in_dim,
                &device,
            )?;
            let up =
                IndexedMoeMatMul::from_qtensor(&up_packed, num_experts, out_dim, in_dim, &device)?;
            let expected = super::super::indexed_moe_epilogue_cuda::silu_mul_exact(
                &gate.forward(&input, &expert_ids)?,
                &up.forward(&input, &expert_ids)?,
            )?;
            let actual = gate
                .silu_gate_up(&up, &input, &expert_ids)?
                .context("eligible fused indexed K-quant SwiGLU")?;
            let expected = expected
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual = actual
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max_diff = actual
                .iter()
                .zip(expected.iter())
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            anyhow::ensure!(
                max_diff <= 1e-6,
                "{dtype:?} indexed SwiGLU drifted by {max_diff}"
            );
        }
        Ok(())
    }
}
