use anyhow::{Context, Result, ensure};

const CUDA_THREADS: usize = 256;
#[cfg(feature = "cuda")]
const CUDA_NORMALIZATION_THREADS: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RouteNormalizationGeometry {
    tokens: usize,
    expert_count: usize,
    top_k: usize,
    probability_elements: usize,
    route_elements: usize,
}

impl RouteNormalizationGeometry {
    fn new(tokens: usize, expert_count: usize, id_tokens: usize, top_k: usize) -> Result<Self> {
        ensure!(tokens > 0, "CUDA route normalization requires tokens");
        ensure!(
            expert_count > 0 && top_k > 0 && top_k <= expert_count,
            "CUDA route normalization expert geometry differs"
        );
        ensure!(
            tokens == id_tokens,
            "CUDA route normalization token counts differ"
        );
        ensure!(
            tokens <= i32::MAX as usize
                && expert_count <= i32::MAX as usize
                && top_k <= i32::MAX as usize,
            "CUDA route normalization exceeds kernel index limits"
        );
        let probability_elements = checked_mul(tokens, expert_count, "probability elements")?;
        let route_elements = checked_mul(tokens, top_k, "route elements")?;
        Ok(Self {
            tokens,
            expert_count,
            top_k,
            probability_elements,
            route_elements,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ElementwiseGeometry {
    shape: Vec<usize>,
    elements: usize,
    rows: usize,
    row_width: usize,
}

impl ElementwiseGeometry {
    fn new(lhs: &[usize], rhs: &[usize]) -> Result<Self> {
        ensure!(lhs == rhs, "CUDA SwiGLU input shapes differ");
        let elements = checked_product(lhs, "SwiGLU elements")?;
        ensure!(elements > 0, "CUDA SwiGLU requires elements");
        ensure_launchable(elements, "SwiGLU")?;
        let row_width = lhs.last().copied().unwrap_or(elements);
        ensure!(row_width > 0, "CUDA SwiGLU requires a nonempty row");
        let rows = elements / row_width;
        Ok(Self {
            shape: lhs.to_vec(),
            elements,
            rows,
            row_width,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WeightedRouteGeometry {
    tokens: usize,
    top_k: usize,
    hidden_dim: usize,
    route_elements: usize,
    weight_elements: usize,
    output_elements: usize,
}

impl WeightedRouteGeometry {
    fn new(
        tokens: usize,
        top_k: usize,
        hidden_dim: usize,
        weight_tokens: usize,
        weight_top_k: usize,
    ) -> Result<Self> {
        ensure!(
            tokens > 0 && top_k > 0 && hidden_dim > 0,
            "CUDA weighted route sum requires nonzero dimensions"
        );
        ensure!(
            (tokens, top_k) == (weight_tokens, weight_top_k),
            "CUDA weighted route sum weight geometry differs"
        );
        ensure!(
            top_k <= i32::MAX as usize && hidden_dim <= i32::MAX as usize,
            "CUDA weighted route sum exceeds kernel index limits"
        );
        let route_elements = checked_mul(
            checked_mul(tokens, top_k, "weighted route rows")?,
            hidden_dim,
            "weighted route elements",
        )?;
        let weight_elements = checked_mul(tokens, top_k, "route weight elements")?;
        let output_elements = checked_mul(tokens, hidden_dim, "weighted output elements")?;
        ensure_launchable(output_elements, "weighted route sum")?;
        Ok(Self {
            tokens,
            top_k,
            hidden_dim,
            route_elements,
            weight_elements,
            output_elements,
        })
    }
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .with_context(|| format!("CUDA indexed MoE {label} overflows usize"))
}

fn checked_product(dims: &[usize], label: &str) -> Result<usize> {
    dims.iter()
        .try_fold(1usize, |elements, dim| checked_mul(elements, *dim, label))
}

fn ensure_launchable(elements: usize, label: &str) -> Result<()> {
    ensure!(
        elements.div_ceil(CUDA_THREADS) <= i32::MAX as usize,
        "CUDA indexed MoE {label} exceeds the grid limit"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::sync::OnceLock;

    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::WrapErr;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    use candle_core::{CpuStorage, CudaStorage, DType, Device, Layout, Shape, Tensor};

    use super::{
        CUDA_NORMALIZATION_THREADS, CUDA_THREADS, ElementwiseGeometry, RouteNormalizationGeometry,
        WeightedRouteGeometry,
    };

    const CUDA_MODULE_NAME: &str = "dinf_indexed_moe_epilogue_v2";
    const EXACT_SILU_MODULE_NAME: &str = "dinf_dense_silu_mul_exact_v2";
    const NORMALIZE_KERNEL_NAME: &str = "dinf_normalized_route_weights_f32";
    const SILU_MUL_KERNEL_NAME: &str = "dinf_silu_mul_f32";
    const EXACT_SILU_KERNEL_NAME: &str = "dinf_silu_mul_exact_f32";
    const WEIGHTED_SUM_KERNEL_NAME: &str = "dinf_weighted_route_sum_f32";

    const CUDA_SOURCE: &str = r#"
#define DINF_NORMALIZATION_THREADS 32
#define DINF_ROUTE_DENOMINATOR_FLOOR 6.103515625e-5f

extern "C" __global__ void dinf_normalized_route_weights_f32(
    const float *probabilities,
    const unsigned int *expert_ids,
    float *output,
    int tokens,
    int expert_count,
    int top_k) {
    const int token = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= tokens) {
        return;
    }

    for (int route = lane; route < top_k; route += DINF_NORMALIZATION_THREADS) {
        const unsigned long long route_offset =
            (unsigned long long) token * top_k + route;
        const unsigned int expert = expert_ids[route_offset];
        const float selected = expert < (unsigned int) expert_count
            ? probabilities[(unsigned long long) token * expert_count + expert]
            : 0.0f;
        output[route_offset] = selected;
    }

    __shared__ float denominator;
    __syncthreads();
    if (lane == 0) {
        // Preserve the host and Metal left-to-right f32 route sum. A tree
        // reduction can change route choice at a later layer during decode.
        volatile float weight_sum = 0.0f;
        for (int route = 0; route < top_k; ++route) {
            weight_sum += output[(unsigned long long) token * top_k + route];
        }
        denominator = fminf(
            fmaxf((float) weight_sum, DINF_ROUTE_DENOMINATOR_FLOOR),
            3.402823466e+38f);
    }
    __syncthreads();
    for (int route = lane; route < top_k; route += DINF_NORMALIZATION_THREADS) {
        const unsigned long long route_offset =
            (unsigned long long) token * top_k + route;
        output[route_offset] /= denominator;
    }
}

extern "C" __global__ void dinf_silu_mul_f32(
    const float *gate,
    const float *up,
    float *output,
    unsigned long long elements,
    unsigned long long row_width,
    unsigned long long gate_row_stride,
    unsigned long long up_row_stride) {
    const unsigned long long index =
        (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const unsigned long long row = index / row_width;
        const unsigned long long column = index - row * row_width;
        const float value = gate[row * gate_row_stride + column];
        output[index] = (value / (1.0f + expf(-value))) *
            up[row * up_row_stride + column];
    }
}

extern "C" __global__ void dinf_weighted_route_sum_f32(
    const float *routes,
    const float *weights,
    float *output,
    int top_k,
    int hidden_dim,
    unsigned long long elements) {
    const unsigned long long index =
        (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const unsigned long long token = index / (unsigned long long) hidden_dim;
    const int column = (int) (index - token * hidden_dim);
    float sum = 0.0f;
    for (int route = 0; route < top_k; ++route) {
        const unsigned long long route_index = token * top_k + route;
        sum = fmaf(
            routes[route_index * hidden_dim + column],
            weights[route_index],
            sum);
    }
    output[index] = sum;
}
"#;

    // Dense transformer decode should preserve the two f32 rounding boundaries
    // from Candle's SiLU kernel followed by its multiply kernel. The indexed
    // MoE epilogue intentionally uses fast math, so compile this small variant
    // separately without fast math and make each boundary explicit.
    const EXACT_SILU_CUDA_SOURCE: &str = r#"
extern "C" __global__ void dinf_silu_mul_exact_f32(
    const float *gate,
    const float *up,
    float *output,
    unsigned long long elements,
    unsigned long long row_width,
    unsigned long long gate_row_stride,
    unsigned long long up_row_stride) {
    const unsigned long long index =
        (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const unsigned long long row = index / row_width;
        const unsigned long long column = index - row * row_width;
        const float value = gate[row * gate_row_stride + column];
        const float denominator = __fadd_rn(1.0f, expf(-value));
        const float silu = __fdiv_rn(value, denominator);
        output[index] = __fmul_rn(
            silu, up[row * up_row_stride + column]);
    }
}
"#;

    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    static EXACT_SILU_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    static CUDA_LOGGED: OnceLock<()> = OnceLock::new();

    fn ptx() -> candle_core::Result<&'static str> {
        match PTX.get_or_init(|| {
            compile_ptx_with_opts(
                CUDA_SOURCE,
                CompileOptions {
                    name: Some("indexed_moe_epilogue.cu".to_string()),
                    use_fast_math: Some(true),
                    ..Default::default()
                },
            )
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
        }) {
            Ok(ptx) => Ok(ptx.as_str()),
            Err(error) => {
                candle_core::bail!("failed to compile CUDA indexed MoE epilogue: {error}")
            }
        }
    }

    fn exact_silu_ptx() -> candle_core::Result<&'static str> {
        match EXACT_SILU_PTX.get_or_init(|| {
            compile_ptx_with_opts(
                EXACT_SILU_CUDA_SOURCE,
                CompileOptions {
                    name: Some("dense_silu_mul_exact.cu".to_string()),
                    use_fast_math: Some(false),
                    ..Default::default()
                },
            )
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
        }) {
            Ok(ptx) => Ok(ptx.as_str()),
            Err(error) => candle_core::bail!("failed to compile exact CUDA SwiGLU: {error}"),
        }
    }

    pub(crate) fn precompile(device: &Device) -> anyhow::Result<()> {
        if !device.is_cuda() {
            return Ok(());
        }
        let device = device.as_cuda_device()?;
        let ptx = ptx()?;
        for kernel in [
            NORMALIZE_KERNEL_NAME,
            SILU_MUL_KERNEL_NAME,
            WEIGHTED_SUM_KERNEL_NAME,
        ] {
            device.get_or_load_custom_func(kernel, CUDA_MODULE_NAME, ptx)?;
        }
        CUDA_LOGGED.get_or_init(|| {
            tracing::info!("CUDA indexed MoE fused epilogue enabled");
        });
        Ok(())
    }

    pub(crate) fn precompile_exact_silu(device: &Device) -> anyhow::Result<()> {
        if !device.is_cuda() {
            return Ok(());
        }
        let device = device.as_cuda_device()?;
        device.get_or_load_custom_func(
            EXACT_SILU_KERNEL_NAME,
            EXACT_SILU_MODULE_NAME,
            exact_silu_ptx()?,
        )?;
        Ok(())
    }

    struct NormalizeRoutes {
        geometry: RouteNormalizationGeometry,
    }

    impl candle_core::CustomOp2 for NormalizeRoutes {
        fn name(&self) -> &'static str {
            "cuda-normalized-route-weights"
        }

        fn cpu_fwd(
            &self,
            _probabilities: &CpuStorage,
            _probabilities_layout: &Layout,
            _expert_ids: &CpuStorage,
            _expert_ids_layout: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("cuda-normalized-route-weights is only dispatched on CUDA")
        }

        fn cuda_fwd(
            &self,
            probabilities: &CudaStorage,
            probabilities_layout: &Layout,
            expert_ids: &CudaStorage,
            expert_ids_layout: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let geometry = self.geometry;
            validate_storage(
                "route probabilities",
                probabilities,
                probabilities_layout,
                DType::F32,
                geometry.probability_elements,
            )?;
            validate_storage(
                "expert IDs",
                expert_ids,
                expert_ids_layout,
                DType::U32,
                geometry.route_elements,
            )?;
            ensure_same_device("route normalization", probabilities, expert_ids)?;
            let probabilities_view = checked_view_f32(
                probabilities.as_cuda_slice::<f32>()?,
                probabilities_layout,
                geometry.probability_elements,
                "route probabilities",
            )?;
            let expert_ids_view = checked_view_u32(
                expert_ids.as_cuda_slice::<u32>()?,
                expert_ids_layout,
                geometry.route_elements,
                "expert IDs",
            )?;
            let device = probabilities.device();
            let mut output = device.alloc_zeros::<f32>(geometry.route_elements)?;
            let function =
                device.get_or_load_custom_func(NORMALIZE_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
            let config = LaunchConfig {
                grid_dim: (geometry.tokens as u32, 1, 1),
                block_dim: (CUDA_NORMALIZATION_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let tokens = geometry.tokens as i32;
            let expert_count = geometry.expert_count as i32;
            let top_k = geometry.top_k as i32;
            let mut builder = function.builder();
            builder
                .arg(&probabilities_view)
                .arg(&expert_ids_view)
                .arg(&mut output)
                .arg(&tokens)
                .arg(&expert_count)
                .arg(&top_k);
            // SAFETY: Shape, dtype, offsets, bounds, and device identity were checked.
            unsafe { builder.launch(config) }.w()?;
            Ok((
                CudaStorage::wrap_cuda_slice(output, device.clone()),
                Shape::from((geometry.tokens, geometry.top_k)),
            ))
        }
    }

    struct SiluMul {
        geometry: ElementwiseGeometry,
        exact: bool,
    }

    impl candle_core::CustomOp2 for SiluMul {
        fn name(&self) -> &'static str {
            "cuda-silu-mul"
        }

        fn cpu_fwd(
            &self,
            _gate: &CpuStorage,
            _gate_layout: &Layout,
            _up: &CpuStorage,
            _up_layout: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("cuda-silu-mul is only dispatched on CUDA")
        }

        fn cuda_fwd(
            &self,
            gate: &CudaStorage,
            gate_layout: &Layout,
            up: &CudaStorage,
            up_layout: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let geometry = &self.geometry;
            let (gate_row_stride, gate_span) =
                elementwise_row_layout("SwiGLU gate", gate, gate_layout, geometry)?;
            let (up_row_stride, up_span) =
                elementwise_row_layout("SwiGLU up", up, up_layout, geometry)?;
            ensure_same_device("SwiGLU", gate, up)?;
            let gate_view = checked_view_f32(
                gate.as_cuda_slice::<f32>()?,
                gate_layout,
                gate_span,
                "SwiGLU gate",
            )?;
            let up_view =
                checked_view_f32(up.as_cuda_slice::<f32>()?, up_layout, up_span, "SwiGLU up")?;
            let device = gate.device();
            let mut output = device.alloc_zeros::<f32>(geometry.elements)?;
            let (kernel_name, module_name, ptx) = if self.exact {
                (
                    EXACT_SILU_KERNEL_NAME,
                    EXACT_SILU_MODULE_NAME,
                    exact_silu_ptx()?,
                )
            } else {
                (SILU_MUL_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)
            };
            let function = device.get_or_load_custom_func(kernel_name, module_name, ptx)?;
            let config = one_dimensional_launch(geometry.elements);
            let elements = geometry.elements as u64;
            let row_width = geometry.row_width as u64;
            let gate_row_stride = gate_row_stride as u64;
            let up_row_stride = up_row_stride as u64;
            let mut builder = function.builder();
            builder
                .arg(&gate_view)
                .arg(&up_view)
                .arg(&mut output)
                .arg(&elements)
                .arg(&row_width)
                .arg(&gate_row_stride)
                .arg(&up_row_stride);
            // SAFETY: Shape, dtype, offsets, bounds, and device identity were checked.
            unsafe { builder.launch(config) }.w()?;
            Ok((
                CudaStorage::wrap_cuda_slice(output, device.clone()),
                Shape::from_dims(&geometry.shape),
            ))
        }
    }

    struct WeightedRouteSum {
        geometry: WeightedRouteGeometry,
    }

    impl candle_core::CustomOp2 for WeightedRouteSum {
        fn name(&self) -> &'static str {
            "cuda-weighted-route-sum"
        }

        fn cpu_fwd(
            &self,
            _routes: &CpuStorage,
            _routes_layout: &Layout,
            _weights: &CpuStorage,
            _weights_layout: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("cuda-weighted-route-sum is only dispatched on CUDA")
        }

        fn cuda_fwd(
            &self,
            routes: &CudaStorage,
            routes_layout: &Layout,
            weights: &CudaStorage,
            weights_layout: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let geometry = self.geometry;
            validate_storage(
                "route outputs",
                routes,
                routes_layout,
                DType::F32,
                geometry.route_elements,
            )?;
            validate_storage(
                "route weights",
                weights,
                weights_layout,
                DType::F32,
                geometry.weight_elements,
            )?;
            ensure_same_device("weighted route sum", routes, weights)?;
            let routes_view = checked_view_f32(
                routes.as_cuda_slice::<f32>()?,
                routes_layout,
                geometry.route_elements,
                "route outputs",
            )?;
            let weights_view = checked_view_f32(
                weights.as_cuda_slice::<f32>()?,
                weights_layout,
                geometry.weight_elements,
                "route weights",
            )?;
            let device = routes.device();
            // SAFETY: The kernel writes every output element.
            let mut output = unsafe { device.alloc::<f32>(geometry.output_elements)? };
            let function = device.get_or_load_custom_func(
                WEIGHTED_SUM_KERNEL_NAME,
                CUDA_MODULE_NAME,
                ptx()?,
            )?;
            let config = one_dimensional_launch(geometry.output_elements);
            let top_k = geometry.top_k as i32;
            let hidden_dim = geometry.hidden_dim as i32;
            let elements = geometry.output_elements as u64;
            let mut builder = function.builder();
            builder
                .arg(&routes_view)
                .arg(&weights_view)
                .arg(&mut output)
                .arg(&top_k)
                .arg(&hidden_dim)
                .arg(&elements);
            // SAFETY: Shape, dtype, offsets, bounds, and device identity were checked.
            unsafe { builder.launch(config) }.w()?;
            Ok((
                CudaStorage::wrap_cuda_slice(output, device.clone()),
                Shape::from((geometry.tokens, geometry.hidden_dim)),
            ))
        }
    }

    fn one_dimensional_launch(elements: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (elements.div_ceil(CUDA_THREADS) as u32, 1, 1),
            block_dim: (CUDA_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn validate_storage(
        name: &str,
        storage: &CudaStorage,
        layout: &Layout,
        dtype: DType,
        elements: usize,
    ) -> candle_core::Result<()> {
        if storage.dtype() != dtype
            || !layout.is_contiguous()
            || layout.shape().elem_count() != elements
        {
            candle_core::bail!(
                "CUDA indexed MoE {name} must be contiguous {dtype:?} with {elements} elements"
            )
        }
        Ok(())
    }

    /// Return the physical stride and span for logically row-major tensors.
    ///
    /// Grouped gate/up MMVQ packs every projection into each batch row. A
    /// narrow view of one projection therefore has contiguous values within a
    /// row but a larger physical stride between rows. The SwiGLU kernels can
    /// consume that layout directly and avoid two materializing copies during
    /// warmup and small-batch inference.
    fn elementwise_row_layout(
        name: &str,
        storage: &CudaStorage,
        layout: &Layout,
        geometry: &ElementwiseGeometry,
    ) -> candle_core::Result<(usize, usize)> {
        if storage.dtype() != DType::F32 || layout.shape().elem_count() != geometry.elements {
            candle_core::bail!(
                "CUDA indexed MoE {name} must be F32 with {} elements",
                geometry.elements
            )
        }
        if geometry.rows == 1 {
            if !layout.is_contiguous() {
                candle_core::bail!("CUDA indexed MoE {name} row is not contiguous")
            }
            return Ok((geometry.row_width, geometry.row_width));
        }

        let dims = layout.dims();
        let strides = layout.stride();
        if dims.last().copied() != Some(geometry.row_width)
            || geometry.row_width > 1 && strides.last().copied() != Some(1)
        {
            candle_core::bail!("CUDA indexed MoE {name} rows are not contiguous")
        }

        let mut logical_rows_after = 1usize;
        let mut row_stride = None;
        for index in (0..dims.len().saturating_sub(1)).rev() {
            if dims[index] > 1 {
                if !strides[index].is_multiple_of(logical_rows_after) {
                    candle_core::bail!("CUDA indexed MoE {name} row strides are irregular")
                }
                let candidate = strides[index] / logical_rows_after;
                if row_stride.is_some_and(|stride| stride != candidate) {
                    candle_core::bail!("CUDA indexed MoE {name} row strides are irregular")
                }
                row_stride = Some(candidate);
            }
            logical_rows_after = logical_rows_after.checked_mul(dims[index]).ok_or_else(|| {
                candle_core::Error::Msg(format!("CUDA indexed MoE {name} row count overflows"))
            })?;
        }
        let row_stride = row_stride.ok_or_else(|| {
            candle_core::Error::Msg(format!("CUDA indexed MoE {name} row stride is missing"))
        })?;
        if row_stride < geometry.row_width {
            candle_core::bail!("CUDA indexed MoE {name} rows overlap")
        }
        let span = (geometry.rows - 1)
            .checked_mul(row_stride)
            .and_then(|offset| offset.checked_add(geometry.row_width))
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("CUDA indexed MoE {name} view span overflows"))
            })?;
        Ok((row_stride, span))
    }

    fn ensure_same_device(
        name: &str,
        lhs: &CudaStorage,
        rhs: &CudaStorage,
    ) -> candle_core::Result<()> {
        if lhs.device().id() != rhs.device().id() {
            candle_core::bail!("CUDA indexed MoE {name} inputs use different devices")
        }
        Ok(())
    }

    fn checked_view_f32<'a>(
        slice: &'a candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
        layout: &Layout,
        elements: usize,
        name: &str,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>> {
        let start = layout.start_offset();
        let end = start
            .checked_add(elements)
            .ok_or_else(|| candle_core::Error::Msg(format!("{name} CUDA view overflow")))?;
        if end > slice.len() {
            candle_core::bail!("{name} CUDA view exceeds its storage")
        }
        Ok(slice.slice(start..end))
    }

    fn checked_view_u32<'a>(
        slice: &'a candle_core::cuda_backend::cudarc::driver::CudaSlice<u32>,
        layout: &Layout,
        elements: usize,
        name: &str,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, u32>> {
        let start = layout.start_offset();
        let end = start
            .checked_add(elements)
            .ok_or_else(|| candle_core::Error::Msg(format!("{name} CUDA view overflow")))?;
        if end > slice.len() {
            candle_core::bail!("{name} CUDA view exceeds its storage")
        }
        Ok(slice.slice(start..end))
    }

    pub(crate) fn normalized_route_weights(
        probabilities: &Tensor,
        expert_ids: &Tensor,
    ) -> anyhow::Result<Tensor> {
        let (tokens, expert_count) = probabilities.dims2()?;
        let (id_tokens, top_k) = expert_ids.dims2()?;
        let geometry = RouteNormalizationGeometry::new(tokens, expert_count, id_tokens, top_k)?;
        anyhow::ensure!(
            probabilities.dtype() == DType::F32 && expert_ids.dtype() == DType::U32,
            "CUDA route normalization requires f32 probabilities and u32 IDs"
        );
        anyhow::ensure!(
            probabilities.device().is_cuda()
                && probabilities.device().same_device(expert_ids.device()),
            "CUDA route normalization inputs must share one CUDA device"
        );
        probabilities
            .apply_op2_no_bwd(expert_ids, &NormalizeRoutes { geometry })
            .map_err(Into::into)
    }

    pub(crate) fn silu_mul(gate: &Tensor, up: &Tensor) -> anyhow::Result<Tensor> {
        let geometry = ElementwiseGeometry::new(gate.shape().dims(), up.shape().dims())?;
        anyhow::ensure!(
            gate.dtype() == DType::F32 && up.dtype() == DType::F32,
            "CUDA SwiGLU requires f32 inputs"
        );
        anyhow::ensure!(
            gate.device().is_cuda() && gate.device().same_device(up.device()),
            "CUDA SwiGLU inputs must share one CUDA device"
        );
        gate.apply_op2_no_bwd(
            up,
            &SiluMul {
                geometry,
                exact: false,
            },
        )
        .map_err(Into::into)
    }

    pub(crate) fn silu_mul_exact(gate: &Tensor, up: &Tensor) -> anyhow::Result<Tensor> {
        let geometry = ElementwiseGeometry::new(gate.shape().dims(), up.shape().dims())?;
        anyhow::ensure!(
            gate.dtype() == DType::F32 && up.dtype() == DType::F32,
            "exact CUDA SwiGLU requires f32 inputs"
        );
        anyhow::ensure!(
            gate.device().is_cuda() && gate.device().same_device(up.device()),
            "exact CUDA SwiGLU inputs must share one CUDA device"
        );
        gate.apply_op2_no_bwd(
            up,
            &SiluMul {
                geometry,
                exact: true,
            },
        )
        .map_err(Into::into)
    }

    pub(crate) fn weighted_route_sum(routes: &Tensor, weights: &Tensor) -> anyhow::Result<Tensor> {
        let (tokens, top_k, hidden_dim) = routes.dims3()?;
        let (weight_tokens, weight_top_k) = weights.dims2()?;
        let geometry =
            WeightedRouteGeometry::new(tokens, top_k, hidden_dim, weight_tokens, weight_top_k)?;
        anyhow::ensure!(
            routes.dtype() == DType::F32 && weights.dtype() == DType::F32,
            "CUDA weighted route sum requires f32 inputs"
        );
        anyhow::ensure!(
            routes.device().is_cuda() && routes.device().same_device(weights.device()),
            "CUDA weighted route sum inputs must share one CUDA device"
        );
        routes
            .apply_op2_no_bwd(weights, &WeightedRouteSum { geometry })
            .map_err(Into::into)
    }
}

#[cfg(feature = "cuda")]
pub(super) use cuda::{
    normalized_route_weights, precompile, precompile_exact_silu, silu_mul, silu_mul_exact,
    weighted_route_sum,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_normalization_geometry_checks_shapes_and_bounds() -> Result<()> {
        let geometry = RouteNormalizationGeometry::new(7, 128, 7, 6)?;
        assert_eq!(geometry.probability_elements, 7 * 128);
        assert_eq!(geometry.route_elements, 7 * 6);
        assert!(RouteNormalizationGeometry::new(0, 4, 0, 2).is_err());
        assert!(RouteNormalizationGeometry::new(2, 4, 3, 2).is_err());
        assert!(RouteNormalizationGeometry::new(2, 4, 2, 5).is_err());
        Ok(())
    }

    #[test]
    fn epilogue_geometry_checks_element_and_route_shapes() -> Result<()> {
        let elementwise = ElementwiseGeometry::new(&[3, 2, 17], &[3, 2, 17])?;
        assert_eq!(elementwise.elements, 102);
        assert_eq!(elementwise.rows, 6);
        assert_eq!(elementwise.row_width, 17);
        assert!(ElementwiseGeometry::new(&[3, 2], &[3, 3]).is_err());
        assert!(ElementwiseGeometry::new(&[3, 0], &[3, 0]).is_err());

        let weighted = WeightedRouteGeometry::new(3, 2, 17, 3, 2)?;
        assert_eq!(weighted.route_elements, 102);
        assert_eq!(weighted.weight_elements, 6);
        assert_eq!(weighted.output_elements, 51);
        assert!(WeightedRouteGeometry::new(3, 2, 17, 2, 2).is_err());
        Ok(())
    }

    #[test]
    fn epilogue_reference_math_tracks_route_order_and_floor() {
        let probabilities = [[0.1f32, 0.6, 0.3], [0.0, 0.0, 0.0]];
        let ids = [[2usize, 1], [0, 2]];
        let mut normalized = [[0.0f32; 2]; 2];
        for token in 0..2 {
            let selected = ids[token].map(|expert| probabilities[token][expert]);
            let denominator = selected.iter().copied().sum::<f32>().max(6.103_515_6e-5);
            normalized[token] = selected.map(|weight| weight / denominator);
        }
        assert_eq!(normalized[0], [1.0 / 3.0, 2.0 / 3.0]);
        assert_eq!(normalized[1], [0.0, 0.0]);

        let gate = [-2.0f32, 0.0, 2.0];
        let up = [0.5f32, 3.0, -0.25];
        let activated = std::array::from_fn::<_, 3, _>(|index| {
            gate[index] / (1.0 + (-gate[index]).exp()) * up[index]
        });
        assert_eq!(activated[1], 0.0);
        assert!(activated[0] < 0.0 && activated[2] < 0.0);

        let routes = [[[3.0f32, 6.0], [9.0, 12.0]]];
        let weights = [[1.0f32 / 3.0, 2.0 / 3.0]];
        let output = std::array::from_fn::<_, 2, _>(|column| {
            (0..2)
                .map(|route| routes[0][route][column] * weights[0][route])
                .sum::<f32>()
        });
        assert_eq!(output, [7.0, 10.0]);
    }

    #[cfg(feature = "cuda")]
    mod cuda_tests {
        use candle_core::{DType, Device, Module, Tensor};

        use super::*;

        fn assert_close(actual: &Tensor, expected: &[f32], tolerance: f32) -> anyhow::Result<()> {
            let actual = actual
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "CUDA indexed MoE epilogue mismatch at {index}: {actual} != {expected}"
                );
            }
            Ok(())
        }

        #[test]
        fn cuda_epilogue_kernels_match_cpu_with_offset_views() -> anyhow::Result<()> {
            let Some(device) = crate::test_cuda_device() else {
                return Ok(());
            };

            let probabilities = Tensor::from_vec(
                vec![99.0f32; 3]
                    .into_iter()
                    .chain([0.1, 0.6, 0.3, 0.0, 0.0, 0.0, 0.25, 0.5, 0.25])
                    .collect::<Vec<_>>(),
                (4, 3),
                &device,
            )?
            .narrow(0, 1, 3)?;
            let expert_ids = Tensor::from_vec(
                vec![0u32, 0]
                    .into_iter()
                    .chain([2, 1, 0, 2, 1, 0])
                    .collect::<Vec<_>>(),
                (4, 2),
                &device,
            )?
            .narrow(0, 1, 3)?;
            assert_ne!(probabilities.layout().start_offset(), 0);
            assert_ne!(expert_ids.layout().start_offset(), 0);
            let normalized = normalized_route_weights(&probabilities, &expert_ids)?;
            assert_close(
                &normalized,
                &[1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0, 2.0 / 3.0, 1.0 / 3.0],
                2e-6,
            )?;

            let gate_values = [-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0];
            let up_values = [0.5f32, 1.0, 2.0, -1.0, -0.25, 0.75];
            let gate = Tensor::from_vec(
                [vec![99.0f32; 6], gate_values.to_vec()].concat(),
                (2, 2, 3),
                &device,
            )?
            .narrow(0, 1, 1)?;
            let up = Tensor::from_vec(
                [vec![99.0f32; 6], up_values.to_vec()].concat(),
                (2, 2, 3),
                &device,
            )?
            .narrow(0, 1, 1)?;
            let expected_silu = gate_values
                .iter()
                .zip(up_values)
                .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
                .collect::<Vec<_>>();
            let activated = silu_mul(&gate, &up)?;
            assert_close(&activated, &expected_silu, 2e-5)?;

            let expected_exact = (candle_nn::Activation::Silu.forward(&gate)? * &up)?;
            let activated_exact = silu_mul_exact(&gate, &up)?;
            let expected_exact = expected_exact
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let activated_exact = activated_exact
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(activated_exact, expected_exact);

            // Grouped gate/up MMVQ stores both projections in each batch row.
            // Narrowing the last dimension produces exactly the row-strided
            // layout used by Llama/Qwen warmup and small-batch prefill.
            let mut packed_gate = Vec::new();
            let mut packed_up = Vec::new();
            for row in 0..4 {
                packed_gate.extend((0..3).map(|column| (row * 3 + column) as f32 / 7.0 - 1.0));
                packed_gate.extend([99.0f32; 3]);
                packed_up.extend([99.0f32; 3]);
                packed_up.extend((0..3).map(|column| (row * 5 + column) as f32 / 11.0 + 0.25));
            }
            let strided_gate = Tensor::from_vec(packed_gate, (2, 2, 6), &device)?.narrow(
                candle_core::D::Minus1,
                0,
                3,
            )?;
            let strided_up = Tensor::from_vec(packed_up, (2, 2, 6), &device)?.narrow(
                candle_core::D::Minus1,
                3,
                3,
            )?;
            assert!(!strided_gate.is_contiguous());
            assert!(!strided_up.is_contiguous());
            let expected_strided =
                (candle_nn::Activation::Silu.forward(&strided_gate)? * &strided_up)?;
            let actual_strided = silu_mul_exact(&strided_gate, &strided_up)?;
            let expected_strided = expected_strided
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual_strided = actual_strided
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual_strided, expected_strided);

            let route_values = (0..3 * 2 * 5)
                .map(|index| (index as f32 - 11.0) / 7.0)
                .collect::<Vec<_>>();
            let weight_values = [0.25f32, 0.75, 0.6, 0.4, 1.0 / 3.0, 2.0 / 3.0];
            let routes = Tensor::from_vec(
                [vec![99.0f32; 10], route_values.clone()].concat(),
                (4, 2, 5),
                &device,
            )?
            .narrow(0, 1, 3)?;
            let weights = Tensor::from_vec(
                [vec![99.0f32; 2], weight_values.to_vec()].concat(),
                (4, 2),
                &device,
            )?
            .narrow(0, 1, 3)?;
            let mut expected_sum = vec![0.0f32; 3 * 5];
            for token in 0..3 {
                for column in 0..5 {
                    for route in 0..2 {
                        expected_sum[token * 5 + column] += route_values
                            [(token * 2 + route) * 5 + column]
                            * weight_values[token * 2 + route];
                    }
                }
            }
            let summed = weighted_route_sum(&routes, &weights)?;
            assert_close(&summed, &expected_sum, 2e-6)
        }
    }
}
