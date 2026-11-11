use anyhow::{Context, Result, ensure};

const Q4_0_BLOCK_SIZE: usize = 32;
const Q4_0_TYPE_SIZE: usize = 18;
#[cfg(feature = "cuda")]
const CUDA_WARP_SIZE: usize = 32;
const CUDA_WARPS_PER_BLOCK: usize = 4;
const CUDA_PROMPT_ROUTE_TILE: usize = 8;
const CUDA_PROMPT_OUTPUT_TILE: usize = 32;
const CUDA_MAX_GRID_Y: usize = 65_535;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Q4IndexedGeometry {
    tokens: usize,
    input_rows: usize,
    top_k: usize,
    num_experts: usize,
    out_dim: usize,
    in_dim: usize,
    routes: usize,
    input_elements: usize,
    output_elements: usize,
    weight_bytes: usize,
}

impl Q4IndexedGeometry {
    #[allow(clippy::too_many_arguments)]
    fn new(
        tokens: usize,
        input_rows: usize,
        in_dim: usize,
        id_tokens: usize,
        top_k: usize,
        num_experts: usize,
        out_dim: usize,
    ) -> Result<Self> {
        ensure!(tokens > 0, "CUDA Q4_0 indexed MoE requires tokens");
        ensure!(top_k > 0, "CUDA Q4_0 indexed MoE requires routes");
        ensure!(
            tokens == id_tokens,
            "CUDA Q4_0 indexed MoE token counts differ"
        );
        ensure!(
            input_rows == 1 || input_rows == top_k,
            "CUDA Q4_0 indexed MoE input rows must be one or top-k"
        );
        ensure!(
            num_experts > 0 && top_k <= num_experts,
            "CUDA Q4_0 indexed MoE expert geometry differs"
        );
        ensure!(
            num_experts <= CUDA_MAX_GRID_Y,
            "CUDA Q4_0 indexed MoE expert count exceeds the CUDA grid limit"
        );
        ensure!(out_dim > 0, "CUDA Q4_0 indexed MoE requires output rows");
        ensure!(
            in_dim > 0 && in_dim.is_multiple_of(Q4_0_BLOCK_SIZE),
            "CUDA Q4_0 indexed MoE input width must be a multiple of {Q4_0_BLOCK_SIZE}"
        );

        let routes = checked_mul(tokens, top_k, "route count")?;
        ensure!(
            routes <= CUDA_MAX_GRID_Y,
            "CUDA Q4_0 indexed MoE route count exceeds the CUDA grid limit"
        );
        let input_elements = checked_mul(
            checked_mul(tokens, input_rows, "input rows")?,
            in_dim,
            "input elements",
        )?;
        let output_elements = checked_mul(routes, out_dim, "output elements")?;
        let blocks_per_row = in_dim / Q4_0_BLOCK_SIZE;
        let weight_bytes = checked_mul(
            checked_mul(
                checked_mul(num_experts, out_dim, "expert output rows")?,
                blocks_per_row,
                "expert Q4_0 blocks",
            )?,
            Q4_0_TYPE_SIZE,
            "expert storage bytes",
        )?;

        for (name, value) in [
            ("tokens", tokens),
            ("input rows", input_rows),
            ("top-k", top_k),
            ("expert count", num_experts),
            ("output width", out_dim),
            ("input width", in_dim),
        ] {
            ensure!(
                value <= i32::MAX as usize,
                "CUDA Q4_0 indexed MoE {name} exceeds the kernel index limit"
            );
        }
        ensure!(
            out_dim.div_ceil(CUDA_WARPS_PER_BLOCK) <= i32::MAX as usize,
            "CUDA Q4_0 indexed MoE output grid exceeds the CUDA limit"
        );

        Ok(Self {
            tokens,
            input_rows,
            top_k,
            num_experts,
            out_dim,
            in_dim,
            routes,
            input_elements,
            output_elements,
            weight_bytes,
        })
    }
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .with_context(|| format!("CUDA Q4_0 indexed MoE {label} overflows usize"))
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::{path::PathBuf, sync::OnceLock};

    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::WrapErr;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{CpuStorage, CudaStorage, D, DType, Device, Layout, Shape, Tensor};

    use super::{
        CUDA_PROMPT_OUTPUT_TILE, CUDA_PROMPT_ROUTE_TILE, CUDA_WARP_SIZE, CUDA_WARPS_PER_BLOCK,
        Q4IndexedGeometry,
    };

    const CUDA_MODULE_NAME: &str = "dinf_indexed_moe_q4_0_v3";
    const DECODE_KERNEL_NAME: &str = "dinf_indexed_moe_q4_0_f32";
    const PROMPT_KERNEL_NAME: &str = "dinf_indexed_moe_q4_0_grouped_f32";

    // Each warp computes one output row. Four warps share each input tile,
    // while Q4_0 weights stay in their canonical 18-byte GGUF blocks.
    const CUDA_SOURCE: &str = r#"
#include <cuda_fp16.h>

#define DINF_Q4_BLOCK_SIZE 32
#define DINF_Q4_TYPE_SIZE 18
#define DINF_WARP_SIZE 32

extern "C" __global__ void dinf_indexed_moe_q4_0_f32(
    const float *input,
    const unsigned int *expert_ids,
    const unsigned char *weights,
    float *output,
    int routes,
    int top_k,
    int input_rows,
    int in_dim,
    int out_dim,
    int num_experts,
    int activation_kind) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int route = blockIdx.y;
    const int output_row = blockIdx.x * blockDim.y + warp;
    if (route >= routes) {
        return;
    }

    const unsigned int expert = expert_ids[route];
    if (expert >= (unsigned int) num_experts) {
        return;
    }
    const bool valid_output = output_row < out_dim;
    const int input_row = input_rows == 1 ? route / top_k : route;
    const float *input_row_ptr = input + (unsigned long long) input_row * in_dim;
    const int blocks_per_row = in_dim / DINF_Q4_BLOCK_SIZE;
    const int safe_output_row = valid_output ? output_row : 0;
    const unsigned long long expert_row =
        (unsigned long long) expert * out_dim + safe_output_row;
    const unsigned char *weight_row = weights
        + expert_row * blocks_per_row * DINF_Q4_TYPE_SIZE;

    __shared__ float input_tile[DINF_Q4_BLOCK_SIZE];
    float sum = 0.0f;
    for (int block = 0; block < blocks_per_row; ++block) {
        if (warp == 0) {
            float value = input_row_ptr[block * DINF_Q4_BLOCK_SIZE + lane];
            if (activation_kind == 1) {
                value = fmaxf(value, 0.0f);
                value *= value;
            }
            input_tile[lane] = value;
        }
        __syncthreads();

        if (valid_output) {
            const unsigned char *packed =
                weight_row + (unsigned long long) block * DINF_Q4_TYPE_SIZE;
            const float scale = __half2float(*reinterpret_cast<const __half *>(packed));
            const unsigned char quant = packed[2 + (lane & 15)];
            const int value = lane < 16 ? (quant & 15) - 8 : (quant >> 4) - 8;
            sum = fmaf(input_tile[lane], scale * (float) value, sum);
        }
        __syncthreads();
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0 && valid_output) {
        output[(unsigned long long) route * out_dim + output_row] = sum;
    }
}

#define DINF_PROMPT_M 8
#define DINF_PROMPT_N 32

__device__ __forceinline__ int dinf_lower_bound_expert(
    const unsigned int *sorted_experts,
    int routes,
    unsigned int expert) {
    int low = 0;
    int high = routes;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (sorted_experts[middle] < expert) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

__device__ __forceinline__ int dinf_upper_bound_expert(
    const unsigned int *sorted_experts,
    int routes,
    unsigned int expert) {
    int low = 0;
    int high = routes;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (sorted_experts[middle] <= expert) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

// Prompt routes are sorted by expert before this launch. One block owns an
// expert, 32 output rows, and eight routes. It expands each 32x32 Q4_0 weight
// tile once, then all eight route warps reuse that tile.
extern "C" __global__ void dinf_indexed_moe_q4_0_grouped_f32(
    const float *input,
    const unsigned int *sorted_route_ids,
    const unsigned int *sorted_experts,
    const unsigned char *weights,
    float *output,
    int routes,
    int top_k,
    int input_rows,
    int in_dim,
    int out_dim,
    int num_experts,
    int activation_kind) {
    const int lane = threadIdx.x;
    const int route_lane = threadIdx.y;
    const int expert = blockIdx.y;
    const int output_row = blockIdx.x * DINF_PROMPT_N + lane;

    if (expert >= num_experts) {
        return;
    }

    __shared__ int segment_start;
    __shared__ int segment_end;
    __shared__ float input_tile[DINF_PROMPT_M][DINF_Q4_BLOCK_SIZE];
    // Padding keeps lanes on separate banks while they read one K column.
    __shared__ float weight_tile[DINF_PROMPT_N][DINF_Q4_BLOCK_SIZE + 1];

    if (lane == 0 && route_lane == 0) {
        segment_start = dinf_lower_bound_expert(
            sorted_experts, routes, (unsigned int) expert);
        segment_end = dinf_upper_bound_expert(
            sorted_experts, routes, (unsigned int) expert);
    }
    __syncthreads();

    const bool valid_output = output_row < out_dim;
    const int blocks_per_row = in_dim / DINF_Q4_BLOCK_SIZE;

    for (int route_tile_start = segment_start;
         route_tile_start < segment_end;
         route_tile_start += DINF_PROMPT_M) {
        const int sorted_index = route_tile_start + route_lane;
        const bool active_route = sorted_index < segment_end;
        const unsigned int candidate_route =
            active_route ? sorted_route_ids[sorted_index] : 0;
        const bool valid_route =
            active_route && candidate_route < (unsigned int) routes;
        const unsigned int route = valid_route ? candidate_route : 0;
        const int input_row = input_rows == 1 ? route / top_k : route;
        float sum = 0.0f;

        for (int block = 0; block < blocks_per_row; ++block) {
            float input_value = valid_route
                ? input[(unsigned long long) input_row * in_dim
                    + block * DINF_Q4_BLOCK_SIZE + lane]
                : 0.0f;
            if (activation_kind == 1) {
                input_value = fmaxf(input_value, 0.0f);
                input_value *= input_value;
            }
            input_tile[route_lane][lane] = input_value;

            const int thread_index = route_lane * DINF_PROMPT_N + lane;
#pragma unroll
            for (int item = thread_index;
                 item < DINF_PROMPT_N * DINF_Q4_BLOCK_SIZE;
                 item += DINF_PROMPT_M * DINF_PROMPT_N) {
                const int tile_row = item / DINF_Q4_BLOCK_SIZE;
                const int inner = item % DINF_Q4_BLOCK_SIZE;
                const int candidate_output_row = blockIdx.x * DINF_PROMPT_N + tile_row;
                const bool valid_weight = candidate_output_row < out_dim;
                const int safe_output_row = valid_weight ? candidate_output_row : 0;
                const unsigned long long expert_row =
                    (unsigned long long) expert * out_dim + safe_output_row;
                const unsigned char *packed = weights
                    + (expert_row * blocks_per_row + block) * DINF_Q4_TYPE_SIZE;
                const float scale = __half2float(*reinterpret_cast<const __half *>(packed));
                const unsigned char quant = packed[2 + (inner & 15)];
                const int value = inner < 16 ? (quant & 15) - 8 : (quant >> 4) - 8;
                weight_tile[tile_row][inner] = valid_weight
                    ? scale * (float) value
                    : 0.0f;
            }
            __syncthreads();

            if (valid_route && valid_output) {
#pragma unroll
                for (int inner = 0; inner < DINF_Q4_BLOCK_SIZE; ++inner) {
                    sum = fmaf(
                        input_tile[route_lane][inner],
                        weight_tile[lane][inner],
                        sum);
                }
            }
            __syncthreads();
        }

        if (valid_route && valid_output) {
            output[(unsigned long long) route * out_dim + output_row] = sum;
        }
    }
}
"#;

    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    static CUDA_LOGGED: OnceLock<()> = OnceLock::new();

    fn cuda_include_paths() -> Vec<String> {
        let mut paths = Vec::new();
        for variable in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
            let Some(root) = std::env::var_os(variable) else {
                continue;
            };
            let root = PathBuf::from(root);
            let include = if root.join("cuda_fp16.h").is_file() {
                root
            } else {
                root.join("include")
            };
            if include.join("cuda_fp16.h").is_file() {
                let include = include.to_string_lossy().into_owned();
                if !paths.contains(&include) {
                    paths.push(include);
                }
            }
        }
        for candidate in ["/usr/local/cuda/include", "/opt/cuda/include"] {
            let include = PathBuf::from(candidate);
            if include.join("cuda_fp16.h").is_file() {
                let include = include.to_string_lossy().into_owned();
                if !paths.contains(&include) {
                    paths.push(include);
                }
            }
        }
        paths
    }

    fn ptx() -> candle_core::Result<&'static str> {
        match PTX.get_or_init(|| {
            let include_paths = cuda_include_paths();
            if include_paths.is_empty() {
                return Err(
                    "CUDA toolkit include directory containing cuda_fp16.h was not found".into(),
                );
            }
            compile_ptx_with_opts(
                CUDA_SOURCE,
                CompileOptions {
                    include_paths,
                    name: Some("indexed_moe_q4_0.cu".to_string()),
                    use_fast_math: Some(true),
                    ..Default::default()
                },
            )
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
        }) {
            Ok(ptx) => Ok(ptx.as_str()),
            Err(error) => candle_core::bail!("failed to compile CUDA Q4_0 indexed MoE: {error}"),
        }
    }

    pub(crate) fn precompile(device: &Device) -> anyhow::Result<()> {
        if !device.is_cuda()
            || std::env::var_os("DINF_DISABLE_INDEXED_MOE_CUDA").is_some()
            || std::env::var_os("DINF_DISABLE_INDEXED_MOE_Q4_0_CUDA").is_some()
        {
            return Ok(());
        }
        let device = device.as_cuda_device()?;
        device.get_or_load_custom_func(DECODE_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        device.get_or_load_custom_func(PROMPT_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        CUDA_LOGGED.get_or_init(|| {
            tracing::info!("CUDA Q4_0 decode and grouped prompt expert matmul enabled");
        });
        Ok(())
    }

    struct DecodeIndexedQ4<'a> {
        weights: &'a QTensor,
        geometry: Q4IndexedGeometry,
        activation_kind: i32,
    }

    impl candle_core::CustomOp2 for DecodeIndexedQ4<'_> {
        fn name(&self) -> &'static str {
            "cuda-indexed-moe-q4-0"
        }

        fn cpu_fwd(
            &self,
            _input: &CpuStorage,
            _input_layout: &Layout,
            _ids: &CpuStorage,
            _ids_layout: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("cuda-indexed-moe-q4-0 is only dispatched on CUDA")
        }

        fn cuda_fwd(
            &self,
            input: &CudaStorage,
            input_layout: &Layout,
            ids: &CudaStorage,
            ids_layout: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let geometry = self.geometry;
            validate_storage(
                "input",
                input,
                input_layout,
                DType::F32,
                geometry.input_elements,
            )?;
            validate_storage("expert IDs", ids, ids_layout, DType::U32, geometry.routes)?;
            if input.device().id() != ids.device().id() {
                candle_core::bail!("CUDA Q4_0 indexed MoE inputs use different devices")
            }
            if self.weights.dtype() != GgmlDType::Q4_0
                || self.weights.shape().dims()
                    != [geometry.num_experts, geometry.out_dim, geometry.in_dim]
                || self.weights.storage_size_in_bytes() != geometry.weight_bytes
                || !self
                    .weights
                    .device()
                    .same_device(&Device::Cuda(input.device().clone()))
            {
                candle_core::bail!("CUDA Q4_0 indexed MoE weight geometry differs")
            }

            let input_view = checked_view_f32(
                input.as_cuda_slice::<f32>()?,
                input_layout,
                geometry.input_elements,
                "input",
            )?;
            let ids_view = checked_view_u32(
                ids.as_cuda_slice::<u32>()?,
                ids_layout,
                geometry.routes,
                "expert IDs",
            )?;
            let device = input.device();
            let stream = device.cuda_stream();
            let (weight_ptr, _weight_guard) = self.weights.device_ptr_with_guard(&stream)?;
            let weight_address = weight_ptr as u64;
            let mut output = device.alloc_zeros::<f32>(geometry.output_elements)?;
            let function =
                device.get_or_load_custom_func(DECODE_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
            let config = LaunchConfig {
                grid_dim: (
                    geometry.out_dim.div_ceil(CUDA_WARPS_PER_BLOCK) as u32,
                    geometry.routes as u32,
                    1,
                ),
                block_dim: (CUDA_WARP_SIZE as u32, CUDA_WARPS_PER_BLOCK as u32, 1),
                shared_mem_bytes: 0,
            };
            let routes = geometry.routes as i32;
            let top_k = geometry.top_k as i32;
            let input_rows = geometry.input_rows as i32;
            let in_dim = geometry.in_dim as i32;
            let out_dim = geometry.out_dim as i32;
            let num_experts = geometry.num_experts as i32;
            let activation_kind = self.activation_kind;
            let mut builder = function.builder();
            builder
                .arg(&input_view)
                .arg(&ids_view)
                .arg(&weight_address)
                .arg(&mut output)
                .arg(&routes)
                .arg(&top_k)
                .arg(&input_rows)
                .arg(&in_dim)
                .arg(&out_dim)
                .arg(&num_experts)
                .arg(&activation_kind);
            // SAFETY: Geometry, storage bounds, device identity, and all kernel
            // index limits were checked before this launch.
            unsafe { builder.launch(config) }.w()?;

            Ok((
                CudaStorage::wrap_cuda_slice(output, device.clone()),
                Shape::from((geometry.tokens, geometry.top_k, geometry.out_dim)),
            ))
        }
    }

    struct GroupedPromptIndexedQ4<'a> {
        weights: &'a QTensor,
        geometry: Q4IndexedGeometry,
        activation_kind: i32,
    }

    impl candle_core::CustomOp3 for GroupedPromptIndexedQ4<'_> {
        fn name(&self) -> &'static str {
            "cuda-indexed-moe-q4-0-grouped-prompt"
        }

        fn cpu_fwd(
            &self,
            _input: &CpuStorage,
            _input_layout: &Layout,
            _sorted_route_ids: &CpuStorage,
            _sorted_route_ids_layout: &Layout,
            _sorted_experts: &CpuStorage,
            _sorted_experts_layout: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("cuda-indexed-moe-q4-0-grouped-prompt is only dispatched on CUDA")
        }

        fn cuda_fwd(
            &self,
            input: &CudaStorage,
            input_layout: &Layout,
            sorted_route_ids: &CudaStorage,
            sorted_route_ids_layout: &Layout,
            sorted_experts: &CudaStorage,
            sorted_experts_layout: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let geometry = self.geometry;
            validate_storage(
                "input",
                input,
                input_layout,
                DType::F32,
                geometry.input_elements,
            )?;
            validate_storage(
                "sorted route IDs",
                sorted_route_ids,
                sorted_route_ids_layout,
                DType::U32,
                geometry.routes,
            )?;
            validate_storage(
                "sorted experts",
                sorted_experts,
                sorted_experts_layout,
                DType::U32,
                geometry.routes,
            )?;
            if input.device().id() != sorted_route_ids.device().id()
                || input.device().id() != sorted_experts.device().id()
            {
                candle_core::bail!("CUDA Q4_0 grouped prompt inputs use different devices")
            }
            if self.weights.dtype() != GgmlDType::Q4_0
                || self.weights.shape().dims()
                    != [geometry.num_experts, geometry.out_dim, geometry.in_dim]
                || self.weights.storage_size_in_bytes() != geometry.weight_bytes
                || !self
                    .weights
                    .device()
                    .same_device(&Device::Cuda(input.device().clone()))
            {
                candle_core::bail!("CUDA Q4_0 grouped prompt weight geometry differs")
            }

            let input_view = checked_view_f32(
                input.as_cuda_slice::<f32>()?,
                input_layout,
                geometry.input_elements,
                "input",
            )?;
            let sorted_route_ids_view = checked_view_u32(
                sorted_route_ids.as_cuda_slice::<u32>()?,
                sorted_route_ids_layout,
                geometry.routes,
                "sorted route IDs",
            )?;
            let sorted_experts_view = checked_view_u32(
                sorted_experts.as_cuda_slice::<u32>()?,
                sorted_experts_layout,
                geometry.routes,
                "sorted experts",
            )?;
            let device = input.device();
            let stream = device.cuda_stream();
            let (weight_ptr, _weight_guard) = self.weights.device_ptr_with_guard(&stream)?;
            let weight_address = weight_ptr as u64;
            let mut output = device.alloc_zeros::<f32>(geometry.output_elements)?;
            let function =
                device.get_or_load_custom_func(PROMPT_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
            let config = LaunchConfig {
                grid_dim: (
                    geometry.out_dim.div_ceil(CUDA_PROMPT_OUTPUT_TILE) as u32,
                    geometry.num_experts as u32,
                    1,
                ),
                block_dim: (
                    CUDA_PROMPT_OUTPUT_TILE as u32,
                    CUDA_PROMPT_ROUTE_TILE as u32,
                    1,
                ),
                shared_mem_bytes: 0,
            };
            let routes = geometry.routes as i32;
            let top_k = geometry.top_k as i32;
            let input_rows = geometry.input_rows as i32;
            let in_dim = geometry.in_dim as i32;
            let out_dim = geometry.out_dim as i32;
            let num_experts = geometry.num_experts as i32;
            let activation_kind = self.activation_kind;
            let mut builder = function.builder();
            builder
                .arg(&input_view)
                .arg(&sorted_route_ids_view)
                .arg(&sorted_experts_view)
                .arg(&weight_address)
                .arg(&mut output)
                .arg(&routes)
                .arg(&top_k)
                .arg(&input_rows)
                .arg(&in_dim)
                .arg(&out_dim)
                .arg(&num_experts)
                .arg(&activation_kind);
            // SAFETY: The sort outputs and all storage views stay on this
            // stream. Geometry, storage bounds, and launch limits were checked.
            unsafe { builder.launch(config) }.w()?;

            Ok((
                CudaStorage::wrap_cuda_slice(output, device.clone()),
                Shape::from((geometry.tokens, geometry.top_k, geometry.out_dim)),
            ))
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
                "CUDA Q4_0 indexed MoE {name} must be contiguous {dtype:?} with {elements} elements"
            )
        }
        Ok(())
    }

    fn checked_view_f32<'a>(
        slice: &'a candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
        layout: &Layout,
        elements: usize,
        name: &str,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>> {
        checked_view(slice, layout, elements, name)
    }

    fn checked_view_u32<'a>(
        slice: &'a candle_core::cuda_backend::cudarc::driver::CudaSlice<u32>,
        layout: &Layout,
        elements: usize,
        name: &str,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, u32>> {
        checked_view(slice, layout, elements, name)
    }

    fn checked_view<'a, T>(
        slice: &'a candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
        layout: &Layout,
        elements: usize,
        name: &str,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, T>> {
        let start = layout.start_offset();
        let end = start
            .checked_add(elements)
            .ok_or_else(|| candle_core::Error::Msg(format!("{name} CUDA view overflow")))?;
        if end > slice.len() {
            candle_core::bail!("{name} CUDA view exceeds its storage")
        }
        Ok(slice.slice(start..end))
    }

    pub(crate) fn forward(
        input: &Tensor,
        expert_ids: &Tensor,
        weights: &QTensor,
        num_experts: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> anyhow::Result<Tensor> {
        forward_impl(input, expert_ids, weights, num_experts, out_dim, in_dim, 0)
    }

    pub(crate) fn forward_relu_squared(
        input: &Tensor,
        expert_ids: &Tensor,
        weights: &QTensor,
        num_experts: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> anyhow::Result<Tensor> {
        forward_impl(input, expert_ids, weights, num_experts, out_dim, in_dim, 1)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_impl(
        input: &Tensor,
        expert_ids: &Tensor,
        weights: &QTensor,
        num_experts: usize,
        out_dim: usize,
        in_dim: usize,
        activation_kind: i32,
    ) -> anyhow::Result<Tensor> {
        let (tokens, input_rows, actual_in_dim) = input.dims3()?;
        let (id_tokens, top_k) = expert_ids.dims2()?;
        anyhow::ensure!(
            actual_in_dim == in_dim,
            "CUDA Q4_0 indexed MoE input width differs"
        );
        let geometry = Q4IndexedGeometry::new(
            tokens,
            input_rows,
            in_dim,
            id_tokens,
            top_k,
            num_experts,
            out_dim,
        )?;
        anyhow::ensure!(
            input.dtype() == DType::F32,
            "CUDA Q4_0 indexed MoE input must be f32"
        );
        anyhow::ensure!(
            expert_ids.dtype() == DType::U32,
            "CUDA Q4_0 indexed MoE expert IDs must be u32"
        );
        anyhow::ensure!(
            input.device().is_cuda()
                && input.device().same_device(expert_ids.device())
                && input.device().same_device(&weights.device()),
            "CUDA Q4_0 indexed MoE inputs and weights must share one CUDA device"
        );
        anyhow::ensure!(
            weights.dtype() == GgmlDType::Q4_0
                && weights.shape().dims() == [num_experts, out_dim, in_dim]
                && weights.storage_size_in_bytes() == geometry.weight_bytes,
            "CUDA Q4_0 indexed MoE weight geometry differs"
        );
        if tokens == 1 {
            return input
                .apply_op2_no_bwd(
                    expert_ids,
                    &DecodeIndexedQ4 {
                        weights,
                        geometry,
                        activation_kind,
                    },
                )
                .map_err(Into::into);
        }

        // Keep route grouping on the accelerator. Sorting once lets every
        // prompt block reuse an expert weight tile across nearby routes.
        let route_experts = expert_ids.flatten_all()?.contiguous()?;
        let sorted_route_ids = route_experts.arg_sort_last_dim(true)?;
        let sorted_experts = route_experts
            .gather(&sorted_route_ids, D::Minus1)?
            .contiguous()?;
        input
            .apply_op3_no_bwd(
                &sorted_route_ids,
                &sorted_experts,
                &GroupedPromptIndexedQ4 {
                    weights,
                    geometry,
                    activation_kind,
                },
            )
            .map_err(Into::into)
    }
}

#[cfg(feature = "cuda")]
pub(super) use cuda::{forward, forward_relu_squared, precompile};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_geometry_tracks_canonical_expert_stride() -> Result<()> {
        let geometry = Q4IndexedGeometry::new(7, 1, 5120, 7, 6, 128, 2688)?;
        assert_eq!(geometry.routes, 42);
        assert_eq!(geometry.input_elements, 7 * 5120);
        assert_eq!(geometry.output_elements, 42 * 2688);
        assert_eq!(geometry.weight_bytes, 128 * 2688 * (5120 / 32) * 18);
        Ok(())
    }

    #[test]
    fn q4_layout_uses_gguf_block_and_expert_strides() -> Result<()> {
        let geometry = Q4IndexedGeometry::new(3, 1, 64, 3, 2, 4, 35)?;
        let blocks_per_row = geometry.in_dim / Q4_0_BLOCK_SIZE;
        let byte_offset = |expert: usize, output_row: usize, block: usize| {
            ((expert * geometry.out_dim + output_row) * blocks_per_row + block) * Q4_0_TYPE_SIZE
        };
        assert_eq!(byte_offset(0, 0, 0), 0);
        assert_eq!(byte_offset(0, 0, 1), 18);
        assert_eq!(byte_offset(0, 1, 0), 36);
        assert_eq!(byte_offset(1, 0, 0), 35 * 2 * 18);
        assert_eq!(byte_offset(3, 34, 1) + 18, geometry.weight_bytes);
        Ok(())
    }

    #[test]
    fn grouped_prompt_tile_math_preserves_original_route_order() {
        let tokens = 10;
        let top_k = 2;
        let routes = tokens * top_k;
        let num_experts = 3;
        let in_dim = 7;
        let out_dim = 35;
        let ids = (0..tokens)
            .flat_map(|token| [2usize, token % 2])
            .collect::<Vec<_>>();
        let input = (0..routes * in_dim)
            .map(|index| (index as f32 - 30.0) / 19.0)
            .collect::<Vec<_>>();
        let weights = (0..num_experts * out_dim * in_dim)
            .map(|index| ((index * 11 % 37) as f32 - 18.0) / 23.0)
            .collect::<Vec<_>>();

        let mut expected = vec![0.0f32; routes * out_dim];
        for route in 0..routes {
            for output_row in 0..out_dim {
                for inner in 0..in_dim {
                    expected[route * out_dim + output_row] += input[route * in_dim + inner]
                        * weights[(ids[route] * out_dim + output_row) * in_dim + inner];
                }
            }
        }

        let mut sorted_routes = (0..routes).collect::<Vec<_>>();
        sorted_routes.sort_by_key(|&route| ids[route]);
        let mut actual = vec![0.0f32; routes * out_dim];
        for expert in 0..num_experts {
            let expert_routes = sorted_routes
                .iter()
                .copied()
                .filter(|&route| ids[route] == expert)
                .collect::<Vec<_>>();
            for route_tile in expert_routes.chunks(CUDA_PROMPT_ROUTE_TILE) {
                for output_start in (0..out_dim).step_by(CUDA_PROMPT_OUTPUT_TILE) {
                    for &route in route_tile {
                        for output_row in
                            output_start..(output_start + CUDA_PROMPT_OUTPUT_TILE).min(out_dim)
                        {
                            for inner in 0..in_dim {
                                actual[route * out_dim + output_row] += input
                                    [route * in_dim + inner]
                                    * weights[(expert * out_dim + output_row) * in_dim + inner];
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(actual, expected);
        assert!(ids.iter().filter(|&&expert| expert == 2).count() > CUDA_PROMPT_ROUTE_TILE);
    }

    #[test]
    fn q4_geometry_rejects_unsafe_shapes() {
        assert!(Q4IndexedGeometry::new(0, 1, 64, 0, 2, 4, 32).is_err());
        assert!(Q4IndexedGeometry::new(2, 3, 64, 2, 2, 4, 32).is_err());
        assert!(Q4IndexedGeometry::new(2, 1, 63, 2, 2, 4, 32).is_err());
        assert!(Q4IndexedGeometry::new(2, 1, 64, 2, 5, 4, 32).is_err());
        assert!(Q4IndexedGeometry::new(40_000, 1, 64, 40_000, 2, 4, 32).is_err());
    }

    #[cfg(feature = "cuda")]
    mod cuda_tests {
        use candle_core::quantized::{GgmlDType, QTensor};
        use candle_core::{Device, Tensor};

        use super::*;

        #[allow(clippy::too_many_arguments)]
        fn expected(
            input: &[f32],
            input_rows: usize,
            ids: &[u32],
            dequantized: &[Vec<Vec<f32>>],
            tokens: usize,
            top_k: usize,
            out_dim: usize,
            in_dim: usize,
            relu_squared: bool,
        ) -> Vec<f32> {
            let mut output = vec![0.0; tokens * top_k * out_dim];
            for token in 0..tokens {
                for route in 0..top_k {
                    let route_index = token * top_k + route;
                    let input_row = if input_rows == 1 { token } else { route_index };
                    let expert = ids[route_index] as usize;
                    for out in 0..out_dim {
                        output[route_index * out_dim + out] = (0..in_dim)
                            .map(|inner| {
                                let input = input[input_row * in_dim + inner];
                                let input = if relu_squared {
                                    input.max(0.0).powi(2)
                                } else {
                                    input
                                };
                                input * dequantized[expert][out][inner]
                            })
                            .sum();
                    }
                }
            }
            output
        }

        fn assert_close(actual: &[f32], expected: &[f32]) {
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                let tolerance = 2e-3 + expected.abs() * 2e-4;
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "CUDA Q4_0 mismatch at {index}: {actual} != {expected}"
                );
            }
        }

        fn test_case(
            route_specific: bool,
            offset_view: bool,
            tokens: usize,
            relu_squared: bool,
        ) -> anyhow::Result<()> {
            let Some(device) = crate::test_cuda_device() else {
                return Ok(());
            };
            if std::env::var_os("DINF_DISABLE_INDEXED_MOE_CUDA").is_some()
                || std::env::var_os("DINF_DISABLE_INDEXED_MOE_Q4_0_CUDA").is_some()
            {
                return Ok(());
            }
            let num_experts = 3;
            let top_k = 2;
            let out_dim = 35;
            let in_dim = 64;
            let input_rows = if route_specific { top_k } else { 1 };
            let weight_values = (0..num_experts * out_dim * in_dim)
                .map(|index| ((index * 13 % 41) as f32 - 20.0) / 17.0)
                .collect::<Vec<_>>();
            let weight_tensor =
                Tensor::from_vec(weight_values, (num_experts, out_dim, in_dim), &Device::Cpu)?;
            let packed = QTensor::quantize(&weight_tensor, GgmlDType::Q4_0)?;
            let dequantized = packed.dequantize(&Device::Cpu)?.to_vec3::<f32>()?;
            let packed = candle_core::quantized::ggml_file::qtensor_from_ggml(
                GgmlDType::Q4_0,
                packed.data()?.as_ref(),
                vec![num_experts, out_dim, in_dim],
                &device,
            )?;

            let active_input = (0..tokens * input_rows * in_dim)
                .map(|index| ((index * 7 % 31) as f32 - 15.0) / 11.0)
                .collect::<Vec<_>>();
            let ids = (0..tokens)
                .flat_map(|token| [2u32, (token % 2) as u32])
                .collect::<Vec<_>>();
            let (input, expert_ids) = if offset_view {
                let mut padded_input = vec![99.0; (tokens + 1) * input_rows * in_dim];
                padded_input[input_rows * in_dim..].copy_from_slice(&active_input);
                let input =
                    Tensor::from_vec(padded_input, (tokens + 1, input_rows, in_dim), &device)?
                        .narrow(0, 1, tokens)?;
                let mut padded_ids = vec![0u32; (tokens + 1) * top_k];
                padded_ids[top_k..].copy_from_slice(&ids);
                let expert_ids = Tensor::from_vec(padded_ids, (tokens + 1, top_k), &device)?
                    .narrow(0, 1, tokens)?;
                assert_ne!(input.layout().start_offset(), 0);
                assert_ne!(expert_ids.layout().start_offset(), 0);
                (input, expert_ids)
            } else {
                (
                    Tensor::from_vec(active_input.clone(), (tokens, input_rows, in_dim), &device)?,
                    Tensor::from_vec(ids.clone(), (tokens, top_k), &device)?,
                )
            };
            let output = if relu_squared {
                forward_relu_squared(&input, &expert_ids, &packed, num_experts, out_dim, in_dim)?
            } else {
                forward(&input, &expert_ids, &packed, num_experts, out_dim, in_dim)?
            }
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            let reference = expected(
                &active_input,
                input_rows,
                &ids,
                &dequantized,
                tokens,
                top_k,
                out_dim,
                in_dim,
                relu_squared,
            );
            assert_close(&output, &reference);
            Ok(())
        }

        #[test]
        fn cuda_q4_decode_matches_dequantized_reference() -> anyhow::Result<()> {
            test_case(false, false, 1, false)
        }

        #[test]
        fn cuda_q4_grouped_prompt_shared_input_matches_reference() -> anyhow::Result<()> {
            test_case(false, false, 10, false)
        }

        #[test]
        fn cuda_q4_grouped_prompt_repeated_experts_and_offsets_match_reference()
        -> anyhow::Result<()> {
            test_case(true, true, 10, false)
        }

        #[test]
        fn cuda_q4_decode_relu_squared_matches_reference() -> anyhow::Result<()> {
            test_case(false, false, 1, true)
        }

        #[test]
        fn cuda_q4_grouped_prompt_relu_squared_matches_reference() -> anyhow::Result<()> {
            test_case(true, true, 10, true)
        }
    }
}
