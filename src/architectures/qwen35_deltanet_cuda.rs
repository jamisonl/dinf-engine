pub(super) use super::qwen35_deltanet_layout::supports;

#[cfg(feature = "cuda")]
mod cuda {
    use std::sync::OnceLock;

    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    use candle_core::cuda_backend::{CudaDevice, WrapErr};
    use candle_core::{CpuStorage, CudaStorage, DType, Device, Layout, Shape, Storage, Tensor};

    use super::super::qwen35_deltanet_layout::{CUDA_WARP_SIZE, DeltaNetGeometry};

    const CUDA_MODULE_NAME: &str = "dinf_qwen35_deltanet_v1";
    const CONV_KERNEL_NAME: &str = "qwen35_conv_scan_f32";

    // The scan keeps one value column's full recurrent state in each lane's
    // registers for the whole token run. One-warp blocks keep the canonical
    // [head, key, value] layout while spreading each head across the GPU.
    const CUDA_SOURCE: &str = r#"
extern "C" __global__ void qwen35_conv_scan_f32(
    const float *input,
    const float *state,
    const float *weight,
    float *output,
    int seq_len,
    int conv_dim,
    int kernel_size) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= conv_dim) {
        return;
    }

    float history[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k < kernel_size) {
            history[k] = state[channel * kernel_size + k];
        }
    }

    for (int token = 0; token < seq_len; ++token) {
#pragma unroll
        for (int k = 1; k < 8; ++k) {
            if (k < kernel_size) {
                history[k - 1] = history[k];
            }
        }
        history[kernel_size - 1] = input[token * conv_dim + channel];

        float value = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k < kernel_size) {
                value = fmaf(
                    history[k],
                    weight[channel * kernel_size + k],
                    value);
            }
        }
        output[token * conv_dim + channel] = value / (1.0f + expf(-value));
    }

    const int state_offset = seq_len * conv_dim + channel * kernel_size;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k < kernel_size) {
            output[state_offset + k] = history[k];
        }
    }
}

__device__ __forceinline__ float qwen35_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template<int HEAD_DIM>
__device__ __forceinline__ void qwen35_deltanet_scan_body(
    const float *state,
    const float *conv,
    const float *alpha_raw,
    const float *beta_raw,
    const float *a,
    const float *dt_bias,
    float *output,
    int seq_len,
    int n_k_heads,
    int n_v_heads) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    const int value_column = blockIdx.y * 32 + lane;
    const bool valid_column = value_column < HEAD_DIM;
    const int qk_head = head % n_k_heads;
    const int total_key_dim = n_k_heads * HEAD_DIM;
    const int total_value_dim = n_v_heads * HEAD_DIM;
    const int conv_dim = 2 * total_key_dim + total_value_dim;
    const int state_output_offset = seq_len * total_value_dim;

    __shared__ float q_values[HEAD_DIM];
    __shared__ float k_values[HEAD_DIM];
    __shared__ float q_scale;
    __shared__ float k_scale;
    __shared__ float beta;
    __shared__ float decay;

    float state_values[HEAD_DIM];
    if (valid_column) {
#pragma unroll
        for (int row = 0; row < HEAD_DIM; ++row) {
            state_values[row] =
                state[(head * HEAD_DIM + row) * HEAD_DIM + value_column];
        }
    }

    for (int token = 0; token < seq_len; ++token) {
        const int q_offset = token * conv_dim + qk_head * HEAD_DIM;
        float q_sum = 0.0f;
        float k_sum = 0.0f;
#pragma unroll
        for (int row = lane; row < HEAD_DIM; row += 32) {
            const float q = conv[q_offset + row];
            const float k = conv[q_offset + total_key_dim + row];
            q_values[row] = q;
            k_values[row] = k;
            q_sum = fmaf(q, q, q_sum);
            k_sum = fmaf(k, k, k_sum);
        }
        q_sum = qwen35_warp_sum(q_sum);
        k_sum = qwen35_warp_sum(k_sum);

        if (lane == 0) {
            q_scale = rsqrtf(q_sum + 1.0e-12f) * rsqrtf((float) HEAD_DIM);
            k_scale = rsqrtf(k_sum + 1.0e-12f);
            const int gate_offset = token * n_v_heads + head;
            beta = 1.0f / (1.0f + expf(-beta_raw[gate_offset]));
            const float dt_input = alpha_raw[gate_offset] + dt_bias[head];
            const float dt = dt_input <= 20.0f ? log1pf(expf(dt_input)) : dt_input;
            decay = expf(a[head] * dt);
        }
        __syncwarp();

        if (valid_column) {
            float retrieved = 0.0f;
#pragma unroll
            for (int row = 0; row < HEAD_DIM; ++row) {
                retrieved = fmaf(
                    decay * state_values[row],
                    k_values[row] * k_scale,
                    retrieved);
            }
            const int value_offset = token * conv_dim + 2 * total_key_dim
                + head * HEAD_DIM + value_column;
            const float delta = beta * (conv[value_offset] - retrieved);
            float y = 0.0f;
#pragma unroll
            for (int row = 0; row < HEAD_DIM; ++row) {
                const float k = k_values[row] * k_scale;
                const float next = fmaf(k, delta, decay * state_values[row]);
                state_values[row] = next;
                y = fmaf(next, q_values[row] * q_scale, y);
            }
            output[token * total_value_dim + head * HEAD_DIM + value_column] = y;
        }
        __syncwarp();
    }

    if (valid_column) {
#pragma unroll
        for (int row = 0; row < HEAD_DIM; ++row) {
            output[state_output_offset + (head * HEAD_DIM + row) * HEAD_DIM
                + value_column] = state_values[row];
        }
    }
}

#define QWEN35_SCAN_KERNEL(HEAD_DIM) \
extern "C" __global__ __launch_bounds__(32, 1) void qwen35_deltanet_scan_f32_##HEAD_DIM( \
    const float *state, const float *conv, const float *alpha_raw, \
    const float *beta_raw, const float *a, const float *dt_bias, \
    float *output, int seq_len, int n_k_heads, int n_v_heads) { \
    qwen35_deltanet_scan_body<HEAD_DIM>( \
        state, conv, alpha_raw, beta_raw, a, dt_bias, output, \
        seq_len, n_k_heads, n_v_heads); \
}

QWEN35_SCAN_KERNEL(32)
QWEN35_SCAN_KERNEL(64)
QWEN35_SCAN_KERNEL(128)
"#;

    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    static CUDA_LOGGED: OnceLock<()> = OnceLock::new();

    fn ptx() -> candle_core::Result<&'static str> {
        match PTX.get_or_init(|| {
            compile_ptx_with_opts(
                CUDA_SOURCE,
                CompileOptions {
                    name: Some("qwen35_deltanet.cu".to_string()),
                    use_fast_math: Some(true),
                    ..Default::default()
                },
            )
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
        }) {
            Ok(ptx) => Ok(ptx.as_str()),
            Err(error) => candle_core::bail!("failed to compile Qwen3.5 CUDA DeltaNet: {error}"),
        }
    }

    pub(crate) fn precompile(device: &Device) -> anyhow::Result<()> {
        if !device.is_cuda() || std::env::var_os("DINF_DISABLE_QWEN35_DELTANET_CUDA").is_some() {
            return Ok(());
        }
        let device = device.as_cuda_device()?;
        let ptx = ptx()?;
        device.get_or_load_custom_func(CONV_KERNEL_NAME, CUDA_MODULE_NAME, ptx)?;
        for kernel in [
            "qwen35_deltanet_scan_f32_32",
            "qwen35_deltanet_scan_f32_64",
            "qwen35_deltanet_scan_f32_128",
        ] {
            device.get_or_load_custom_func(kernel, CUDA_MODULE_NAME, ptx)?;
        }
        CUDA_LOGGED.get_or_init(|| {
            tracing::info!("fused CUDA DeltaNet scan enabled for Qwen3.5");
        });
        Ok(())
    }

    struct ConvScan<'a> {
        weight: &'a Tensor,
        geometry: DeltaNetGeometry,
    }

    impl candle_core::CustomOp2 for ConvScan<'_> {
        fn name(&self) -> &'static str {
            "qwen35-cuda-conv-scan"
        }

        fn cpu_fwd(
            &self,
            _input: &CpuStorage,
            _input_layout: &Layout,
            _state: &CpuStorage,
            _state_layout: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("qwen35-cuda-conv-scan is only dispatched on CUDA")
        }

        fn cuda_fwd(
            &self,
            input: &CudaStorage,
            input_layout: &Layout,
            state: &CudaStorage,
            state_layout: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let geometry = self.geometry;
            validate_cuda_storage("input", input, input_layout, geometry.conv_elements)?;
            validate_cuda_storage(
                "convolution state",
                state,
                state_layout,
                geometry.conv_state_elements,
            )?;
            ensure_same_cuda_device("convolution state", input, state)?;

            let (weight_guard, weight_layout) = self.weight.storage_and_layout();
            let weight = captured_cuda_storage(
                "convolution weight",
                &weight_guard,
                weight_layout,
                geometry.conv_state_elements,
                input.device(),
            )?;

            let input_slice = input.as_cuda_slice::<f32>()?;
            let state_slice = state.as_cuda_slice::<f32>()?;
            let weight_slice = weight.as_cuda_slice::<f32>()?;
            let input_view = checked_view(input_slice, input_layout, geometry.conv_elements)?;
            let state_view = checked_view(state_slice, state_layout, geometry.conv_state_elements)?;
            let weight_view =
                checked_view(weight_slice, weight_layout, geometry.conv_state_elements)?;

            let output_elements = geometry
                .conv_packed_elements()
                .map_err(candle_core::Error::wrap)?;
            let device = input.device();
            // SAFETY: The kernel writes every output and final-state element.
            let mut output = unsafe { device.alloc::<f32>(output_elements)? };
            let function =
                device.get_or_load_custom_func(CONV_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
            let seq_len = geometry.seq_len as i32;
            let conv_dim = geometry.conv_dim as i32;
            let kernel_size = geometry.kernel_size as i32;
            let config = LaunchConfig {
                grid_dim: (geometry.conv_dim.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = function.builder();
            builder
                .arg(&input_view)
                .arg(&state_view)
                .arg(&weight_view)
                .arg(&mut output)
                .arg(&seq_len)
                .arg(&conv_dim)
                .arg(&kernel_size);
            // SAFETY: All pointers and dimensions were checked above.
            unsafe { builder.launch(config) }.w()?;

            Ok((
                CudaStorage::wrap_cuda_slice(output, device.clone()),
                Shape::from(output_elements),
            ))
        }
    }

    struct DeltaNetScan<'a> {
        beta_raw: &'a Tensor,
        a: &'a Tensor,
        dt_bias: &'a Tensor,
        geometry: DeltaNetGeometry,
    }

    impl candle_core::CustomOp3 for DeltaNetScan<'_> {
        fn name(&self) -> &'static str {
            "qwen35-cuda-deltanet-scan"
        }

        fn cpu_fwd(
            &self,
            _state: &CpuStorage,
            _state_layout: &Layout,
            _conv: &CpuStorage,
            _conv_layout: &Layout,
            _alpha_raw: &CpuStorage,
            _alpha_layout: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("qwen35-cuda-deltanet-scan is only dispatched on CUDA")
        }

        fn cuda_fwd(
            &self,
            state: &CudaStorage,
            state_layout: &Layout,
            conv: &CudaStorage,
            conv_layout: &Layout,
            alpha_raw: &CudaStorage,
            alpha_layout: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let geometry = self.geometry;
            let gate_elements = geometry
                .seq_len
                .checked_mul(geometry.n_v_heads)
                .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 gate size overflow".into()))?;
            validate_cuda_storage(
                "recurrent state",
                state,
                state_layout,
                geometry.recurrent_state_elements,
            )?;
            validate_cuda_storage(
                "convolution output",
                conv,
                conv_layout,
                geometry.conv_elements,
            )?;
            validate_cuda_storage("alpha gate", alpha_raw, alpha_layout, gate_elements)?;
            ensure_same_cuda_device("convolution output", state, conv)?;
            ensure_same_cuda_device("alpha gate", state, alpha_raw)?;

            let (beta_guard, beta_layout) = self.beta_raw.storage_and_layout();
            let (a_guard, a_layout) = self.a.storage_and_layout();
            let (dt_guard, dt_layout) = self.dt_bias.storage_and_layout();
            let beta = captured_cuda_storage(
                "beta gate",
                &beta_guard,
                beta_layout,
                gate_elements,
                state.device(),
            )?;
            let a = captured_cuda_storage(
                "decay rate",
                &a_guard,
                a_layout,
                geometry.n_v_heads,
                state.device(),
            )?;
            let dt = captured_cuda_storage(
                "time-step bias",
                &dt_guard,
                dt_layout,
                geometry.n_v_heads,
                state.device(),
            )?;

            let state_view = checked_view(
                state.as_cuda_slice::<f32>()?,
                state_layout,
                geometry.recurrent_state_elements,
            )?;
            let conv_view = checked_view(
                conv.as_cuda_slice::<f32>()?,
                conv_layout,
                geometry.conv_elements,
            )?;
            let alpha_view = checked_view(
                alpha_raw.as_cuda_slice::<f32>()?,
                alpha_layout,
                gate_elements,
            )?;
            let beta_view = checked_view(beta.as_cuda_slice::<f32>()?, beta_layout, gate_elements)?;
            let a_view = checked_view(a.as_cuda_slice::<f32>()?, a_layout, geometry.n_v_heads)?;
            let dt_view = checked_view(dt.as_cuda_slice::<f32>()?, dt_layout, geometry.n_v_heads)?;

            let output_elements = geometry
                .recurrent_packed_elements()
                .map_err(candle_core::Error::wrap)?;
            let device = state.device();
            // SAFETY: The scan writes every output and final-state element.
            let mut output = unsafe { device.alloc::<f32>(output_elements)? };
            let function = device.get_or_load_custom_func(
                geometry.scan_kernel_name(),
                CUDA_MODULE_NAME,
                ptx()?,
            )?;
            let seq_len = geometry.seq_len as i32;
            let n_k_heads = geometry.n_k_heads as i32;
            let n_v_heads = geometry.n_v_heads as i32;
            let config = LaunchConfig {
                grid_dim: (
                    geometry.n_v_heads as u32,
                    geometry.head_dim.div_ceil(CUDA_WARP_SIZE) as u32,
                    1,
                ),
                block_dim: (CUDA_WARP_SIZE as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = function.builder();
            builder
                .arg(&state_view)
                .arg(&conv_view)
                .arg(&alpha_view)
                .arg(&beta_view)
                .arg(&a_view)
                .arg(&dt_view)
                .arg(&mut output)
                .arg(&seq_len)
                .arg(&n_k_heads)
                .arg(&n_v_heads);
            // SAFETY: All pointers and dimensions were checked above.
            unsafe { builder.launch(config) }.w()?;

            Ok((
                CudaStorage::wrap_cuda_slice(output, device.clone()),
                Shape::from(output_elements),
            ))
        }
    }

    fn validate_cuda_storage(
        name: &str,
        storage: &CudaStorage,
        layout: &Layout,
        expected_elements: usize,
    ) -> candle_core::Result<()> {
        if storage.dtype() != DType::F32
            || !layout.is_contiguous()
            || layout.shape().elem_count() != expected_elements
        {
            candle_core::bail!(
                "{name} must be contiguous f32 with {expected_elements} elements for Qwen3.5 CUDA DeltaNet"
            )
        }
        let slice = storage.as_cuda_slice::<f32>()?;
        let end = layout
            .start_offset()
            .checked_add(expected_elements)
            .ok_or_else(|| candle_core::Error::Msg(format!("{name} layout overflow")))?;
        if end > slice.len() {
            candle_core::bail!("{name} layout exceeds its CUDA storage")
        }
        Ok(())
    }

    fn ensure_same_cuda_device(
        name: &str,
        expected: &CudaStorage,
        actual: &CudaStorage,
    ) -> candle_core::Result<()> {
        if expected.device().id() != actual.device().id() {
            candle_core::bail!("{name} is on a different CUDA device")
        }
        Ok(())
    }

    fn captured_cuda_storage<'a>(
        name: &str,
        storage: &'a Storage,
        layout: &Layout,
        expected_elements: usize,
        expected_device: &CudaDevice,
    ) -> candle_core::Result<&'a CudaStorage> {
        let storage = match storage {
            Storage::Cuda(storage) => storage,
            _ => candle_core::bail!("{name} is not on CUDA"),
        };
        validate_cuda_storage(name, storage, layout, expected_elements)?;
        if storage.device().id() != expected_device.id() {
            candle_core::bail!("{name} is on a different CUDA device")
        }
        Ok(storage)
    }

    fn checked_view<'a>(
        slice: &'a candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
        layout: &Layout,
        elements: usize,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>> {
        let start = layout.start_offset();
        let end = start
            .checked_add(elements)
            .ok_or_else(|| candle_core::Error::Msg("CUDA view offset overflow".into()))?;
        if end > slice.len() {
            candle_core::bail!("CUDA view exceeds its storage")
        }
        Ok(slice.slice(start..end))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill(
        qkv: &Tensor,
        conv_state: &Tensor,
        recurrent_state: &Tensor,
        conv_weight: &Tensor,
        alpha_raw: &Tensor,
        beta_raw: &Tensor,
        a: &Tensor,
        dt_bias: &Tensor,
        n_k_heads: usize,
        n_v_heads: usize,
        key_dim: usize,
        value_dim: usize,
        kernel_size: usize,
    ) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let (seq_len, conv_dim) = qkv.dims2()?;
        let geometry = DeltaNetGeometry::new(
            seq_len,
            conv_dim,
            kernel_size,
            n_k_heads,
            n_v_heads,
            key_dim,
            value_dim,
        )?;
        anyhow::ensure!(
            conv_state.dims2()? == (conv_dim, kernel_size),
            "invalid Qwen3.5 CUDA convolution state shape"
        );
        anyhow::ensure!(
            recurrent_state.dims3()? == (n_v_heads, key_dim, value_dim),
            "invalid Qwen3.5 CUDA recurrent state shape"
        );
        anyhow::ensure!(
            conv_weight.dims2()? == (conv_dim, kernel_size),
            "invalid Qwen3.5 CUDA convolution weight shape"
        );
        anyhow::ensure!(
            alpha_raw.dims2()? == (seq_len, n_v_heads) && beta_raw.dims2()? == (seq_len, n_v_heads),
            "invalid Qwen3.5 CUDA gate shape"
        );
        anyhow::ensure!(
            a.dims1()? == n_v_heads && dt_bias.dims1()? == n_v_heads,
            "invalid Qwen3.5 CUDA decay shape"
        );

        let conv_packed = qkv.apply_op2_no_bwd(
            conv_state,
            &ConvScan {
                weight: conv_weight,
                geometry,
            },
        )?;
        let conv = conv_packed
            .narrow(0, 0, geometry.conv_elements)?
            .reshape((seq_len, conv_dim))?;
        let next_conv_state = conv_packed
            .narrow(0, geometry.conv_elements, geometry.conv_state_elements)?
            .reshape((conv_dim, kernel_size))?;

        let recurrent_packed = recurrent_state.apply_op3_no_bwd(
            &conv,
            alpha_raw,
            &DeltaNetScan {
                beta_raw,
                a,
                dt_bias,
                geometry,
            },
        )?;
        let recurrent_output = recurrent_packed
            .narrow(0, 0, geometry.recurrent_output_elements)?
            .reshape((seq_len, n_v_heads, value_dim))?;
        let next_recurrent_state = recurrent_packed
            .narrow(
                0,
                geometry.recurrent_output_elements,
                geometry.recurrent_state_elements,
            )?
            .reshape((n_v_heads, key_dim, value_dim))?;
        Ok((recurrent_output, next_conv_state, next_recurrent_state))
    }
}

#[cfg(feature = "cuda")]
pub(super) use cuda::{precompile, prefill};

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    #[test]
    fn cuda_kernels_precompile() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        super::precompile(&device)
    }
}
