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

    const CUDA_MODULE_NAME: &str = "dinf_qwen35_deltanet_v3";
    const CONV_KERNEL_NAME: &str = "qwen35_conv_scan_f32";
    /// Value columns per warp in the prompt scan: two groups of sixteen
    /// lanes, each lane carrying `QWEN35_LANE_COLUMNS`.
    const PROMPT_SCAN_COLUMNS: usize = 2 * 2;

    // Both scans keep the recurrent state in registers for the whole token
    // run and the canonical [head, key, value] layout in memory.
    //
    // The single-token scan gives each lane one value column's full state in
    // a one-warp block.
    //
    // A prompt scan is bound by how fast a warp can step through one token
    // and by its memory loads, not by arithmetic. So a separate kernel first
    // computes every token's normalized query and key and its gates, which
    // do not depend on the state, and the prompt scan keeps the step short:
    //
    // - Sixteen lanes share a group of value columns. A lane owns a run of
    //   consecutive key rows of those columns, and the lanes add their
    //   partial dot products with shuffles, so no step waits on shared
    //   memory or a block barrier.
    // - A lane fetches its run of the query and of the key with vector
    //   loads, once for all of its columns.
    // - Each token's loads are issued a few tokens early and held in
    //   registers, so the scan never waits for memory.
    //
    // Both scans add each dot product as the same sixteen partial sums in
    // the same order, so a prompt leaves exactly the state that the same
    // tokens leave one at a time.
    const CUDA_SOURCE: &str = r#"
// Causal convolution with SiLU. Each output depends only on the last
// `kernel_size` inputs of its channel, so tokens run in parallel: block y
// is a token, and one extra row of blocks writes the next state.
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
    const int token = blockIdx.y;

    // The inputs continue the state: position -1 is the state's last entry.
    // A token's window ends at that token, and the next state is the window
    // that ends at the last token.
    const bool next_state = token == seq_len;
    const int first = (next_state ? seq_len : token + 1) - kernel_size;
    float history[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k < kernel_size) {
            const int position = first + k;
            history[k] = position >= 0
                ? input[position * conv_dim + channel]
                : state[channel * kernel_size + kernel_size + position];
        }
    }

    if (next_state) {
        const int state_offset = seq_len * conv_dim + channel * kernel_size;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k < kernel_size) {
                output[state_offset + k] = history[k];
            }
        }
        return;
    }

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

// Every recurrent dot product is 16 partial sums over the key rows. Sum `l`
// takes the `l`-th run of HEAD_DIM / 16 consecutive rows in order, and the
// sums are combined in the order of an xor-shuffle reduction over 16 lanes.
#define QWEN35_SUMS 16

// Load `ROWS` consecutive floats with the widest aligned loads.
template<int ROWS>
__device__ __forceinline__ void qwen35_load_rows(const float *source, float *rows) {
    if (ROWS == 2) {
        const float2 pair = *(const float2 *) source;
        rows[0] = pair.x;
        rows[1] = pair.y;
        return;
    }
#pragma unroll
    for (int index = 0; index < ROWS; index += 4) {
        const float4 quad = *(const float4 *) (source + index);
        rows[index] = quad.x;
        rows[index + 1] = quad.y;
        rows[index + 2] = quad.z;
        rows[index + 3] = quad.w;
    }
}

__device__ __forceinline__ float qwen35_combine_sums(const float *sums) {
    float pairs[8];
#pragma unroll
    for (int index = 0; index < 8; ++index) {
        pairs[index] = sums[index] + sums[index + 8];
    }
    return ((pairs[0] + pairs[4]) + (pairs[2] + pairs[6]))
        + ((pairs[1] + pairs[5]) + (pairs[3] + pairs[7]));
}

// The same combination across the 16 lanes of one value column.
__device__ __forceinline__ float qwen35_column_sum(float value) {
#pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, mask);
    }
    return value;
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
        // Scale the key and query once per token instead of in both loops.
#pragma unroll
        for (int row = lane; row < HEAD_DIM; row += 32) {
            q_values[row] *= q_scale;
            k_values[row] *= k_scale;
        }
        __syncwarp();

        if (valid_column) {
            // Decay the state once.
            const float token_decay = decay;
            float retrieved[QWEN35_SUMS];
#pragma unroll
            for (int sum = 0; sum < QWEN35_SUMS; ++sum) {
                retrieved[sum] = 0.0f;
            }
#pragma unroll
            for (int row = 0; row < HEAD_DIM; ++row) {
                const float decayed = token_decay * state_values[row];
                state_values[row] = decayed;
                retrieved[row / (HEAD_DIM / QWEN35_SUMS)] = fmaf(
                    decayed,
                    k_values[row],
                    retrieved[row / (HEAD_DIM / QWEN35_SUMS)]);
            }
            const int value_offset = token * conv_dim + 2 * total_key_dim
                + head * HEAD_DIM + value_column;
            const float delta =
                beta * (conv[value_offset] - qwen35_combine_sums(retrieved));
            float y[QWEN35_SUMS];
#pragma unroll
            for (int sum = 0; sum < QWEN35_SUMS; ++sum) {
                y[sum] = 0.0f;
            }
#pragma unroll
            for (int row = 0; row < HEAD_DIM; ++row) {
                const float next = fmaf(k_values[row], delta, state_values[row]);
                state_values[row] = next;
                y[row / (HEAD_DIM / QWEN35_SUMS)] = fmaf(
                    next,
                    q_values[row],
                    y[row / (HEAD_DIM / QWEN35_SUMS)]);
            }
            output[token * total_value_dim + head * HEAD_DIM + value_column] =
                qwen35_combine_sums(y);
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

template<int HEAD_DIM>
__device__ __forceinline__ void qwen35_deltanet_terms_body(
    const float *conv,
    const float *alpha_raw,
    const float *beta_raw,
    const float *a,
    const float *dt_bias,
    float *terms,
    int seq_len,
    int n_k_heads,
    int n_v_heads) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int lane = threadIdx.x;
    const int total_key_dim = n_k_heads * HEAD_DIM;
    const int conv_dim = 2 * total_key_dim + n_v_heads * HEAD_DIM;
    const int gate_offset = token * n_v_heads + head;

    if (lane == 0) {
        const float beta = 1.0f / (1.0f + expf(-beta_raw[gate_offset]));
        const float dt_input = alpha_raw[gate_offset] + dt_bias[head];
        const float dt = dt_input <= 20.0f ? log1pf(expf(dt_input)) : dt_input;
        float *gates = terms + seq_len * n_k_heads * 2 * HEAD_DIM + 2 * gate_offset;
        gates[0] = beta;
        gates[1] = expf(a[head] * dt);
    }
    if (head >= n_k_heads) {
        return;
    }

    const int q_offset = token * conv_dim + head * HEAD_DIM;
    float q[HEAD_DIM / 32];
    float k[HEAD_DIM / 32];
    float q_sum = 0.0f;
    float k_sum = 0.0f;
#pragma unroll
    for (int index = 0; index < HEAD_DIM / 32; ++index) {
        const int row = lane + 32 * index;
        q[index] = conv[q_offset + row];
        k[index] = conv[q_offset + total_key_dim + row];
        q_sum = fmaf(q[index], q[index], q_sum);
        k_sum = fmaf(k[index], k[index], k_sum);
    }
    q_sum = __shfl_sync(0xffffffffu, qwen35_warp_sum(q_sum), 0);
    k_sum = __shfl_sync(0xffffffffu, qwen35_warp_sum(k_sum), 0);
    const float q_scale = rsqrtf(q_sum + 1.0e-12f) * rsqrtf((float) HEAD_DIM);
    const float k_scale = rsqrtf(k_sum + 1.0e-12f);

    float *scaled = terms + (token * n_k_heads + head) * 2 * HEAD_DIM;
#pragma unroll
    for (int index = 0; index < HEAD_DIM / 32; ++index) {
        const int row = lane + 32 * index;
        scaled[row] = q[index] * q_scale;
        scaled[HEAD_DIM + row] = k[index] * k_scale;
    }
}

// Value columns each lane carries in the prompt scan.
#define QWEN35_LANE_COLUMNS 2

// Tokens whose inputs the prompt scan holds in registers. A token's loads
// are issued this many steps before it is scanned, which hides their
// latency behind the arithmetic of the tokens in between.
#define QWEN35_TOKEN_SLOTS 4

template<int HEAD_DIM, int COLUMNS>
__device__ __forceinline__ void qwen35_deltanet_prompt_scan_body(
    const float *state,
    const float *conv,
    const float *terms,
    float *output,
    int seq_len,
    int n_k_heads,
    int n_v_heads) {
    enum { ROWS = HEAD_DIM / QWEN35_SUMS };
    const int head = blockIdx.x;
    const int sum = threadIdx.x % QWEN35_SUMS;
    const int first_column = (blockIdx.y * 2 + threadIdx.x / QWEN35_SUMS) * COLUMNS;
    const int total_key_dim = n_k_heads * HEAD_DIM;
    const int total_value_dim = n_v_heads * HEAD_DIM;
    const int conv_dim = 2 * total_key_dim + total_value_dim;
    const int terms_stride = n_k_heads * 2 * HEAD_DIM;

    // This thread's inputs for token 0; each token advances them. The
    // terms hold every token's scaled queries and keys, then every token's
    // gates, so each run of rows and each gate pair is aligned for its load.
    const float *scaled = terms + (head % n_k_heads) * 2 * HEAD_DIM + sum * ROWS;
    const float *gates = terms + seq_len * terms_stride + 2 * head;
    const float *value = conv + 2 * total_key_dim + head * HEAD_DIM + first_column;
    float *y_output = output + head * HEAD_DIM + first_column;
    float *state_output = output + seq_len * total_value_dim
        + (head * HEAD_DIM + sum * ROWS) * HEAD_DIM + first_column;
    state += (head * HEAD_DIM + sum * ROWS) * HEAD_DIM + first_column;

    float state_values[COLUMNS][ROWS];
#pragma unroll
    for (int column = 0; column < COLUMNS; ++column) {
#pragma unroll
        for (int index = 0; index < ROWS; ++index) {
            state_values[column][index] = state[index * HEAD_DIM + column];
        }
    }

    float q_values[QWEN35_TOKEN_SLOTS][ROWS];
    float k_values[QWEN35_TOKEN_SLOTS][ROWS];
    float2 gate[QWEN35_TOKEN_SLOTS];
    float targets[QWEN35_TOKEN_SLOTS][COLUMNS];
#pragma unroll
    for (int slot = 0; slot < QWEN35_TOKEN_SLOTS; ++slot) {
        if (slot < seq_len) {
            qwen35_load_rows<ROWS>(scaled, q_values[slot]);
            qwen35_load_rows<ROWS>(scaled + HEAD_DIM, k_values[slot]);
            gate[slot] = *(const float2 *) gates;
#pragma unroll
            for (int column = 0; column < COLUMNS; ++column) {
                targets[slot][column] = value[column];
            }
        }
        // After the last slot these point at token QWEN35_TOKEN_SLOTS.
        scaled += terms_stride;
        gates += 2 * n_v_heads;
        value += conv_dim;
    }

    for (int first = 0; first < seq_len; first += QWEN35_TOKEN_SLOTS) {
#pragma unroll
        for (int slot = 0; slot < QWEN35_TOKEN_SLOTS; ++slot) {
            const int token = first + slot;
            if (token < seq_len) {
                const float beta = gate[slot].x;
                const float decay = gate[slot].y;
#pragma unroll
                for (int column = 0; column < COLUMNS; ++column) {
                    float retrieved = 0.0f;
#pragma unroll
                    for (int index = 0; index < ROWS; ++index) {
                        const float decayed = decay * state_values[column][index];
                        state_values[column][index] = decayed;
                        retrieved = fmaf(decayed, k_values[slot][index], retrieved);
                    }
                    const float delta = beta
                        * (targets[slot][column] - qwen35_column_sum(retrieved));
                    float y = 0.0f;
#pragma unroll
                    for (int index = 0; index < ROWS; ++index) {
                        const float next = fmaf(
                            k_values[slot][index], delta, state_values[column][index]);
                        state_values[column][index] = next;
                        y = fmaf(next, q_values[slot][index], y);
                    }
                    y = qwen35_column_sum(y);
                    if (sum == 0) {
                        y_output[column] = y;
                    }
                }
                y_output += total_value_dim;

                // Refill this slot with the token that will use it next.
                if (token + QWEN35_TOKEN_SLOTS < seq_len) {
                    qwen35_load_rows<ROWS>(scaled, q_values[slot]);
                    qwen35_load_rows<ROWS>(scaled + HEAD_DIM, k_values[slot]);
                    gate[slot] = *(const float2 *) gates;
#pragma unroll
                    for (int column = 0; column < COLUMNS; ++column) {
                        targets[slot][column] = value[column];
                    }
                }
                scaled += terms_stride;
                gates += 2 * n_v_heads;
                value += conv_dim;
            }
        }
    }

#pragma unroll
    for (int column = 0; column < COLUMNS; ++column) {
#pragma unroll
        for (int index = 0; index < ROWS; ++index) {
            state_output[index * HEAD_DIM + column] = state_values[column][index];
        }
    }
}

#define QWEN35_PROMPT_SCAN_KERNELS(HEAD_DIM) \
extern "C" __global__ __launch_bounds__(32) void qwen35_deltanet_terms_f32_##HEAD_DIM( \
    const float *conv, const float *alpha_raw, const float *beta_raw, \
    const float *a, const float *dt_bias, float *terms, \
    int seq_len, int n_k_heads, int n_v_heads) { \
    qwen35_deltanet_terms_body<HEAD_DIM>( \
        conv, alpha_raw, beta_raw, a, dt_bias, terms, \
        seq_len, n_k_heads, n_v_heads); \
} \
extern "C" __global__ __launch_bounds__(32) void qwen35_deltanet_prompt_scan_f32_##HEAD_DIM( \
    const float *state, const float *conv, const float *terms, \
    float *output, int seq_len, int n_k_heads, int n_v_heads) { \
    qwen35_deltanet_prompt_scan_body<HEAD_DIM, QWEN35_LANE_COLUMNS>( \
        state, conv, terms, output, seq_len, n_k_heads, n_v_heads); \
}

QWEN35_PROMPT_SCAN_KERNELS(32)
QWEN35_PROMPT_SCAN_KERNELS(64)
QWEN35_PROMPT_SCAN_KERNELS(128)

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
        if !device.is_cuda() {
            return Ok(());
        }
        let device = device.as_cuda_device()?;
        let ptx = ptx()?;
        device.get_or_load_custom_func(CONV_KERNEL_NAME, CUDA_MODULE_NAME, ptx)?;
        for head_dim in [32, 64, 128] {
            for kernel in ["scan", "terms", "prompt_scan"] {
                device.get_or_load_custom_func(
                    &format!("qwen35_deltanet_{kernel}_f32_{head_dim}"),
                    CUDA_MODULE_NAME,
                    ptx,
                )?;
            }
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
                grid_dim: (
                    geometry.conv_dim.div_ceil(256) as u32,
                    geometry.seq_len as u32 + 1,
                    1,
                ),
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
            let seq_len = geometry.seq_len as i32;
            let n_k_heads = geometry.n_k_heads as i32;
            let n_v_heads = geometry.n_v_heads as i32;
            let columns = geometry.head_dim.div_ceil(CUDA_WARP_SIZE) as u32;
            if geometry.seq_len == 1 {
                let function = device.get_or_load_custom_func(
                    geometry.scan_kernel_name(),
                    CUDA_MODULE_NAME,
                    ptx()?,
                )?;
                let config = LaunchConfig {
                    grid_dim: (geometry.n_v_heads as u32, columns, 1),
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
            } else {
                // SAFETY: The terms kernel writes both gates of every value
                // head and the scaled query and key of every key head, for
                // every token.
                let mut terms = unsafe { device.alloc::<f32>(geometry.terms_elements)? };
                let function = device.get_or_load_custom_func(
                    geometry.terms_kernel_name(),
                    CUDA_MODULE_NAME,
                    ptx()?,
                )?;
                let config = LaunchConfig {
                    grid_dim: (geometry.n_v_heads as u32, geometry.seq_len as u32, 1),
                    block_dim: (CUDA_WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut builder = function.builder();
                builder
                    .arg(&conv_view)
                    .arg(&alpha_view)
                    .arg(&beta_view)
                    .arg(&a_view)
                    .arg(&dt_view)
                    .arg(&mut terms)
                    .arg(&seq_len)
                    .arg(&n_k_heads)
                    .arg(&n_v_heads);
                // SAFETY: All pointers and dimensions were checked above.
                unsafe { builder.launch(config) }.w()?;

                let function = device.get_or_load_custom_func(
                    geometry.prompt_scan_kernel_name(),
                    CUDA_MODULE_NAME,
                    ptx()?,
                )?;
                let config = LaunchConfig {
                    grid_dim: (
                        geometry.n_v_heads as u32,
                        (geometry.head_dim / PROMPT_SCAN_COLUMNS) as u32,
                        1,
                    ),
                    block_dim: (CUDA_WARP_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut builder = function.builder();
                builder
                    .arg(&state_view)
                    .arg(&conv_view)
                    .arg(&terms)
                    .arg(&mut output)
                    .arg(&seq_len)
                    .arg(&n_k_heads)
                    .arg(&n_v_heads);
                // SAFETY: All pointers and dimensions were checked above, and
                // the terms kernel runs first on the same stream.
                unsafe { builder.launch(config) }.w()?;
            }

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

    fn values(elements: usize, multiplier: usize, scale: f64) -> Vec<f64> {
        (0..elements)
            .map(|index| ((index * multiplier % 1009) as f64 - 504.0) / scale)
            .collect()
    }

    /// A prompt leaves exactly the output and states that the same tokens
    /// leave when they are scanned one at a time.
    #[test]
    fn cuda_prompt_scan_matches_single_token_scans_bit_for_bit() -> anyhow::Result<()> {
        use candle_core::{Device, Tensor};

        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        for head_dim in [32usize, 64, 128] {
            let (seq_len, kernel_size, n_k_heads, n_v_heads) = (9usize, 4usize, 2usize, 4usize);
            let conv_dim = (2 * n_k_heads + n_v_heads) * head_dim;
            let tensor = |elements: usize, multiplier: usize, scale: f64, shape: &[usize]| {
                let data = values(elements, multiplier, scale)
                    .into_iter()
                    .map(|value| value as f32)
                    .collect::<Vec<_>>();
                Tensor::from_vec(data, shape, &device)
            };
            let input = tensor(seq_len * conv_dim, 17, 211.0, &[seq_len, conv_dim])?;
            let conv_weight = tensor(conv_dim * kernel_size, 29, 401.0, &[conv_dim, kernel_size])?;
            let alpha_raw = tensor(seq_len * n_v_heads, 37, 151.0, &[seq_len, n_v_heads])?;
            let beta_raw = tensor(seq_len * n_v_heads, 41, 173.0, &[seq_len, n_v_heads])?;
            let a = Tensor::from_vec(
                (0..n_v_heads)
                    .map(|head| -0.25 - head as f32 * 0.125)
                    .collect::<Vec<_>>(),
                n_v_heads,
                &device,
            )?;
            let dt_bias = Tensor::from_vec(
                (0..n_v_heads)
                    .map(|head| 0.1 + head as f32 * 0.2)
                    .collect::<Vec<_>>(),
                n_v_heads,
                &device,
            )?;
            let conv_state = tensor(conv_dim * kernel_size, 23, 307.0, &[conv_dim, kernel_size])?;
            let state = tensor(
                n_v_heads * head_dim * head_dim,
                31,
                997.0,
                &[n_v_heads, head_dim, head_dim],
            )?;
            let scan = |input: &Tensor,
                        conv_state: &Tensor,
                        state: &Tensor,
                        alpha_raw: &Tensor,
                        beta_raw: &Tensor| {
                super::prefill(
                    input,
                    conv_state,
                    state,
                    &conv_weight,
                    alpha_raw,
                    beta_raw,
                    &a,
                    &dt_bias,
                    n_k_heads,
                    n_v_heads,
                    head_dim,
                    head_dim,
                    kernel_size,
                )
            };

            let (prompt_output, prompt_conv, prompt_state) =
                scan(&input, &conv_state, &state, &alpha_raw, &beta_raw)?;
            let mut outputs = Vec::new();
            let (mut step_conv, mut step_state) = (conv_state, state);
            for token in 0..seq_len {
                let (output, next_conv, next_state) = scan(
                    &input.narrow(0, token, 1)?.contiguous()?,
                    &step_conv.contiguous()?,
                    &step_state.contiguous()?,
                    &alpha_raw.narrow(0, token, 1)?.contiguous()?,
                    &beta_raw.narrow(0, token, 1)?.contiguous()?,
                )?;
                outputs.push(output);
                (step_conv, step_state) = (next_conv, next_state);
            }
            let step_output = Tensor::cat(&outputs, 0)?;

            let bits = |tensor: &Tensor| -> anyhow::Result<Vec<u32>> {
                Ok(tensor
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            assert_eq!(
                bits(&prompt_output)?,
                bits(&step_output)?,
                "head_dim {head_dim} output"
            );
            assert_eq!(
                bits(&prompt_conv)?,
                bits(&step_conv)?,
                "head_dim {head_dim} convolution state"
            );
            assert_eq!(
                bits(&prompt_state)?,
                bits(&step_state)?,
                "head_dim {head_dim} recurrent state"
            );
        }
        Ok(())
    }

    /// The convolution and recurrent scans against a direct f64 evaluation
    /// of the same recurrences, for every head width and for both the
    /// single-token scan and the prompt scan.
    #[test]
    fn cuda_scans_match_f64_reference() -> anyhow::Result<()> {
        use candle_core::{Device, Tensor};

        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let mut checked = 0;
        for (head_dim, seq_len, n_k_heads, n_v_heads) in [
            (32usize, 1usize, 1usize, 2usize),
            (32, 5, 1, 2),
            (64, 5, 1, 2),
            (128, 1, 2, 4),
            (128, 2, 2, 4),
            (128, 7, 2, 6),
        ] {
            let kernel_size = 4usize;
            let total_key = n_k_heads * head_dim;
            let total_value = n_v_heads * head_dim;
            let conv_dim = 2 * total_key + total_value;
            if !super::supports(
                seq_len,
                conv_dim,
                kernel_size,
                n_k_heads,
                n_v_heads,
                head_dim,
                head_dim,
            ) {
                continue;
            }
            let input = values(seq_len * conv_dim, 17, 211.0);
            let conv_state = values(conv_dim * kernel_size, 23, 307.0);
            let conv_weight = values(conv_dim * kernel_size, 29, 401.0);
            let state = values(n_v_heads * head_dim * head_dim, 31, 997.0);
            let alpha_raw = values(seq_len * n_v_heads, 37, 151.0);
            let beta_raw = values(seq_len * n_v_heads, 41, 173.0);
            let a = (0..n_v_heads)
                .map(|h| -0.25 - h as f64 * 0.125)
                .collect::<Vec<_>>();
            let dt_bias = (0..n_v_heads)
                .map(|h| 0.1 + h as f64 * 0.2)
                .collect::<Vec<_>>();

            let tensor = |data: &[f64], shape: &[usize]| {
                let data = data.iter().map(|value| *value as f32).collect::<Vec<_>>();
                Tensor::from_vec(data, shape, &device)
            };
            let (output, next_conv, next_state) = super::prefill(
                &tensor(&input, &[seq_len, conv_dim])?,
                &tensor(&conv_state, &[conv_dim, kernel_size])?,
                &tensor(&state, &[n_v_heads, head_dim, head_dim])?,
                &tensor(&conv_weight, &[conv_dim, kernel_size])?,
                &tensor(&alpha_raw, &[seq_len, n_v_heads])?,
                &tensor(&beta_raw, &[seq_len, n_v_heads])?,
                &tensor(&a, &[n_v_heads])?,
                &tensor(&dt_bias, &[n_v_heads])?,
                n_k_heads,
                n_v_heads,
                head_dim,
                head_dim,
                kernel_size,
            )?;

            // Reference: causal convolution with SiLU, then the delta rule.
            let round = |value: f64| f64::from(value as f32);
            let mut history = conv_state.iter().map(|v| round(*v)).collect::<Vec<_>>();
            let mut conv = vec![0.0f64; seq_len * conv_dim];
            for token in 0..seq_len {
                for channel in 0..conv_dim {
                    let row = &mut history[channel * kernel_size..(channel + 1) * kernel_size];
                    row.rotate_left(1);
                    row[kernel_size - 1] = round(input[token * conv_dim + channel]);
                    let value = (0..kernel_size)
                        .map(|k| row[k] * round(conv_weight[channel * kernel_size + k]))
                        .sum::<f64>();
                    conv[token * conv_dim + channel] = value / (1.0 + (-value).exp());
                }
            }
            let mut recurrent = state.iter().map(|v| round(*v)).collect::<Vec<_>>();
            let mut expected = vec![0.0f64; seq_len * total_value];
            for token in 0..seq_len {
                for head in 0..n_v_heads {
                    let qk_head = head % n_k_heads;
                    let q_offset = token * conv_dim + qk_head * head_dim;
                    let q = &conv[q_offset..q_offset + head_dim];
                    let k = &conv[q_offset + total_key..q_offset + total_key + head_dim];
                    let q_scale = 1.0
                        / (q.iter().map(|v| v * v).sum::<f64>() + 1e-12).sqrt()
                        / (head_dim as f64).sqrt();
                    let k_scale = 1.0 / (k.iter().map(|v| v * v).sum::<f64>() + 1e-12).sqrt();
                    let gate = token * n_v_heads + head;
                    let beta = 1.0 / (1.0 + (-round(beta_raw[gate])).exp());
                    let dt_input = round(alpha_raw[gate]) + round(dt_bias[head]);
                    let decay = (round(a[head]) * dt_input.exp().ln_1p()).exp();
                    for column in 0..head_dim {
                        let cell = |row: usize| (head * head_dim + row) * head_dim + column;
                        let retrieved = (0..head_dim)
                            .map(|row| decay * recurrent[cell(row)] * k[row] * k_scale)
                            .sum::<f64>();
                        let value =
                            conv[token * conv_dim + 2 * total_key + head * head_dim + column];
                        let delta = beta * (value - retrieved);
                        let mut y = 0.0;
                        for row in 0..head_dim {
                            let next = k[row] * k_scale * delta + decay * recurrent[cell(row)];
                            recurrent[cell(row)] = next;
                            y += next * q[row] * q_scale;
                        }
                        expected[token * total_value + head * head_dim + column] = y;
                    }
                }
            }

            let close = |name: &str, actual: &Tensor, expected: &[f64]| -> anyhow::Result<()> {
                let actual = actual
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(actual.len(), expected.len(), "{name} length");
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    let tolerance = 2e-4 + expected.abs() * 2e-4;
                    assert!(
                        (f64::from(*actual) - expected).abs() <= tolerance,
                        "head_dim {head_dim} tokens {seq_len} {name} {index}: {actual} != {expected}"
                    );
                }
                Ok(())
            };
            close("output", &output, &expected)?;
            close("conv state", &next_conv, &history)?;
            close("recurrent state", &next_state, &recurrent)?;
            checked += 1;
        }
        assert_eq!(checked, 6, "a test geometry was not supported");
        Ok(())
    }
}
