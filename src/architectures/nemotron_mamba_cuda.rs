use anyhow::{Context, Result, ensure};

const MAX_CONV_KERNEL: usize = 8;
const MAX_D_STATE: usize = 256;
const CUDA_SCAN_WARPS: usize = 4;
#[cfg(feature = "cuda")]
const CUDA_WARP_SIZE: usize = 32;
#[cfg(feature = "cuda")]
const CUDA_SCAN_THREADS: usize = CUDA_WARP_SIZE * CUDA_SCAN_WARPS;
#[cfg(feature = "cuda")]
const CUDA_NORM_THREADS: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Mamba2Geometry {
    seq_len: usize,
    conv_dim: usize,
    kernel_size: usize,
    n_heads: usize,
    d_state: usize,
    head_dim: usize,
    n_group: usize,
    inner_dim: usize,
    group_size: usize,
    conv_elements: usize,
    conv_state_elements: usize,
    output_elements: usize,
    ssm_state_elements: usize,
    packed_elements: usize,
}

impl Mamba2Geometry {
    #[allow(clippy::too_many_arguments)]
    fn new(
        seq_len: usize,
        conv_dim: usize,
        kernel_size: usize,
        n_heads: usize,
        d_state: usize,
        head_dim: usize,
        n_group: usize,
    ) -> Result<Self> {
        ensure!(
            seq_len > 0,
            "Nemotron CUDA Mamba2 requires at least one token"
        );
        ensure!(
            (1..=MAX_CONV_KERNEL).contains(&kernel_size),
            "Nemotron CUDA Mamba2 supports convolution widths 1 through {MAX_CONV_KERNEL}"
        );
        ensure!(n_heads > 0, "Nemotron CUDA Mamba2 requires heads");
        ensure!(head_dim > 0, "Nemotron CUDA Mamba2 requires a head width");
        ensure!(
            (1..=MAX_D_STATE).contains(&d_state),
            "Nemotron CUDA Mamba2 supports state widths 1 through {MAX_D_STATE}"
        );
        ensure!(
            n_group > 0 && n_heads.is_multiple_of(n_group),
            "Nemotron CUDA Mamba2 heads must divide evenly into groups"
        );

        let inner_dim = checked_mul(n_heads, head_dim, "inner width")?;
        let grouped_state = checked_mul(n_group, d_state, "grouped state width")?;
        let expected_conv_dim = checked_add(
            inner_dim,
            checked_mul(grouped_state, 2, "B/C convolution width")?,
            "convolution width",
        )?;
        ensure!(
            conv_dim == expected_conv_dim,
            "Nemotron CUDA Mamba2 convolution width is {conv_dim}, expected {expected_conv_dim}"
        );

        for (name, value) in [
            ("sequence length", seq_len),
            ("convolution width", conv_dim),
            ("head count", n_heads),
            ("head width", head_dim),
            ("state width", d_state),
            ("group count", n_group),
            ("group width", inner_dim / n_group),
        ] {
            ensure!(
                value <= i32::MAX as usize,
                "Nemotron CUDA Mamba2 {name} exceeds the kernel index limit"
            );
        }
        ensure!(
            n_group <= u16::MAX as usize,
            "Nemotron CUDA Mamba2 group count exceeds the CUDA grid limit"
        );

        let conv_elements = checked_mul(seq_len, conv_dim, "convolution output")?;
        let conv_state_elements = checked_mul(conv_dim, kernel_size, "convolution state")?;
        let output_elements = checked_mul(seq_len, inner_dim, "normalized output")?;
        let ssm_state_elements = checked_mul(
            checked_mul(n_heads, d_state, "recurrent state heads")?,
            head_dim,
            "recurrent state",
        )?;
        let packed_elements = checked_add(
            checked_add(
                output_elements,
                conv_state_elements,
                "packed convolution state",
            )?,
            ssm_state_elements,
            "packed recurrent state",
        )?;
        for (name, elements) in [
            ("convolution output", conv_elements),
            ("convolution state", conv_state_elements),
            ("normalized output", output_elements),
            ("recurrent state", ssm_state_elements),
            ("packed output", packed_elements),
        ] {
            ensure!(
                elements <= i32::MAX as usize,
                "Nemotron CUDA Mamba2 {name} exceeds the signed kernel index limit"
            );
        }
        ensure!(
            packed_elements <= i64::MAX as usize,
            "Nemotron CUDA Mamba2 packed output exceeds the kernel offset limit"
        );

        Ok(Self {
            seq_len,
            conv_dim,
            kernel_size,
            n_heads,
            d_state,
            head_dim,
            n_group,
            inner_dim,
            group_size: inner_dim / n_group,
            conv_elements,
            conv_state_elements,
            output_elements,
            ssm_state_elements,
            packed_elements,
        })
    }

    fn next_conv_offset(self) -> usize {
        self.output_elements
    }

    fn next_ssm_offset(self) -> usize {
        self.output_elements + self.conv_state_elements
    }

    fn scan_blocks(self) -> usize {
        self.inner_dim.div_ceil(CUDA_SCAN_WARPS)
    }
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .with_context(|| format!("Nemotron CUDA Mamba2 {label} overflows usize"))
}

fn checked_add(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_add(rhs)
        .with_context(|| format!("Nemotron CUDA Mamba2 {label} overflows usize"))
}

fn checked_row_span(rows: usize, columns: usize, row_stride: usize) -> Result<usize> {
    ensure!(rows > 0, "Nemotron CUDA Mamba2 row span requires rows");
    ensure!(
        columns > 0,
        "Nemotron CUDA Mamba2 row span requires columns"
    );
    ensure!(
        row_stride <= i32::MAX as usize,
        "Nemotron CUDA Mamba2 row stride exceeds the signed kernel index limit"
    );
    ensure!(
        rows == 1 || row_stride >= columns,
        "Nemotron CUDA Mamba2 row stride overlaps rows"
    );
    let span = checked_add(
        checked_mul(rows - 1, row_stride, "strided row span")?,
        columns,
        "strided row span",
    )?;
    ensure!(
        span <= i32::MAX as usize,
        "Nemotron CUDA Mamba2 row span exceeds the signed kernel index limit"
    );
    Ok(span)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn supports(
    seq_len: usize,
    conv_dim: usize,
    kernel_size: usize,
    n_heads: usize,
    d_state: usize,
    head_dim: usize,
    n_group: usize,
) -> bool {
    Mamba2Geometry::new(
        seq_len,
        conv_dim,
        kernel_size,
        n_heads,
        d_state,
        head_dim,
        n_group,
    )
    .is_ok()
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::sync::OnceLock;

    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    use candle_core::cuda_backend::{CudaDevice, WrapErr};
    use candle_core::{CpuStorage, CudaStorage, DType, Device, Layout, Shape, Storage, Tensor};

    use super::{CUDA_NORM_THREADS, CUDA_SCAN_THREADS, Mamba2Geometry};

    const CUDA_MODULE_NAME: &str = "dinf_nemotron_mamba2_v1";
    const PREFILL_CONV_KERNEL: &str = "nemotron_mamba2_prefill_conv_f32";
    const PREFILL_SCAN_KERNEL: &str = "nemotron_mamba2_prefill_scan_f32";
    const GATE_NORM_KERNEL: &str = "nemotron_mamba2_gate_group_norm_f32";
    const DECODE_KERNEL: &str = "nemotron_mamba2_decode_f32";

    // Prompt scan follows llama.cpp's grouped SSM layout: one warp owns one
    // [head, head-dimension] state vector and retains up to 256 state values in
    // registers across the token loop. Decode is arranged by Mamba group so a
    // single block can also finish gate-first RMS normalization.
    const CUDA_SOURCE: &str = r#"
#define NEMOTRON_MAX_CONV 8
#define NEMOTRON_MAX_STATE 256
#define NEMOTRON_WARP_SIZE 32

__device__ __forceinline__ float nemotron_softplus(float value) {
    return value <= 20.0f ? log1pf(expf(value)) : value;
}

__device__ __forceinline__ float nemotron_silu(float value) {
    return value / (1.0f + expf(-value));
}

__device__ __forceinline__ float nemotron_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

extern "C" __global__ void nemotron_mamba2_prefill_conv_f32(
    const float *input,
    const float *state,
    const float *weight,
    const float *bias,
    float *conv,
    float *packed,
    int seq_len,
    int conv_dim,
    int kernel_size,
    int input_stride,
    long long next_conv_offset) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= conv_dim) {
        return;
    }

    float history[NEMOTRON_MAX_CONV];
#pragma unroll
    for (int k = 0; k < NEMOTRON_MAX_CONV; ++k) {
        if (k < kernel_size) {
            history[k] = state[channel * kernel_size + k];
        }
    }

    for (int token = 0; token < seq_len; ++token) {
#pragma unroll
        for (int k = 1; k < NEMOTRON_MAX_CONV; ++k) {
            if (k < kernel_size) {
                history[k - 1] = history[k];
            }
        }
        history[kernel_size - 1] = input[token * input_stride + channel];

        float value = bias[channel];
#pragma unroll
        for (int k = 0; k < NEMOTRON_MAX_CONV; ++k) {
            if (k < kernel_size) {
                value = fmaf(history[k], weight[channel * kernel_size + k], value);
            }
        }
        conv[token * conv_dim + channel] = nemotron_silu(value);
    }

#pragma unroll
    for (int k = 0; k < NEMOTRON_MAX_CONV; ++k) {
        if (k < kernel_size) {
            packed[next_conv_offset + channel * kernel_size + k] = history[k];
        }
    }
}

extern "C" __global__ void nemotron_mamba2_prefill_scan_f32(
    const float *state,
    const float *conv,
    const float *dt_raw,
    const float *a,
    const float *d,
    const float *dt_bias,
    float *raw_y,
    float *packed,
    int seq_len,
    int conv_dim,
    int n_heads,
    int d_state,
    int head_dim,
    int n_group,
    int dt_stride,
    long long next_ssm_offset) {
    const int warp = threadIdx.x / NEMOTRON_WARP_SIZE;
    const int lane = threadIdx.x % NEMOTRON_WARP_SIZE;
    const int output_index = blockIdx.x * (blockDim.x / NEMOTRON_WARP_SIZE) + warp;
    const int inner_dim = n_heads * head_dim;
    if (output_index >= inner_dim) {
        return;
    }

    const int head = output_index / head_dim;
    const int dim = output_index - head * head_dim;
    const int heads_per_group = n_heads / n_group;
    const int group = head / heads_per_group;
    const int state_slots = (d_state + NEMOTRON_WARP_SIZE - 1) / NEMOTRON_WARP_SIZE;
    float history[NEMOTRON_MAX_STATE / NEMOTRON_WARP_SIZE];
#pragma unroll
    for (int slot = 0; slot < NEMOTRON_MAX_STATE / NEMOTRON_WARP_SIZE; ++slot) {
        const int state_index = lane + slot * NEMOTRON_WARP_SIZE;
        if (slot < state_slots && state_index < d_state) {
            history[slot] = state[(head * d_state + state_index) * head_dim + dim];
        }
    }

    for (int token = 0; token < seq_len; ++token) {
        const float dt = nemotron_softplus(dt_raw[token * dt_stride + head] + dt_bias[head]);
        const float decay = expf(dt * a[head]);
        const float x = conv[token * conv_dim + output_index];
        const float x_dt = x * dt;
        const int group_base = token * conv_dim + inner_dim + group * d_state;
        float weighted = 0.0f;
#pragma unroll
        for (int slot = 0; slot < NEMOTRON_MAX_STATE / NEMOTRON_WARP_SIZE; ++slot) {
            const int state_index = lane + slot * NEMOTRON_WARP_SIZE;
            if (slot < state_slots && state_index < d_state) {
                const float next = fmaf(
                    conv[group_base + state_index],
                    x_dt,
                    history[slot] * decay);
                history[slot] = next;
                weighted = fmaf(next, conv[group_base + n_group * d_state + state_index], weighted);
            }
        }
        const float sum = nemotron_warp_sum(weighted);
        if (lane == 0) {
            raw_y[token * inner_dim + output_index] = fmaf(d[head], x, sum);
        }
    }

#pragma unroll
    for (int slot = 0; slot < NEMOTRON_MAX_STATE / NEMOTRON_WARP_SIZE; ++slot) {
        const int state_index = lane + slot * NEMOTRON_WARP_SIZE;
        if (slot < state_slots && state_index < d_state) {
            packed[next_ssm_offset + (head * d_state + state_index) * head_dim + dim] = history[slot];
        }
    }
}

extern "C" __global__ void nemotron_mamba2_gate_group_norm_f32(
    const float *z,
    const float *raw_y,
    const float *weight,
    float *packed,
    int seq_len,
    int n_group,
    int group_size,
    int z_stride) {
    const int token = blockIdx.x;
    const int group = blockIdx.y;
    if (token >= seq_len || group >= n_group) {
        return;
    }

    const int base = (token * n_group + group) * group_size;
    const int z_base = token * z_stride + group * group_size;
    float local_sum = 0.0f;
    for (int index = threadIdx.x; index < group_size; index += blockDim.x) {
        const float gated = nemotron_silu(z[z_base + index]) * raw_y[base + index];
        packed[base + index] = gated;
        local_sum = fmaf(gated, gated, local_sum);
    }

    __shared__ float warp_sums[8];
    __shared__ float scale;
    const int lane = threadIdx.x % NEMOTRON_WARP_SIZE;
    const int warp = threadIdx.x / NEMOTRON_WARP_SIZE;
    const float warp_sum = nemotron_warp_sum(local_sum);
    if (lane == 0) {
        warp_sums[warp] = warp_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int index = 0; index < blockDim.x / NEMOTRON_WARP_SIZE; ++index) {
            sum += warp_sums[index];
        }
        scale = rsqrtf(sum / (float) group_size + 1.0e-5f);
    }
    __syncthreads();

    const int weight_base = group * group_size;
    for (int index = threadIdx.x; index < group_size; index += blockDim.x) {
        packed[base + index] *= scale * weight[weight_base + index];
    }
}

__device__ __forceinline__ float nemotron_decode_conv(
    int channel,
    float input,
    const float *state,
    const float *weight,
    const float *bias,
    float *packed,
    int kernel_size,
    long long next_conv_offset) {
    float value = bias[channel];
#pragma unroll
    for (int k = 0; k < NEMOTRON_MAX_CONV; ++k) {
        if (k < kernel_size) {
            const float history = k + 1 < kernel_size
                ? state[channel * kernel_size + k + 1]
                : input;
            packed[next_conv_offset + channel * kernel_size + k] = history;
            value = fmaf(history, weight[channel * kernel_size + k], value);
        }
    }
    return nemotron_silu(value);
}

extern "C" __global__ void nemotron_mamba2_decode_f32(
    const float *z,
    const float *conv_input,
    const float *dt_raw,
    const float *conv_state,
    const float *ssm_state,
    const float *conv_weight,
    const float *conv_bias,
    const float *a,
    const float *d,
    const float *dt_bias,
    const float *norm_weight,
    float *packed,
    int conv_dim,
    int kernel_size,
    int n_heads,
    int d_state,
    int head_dim,
    int n_group,
    int group_size,
    long long next_conv_offset,
    long long next_ssm_offset) {
    const int group = blockIdx.x;
    if (group >= n_group) {
        return;
    }
    const int inner_dim = n_heads * head_dim;
    const int group_state_base = group * d_state;

    __shared__ float b_values[NEMOTRON_MAX_STATE];
    __shared__ float c_values[NEMOTRON_MAX_STATE];
    __shared__ float warp_sums[8];
    __shared__ float scale;

    for (int state_index = threadIdx.x; state_index < d_state; state_index += blockDim.x) {
        const int b_channel = inner_dim + group_state_base + state_index;
        const int c_channel = inner_dim + n_group * d_state + group_state_base + state_index;
        b_values[state_index] = nemotron_decode_conv(
            b_channel, conv_input[b_channel], conv_state, conv_weight, conv_bias,
            packed, kernel_size, next_conv_offset);
        c_values[state_index] = nemotron_decode_conv(
            c_channel, conv_input[c_channel], conv_state, conv_weight, conv_bias,
            packed, kernel_size, next_conv_offset);
    }
    const int group_output_base = group * group_size;
    for (int local = threadIdx.x; local < group_size; local += blockDim.x) {
        const int channel = group_output_base + local;
        packed[channel] = nemotron_decode_conv(
            channel, conv_input[channel], conv_state, conv_weight, conv_bias,
            packed, kernel_size, next_conv_offset);
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int local = threadIdx.x; local < group_size; local += blockDim.x) {
        const int channel = group_output_base + local;
        const int head = channel / head_dim;
        const int dim = channel - head * head_dim;
        const float x = packed[channel];
        const float dt = nemotron_softplus(dt_raw[head] + dt_bias[head]);
        const float decay = expf(dt * a[head]);
        const float x_dt = x * dt;
        float y = 0.0f;
        for (int state_index = 0; state_index < d_state; ++state_index) {
            const long long index = (long long) (head * d_state + state_index) * head_dim + dim;
            const float next = fmaf(
                b_values[state_index], x_dt, ssm_state[index] * decay);
            packed[next_ssm_offset + index] = next;
            y = fmaf(next, c_values[state_index], y);
        }
        y = fmaf(d[head], x, y);
        const float gated = nemotron_silu(z[channel]) * y;
        packed[channel] = gated;
        local_sum = fmaf(gated, gated, local_sum);
    }

    const int lane = threadIdx.x % NEMOTRON_WARP_SIZE;
    const int warp = threadIdx.x / NEMOTRON_WARP_SIZE;
    const float warp_sum = nemotron_warp_sum(local_sum);
    if (lane == 0) {
        warp_sums[warp] = warp_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int index = 0; index < blockDim.x / NEMOTRON_WARP_SIZE; ++index) {
            sum += warp_sums[index];
        }
        scale = rsqrtf(sum / (float) group_size + 1.0e-5f);
    }
    __syncthreads();

    for (int local = threadIdx.x; local < group_size; local += blockDim.x) {
        const int channel = group_output_base + local;
        packed[channel] *= scale * norm_weight[channel];
    }
}
"#;

    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    static CUDA_LOGGED: OnceLock<()> = OnceLock::new();

    fn ptx() -> candle_core::Result<&'static str> {
        match PTX.get_or_init(|| {
            compile_ptx_with_opts(
                CUDA_SOURCE,
                CompileOptions {
                    name: Some("nemotron_mamba2.cu".to_string()),
                    use_fast_math: Some(true),
                    ..Default::default()
                },
            )
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
        }) {
            Ok(ptx) => Ok(ptx.as_str()),
            Err(error) => candle_core::bail!("failed to compile Nemotron CUDA Mamba2: {error}"),
        }
    }

    pub(crate) fn precompile(device: &Device) -> anyhow::Result<()> {
        if !device.is_cuda() || std::env::var_os("DINF_DISABLE_NEMOTRON_MAMBA_CUDA").is_some() {
            return Ok(());
        }
        let device = device.as_cuda_device()?;
        let ptx = ptx()?;
        for kernel in [
            PREFILL_CONV_KERNEL,
            PREFILL_SCAN_KERNEL,
            GATE_NORM_KERNEL,
            DECODE_KERNEL,
        ] {
            device.get_or_load_custom_func(kernel, CUDA_MODULE_NAME, ptx)?;
        }
        CUDA_LOGGED.get_or_init(|| {
            tracing::info!("fused CUDA Mamba2 prompt and decode paths enabled for Nemotron-H");
        });
        Ok(())
    }

    struct FusedMamba2<'a> {
        z: &'a Tensor,
        dt_raw: &'a Tensor,
        conv_weight: &'a Tensor,
        conv_bias: &'a Tensor,
        a: &'a Tensor,
        d: &'a Tensor,
        dt_bias: &'a Tensor,
        norm_weight: &'a Tensor,
        geometry: Mamba2Geometry,
    }

    impl candle_core::CustomOp3 for FusedMamba2<'_> {
        fn name(&self) -> &'static str {
            "nemotron-cuda-mamba2"
        }

        fn cpu_fwd(
            &self,
            _conv_input: &CpuStorage,
            _conv_layout: &Layout,
            _conv_state: &CpuStorage,
            _conv_state_layout: &Layout,
            _ssm_state: &CpuStorage,
            _ssm_state_layout: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("nemotron-cuda-mamba2 is only dispatched on CUDA")
        }

        fn cuda_fwd(
            &self,
            conv_input: &CudaStorage,
            conv_layout: &Layout,
            conv_state: &CudaStorage,
            conv_state_layout: &Layout,
            ssm_state: &CudaStorage,
            ssm_state_layout: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let geometry = self.geometry;
            let conv_input_view = checked_row_view(
                "convolution input",
                conv_input,
                conv_layout,
                geometry.seq_len,
                geometry.conv_dim,
            )?;
            validate_cuda_storage(
                "convolution state",
                conv_state,
                conv_state_layout,
                geometry.conv_state_elements,
            )?;
            validate_cuda_storage(
                "recurrent state",
                ssm_state,
                ssm_state_layout,
                geometry.ssm_state_elements,
            )?;
            ensure_same_cuda_device("convolution state", conv_input, conv_state)?;
            ensure_same_cuda_device("recurrent state", conv_input, ssm_state)?;

            let device = conv_input.device();
            let (z_guard, z_layout) = self.z.storage_and_layout();
            let z = captured_cuda_storage("gate", &z_guard, z_layout, device)?;
            let z_view =
                checked_row_view("gate", z, z_layout, geometry.seq_len, geometry.inner_dim)?;
            let (dt_guard, dt_layout) = self.dt_raw.storage_and_layout();
            let dt = captured_cuda_storage("time step", &dt_guard, dt_layout, device)?;
            let dt_view = checked_row_view(
                "time step",
                dt,
                dt_layout,
                geometry.seq_len,
                geometry.n_heads,
            )?;
            let (conv_weight_guard, conv_weight_layout) = self.conv_weight.storage_and_layout();
            let conv_weight = captured_contiguous_view(
                "convolution weight",
                &conv_weight_guard,
                conv_weight_layout,
                geometry.conv_state_elements,
                device,
            )?;
            let (conv_bias_guard, conv_bias_layout) = self.conv_bias.storage_and_layout();
            let conv_bias = captured_contiguous_view(
                "convolution bias",
                &conv_bias_guard,
                conv_bias_layout,
                geometry.conv_dim,
                device,
            )?;
            let (a_guard, a_layout) = self.a.storage_and_layout();
            let a = captured_contiguous_view(
                "decay rate",
                &a_guard,
                a_layout,
                geometry.n_heads,
                device,
            )?;
            let (d_guard, d_layout) = self.d.storage_and_layout();
            let d = captured_contiguous_view(
                "skip weight",
                &d_guard,
                d_layout,
                geometry.n_heads,
                device,
            )?;
            let (dt_bias_guard, dt_bias_layout) = self.dt_bias.storage_and_layout();
            let dt_bias = captured_contiguous_view(
                "time-step bias",
                &dt_bias_guard,
                dt_bias_layout,
                geometry.n_heads,
                device,
            )?;
            let (norm_guard, norm_layout) = self.norm_weight.storage_and_layout();
            let norm = captured_contiguous_view(
                "group norm weight",
                &norm_guard,
                norm_layout,
                geometry.inner_dim,
                device,
            )?;

            let conv_state_view = checked_view(
                conv_state.as_cuda_slice::<f32>()?,
                conv_state_layout,
                geometry.conv_state_elements,
            )?;
            let ssm_state_view = checked_view(
                ssm_state.as_cuda_slice::<f32>()?,
                ssm_state_layout,
                geometry.ssm_state_elements,
            )?;
            let mut packed = unsafe { device.alloc::<f32>(geometry.packed_elements)? };

            let conv_stride = checked_i32(conv_layout.stride()[0], "convolution row stride")?;
            let dt_stride = checked_i32(dt_layout.stride()[0], "time-step row stride")?;
            let z_stride = checked_i32(z_layout.stride()[0], "gate row stride")?;
            let conv_dim = geometry.conv_dim as i32;
            let kernel_size = geometry.kernel_size as i32;
            let n_heads = geometry.n_heads as i32;
            let d_state = geometry.d_state as i32;
            let head_dim = geometry.head_dim as i32;
            let n_group = geometry.n_group as i32;
            let group_size = geometry.group_size as i32;
            let next_conv_offset = geometry.next_conv_offset() as i64;
            let next_ssm_offset = geometry.next_ssm_offset() as i64;

            if geometry.seq_len == 1 {
                let function =
                    device.get_or_load_custom_func(DECODE_KERNEL, CUDA_MODULE_NAME, ptx()?)?;
                let config = LaunchConfig {
                    grid_dim: (geometry.n_group as u32, 1, 1),
                    block_dim: (CUDA_NORM_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut builder = function.builder();
                builder
                    .arg(&z_view)
                    .arg(&conv_input_view)
                    .arg(&dt_view)
                    .arg(&conv_state_view)
                    .arg(&ssm_state_view)
                    .arg(&conv_weight)
                    .arg(&conv_bias)
                    .arg(&a)
                    .arg(&d)
                    .arg(&dt_bias)
                    .arg(&norm)
                    .arg(&mut packed)
                    .arg(&conv_dim)
                    .arg(&kernel_size)
                    .arg(&n_heads)
                    .arg(&d_state)
                    .arg(&head_dim)
                    .arg(&n_group)
                    .arg(&group_size)
                    .arg(&next_conv_offset)
                    .arg(&next_ssm_offset);
                unsafe { builder.launch(config) }.w()?;
            } else {
                let mut conv = unsafe { device.alloc::<f32>(geometry.conv_elements)? };
                let conv_function = device.get_or_load_custom_func(
                    PREFILL_CONV_KERNEL,
                    CUDA_MODULE_NAME,
                    ptx()?,
                )?;
                let seq_len = geometry.seq_len as i32;
                let conv_config = LaunchConfig {
                    grid_dim: (geometry.conv_dim.div_ceil(256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut conv_builder = conv_function.builder();
                conv_builder
                    .arg(&conv_input_view)
                    .arg(&conv_state_view)
                    .arg(&conv_weight)
                    .arg(&conv_bias)
                    .arg(&mut conv)
                    .arg(&mut packed)
                    .arg(&seq_len)
                    .arg(&conv_dim)
                    .arg(&kernel_size)
                    .arg(&conv_stride)
                    .arg(&next_conv_offset);
                unsafe { conv_builder.launch(conv_config) }.w()?;

                let mut raw_y = unsafe { device.alloc::<f32>(geometry.output_elements)? };
                let scan_function = device.get_or_load_custom_func(
                    PREFILL_SCAN_KERNEL,
                    CUDA_MODULE_NAME,
                    ptx()?,
                )?;
                let scan_config = LaunchConfig {
                    grid_dim: (geometry.scan_blocks() as u32, 1, 1),
                    block_dim: (CUDA_SCAN_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut scan_builder = scan_function.builder();
                scan_builder
                    .arg(&ssm_state_view)
                    .arg(&conv)
                    .arg(&dt_view)
                    .arg(&a)
                    .arg(&d)
                    .arg(&dt_bias)
                    .arg(&mut raw_y)
                    .arg(&mut packed)
                    .arg(&seq_len)
                    .arg(&conv_dim)
                    .arg(&n_heads)
                    .arg(&d_state)
                    .arg(&head_dim)
                    .arg(&n_group)
                    .arg(&dt_stride)
                    .arg(&next_ssm_offset);
                unsafe { scan_builder.launch(scan_config) }.w()?;

                let gate_function =
                    device.get_or_load_custom_func(GATE_NORM_KERNEL, CUDA_MODULE_NAME, ptx()?)?;
                let gate_config = LaunchConfig {
                    grid_dim: (geometry.seq_len as u32, geometry.n_group as u32, 1),
                    block_dim: (CUDA_NORM_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut gate_builder = gate_function.builder();
                gate_builder
                    .arg(&z_view)
                    .arg(&raw_y)
                    .arg(&norm)
                    .arg(&mut packed)
                    .arg(&seq_len)
                    .arg(&n_group)
                    .arg(&group_size)
                    .arg(&z_stride);
                unsafe { gate_builder.launch(gate_config) }.w()?;
            }

            Ok((
                CudaStorage::wrap_cuda_slice(packed, device.clone()),
                Shape::from(geometry.packed_elements),
            ))
        }
    }

    fn checked_i32(value: usize, label: &str) -> candle_core::Result<i32> {
        i32::try_from(value).map_err(|_| {
            candle_core::Error::Msg(format!("Nemotron CUDA Mamba2 {label} exceeds i32"))
        })
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
                "{name} must be contiguous f32 with {expected_elements} elements for Nemotron CUDA Mamba2"
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

    fn checked_row_view<'a>(
        name: &str,
        storage: &'a CudaStorage,
        layout: &Layout,
        rows: usize,
        columns: usize,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>> {
        if storage.dtype() != DType::F32
            || layout.shape().dims2()? != (rows, columns)
            || layout.stride()[1] != 1
            || (rows > 1 && layout.stride()[0] < columns)
        {
            candle_core::bail!(
                "{name} must be row-contiguous f32 [{rows}, {columns}] for Nemotron CUDA Mamba2"
            )
        }
        let span = super::checked_row_span(rows, columns, layout.stride()[0])
            .map_err(candle_core::Error::wrap)?;
        checked_slice(
            storage.as_cuda_slice::<f32>()?,
            layout.start_offset(),
            span,
            name,
        )
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
        _layout: &Layout,
        expected_device: &CudaDevice,
    ) -> candle_core::Result<&'a CudaStorage> {
        let storage = match storage {
            Storage::Cuda(storage) => storage,
            _ => candle_core::bail!("{name} is not on CUDA"),
        };
        if storage.device().id() != expected_device.id() {
            candle_core::bail!("{name} is on a different CUDA device")
        }
        Ok(storage)
    }

    fn captured_contiguous_view<'a>(
        name: &str,
        storage: &'a Storage,
        layout: &Layout,
        expected_elements: usize,
        expected_device: &CudaDevice,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>> {
        let storage = captured_cuda_storage(name, storage, layout, expected_device)?;
        validate_cuda_storage(name, storage, layout, expected_elements)?;
        checked_view(storage.as_cuda_slice::<f32>()?, layout, expected_elements)
    }

    fn checked_view<'a>(
        slice: &'a candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
        layout: &Layout,
        elements: usize,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>> {
        checked_slice(slice, layout.start_offset(), elements, "CUDA tensor")
    }

    fn checked_slice<'a>(
        slice: &'a candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
        start: usize,
        elements: usize,
        name: &str,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>> {
        let end = start
            .checked_add(elements)
            .ok_or_else(|| candle_core::Error::Msg(format!("{name} CUDA view overflow")))?;
        if end > slice.len() {
            candle_core::bail!("{name} CUDA view exceeds its storage")
        }
        Ok(slice.slice(start..end))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        z: &Tensor,
        conv_input: &Tensor,
        dt_raw: &Tensor,
        conv_state: &Tensor,
        ssm_state: &Tensor,
        conv_weight: &Tensor,
        conv_bias: &Tensor,
        a: &Tensor,
        d: &Tensor,
        dt_bias: &Tensor,
        norm_weight: &Tensor,
        n_heads: usize,
        d_state: usize,
        head_dim: usize,
        n_group: usize,
        kernel_size: usize,
    ) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let (seq_len, conv_dim) = conv_input.dims2()?;
        let geometry = Mamba2Geometry::new(
            seq_len,
            conv_dim,
            kernel_size,
            n_heads,
            d_state,
            head_dim,
            n_group,
        )?;
        anyhow::ensure!(
            z.dims2()? == (seq_len, geometry.inner_dim),
            "invalid Nemotron CUDA gate shape"
        );
        anyhow::ensure!(
            dt_raw.dims2()? == (seq_len, n_heads),
            "invalid Nemotron CUDA time-step shape"
        );
        anyhow::ensure!(
            conv_state.dims2()? == (conv_dim, kernel_size),
            "invalid Nemotron CUDA convolution state shape"
        );
        anyhow::ensure!(
            ssm_state.dims3()? == (n_heads, d_state, head_dim),
            "invalid Nemotron CUDA recurrent state shape"
        );
        anyhow::ensure!(
            conv_weight.dims2()? == (conv_dim, kernel_size),
            "invalid Nemotron CUDA convolution weight shape"
        );
        anyhow::ensure!(
            conv_bias.elem_count() == conv_dim,
            "invalid Nemotron CUDA convolution bias shape"
        );
        anyhow::ensure!(
            a.elem_count() == n_heads
                && d.elem_count() == n_heads
                && dt_bias.elem_count() == n_heads,
            "invalid Nemotron CUDA head parameter shape"
        );
        anyhow::ensure!(
            norm_weight.dims2()? == (n_group, geometry.group_size),
            "invalid Nemotron CUDA group norm shape"
        );
        for (name, tensor) in [
            ("gate", z),
            ("time step", dt_raw),
            ("convolution state", conv_state),
            ("recurrent state", ssm_state),
            ("convolution weight", conv_weight),
            ("convolution bias", conv_bias),
            ("decay rate", a),
            ("skip weight", d),
            ("time-step bias", dt_bias),
            ("group norm weight", norm_weight),
        ] {
            anyhow::ensure!(tensor.dtype() == DType::F32, "{name} must be f32");
            anyhow::ensure!(
                conv_input.device().same_device(tensor.device()),
                "{name} is on a different device"
            );
        }

        let packed = conv_input.apply_op3_no_bwd(
            conv_state,
            ssm_state,
            &FusedMamba2 {
                z,
                dt_raw,
                conv_weight,
                conv_bias,
                a,
                d,
                dt_bias,
                norm_weight,
                geometry,
            },
        )?;
        let normalized = packed
            .narrow(0, 0, geometry.output_elements)?
            .reshape((seq_len, geometry.inner_dim))?;
        let next_conv_state = packed
            .narrow(0, geometry.next_conv_offset(), geometry.conv_state_elements)?
            .reshape((conv_dim, kernel_size))?;
        let next_ssm_state = packed
            .narrow(0, geometry.next_ssm_offset(), geometry.ssm_state_elements)?
            .reshape((n_heads, d_state, head_dim))?;
        Ok((normalized, next_conv_state, next_ssm_state))
    }
}

#[cfg(feature = "cuda")]
pub(super) use cuda::{forward, precompile};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct ReferenceInputs {
        geometry: Mamba2Geometry,
        z: Vec<f32>,
        conv_input: Vec<f32>,
        conv_state: Vec<f32>,
        ssm_state: Vec<f32>,
        conv_weight: Vec<f32>,
        conv_bias: Vec<f32>,
        dt_raw: Vec<f32>,
        a: Vec<f32>,
        d: Vec<f32>,
        dt_bias: Vec<f32>,
        norm_weight: Vec<f32>,
    }

    fn sample_inputs(seq_len: usize) -> ReferenceInputs {
        let geometry = Mamba2Geometry::new(seq_len, 20, 3, 4, 3, 2, 2).unwrap();
        let values = |count: usize, factor: usize, scale: f32| {
            (0..count)
                .map(|index| (((index * factor) % 31) as f32 - 15.0) * scale)
                .collect::<Vec<_>>()
        };
        ReferenceInputs {
            geometry,
            z: values(geometry.output_elements, 7, 0.03),
            conv_input: values(geometry.conv_elements, 11, 0.02),
            conv_state: values(geometry.conv_state_elements, 13, 0.01),
            ssm_state: values(geometry.ssm_state_elements, 17, 0.005),
            conv_weight: values(geometry.conv_state_elements, 19, 0.01),
            conv_bias: values(geometry.conv_dim, 23, 0.005),
            dt_raw: values(seq_len * geometry.n_heads, 5, 0.04),
            a: vec![-0.1, -0.2, -0.3, -0.4],
            d: vec![0.5, 0.6, 0.7, 0.8],
            dt_bias: vec![0.1, -0.1, 0.2, -0.2],
            norm_weight: values(geometry.inner_dim, 29, 0.02),
        }
    }

    fn silu(value: f32) -> f32 {
        value / (1.0 + (-value).exp())
    }

    fn softplus(value: f32) -> f32 {
        if value <= 20.0 {
            value.exp().ln_1p()
        } else {
            value
        }
    }

    fn portable_reference(input: &ReferenceInputs) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let g = input.geometry;
        let mut conv_state = input.conv_state.clone();
        let mut ssm_state = input.ssm_state.clone();
        let mut output = vec![0.0; g.output_elements];
        for token in 0..g.seq_len {
            let mut conv = vec![0.0; g.conv_dim];
            for (channel, conv_value) in conv.iter_mut().enumerate() {
                let base = channel * g.kernel_size;
                conv_state.copy_within(base + 1..base + g.kernel_size, base);
                conv_state[base + g.kernel_size - 1] =
                    input.conv_input[token * g.conv_dim + channel];
                let value = (0..g.kernel_size).fold(input.conv_bias[channel], |sum, kernel| {
                    sum + conv_state[base + kernel] * input.conv_weight[base + kernel]
                });
                *conv_value = silu(value);
            }

            for head in 0..g.n_heads {
                let group = head / (g.n_heads / g.n_group);
                let dt = softplus(input.dt_raw[token * g.n_heads + head] + input.dt_bias[head]);
                let decay = (dt * input.a[head]).exp();
                for dim in 0..g.head_dim {
                    let x_index = head * g.head_dim + dim;
                    let x = conv[x_index];
                    let mut y = 0.0;
                    for state_index in 0..g.d_state {
                        let index = (head * g.d_state + state_index) * g.head_dim + dim;
                        let b = conv[g.inner_dim + group * g.d_state + state_index];
                        let c = conv
                            [g.inner_dim + g.n_group * g.d_state + group * g.d_state + state_index];
                        let next = ssm_state[index] * decay + b * x * dt;
                        ssm_state[index] = next;
                        y += next * c;
                    }
                    output[token * g.inner_dim + x_index] = y + input.d[head] * x;
                }
            }

            for group in 0..g.n_group {
                let base = token * g.inner_dim + group * g.group_size;
                let mut sum = 0.0;
                for index in 0..g.group_size {
                    let gated = silu(input.z[base + index]) * output[base + index];
                    output[base + index] = gated;
                    sum += gated * gated;
                }
                let scale = (sum / g.group_size as f32 + 1e-5).sqrt().recip();
                for index in 0..g.group_size {
                    output[base + index] *= scale * input.norm_weight[group * g.group_size + index];
                }
            }
        }
        (output, conv_state, ssm_state)
    }

    // Mirrors the CUDA grouping/indexing independently from the token-first
    // portable reference above. This runs on every platform.
    fn grouped_reference(input: &ReferenceInputs) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let g = input.geometry;
        let mut conv = vec![0.0; g.conv_elements];
        let mut next_conv = vec![0.0; g.conv_state_elements];
        for channel in 0..g.conv_dim {
            let mut history = (0..g.kernel_size)
                .map(|kernel| input.conv_state[channel * g.kernel_size + kernel])
                .collect::<Vec<_>>();
            for token in 0..g.seq_len {
                history.rotate_left(1);
                history[g.kernel_size - 1] = input.conv_input[token * g.conv_dim + channel];
                let value = history.iter().enumerate().fold(
                    input.conv_bias[channel],
                    |sum, (kernel, value)| {
                        sum + value * input.conv_weight[channel * g.kernel_size + kernel]
                    },
                );
                conv[token * g.conv_dim + channel] = silu(value);
            }
            next_conv[channel * g.kernel_size..(channel + 1) * g.kernel_size]
                .copy_from_slice(&history);
        }

        let mut raw_y = vec![0.0; g.output_elements];
        let mut next_ssm = input.ssm_state.clone();
        for output_index in 0..g.inner_dim {
            let head = output_index / g.head_dim;
            let dim = output_index % g.head_dim;
            let group = head / (g.n_heads / g.n_group);
            for token in 0..g.seq_len {
                let dt = softplus(input.dt_raw[token * g.n_heads + head] + input.dt_bias[head]);
                let decay = (dt * input.a[head]).exp();
                let x = conv[token * g.conv_dim + output_index];
                let mut y = 0.0;
                for state_index in 0..g.d_state {
                    let index = (head * g.d_state + state_index) * g.head_dim + dim;
                    let group_base = token * g.conv_dim + g.inner_dim + group * g.d_state;
                    let next = next_ssm[index] * decay + conv[group_base + state_index] * x * dt;
                    next_ssm[index] = next;
                    y += next * conv[group_base + g.n_group * g.d_state + state_index];
                }
                raw_y[token * g.inner_dim + output_index] = y + input.d[head] * x;
            }
        }

        let mut output = raw_y;
        for token in 0..g.seq_len {
            for group in 0..g.n_group {
                let base = token * g.inner_dim + group * g.group_size;
                let sum = (0..g.group_size).fold(0.0, |sum, index| {
                    let gated = silu(input.z[base + index]) * output[base + index];
                    output[base + index] = gated;
                    sum + gated * gated
                });
                let scale = (sum / g.group_size as f32 + 1e-5).sqrt().recip();
                for index in 0..g.group_size {
                    output[base + index] *= scale * input.norm_weight[group * g.group_size + index];
                }
            }
        }
        (output, next_conv, next_ssm)
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{label}[{index}] got {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn geometry_computes_disjoint_packed_regions() {
        let geometry = Mamba2Geometry::new(17, 2_048, 4, 16, 128, 64, 4).unwrap();
        assert_eq!(geometry.inner_dim, 1_024);
        assert_eq!(geometry.group_size, 256);
        assert_eq!(geometry.conv_elements, 34_816);
        assert_eq!(geometry.conv_state_elements, 8_192);
        assert_eq!(geometry.output_elements, 17_408);
        assert_eq!(geometry.ssm_state_elements, 131_072);
        assert_eq!(geometry.next_conv_offset(), geometry.output_elements);
        assert_eq!(
            geometry.next_ssm_offset(),
            geometry.output_elements + geometry.conv_state_elements
        );
        assert_eq!(
            geometry.packed_elements,
            geometry.output_elements + geometry.conv_state_elements + geometry.ssm_state_elements
        );
        assert_eq!(geometry.scan_blocks(), 256);
    }

    #[test]
    fn geometry_rejects_unsafe_or_inconsistent_shapes() {
        assert!(!supports(0, 20, 3, 4, 3, 2, 2));
        assert!(!supports(1, 20, 0, 4, 3, 2, 2));
        assert!(!supports(1, 20, MAX_CONV_KERNEL + 1, 4, 3, 2, 2));
        assert!(!supports(1, 20, 3, 4, 0, 2, 2));
        assert!(!supports(1, 20, 3, 4, MAX_D_STATE + 1, 2, 2));
        assert!(!supports(1, 20, 3, 3, 3, 2, 2));
        assert!(!supports(1, 19, 3, 4, 3, 2, 2));
        assert!(!supports(i32::MAX as usize, 3, 1, 1, 1, 1, 1));
        assert!(supports(1, 20, 3, 4, 3, 2, 2));
    }

    #[test]
    fn strided_row_span_enforces_signed_kernel_index_bounds() {
        let limit = i32::MAX as usize;
        assert_eq!(checked_row_span(2, 4, limit - 4).unwrap(), limit);
        assert!(checked_row_span(2, 4, limit - 3).is_err());
        assert!(checked_row_span(2, 4, 3).is_err());
        assert!(checked_row_span(1, 4, limit + 1).is_err());
        assert!(checked_row_span(0, 4, 4).is_err());
    }

    #[test]
    fn grouped_prompt_reference_matches_portable_recurrence() {
        let input = sample_inputs(4);
        let expected = portable_reference(&input);
        let actual = grouped_reference(&input);
        assert_close(&actual.0, &expected.0, 1e-6, "prompt output");
        assert_close(&actual.1, &expected.1, 1e-6, "prompt conv state");
        assert_close(&actual.2, &expected.2, 1e-6, "prompt recurrent state");
    }

    #[test]
    fn fused_decode_reference_matches_portable_recurrence() {
        let input = sample_inputs(1);
        let expected = portable_reference(&input);
        let actual = grouped_reference(&input);
        assert_close(&actual.0, &expected.0, 1e-6, "decode output");
        assert_close(&actual.1, &expected.1, 1e-6, "decode conv state");
        assert_close(&actual.2, &expected.2, 1e-6, "decode recurrent state");
    }
}
