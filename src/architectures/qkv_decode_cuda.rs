//! Fused CUDA epilogue for single-token Qwen2 attention projections.
//!
//! Qwen2 applies a bias to Q, K, and V, then split-half (NEOX) RoPE to Q and
//! K. The portable Candle path dispatches five elementwise kernels and
//! allocates five intermediate outputs. Grouped decode projections already
//! produce packed Q/K/V storage, so one epilogue launch can perform all five
//! operations and, when FlashAttention is active, fold the three following
//! f32-to-f16 casts into the same packed output.

use std::sync::OnceLock;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaView, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use candle_core::{CudaDevice, DType, Device, Layout, Storage, Tensor};

const CUDA_THREADS: usize = 256;
const CUDA_MODULE_NAME: &str = "dinf_qkv_decode_epilogue_v1";
const CUDA_F32_KERNEL_NAME: &str = "dinf_qkv_bias_rope_neox_f32";
const CUDA_F16_KERNEL_NAME: &str = "dinf_qkv_bias_rope_neox_f16";

const CUDA_SOURCE: &str = r#"
extern "C" __global__ void dinf_qkv_bias_rope_neox_f32(
    float *qkv,
    const float *q_bias,
    const float *k_bias,
    const float *v_bias,
    const float *cos,
    const float *sin,
    unsigned long long q_dim,
    unsigned long long k_dim,
    unsigned long long v_dim,
    unsigned int head_dim) {
    const unsigned long long index =
        (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long half_dim = head_dim / 2;
    const unsigned long long q_pairs = q_dim / 2;
    const unsigned long long k_pairs = k_dim / 2;

    if (index < q_pairs) {
        const unsigned long long head = index / half_dim;
        const unsigned long long lane = index - head * half_dim;
        const unsigned long long first = head * head_dim + lane;
        const unsigned long long second = first + half_dim;
        // Bias is a separate f32 kernel in Candle's portable path. Explicit
        // round-to-nearest additions prevent NVRTC from reassociating the bias
        // into the following RoPE multiplies.
        const float first_value = __fadd_rn(qkv[first], q_bias[first]);
        const float second_value = __fadd_rn(qkv[second], q_bias[second]);
        const float c = cos[lane];
        const float s = sin[lane];
        const float second_sin = __fmul_rn(second_value, s);
        const float second_cos = __fmul_rn(second_value, c);
        qkv[first] = __fmaf_rn(first_value, c, -second_sin);
        qkv[second] = __fmaf_rn(first_value, s, second_cos);
        return;
    }

    if (index < q_pairs + k_pairs) {
        const unsigned long long pair = index - q_pairs;
        const unsigned long long head = pair / half_dim;
        const unsigned long long lane = pair - head * half_dim;
        const unsigned long long first = q_dim + head * head_dim + lane;
        const unsigned long long second = first + half_dim;
        const unsigned long long bias_first = head * head_dim + lane;
        const unsigned long long bias_second = bias_first + half_dim;
        const float first_value = __fadd_rn(qkv[first], k_bias[bias_first]);
        const float second_value = __fadd_rn(qkv[second], k_bias[bias_second]);
        const float c = cos[lane];
        const float s = sin[lane];
        const float second_sin = __fmul_rn(second_value, s);
        const float second_cos = __fmul_rn(second_value, c);
        qkv[first] = __fmaf_rn(first_value, c, -second_sin);
        qkv[second] = __fmaf_rn(first_value, s, second_cos);
        return;
    }

    const unsigned long long value_index = index - q_pairs - k_pairs;
    if (value_index < v_dim) {
        const unsigned long long output_index = q_dim + k_dim + value_index;
        qkv[output_index] = __fadd_rn(qkv[output_index], v_bias[value_index]);
    }
}

__device__ __forceinline__ unsigned short dinf_f32_to_f16(float value) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
    return result;
}

extern "C" __global__ void dinf_qkv_bias_rope_neox_f16(
    const float *qkv,
    unsigned short *output,
    const float *q_bias,
    const float *k_bias,
    const float *v_bias,
    const float *cos,
    const float *sin,
    unsigned long long q_dim,
    unsigned long long k_dim,
    unsigned long long v_dim,
    unsigned int head_dim) {
    const unsigned long long index =
        (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long half_dim = head_dim / 2;
    const unsigned long long q_pairs = q_dim / 2;
    const unsigned long long k_pairs = k_dim / 2;

    if (index < q_pairs) {
        const unsigned long long head = index / half_dim;
        const unsigned long long lane = index - head * half_dim;
        const unsigned long long first = head * head_dim + lane;
        const unsigned long long second = first + half_dim;
        const float first_value = __fadd_rn(qkv[first], q_bias[first]);
        const float second_value = __fadd_rn(qkv[second], q_bias[second]);
        const float c = cos[lane];
        const float s = sin[lane];
        const float second_sin = __fmul_rn(second_value, s);
        const float second_cos = __fmul_rn(second_value, c);
        output[first] = dinf_f32_to_f16(
            __fmaf_rn(first_value, c, -second_sin));
        output[second] = dinf_f32_to_f16(
            __fmaf_rn(first_value, s, second_cos));
        return;
    }

    if (index < q_pairs + k_pairs) {
        const unsigned long long pair = index - q_pairs;
        const unsigned long long head = pair / half_dim;
        const unsigned long long lane = pair - head * half_dim;
        const unsigned long long first = q_dim + head * head_dim + lane;
        const unsigned long long second = first + half_dim;
        const unsigned long long bias_first = head * head_dim + lane;
        const unsigned long long bias_second = bias_first + half_dim;
        const float first_value = __fadd_rn(qkv[first], k_bias[bias_first]);
        const float second_value = __fadd_rn(qkv[second], k_bias[bias_second]);
        const float c = cos[lane];
        const float s = sin[lane];
        const float second_sin = __fmul_rn(second_value, s);
        const float second_cos = __fmul_rn(second_value, c);
        output[first] = dinf_f32_to_f16(
            __fmaf_rn(first_value, c, -second_sin));
        output[second] = dinf_f32_to_f16(
            __fmaf_rn(first_value, s, second_cos));
        return;
    }

    const unsigned long long value_index = index - q_pairs - k_pairs;
    if (value_index < v_dim) {
        const unsigned long long input_index = q_dim + k_dim + value_index;
        output[input_index] = dinf_f32_to_f16(
            __fadd_rn(qkv[input_index], v_bias[value_index]));
    }
}
"#;

static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
static FAST_PATH_LOGGED: OnceLock<()> = OnceLock::new();

fn ptx() -> candle_core::Result<&'static str> {
    match PTX.get_or_init(|| {
        compile_ptx_with_opts(
            CUDA_SOURCE,
            CompileOptions {
                name: Some("qkv_decode_epilogue.cu".to_string()),
                // Candle's binary and RoPE PTX is built without
                // `--use_fast_math`; match it so the fused path preserves the
                // same f32 rounding behavior.
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(error) => candle_core::bail!("failed to compile CUDA QKV decode epilogue: {error}"),
    }
}

pub(super) fn precompile(device: &Device) -> anyhow::Result<()> {
    if !device.is_cuda() || std::env::var_os("DINF_DISABLE_CUDA_QKV_EPILOGUE").is_some() {
        return Ok(());
    }
    let device = device.as_cuda_device()?;
    for kernel in [CUDA_F32_KERNEL_NAME, CUDA_F16_KERNEL_NAME] {
        device.get_or_load_custom_func(kernel, CUDA_MODULE_NAME, ptx()?)?;
    }
    Ok(())
}

pub(super) enum QkvDecodeOutput {
    F32(CudaSlice<f32>),
    F16(CudaSlice<half::f16>),
}

pub(super) struct QkvDecodeEpilogue {
    q_bias: Tensor,
    k_bias: Tensor,
    v_bias: Tensor,
    cos: Tensor,
    sin: Tensor,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
    head_dim: usize,
    rope_offset: usize,
    output_f16: bool,
}

impl QkvDecodeEpilogue {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_new(
        q_bias: &Tensor,
        k_bias: &Tensor,
        v_bias: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        start_pos: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        allow_f16_output: bool,
    ) -> anyhow::Result<Option<Self>> {
        if std::env::var_os("DINF_DISABLE_CUDA_QKV_EPILOGUE").is_some()
            || num_heads == 0
            || num_kv_heads == 0
            || head_dim == 0
            || !head_dim.is_multiple_of(2)
        {
            return Ok(None);
        }

        let q_dim = num_heads
            .checked_mul(head_dim)
            .ok_or_else(|| anyhow::anyhow!("QKV decode Q size overflow"))?;
        let k_dim = num_kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| anyhow::anyhow!("QKV decode K size overflow"))?;
        let v_dim = k_dim;
        let half_dim = head_dim / 2;
        let rope_offset = start_pos
            .checked_mul(half_dim)
            .ok_or_else(|| anyhow::anyhow!("QKV decode RoPE offset overflow"))?;
        let (cos_positions, cos_dim) = cos.dims2()?;
        let (sin_positions, sin_dim) = sin.dims2()?;
        if q_bias.dtype() != DType::F32
            || k_bias.dtype() != DType::F32
            || v_bias.dtype() != DType::F32
            || cos.dtype() != DType::F32
            || sin.dtype() != DType::F32
            || q_bias.elem_count() != q_dim
            || k_bias.elem_count() != k_dim
            || v_bias.elem_count() != v_dim
            || cos_dim != half_dim
            || sin_dim != half_dim
            || start_pos >= cos_positions
            || start_pos >= sin_positions
            || !q_bias.is_contiguous()
            || !k_bias.is_contiguous()
            || !v_bias.is_contiguous()
            || !cos.is_contiguous()
            || !sin.is_contiguous()
            || !q_bias.device().is_cuda()
            || !q_bias.device().same_device(k_bias.device())
            || !q_bias.device().same_device(v_bias.device())
            || !q_bias.device().same_device(cos.device())
            || !q_bias.device().same_device(sin.device())
        {
            return Ok(None);
        }

        let launch_elements = q_dim / 2 + k_dim / 2 + v_dim;
        if launch_elements.div_ceil(CUDA_THREADS) > u32::MAX as usize
            || head_dim > u32::MAX as usize
        {
            return Ok(None);
        }

        let output_f16 = allow_f16_output
            && std::env::var_os("DINF_DISABLE_CUDA_QKV_F16_EPILOGUE").is_none()
            && std::env::var_os("DINF_DISABLE_CUDA_FLASH_ATTN").is_none()
            && head_dim <= 512
            && head_dim.is_multiple_of(8)
            && num_heads.is_multiple_of(num_kv_heads)
            && match q_bias.device() {
                Device::Cuda(device) => device
                    .cuda_stream()
                    .context()
                    .compute_capability()
                    .is_ok_and(|(major, _)| major >= 8),
                _ => false,
            };

        Ok(Some(Self {
            q_bias: q_bias.clone(),
            k_bias: k_bias.clone(),
            v_bias: v_bias.clone(),
            cos: cos.clone(),
            sin: sin.clone(),
            q_dim,
            k_dim,
            v_dim,
            head_dim,
            rope_offset,
            output_f16,
        }))
    }

    pub(super) fn output_dims(&self) -> [usize; 3] {
        [self.q_dim, self.k_dim, self.v_dim]
    }

    pub(super) fn launch(
        &self,
        mut output: CudaSlice<f32>,
        device: &CudaDevice,
    ) -> candle_core::Result<QkvDecodeOutput> {
        let total_dim = self
            .q_dim
            .checked_add(self.k_dim)
            .and_then(|value| value.checked_add(self.v_dim))
            .ok_or_else(|| candle_core::Error::Msg("QKV decode output size overflow".into()))?;
        if output.len() != total_dim {
            candle_core::bail!(
                "CUDA QKV decode epilogue expected {total_dim} output elements, got {}",
                output.len()
            )
        }

        let (q_bias_guard, q_bias_layout) = self.q_bias.storage_and_layout();
        let (k_bias_guard, k_bias_layout) = self.k_bias.storage_and_layout();
        let (v_bias_guard, v_bias_layout) = self.v_bias.storage_and_layout();
        let (cos_guard, cos_layout) = self.cos.storage_and_layout();
        let (sin_guard, sin_layout) = self.sin.storage_and_layout();
        let q_bias = captured_view(
            "Q bias",
            &q_bias_guard,
            q_bias_layout,
            q_bias_layout.start_offset(),
            self.q_dim,
            device,
        )?;
        let k_bias = captured_view(
            "K bias",
            &k_bias_guard,
            k_bias_layout,
            k_bias_layout.start_offset(),
            self.k_dim,
            device,
        )?;
        let v_bias = captured_view(
            "V bias",
            &v_bias_guard,
            v_bias_layout,
            v_bias_layout.start_offset(),
            self.v_dim,
            device,
        )?;
        let cos_start = cos_layout
            .start_offset()
            .checked_add(self.rope_offset)
            .ok_or_else(|| candle_core::Error::Msg("RoPE cosine offset overflow".into()))?;
        let sin_start = sin_layout
            .start_offset()
            .checked_add(self.rope_offset)
            .ok_or_else(|| candle_core::Error::Msg("RoPE sine offset overflow".into()))?;
        let cos = captured_view(
            "RoPE cosine",
            &cos_guard,
            cos_layout,
            cos_start,
            self.head_dim / 2,
            device,
        )?;
        let sin = captured_view(
            "RoPE sine",
            &sin_guard,
            sin_layout,
            sin_start,
            self.head_dim / 2,
            device,
        )?;

        let launch_elements = self.q_dim / 2 + self.k_dim / 2 + self.v_dim;
        let config = LaunchConfig {
            grid_dim: (launch_elements.div_ceil(CUDA_THREADS) as u32, 1, 1),
            block_dim: (CUDA_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let q_dim = self.q_dim as u64;
        let k_dim = self.k_dim as u64;
        let v_dim = self.v_dim as u64;
        let head_dim = self.head_dim as u32;
        let result = if self.output_f16 {
            // SAFETY: The kernel writes every packed f16 output element.
            let mut converted = unsafe { device.alloc::<half::f16>(total_dim)? };
            let function =
                device.get_or_load_custom_func(CUDA_F16_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
            let mut builder = function.builder();
            builder
                .arg(&output)
                .arg(&mut converted)
                .arg(&q_bias)
                .arg(&k_bias)
                .arg(&v_bias)
                .arg(&cos)
                .arg(&sin)
                .arg(&q_dim)
                .arg(&k_dim)
                .arg(&v_dim)
                .arg(&head_dim);
            // SAFETY: All buffers, views, dimensions, and device identities were checked above.
            unsafe { builder.launch(config) }.w()?;
            QkvDecodeOutput::F16(converted)
        } else {
            let function =
                device.get_or_load_custom_func(CUDA_F32_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
            let mut builder = function.builder();
            builder
                .arg(&mut output)
                .arg(&q_bias)
                .arg(&k_bias)
                .arg(&v_bias)
                .arg(&cos)
                .arg(&sin)
                .arg(&q_dim)
                .arg(&k_dim)
                .arg(&v_dim)
                .arg(&head_dim);
            // SAFETY: All buffers, views, dimensions, and device identities were checked above.
            unsafe { builder.launch(config) }.w()?;
            QkvDecodeOutput::F32(output)
        };
        FAST_PATH_LOGGED.get_or_init(|| {
            tracing::info!("CUDA fused QKV bias + RoPE/cast decode epilogue enabled");
        });
        Ok(result)
    }
}

fn captured_view<'a>(
    name: &str,
    storage: &'a Storage,
    layout: &Layout,
    start: usize,
    elements: usize,
    expected_device: &CudaDevice,
) -> candle_core::Result<CudaView<'a, f32>> {
    let storage = match storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{name} is not on CUDA"),
    };
    if storage.dtype() != DType::F32 || !layout.is_contiguous() {
        candle_core::bail!("{name} must be contiguous f32")
    }
    if storage.device().id() != expected_device.id() {
        candle_core::bail!("{name} is on a different CUDA device")
    }
    let slice = storage.as_cuda_slice::<f32>()?;
    let end = start
        .checked_add(elements)
        .ok_or_else(|| candle_core::Error::Msg(format!("{name} CUDA view overflow")))?;
    if start < layout.start_offset() || end > slice.len() {
        candle_core::bail!("{name} CUDA view exceeds its storage")
    }
    Ok(slice.slice(start..end))
}
