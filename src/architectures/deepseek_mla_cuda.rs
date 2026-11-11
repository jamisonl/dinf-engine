use std::sync::OnceLock;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use candle_core::{CpuStorage, CudaStorage, DType, Layout, Shape, Storage, Tensor};

const CUDA_MODULE_NAME: &str = "dinf_deepseek_mla_layout_v1";
const CUDA_KERNEL_NAME: &str = "dinf_deepseek_mla_layout_rope_f32";
const CUDA_THREADS: usize = 256;

const CUDA_SOURCE: &str = r#"
extern "C" __global__ void dinf_deepseek_mla_layout_rope_f32(
    const float *q_input,
    const float *kv_input,
    const float *kv_mixed,
    const float *cos,
    const float *sin,
    float *output,
    int batch,
    int seq_len,
    int num_heads,
    int nope_dim,
    int rope_dim,
    int value_dim,
    int kv_rank,
    int start_pos,
    unsigned long long elements) {
    const unsigned long long gid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= elements) {
        return;
    }

    const unsigned long long qk_dim = nope_dim + rope_dim;
    const unsigned long long qk_elements =
        (unsigned long long)batch * num_heads * seq_len * qk_dim;
    unsigned long long local = gid;
    int output_kind = 0;
    unsigned long long head_dim = qk_dim;
    if (local >= qk_elements) {
        local -= qk_elements;
        output_kind = 1;
        if (local >= qk_elements) {
            local -= qk_elements;
            output_kind = 2;
            head_dim = value_dim;
        }
    }

    const unsigned long long column = local % head_dim;
    unsigned long long index = local / head_dim;
    const unsigned long long token = index % seq_len;
    index /= seq_len;
    const unsigned long long head = index % num_heads;
    const unsigned long long lane = index / num_heads;
    const unsigned long long token_row = lane * seq_len + token;

    if (output_kind == 2) {
        const unsigned long long kv_head_dim = nope_dim + value_dim;
        const unsigned long long source = token_row * num_heads * kv_head_dim
            + head * kv_head_dim + nope_dim + column;
        output[gid] = kv_input[source];
        return;
    }

    if (column < (unsigned long long)nope_dim) {
        if (output_kind == 0) {
            const unsigned long long source = token_row * num_heads * qk_dim
                + head * qk_dim + column;
            output[gid] = q_input[source];
        } else {
            const unsigned long long kv_head_dim = nope_dim + value_dim;
            const unsigned long long source = token_row * num_heads * kv_head_dim
                + head * kv_head_dim + column;
            output[gid] = kv_input[source];
        }
        return;
    }

    const unsigned long long rotary_column = column - nope_dim;
    const unsigned long long pair = rotary_column / 2;
    const unsigned long long first_column = pair * 2;
    const unsigned long long source_base = output_kind == 0
        ? token_row * num_heads * qk_dim + head * qk_dim + nope_dim
        : token_row * (kv_rank + rope_dim) + kv_rank;
    const float *source = output_kind == 0 ? q_input : kv_mixed;
    const float first = source[source_base + first_column];
    const float second = source[source_base + first_column + 1];
    const unsigned long long rope_index =
        (unsigned long long)(start_pos + token) * (rope_dim / 2) + pair;
    const float c = cos[rope_index];
    const float s = sin[rope_index];
    output[gid] = (rotary_column & 1) == 0
        ? first * c - second * s
        : first * s + second * c;
}
"#;

static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
static FAST_PATH_LOGGED: OnceLock<()> = OnceLock::new();

fn ptx() -> candle_core::Result<&'static str> {
    match PTX.get_or_init(|| {
        compile_ptx_with_opts(
            CUDA_SOURCE,
            CompileOptions {
                name: Some("deepseek_mla_layout.cu".to_string()),
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(error) => candle_core::bail!("failed to compile DeepSeek MLA CUDA layout: {error}"),
    }
}

pub(super) fn precompile(device: &candle_core::Device) -> anyhow::Result<()> {
    if !device.is_cuda() || std::env::var_os("DINF_DISABLE_DEEPSEEK_FUSED_MLA_CUDA").is_some() {
        return Ok(());
    }
    device
        .as_cuda_device()?
        .get_or_load_custom_func(CUDA_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
    Ok(())
}

struct DeepseekMlaLayout<'a> {
    cos: &'a Tensor,
    sin: &'a Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    value_dim: usize,
    kv_rank: usize,
    start_pos: usize,
}

fn checked_view<'a>(
    storage: &'a CudaStorage,
    layout: &Layout,
    elements: usize,
    name: &str,
) -> candle_core::Result<CudaView<'a, f32>> {
    if storage.dtype() != DType::F32 || !layout.is_contiguous() {
        candle_core::bail!("DeepSeek MLA {name} must be contiguous f32")
    }
    let slice = storage.as_cuda_slice::<f32>()?;
    let start = layout.start_offset();
    let end = start
        .checked_add(elements)
        .ok_or_else(|| candle_core::Error::Msg(format!("DeepSeek MLA {name} view overflow")))?;
    if end > slice.len() {
        candle_core::bail!("DeepSeek MLA {name} view exceeds its storage")
    }
    Ok(slice.slice(start..end))
}

impl candle_core::CustomOp3 for DeepseekMlaLayout<'_> {
    fn name(&self) -> &'static str {
        "dinf-cuda-deepseek-mla-layout"
    }

    fn cpu_fwd(
        &self,
        _q_storage: &CpuStorage,
        _q_layout: &Layout,
        _kv_storage: &CpuStorage,
        _kv_layout: &Layout,
        _mixed_storage: &CpuStorage,
        _mixed_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused DeepSeek MLA layout is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        q_storage: &CudaStorage,
        q_layout: &Layout,
        kv_storage: &CudaStorage,
        kv_layout: &Layout,
        mixed_storage: &CudaStorage,
        mixed_layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        if self.rope_dim == 0 || !self.rope_dim.is_multiple_of(2) {
            candle_core::bail!("fused DeepSeek MLA CUDA RoPE dimension is invalid")
        }
        let qk_dim = self.nope_dim + self.rope_dim;
        let q_elements = self.batch * self.seq_len * self.num_heads * qk_dim;
        let kv_head_dim = self.nope_dim + self.value_dim;
        let kv_elements = self.batch * self.seq_len * self.num_heads * kv_head_dim;
        let mixed_elements = self.batch * self.seq_len * (self.kv_rank + self.rope_dim);
        if q_layout.shape().dims() != [self.batch, self.seq_len, self.num_heads * qk_dim]
            || kv_layout.shape().dims() != [self.batch, self.seq_len, self.num_heads * kv_head_dim]
            || mixed_layout.shape().dims()
                != [self.batch, self.seq_len, self.kv_rank + self.rope_dim]
            || q_storage.device().id() != kv_storage.device().id()
            || q_storage.device().id() != mixed_storage.device().id()
        {
            candle_core::bail!("fused DeepSeek MLA CUDA input geometry differs")
        }
        let q = checked_view(q_storage, q_layout, q_elements, "query")?;
        let kv = checked_view(kv_storage, kv_layout, kv_elements, "expanded KV")?;
        let mixed = checked_view(mixed_storage, mixed_layout, mixed_elements, "compressed KV")?;

        let (cos_guard, cos_layout) = self.cos.storage_and_layout();
        let (sin_guard, sin_layout) = self.sin.storage_and_layout();
        let Storage::Cuda(cos_storage) = &*cos_guard else {
            candle_core::bail!("DeepSeek MLA cosine table is not on CUDA")
        };
        let Storage::Cuda(sin_storage) = &*sin_guard else {
            candle_core::bail!("DeepSeek MLA sine table is not on CUDA")
        };
        let (context_len, half_dim) = cos_layout.shape().dims2()?;
        if cos_layout.shape() != sin_layout.shape()
            || half_dim * 2 != self.rope_dim
            || self.start_pos.saturating_add(self.seq_len) > context_len
            || q_storage.device().id() != cos_storage.device().id()
            || q_storage.device().id() != sin_storage.device().id()
        {
            candle_core::bail!("DeepSeek MLA CUDA RoPE table geometry differs")
        }
        let table_elements = context_len * half_dim;
        let cos = checked_view(cos_storage, cos_layout, table_elements, "cosine table")?;
        let sin = checked_view(sin_storage, sin_layout, table_elements, "sine table")?;

        let qk_elements = self.batch * self.num_heads * self.seq_len * qk_dim;
        let value_elements = self.batch * self.num_heads * self.seq_len * self.value_dim;
        let elements = qk_elements
            .checked_mul(2)
            .and_then(|value| value.checked_add(value_elements))
            .ok_or_else(|| candle_core::Error::Msg("DeepSeek MLA output overflow".into()))?;
        if elements == 0
            || elements.div_ceil(CUDA_THREADS) > u32::MAX as usize
            || [
                self.batch,
                self.seq_len,
                self.num_heads,
                self.nope_dim,
                self.rope_dim,
                self.value_dim,
                self.kv_rank,
                self.start_pos,
            ]
            .into_iter()
            .any(|value| value > i32::MAX as usize)
        {
            candle_core::bail!("fused DeepSeek MLA CUDA geometry is unsupported")
        }
        let device = q_storage.device();
        let mut output = unsafe { device.alloc::<f32>(elements)? };
        let function =
            device.get_or_load_custom_func(CUDA_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        let config = LaunchConfig {
            grid_dim: (elements.div_ceil(CUDA_THREADS) as u32, 1, 1),
            block_dim: (CUDA_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let batch = self.batch as i32;
        let seq_len = self.seq_len as i32;
        let num_heads = self.num_heads as i32;
        let nope_dim = self.nope_dim as i32;
        let rope_dim = self.rope_dim as i32;
        let value_dim = self.value_dim as i32;
        let kv_rank = self.kv_rank as i32;
        let start_pos = self.start_pos as i32;
        let elements = elements as u64;
        let mut builder = function.builder();
        builder
            .arg(&q)
            .arg(&kv)
            .arg(&mixed)
            .arg(&cos)
            .arg(&sin)
            .arg(&mut output)
            .arg(&batch)
            .arg(&seq_len)
            .arg(&num_heads)
            .arg(&nope_dim)
            .arg(&rope_dim)
            .arg(&value_dim)
            .arg(&kv_rank)
            .arg(&start_pos)
            .arg(&elements);
        // SAFETY: Every input view, output bound, device, and index limit was
        // validated above and all allocations remain live through the launch.
        unsafe { builder.launch(config) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from(elements as usize),
        ))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_fused_mla_layout(
    q: &Tensor,
    kv: &Tensor,
    kv_mixed: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    value_dim: usize,
    kv_rank: usize,
    start_pos: usize,
) -> candle_core::Result<Option<(Tensor, Tensor, Tensor)>> {
    if std::env::var_os("DINF_DISABLE_DEEPSEEK_FUSED_MLA_CUDA").is_some()
        || !q.device().is_cuda()
        || q.dtype() != DType::F32
        || kv.dtype() != DType::F32
        || kv_mixed.dtype() != DType::F32
        || !q.is_contiguous()
        || !kv.is_contiguous()
        || !kv_mixed.is_contiguous()
    {
        return Ok(None);
    }
    let packed = q.apply_op3_no_bwd(
        kv,
        kv_mixed,
        &DeepseekMlaLayout {
            cos,
            sin,
            batch,
            seq_len,
            num_heads,
            nope_dim,
            rope_dim,
            value_dim,
            kv_rank,
            start_pos,
        },
    )?;
    let qk_dim = nope_dim + rope_dim;
    let qk_elements = batch * num_heads * seq_len * qk_dim;
    let value_elements = batch * num_heads * seq_len * value_dim;
    let q = packed
        .narrow(0, 0, qk_elements)?
        .reshape((batch, num_heads, seq_len, qk_dim))?;
    let k = packed
        .narrow(0, qk_elements, qk_elements)?
        .reshape((batch, num_heads, seq_len, qk_dim))?;
    let v = packed
        .narrow(0, qk_elements * 2, value_elements)?
        .reshape((batch, num_heads, seq_len, value_dim))?;
    FAST_PATH_LOGGED.get_or_init(|| {
        tracing::info!("CUDA fused DeepSeek MLA layout and RoPE enabled");
    });
    Ok(Some((q, k, v)))
}

#[cfg(test)]
mod tests {
    use candle_core::{D, Device, Tensor};

    use super::try_fused_mla_layout;

    #[test]
    fn fused_mla_layout_matches_portable_assembly() -> candle_core::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let (batch, seq_len, heads) = (2, 7, 4);
        let (nope_dim, rope_dim, value_dim, kv_rank) = (6, 4, 5, 8);
        let start_pos = 2;
        let qk_dim = nope_dim + rope_dim;
        let kv_head_dim = nope_dim + value_dim;
        let q = Tensor::arange(
            -200f32,
            (batch * seq_len * heads * qk_dim) as f32 - 200.0,
            &device,
        )?
        .affine(0.01, 0.0)?
        .reshape((batch, seq_len, heads * qk_dim))?;
        let kv = Tensor::arange(
            -100f32,
            (batch * seq_len * heads * kv_head_dim) as f32 - 100.0,
            &device,
        )?
        .affine(0.02, 0.0)?
        .reshape((batch, seq_len, heads * kv_head_dim))?;
        let mixed = Tensor::arange(
            -50f32,
            (batch * seq_len * (kv_rank + rope_dim)) as f32 - 50.0,
            &device,
        )?
        .affine(0.015, 0.0)?
        .reshape((batch, seq_len, kv_rank + rope_dim))?;
        let table_len = start_pos + seq_len + 1;
        let angles = Tensor::arange(0f32, (table_len * rope_dim / 2) as f32, &device)?
            .affine(0.03125, 0.0)?
            .reshape((table_len, rope_dim / 2))?;
        let cos = angles.cos()?;
        let sin = angles.sin()?;

        let q_heads = q
            .reshape((batch, seq_len, heads, qk_dim))?
            .transpose(1, 2)?;
        let q_nope = q_heads.narrow(D::Minus1, 0, nope_dim)?;
        let q_pe = q_heads.narrow(D::Minus1, nope_dim, rope_dim)?;
        let q_pe = candle_nn::rotary_emb::rope_i(
            &q_pe.contiguous()?,
            &cos.narrow(0, start_pos, seq_len)?,
            &sin.narrow(0, start_pos, seq_len)?,
        )?;
        let expected_q = Tensor::cat(&[q_nope, q_pe], D::Minus1)?;
        let kv_heads = kv
            .reshape((batch, seq_len, heads, kv_head_dim))?
            .transpose(1, 2)?;
        let k_nope = kv_heads.narrow(D::Minus1, 0, nope_dim)?;
        let expected_v = kv_heads.narrow(D::Minus1, nope_dim, value_dim)?;
        let k_pe = mixed
            .narrow(D::Minus1, kv_rank, rope_dim)?
            .reshape((batch, seq_len, 1, rope_dim))?
            .transpose(1, 2)?;
        let k_pe = candle_nn::rotary_emb::rope_i(
            &k_pe.contiguous()?,
            &cos.narrow(0, start_pos, seq_len)?,
            &sin.narrow(0, start_pos, seq_len)?,
        )?;
        let expected_k = Tensor::cat(&[k_nope, k_pe.repeat((1, heads, 1, 1))?], D::Minus1)?;
        let (actual_q, actual_k, actual_v) = try_fused_mla_layout(
            &q, &kv, &mixed, &cos, &sin, batch, seq_len, heads, nope_dim, rope_dim, value_dim,
            kv_rank, start_pos,
        )?
        .expect("eligible CUDA MLA layout");
        for (actual, expected) in [
            (actual_q, expected_q),
            (actual_k, expected_k),
            (actual_v, expected_v),
        ] {
            let actual = actual
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected = expected
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max_diff = actual
                .iter()
                .zip(expected.iter())
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(max_diff <= 2e-6, "MLA layout drifted by {max_diff}");
        }
        Ok(())
    }
}
