//! Expert banks in GGUF formats Candle lacks, kept in their GGUF bytes.
//!
//! - NVFP4 (GGML type 40): 64 weights in 36 bytes, four UE4M3 scales, one per
//!   16 weights, then 32 bytes of E2M1 pairs. Byte `j` of a 16-weight group
//!   holds weight `j` in its low nibble and weight `j + 8` in its high one
//!   (`dequantize_row_nvfp4` in ggml).
//! - IQ4_NL (type 20): Q4_0's 18-byte block of 32 weights, an F16 scale and
//!   16 bytes of nibbles (low nibbles first), through a nonlinear value table
//!   (`dequantize_row_iq4_nl`).
//!
//! - MXFP8 (the official checkpoint's last four layers): E4M3 weights with an
//!   E8M0 scale per 32, stored in the GGUF as an `F8_E4M3` tensor and an
//!   `I8` `.mx_scale` tensor. In memory each 32 weights become a 36-byte
//!   block: the scale byte, three zero bytes, then the 32 weight bytes.
//!
//! A bank may also carry one F32 scale per expert, which multiplies that
//! expert's output (`ffn_*_exps.scale`, which comes with NVFP4 banks).
//!
//! CUDA runs banks through the indexed expert kernels in `indexed_moe_cuda`,
//! Metal through `raw_experts_metal`. Elsewhere each expert a call uses is
//! dequantized on the host for that call: the reference for the kernels, and
//! a slow path for devices without kernels for the format.

use anyhow::{Context, Result};
use candle_core::{CpuStorage, DType, Device, Storage, Tensor};

use crate::deepseek4_gguf::{DeepSeek4DType, RawTensor};

/// NVFP4 weights per block.
const BLOCK: usize = 64;
/// NVFP4 bytes per block.
const BLOCK_BYTES: usize = 36;
/// NVFP4 weights per scale.
const GROUP: usize = 16;
/// E2M1 values, doubled as in ggml's `kvalues_fp4`; [`ue4m3`] halves the scale.
const FP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
/// ggml's `kvalues_iq4nl`.
const IQ4_NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// A raw expert format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RawFormat {
    Nvfp4,
    Iq4Nl,
    Mxfp8,
}

impl RawFormat {
    pub(crate) fn of(dtype: DeepSeek4DType) -> Option<Self> {
        match dtype {
            DeepSeek4DType::NVFP4 => Some(Self::Nvfp4),
            DeepSeek4DType::IQ4NL => Some(Self::Iq4Nl),
            _ => None,
        }
    }

    #[cfg(test)]
    fn dtype(self) -> DeepSeek4DType {
        match self {
            Self::Nvfp4 => DeepSeek4DType::NVFP4,
            Self::Iq4Nl => DeepSeek4DType::IQ4NL,
            Self::Mxfp8 => DeepSeek4DType::F8E4M3,
        }
    }

    /// `(weights, bytes)` per block, as held in memory.
    fn block(self) -> (usize, usize) {
        match self {
            Self::Nvfp4 => (BLOCK, BLOCK_BYTES),
            Self::Iq4Nl => (32, 18),
            Self::Mxfp8 => (32, MXFP8_BLOCK_BYTES),
        }
    }

    fn dequantize(self, bytes: &[u8], out: &mut [f32]) -> Result<()> {
        match self {
            Self::Nvfp4 => dequantize(bytes, out),
            Self::Iq4Nl => dequantize_iq4_nl(bytes, out),
            Self::Mxfp8 => dequantize_mxfp8(bytes, out),
        }
    }

    #[cfg(feature = "cuda")]
    fn kernel_format(self) -> super::indexed_moe_cuda::RawExpertFormat {
        match self {
            Self::Nvfp4 => super::indexed_moe_cuda::RawExpertFormat::Nvfp4,
            Self::Iq4Nl => super::indexed_moe_cuda::RawExpertFormat::Iq4Nl,
            Self::Mxfp8 => super::indexed_moe_cuda::RawExpertFormat::Mxfp8,
        }
    }
}

/// Bytes of one in-memory MXFP8 block of 32 weights: scale, padding, weights.
const MXFP8_BLOCK_BYTES: usize = 36;

/// An E4M3 (FN) byte. 0x7f and 0xff are NaN, which the loader rejects.
pub(crate) fn e4m3(x: u8) -> f32 {
    let exponent = i32::from((x >> 3) & 15);
    let mantissa = f32::from(x & 7);
    let magnitude = if exponent == 0 {
        mantissa * 2f32.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    };
    if x & 0x80 != 0 { -magnitude } else { magnitude }
}

/// A UE8M0 scale byte: `2^(x - 127)`. 255 is NaN, which the loader rejects.
pub(crate) fn e8m0(x: u8) -> f32 {
    2f32.powi(i32::from(x) - 127)
}

/// Dequantize in-memory MXFP8 blocks into `out`.
pub(crate) fn dequantize_mxfp8(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    anyhow::ensure!(
        out.len().is_multiple_of(32) && bytes.len() == out.len() / 32 * MXFP8_BLOCK_BYTES,
        "MXFP8 dequantization of {} bytes into {} values",
        bytes.len(),
        out.len()
    );
    for (block, values) in bytes
        .chunks_exact(MXFP8_BLOCK_BYTES)
        .zip(out.chunks_exact_mut(32))
    {
        let scale = e8m0(block[0]);
        for (value, &byte) in values.iter_mut().zip(&block[4..]) {
            *value = e4m3(byte) * scale;
        }
    }
    Ok(())
}

/// The GGUF's MXFP8 weight bytes `[rows, cols]` and scale bytes `[rows, cols /
/// 32]` as in-memory blocks, rejecting NaN codes.
fn mxfp8_blocks(values: &[u8], scales: &[u8], cols: usize) -> Result<Vec<u8>> {
    anyhow::ensure!(
        cols.is_multiple_of(32)
            && values.len() == scales.len() * 32
            && values.len().is_multiple_of(cols),
        "MXFP8 values and scales differ in shape"
    );
    anyhow::ensure!(
        values.iter().all(|&x| x & 0x7f != 0x7f) && scales.iter().all(|&x| x != 0xff),
        "MXFP8 weights hold NaN codes"
    );
    let mut blocks = vec![0u8; scales.len() * MXFP8_BLOCK_BYTES];
    for ((block, &scale), weights) in blocks
        .chunks_exact_mut(MXFP8_BLOCK_BYTES)
        .zip(scales)
        .zip(values.chunks_exact(32))
    {
        block[0] = scale;
        block[4..].copy_from_slice(weights);
    }
    Ok(blocks)
}

fn host_bytes(tensor: &Tensor) -> Result<Vec<u8>> {
    Ok(tensor
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<u8>()?)
}

/// ggml's `ggml_ue4m3_to_fp32`: an unsigned E4M3 scale, halved. 0 and 0x7f
/// read as zero and the sign bit is ignored.
pub(crate) fn ue4m3(x: u8) -> f32 {
    if x == 0 || x == 0x7f {
        return 0.0;
    }
    let exponent = i32::from((x >> 3) & 15);
    let mantissa = f32::from(x & 7);
    let raw = if exponent == 0 {
        mantissa * 2f32.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    };
    raw * 0.5
}

/// Dequantize whole IQ4_NL blocks into `out`, as `dequantize_row_iq4_nl` does.
pub(crate) fn dequantize_iq4_nl(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    anyhow::ensure!(
        out.len().is_multiple_of(32) && bytes.len() == out.len() / 32 * 18,
        "IQ4_NL dequantization of {} bytes into {} values",
        bytes.len(),
        out.len()
    );
    for (block, values) in bytes.chunks_exact(18).zip(out.chunks_exact_mut(32)) {
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        for (j, &quant) in block[2..].iter().enumerate() {
            values[j] = scale * f32::from(IQ4_NL_VALUES[usize::from(quant & 15)]);
            values[j + 16] = scale * f32::from(IQ4_NL_VALUES[usize::from(quant >> 4)]);
        }
    }
    Ok(())
}

/// Dequantize whole NVFP4 blocks into `out`, as `dequantize_row_nvfp4` does.
pub(crate) fn dequantize(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    anyhow::ensure!(
        out.len().is_multiple_of(BLOCK) && bytes.len() == out.len() / BLOCK * BLOCK_BYTES,
        "NVFP4 dequantization of {} bytes into {} values",
        bytes.len(),
        out.len()
    );
    for (block, values) in bytes
        .chunks_exact(BLOCK_BYTES)
        .zip(out.chunks_exact_mut(BLOCK))
    {
        for group in 0..BLOCK / GROUP {
            let scale = ue4m3(block[group]);
            let quants = &block[4 + group * GROUP / 2..4 + (group + 1) * GROUP / 2];
            let values = &mut values[group * GROUP..(group + 1) * GROUP];
            for (j, &quant) in quants.iter().enumerate() {
                values[j] = f32::from(FP4_VALUES[usize::from(quant & 15)]) * scale;
                values[j + GROUP / 2] = f32::from(FP4_VALUES[usize::from(quant >> 4)]) * scale;
            }
        }
    }
    Ok(())
}

/// `[experts, out_dim, in_dim]` raw weights with optional per-expert output
/// scales.
pub(crate) struct RawExperts {
    format: RawFormat,
    /// GGUF bytes: on the device for CUDA and Metal, in host memory
    /// otherwise.
    bytes: Tensor,
    /// `[experts]` F32 on the bytes' device.
    #[cfg_attr(not(any(feature = "cuda", target_os = "macos")), allow(dead_code))]
    scales: Option<Tensor>,
    host_scales: Option<Vec<f32>>,
    experts: usize,
    out_dim: usize,
    in_dim: usize,
}

impl RawExperts {
    pub(crate) fn new(
        raw: &RawTensor,
        scales: Option<Vec<f32>>,
        experts: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self> {
        let format = RawFormat::of(raw.dtype())
            .with_context(|| format!("expert type {:?} is not supported", raw.dtype()))?;
        let (block, block_bytes) = format.block();
        anyhow::ensure!(
            raw.shape() == [experts, out_dim, in_dim] && in_dim.is_multiple_of(block),
            "{format:?} expert shape {:?} differs from [{experts}, {out_dim}, {in_dim}]",
            raw.shape()
        );
        let bytes = raw.bytes().clone();
        let expected = experts
            .checked_mul(out_dim)
            .and_then(|rows| rows.checked_mul(in_dim / block * block_bytes))
            .context("expert bank size overflows")?;
        anyhow::ensure!(
            bytes.dtype() == DType::U8 && bytes.dims() == [expected],
            "{format:?} expert storage holds {:?}, expected {expected} bytes",
            bytes.dims()
        );
        Self::from_parts(format, bytes, scales, experts, out_dim, in_dim)
    }

    /// An MXFP8 bank from the GGUF's `F8_E4M3` weights `[experts, out_dim,
    /// in_dim]` and UE8M0 scales `[experts, out_dim, in_dim / 32]`, held on
    /// `device` for CUDA and Metal, in host memory otherwise.
    pub(crate) fn mxfp8(
        values: &RawTensor,
        block_scales: &RawTensor,
        experts: usize,
        out_dim: usize,
        in_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        anyhow::ensure!(
            values.dtype() == DeepSeek4DType::F8E4M3
                && values.shape() == [experts, out_dim, in_dim]
                && block_scales.shape() == [experts, out_dim, in_dim / 32]
                && in_dim.is_multiple_of(32),
            "MXFP8 expert shapes {:?} / {:?} differ from [{experts}, {out_dim}, {in_dim}]",
            values.shape(),
            block_scales.shape()
        );
        // The Metal loader leaves values and scales on the GPU, which builds
        // the blocks itself.
        #[cfg(target_os = "macos")]
        if values.bytes().device().is_metal() && block_scales.bytes().device().is_metal() {
            let bytes =
                super::raw_experts_metal::mxfp8_blocks(values.bytes(), block_scales.bytes())?;
            return Self::from_parts(RawFormat::Mxfp8, bytes, None, experts, out_dim, in_dim);
        }
        let blocks = mxfp8_blocks(
            &host_bytes(values.bytes())?,
            &host_bytes(block_scales.bytes())?,
            in_dim,
        )?;
        let len = blocks.len();
        let place = if device.is_cuda() || device.is_metal() {
            device
        } else {
            &Device::Cpu
        };
        let bytes = Tensor::from_vec(blocks, len, place)?;
        Self::from_parts(RawFormat::Mxfp8, bytes, None, experts, out_dim, in_dim)
    }

    fn from_parts(
        format: RawFormat,
        bytes: Tensor,
        scales: Option<Vec<f32>>,
        experts: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self> {
        if let Some(scales) = &scales {
            anyhow::ensure!(
                scales.len() == experts && scales.iter().all(|scale| scale.is_finite()),
                "expert scales must be {experts} finite values"
            );
        }
        let device_scales = scales
            .as_ref()
            .map(|scales| Tensor::from_slice(scales, experts, bytes.device()))
            .transpose()?;
        #[cfg(feature = "cuda")]
        if bytes.device().is_cuda() {
            super::indexed_moe_cuda::precompile_raw(format.kernel_format(), bytes.device())?;
        }
        #[cfg(target_os = "macos")]
        if let Device::Metal(metal) = bytes.device() {
            super::raw_experts_metal::precompile(format, device_scales.is_some(), metal)?;
        }
        Ok(Self {
            format,
            bytes,
            scales: device_scales,
            host_scales: scales,
            experts,
            out_dim,
            in_dim,
        })
    }

    /// Whether `device` runs this bank in kernels rather than on the host.
    #[cfg_attr(not(any(feature = "cuda", target_os = "macos")), allow(dead_code))]
    pub(crate) fn runs_on(&self, device: &Device) -> bool {
        (self.bytes.device().is_cuda() || self.bytes.device().is_metal())
            && self.bytes.device().same_device(device)
    }

    /// `input` `[tokens, 1 | top_k, in_dim]` F32 through the experts named
    /// by `ids` `[tokens, top_k]` U32, giving `[tokens, top_k, out_dim]`.
    #[cfg(target_os = "macos")]
    pub(crate) fn group_routes(
        &self,
        ids: &Tensor,
    ) -> Result<Option<super::raw_experts_metal::RawRoutePlan>> {
        if !self.bytes.device().is_metal() || !self.runs_on(ids.device()) {
            return Ok(None);
        }
        Ok(super::raw_experts_metal::group_routes(
            ids,
            self.experts,
            (self.out_dim / 2).div_ceil(4),
            self.in_dim.div_ceil(8),
        )?)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn forward_with_plan(
        &self,
        input: &Tensor,
        ids: &Tensor,
        plan: Option<&super::raw_experts_metal::RawRoutePlan>,
    ) -> Result<Tensor> {
        if plan.is_some() && self.bytes.device().is_metal() && self.runs_on(input.device()) {
            return Ok(super::raw_experts_metal::forward_with_plan(
                self.format,
                &self.bytes,
                self.scales.as_ref(),
                input,
                ids,
                plan,
                self.experts,
                self.out_dim,
                self.in_dim,
            )?);
        }
        self.forward(input, ids)
    }

    /// Single-token fused `down` + route-weighted sum + static `post_conv` + `residual` on Metal.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_down_post_conv_1(
        &self,
        input: &Tensor,
        ids: &Tensor,
        route_weights: &Tensor,
        past: Option<&Tensor>,
        conv_weight: &Tensor,
        residual: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if self.bytes.device().is_metal()
            && self.runs_on(input.device())
            && input.dim(0)? == 1
            && input.dim(1)? <= 8
        {
            return Ok(Some(super::raw_experts_metal::forward_down_post_conv_1(
                self.format,
                &self.bytes,
                self.scales.as_ref(),
                input,
                ids,
                route_weights,
                past,
                conv_weight,
                residual,
                [self.experts, self.out_dim, self.in_dim],
            )?));
        }
        Ok(None)
    }

    /// The whole prompt MoE of this `[gate; up]` bank and its `down` bank on
    /// Metal 4, with routes sorted by expert: the weighted route sum
    /// `[tokens, down.out_dim]` for `x` `[tokens, in_dim]`. `None` where
    /// those kernels do not apply.
    #[cfg(target_os = "macos")]
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prompt_moe(
        &self,
        down: &RawExperts,
        x: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
        beta: f32,
        linear_beta: f32,
    ) -> Result<Option<Tensor>> {
        self.prompt_moe_with_half(down, x, None, ids, weights, beta, linear_beta)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prompt_moe_with_half(
        &self,
        down: &RawExperts,
        x: &Tensor,
        x_f16: Option<&Tensor>,
        ids: &Tensor,
        weights: &Tensor,
        beta: f32,
        linear_beta: f32,
    ) -> Result<Option<Tensor>> {
        use super::raw_prompt_moe_metal::Bank;
        if !(self.bytes.device().is_metal() && self.runs_on(x.device()) && down.runs_on(x.device()))
        {
            return Ok(None);
        }
        fn bank(experts: &RawExperts) -> Bank<'_> {
            Bank {
                format: experts.format,
                bytes: &experts.bytes,
                scales: experts.scales.as_ref(),
                experts: experts.experts,
                out_dim: experts.out_dim,
                in_dim: experts.in_dim,
            }
        }
        Ok(super::raw_prompt_moe_metal::forward(
            &bank(self),
            &bank(down),
            x,
            x_f16,
            ids,
            weights,
            beta,
            linear_beta,
        )?)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn forward_situ_with_plan(
        &self,
        input: &Tensor,
        ids: &Tensor,
        plan: Option<&super::raw_experts_metal::RawRoutePlan>,
        beta: f32,
        linear_beta: f32,
    ) -> Result<Option<Tensor>> {
        if self.bytes.device().is_metal()
            && self.runs_on(input.device())
            && super::raw_experts_metal::situ_serves(input.dim(0)?)
        {
            return Ok(Some(super::raw_experts_metal::forward_situ_with_plan(
                self.format,
                &self.bytes,
                self.scales.as_ref(),
                input,
                ids,
                plan,
                [self.experts, self.out_dim, self.in_dim],
                beta,
                linear_beta,
            )?));
        }
        Ok(None)
    }

    pub(crate) fn forward(&self, input: &Tensor, ids: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.runs_on(input.device()) {
            return super::indexed_moe_cuda::forward_raw(
                self.format.kernel_format(),
                &input.contiguous()?,
                &ids.contiguous()?,
                &self.bytes,
                self.scales.as_ref(),
                self.experts,
                self.out_dim,
                self.in_dim,
            );
        }
        #[cfg(target_os = "macos")]
        if self.bytes.device().is_metal() && self.runs_on(input.device()) {
            return self.forward_metal(input, ids);
        }
        self.forward_host(input, ids)
    }

    /// The Metal kernels, in chunks of tokens: the prompt matmul keeps each
    /// expert's routes in threadgroup memory.
    #[cfg(target_os = "macos")]
    fn forward_metal(&self, input: &Tensor, ids: &Tensor) -> Result<Tensor> {
        const CHUNK: usize = 4096;
        let tokens = input.dim(0)?;
        let run = |input: &Tensor, ids: &Tensor| -> Result<Tensor> {
            Ok(super::raw_experts_metal::forward(
                self.format,
                &self.bytes,
                self.scales.as_ref(),
                input,
                ids,
                self.experts,
                self.out_dim,
                self.in_dim,
            )?)
        };
        if tokens <= CHUNK {
            return run(input, ids);
        }
        let parts = (0..tokens)
            .step_by(CHUNK)
            .map(|start| {
                let len = CHUNK.min(tokens - start);
                run(&input.narrow(0, start, len)?, &ids.narrow(0, start, len)?)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Tensor::cat(&parts, 0)?)
    }

    /// [`Self::forward`] of a `[gate; up]` bank for one token on CUDA (fewer
    /// than eight on Metal), with SiTU (`beta`, `linear_beta`) applied to each
    /// route's row pairs in the same kernel: `[tokens, top_k, out_dim / 2]`.
    /// `None` elsewhere or for more tokens.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn forward_situ(
        &self,
        input: &Tensor,
        ids: &Tensor,
        beta: f32,
        linear_beta: f32,
    ) -> Result<Option<Tensor>> {
        #[cfg(feature = "cuda")]
        if self.runs_on(input.device()) && input.dim(0)? == 1 {
            return Ok(Some(super::indexed_moe_cuda::forward_raw_situ(
                self.format.kernel_format(),
                &input.contiguous()?,
                &ids.contiguous()?,
                &self.bytes,
                self.scales.as_ref(),
                self.experts,
                self.out_dim,
                self.in_dim,
                beta,
                linear_beta,
            )?));
        }
        #[cfg(target_os = "macos")]
        if self.bytes.device().is_metal()
            && self.runs_on(input.device())
            && super::raw_experts_metal::situ_serves(input.dim(0)?)
        {
            return Ok(Some(super::raw_experts_metal::forward_situ(
                self.format,
                &self.bytes,
                self.scales.as_ref(),
                input,
                ids,
                [self.experts, self.out_dim, self.in_dim],
                beta,
                linear_beta,
            )?));
        }
        let _ = (input, ids, beta, linear_beta);
        Ok(None)
    }

    /// [`Self::forward`] of a prompt's `[gate | up]` rows (`[tokens, top_k,
    /// 2 * in_dim]`) on CUDA, applying SiTU (`beta`, `linear_beta`) as the
    /// inputs are gathered for the tensor-core kernel. `None` off CUDA, for
    /// one token, or without that kernel.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn forward_after_situ(
        &self,
        gate_up: &Tensor,
        ids: &Tensor,
        beta: f32,
        linear_beta: f32,
    ) -> Result<Option<Tensor>> {
        #[cfg(feature = "cuda")]
        if self.runs_on(gate_up.device()) {
            return super::indexed_moe_cuda::forward_raw_after_situ(
                self.format.kernel_format(),
                &gate_up.contiguous()?,
                &ids.contiguous()?,
                &self.bytes,
                self.scales.as_ref(),
                self.experts,
                self.out_dim,
                self.in_dim,
                beta,
                linear_beta,
            );
        }
        let _ = (gate_up, ids, beta, linear_beta);
        Ok(None)
    }

    /// Dequantize one expert's `[out_dim, in_dim]` weights.
    pub(crate) fn dequantize_expert(&self, expert: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(expert < self.experts, "expert {expert} is out of range");
        let (block, block_bytes) = self.format.block();
        let row_bytes = self.in_dim / block * block_bytes;
        let bytes = if self.bytes.device().is_cpu() {
            self.bytes.clone()
        } else {
            let len = self.out_dim * row_bytes;
            self.bytes
                .narrow(0, expert * len, len)?
                .to_device(&Device::Cpu)?
        };
        let (storage, layout) = bytes.storage_and_layout();
        let Storage::Cpu(CpuStorage::U8(data)) = &*storage else {
            anyhow::bail!("raw expert host bytes are not u8")
        };
        let start = layout.start_offset()
            + if self.bytes.device().is_cpu() {
                expert * self.out_dim * row_bytes
            } else {
                0
            };
        let mut values = vec![0f32; self.out_dim * self.in_dim];
        self.format
            .dequantize(&data[start..start + self.out_dim * row_bytes], &mut values)?;
        Ok(values)
    }

    fn forward_host(&self, input: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let (tokens, input_rows, in_dim) = input.dims3()?;
        let (id_tokens, top_k) = ids.dims2()?;
        anyhow::ensure!(
            in_dim == self.in_dim
                && id_tokens == tokens
                && (input_rows == 1 || input_rows == top_k),
            "expert input {:?} does not fit ids {:?}",
            input.dims(),
            ids.dims()
        );
        let ids = ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let input = input
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape((tokens * input_rows, in_dim))?;
        let mut routes_of = vec![Vec::new(); self.experts];
        for (token, experts) in ids.iter().enumerate() {
            for (slot, &expert) in experts.iter().enumerate() {
                let expert = expert as usize;
                anyhow::ensure!(expert < self.experts, "expert {expert} is out of range");
                routes_of[expert].push(token * top_k + slot);
            }
        }
        let mut output = vec![0f32; tokens * top_k * self.out_dim];
        for (expert, routes) in routes_of.iter().enumerate() {
            if routes.is_empty() {
                continue;
            }
            let weight = Tensor::from_vec(
                self.dequantize_expert(expert)?,
                (self.out_dim, self.in_dim),
                &Device::Cpu,
            )?;
            let rows = routes
                .iter()
                .map(|&route| {
                    if input_rows == 1 {
                        (route / top_k) as u32
                    } else {
                        route as u32
                    }
                })
                .collect::<Vec<_>>();
            let rows = Tensor::from_vec(rows, routes.len(), &Device::Cpu)?;
            let mut product = input.index_select(&rows, 0)?.matmul(&weight.t()?)?;
            if let Some(scales) = &self.host_scales {
                product = (product * f64::from(scales[expert]))?;
            }
            for (&route, values) in routes.iter().zip(product.to_vec2::<f32>()?) {
                output[route * self.out_dim..(route + 1) * self.out_dim].copy_from_slice(&values);
            }
        }
        Ok(Tensor::from_vec(
            output,
            (tokens, top_k, self.out_dim),
            &Device::Cpu,
        )?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A bank of `experts` x `out_dim` x `in_dim` pseudo-random NVFP4 blocks,
    /// with every scale byte the format can hold appearing somewhere.
    pub(crate) fn random_bank(experts: usize, out_dim: usize, in_dim: usize, seed: u32) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(7);
        let blocks = experts * out_dim * in_dim / BLOCK;
        let mut bytes = Vec::with_capacity(blocks * BLOCK_BYTES);
        for _ in 0..blocks {
            for byte in 0..BLOCK_BYTES {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = (state >> 13) as u8;
                // Keep the scales near 1 most of the time so products stay
                // in a range that a relative tolerance can check.
                bytes.push(if byte < 4 && !value.is_multiple_of(8) {
                    0x30 + value % 24
                } else {
                    value
                });
            }
        }
        bytes
    }

    #[test]
    fn ue4m3_matches_ggml() {
        assert_eq!(ue4m3(0), 0.0);
        assert_eq!(ue4m3(0x7f), 0.0);
        // 0x38: exponent 7, mantissa 0 -> 1.0, halved.
        assert_eq!(ue4m3(0x38), 0.5);
        // 0x39: (1 + 1/8) * 2^0 / 2.
        assert_eq!(ue4m3(0x39), 0.5625);
        // Subnormal: mantissa 1 * 2^-9 / 2.
        assert_eq!(ue4m3(0x01), 2f32.powi(-10));
        // Largest finite (0x7f is the NaN code): (1 + 6/8) * 2^8 / 2.
        assert_eq!(ue4m3(0x7e), 224.0);
        // The sign bit is ignored.
        assert_eq!(ue4m3(0xb8), ue4m3(0x38));
    }

    #[test]
    fn dequantize_places_low_and_high_nibbles_per_group() -> Result<()> {
        let mut block = [0u8; BLOCK_BYTES];
        // Group scales 0.5, 1, 2, 4 (unhalved 1, 2, 4, 8).
        block[..4].copy_from_slice(&[0x38, 0x40, 0x48, 0x50]);
        for group in 0..4 {
            for j in 0..8 {
                // Low nibble: value index j % 8; high nibble: 8 + j % 8 (negatives).
                block[4 + group * 8 + j] = ((8 + j as u8) << 4) | j as u8;
            }
        }
        let mut values = [0f32; BLOCK];
        dequantize(&block, &mut values)?;
        for group in 0..4 {
            let scale = [0.5f32, 1.0, 2.0, 4.0][group];
            for j in 0..8 {
                assert_eq!(values[group * 16 + j], f32::from(FP4_VALUES[j]) * scale);
                assert_eq!(
                    values[group * 16 + 8 + j],
                    f32::from(FP4_VALUES[8 + j]) * scale
                );
            }
        }
        Ok(())
    }

    /// A bank of pseudo-random IQ4_NL blocks with small scales.
    #[cfg(any(feature = "cuda", target_os = "macos"))]
    pub(crate) fn random_iq4_nl_bank(
        experts: usize,
        out_dim: usize,
        in_dim: usize,
        seed: u32,
    ) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(3);
        let blocks = experts * out_dim * in_dim / 32;
        let mut bytes = Vec::with_capacity(blocks * 18);
        for _ in 0..blocks {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let scale =
                half::f16::from_f32(((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) / 32.0);
            bytes.extend_from_slice(&scale.to_le_bytes());
            for _ in 0..16 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                bytes.push((state >> 13) as u8);
            }
        }
        bytes
    }

    fn bank_on(
        format: RawFormat,
        bytes: &[u8],
        scales: Option<Vec<f32>>,
        dims: [usize; 3],
        device: &Device,
    ) -> Result<RawExperts> {
        let raw = RawTensor::new(
            dims.to_vec(),
            format.dtype(),
            Tensor::from_slice(bytes, bytes.len(), device)?,
        )?;
        RawExperts::new(&raw, scales, dims[0], dims[1], dims[2])
    }

    /// Pseudo-random MXFP8 weight bytes (no NaN codes) and scales near 2^-6.
    pub(crate) fn random_mxfp8(
        experts: usize,
        out_dim: usize,
        in_dim: usize,
        seed: u32,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(5);
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 13) as u8
        };
        let values = (0..experts * out_dim * in_dim)
            .map(|_| {
                let byte = next();
                if byte & 0x7f == 0x7f { byte ^ 1 } else { byte }
            })
            .collect();
        let scales = (0..experts * out_dim * in_dim / 32)
            .map(|_| 118 + next() % 6)
            .collect();
        (values, scales)
    }

    fn mxfp8_on(
        values: &[u8],
        scales: &[u8],
        dims: [usize; 3],
        device: &Device,
    ) -> Result<RawExperts> {
        let values = RawTensor::new(
            dims.to_vec(),
            DeepSeek4DType::F8E4M3,
            Tensor::from_slice(values, values.len(), &Device::Cpu)?,
        )?;
        let scales = RawTensor::new(
            vec![dims[0], dims[1], dims[2] / 32],
            DeepSeek4DType::I8,
            Tensor::from_slice(scales, scales.len(), &Device::Cpu)?,
        )?;
        RawExperts::mxfp8(&values, &scales, dims[0], dims[1], dims[2], device)
    }

    #[test]
    fn mxfp8_matches_e4m3_and_e8m0() -> Result<()> {
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0xb8), -1.0);
        assert_eq!(e4m3(0x7e), 448.0);
        assert_eq!(e4m3(0x01), 2f32.powi(-9));
        assert_eq!(e8m0(127), 1.0);
        assert_eq!(e8m0(0), 2f32.powi(-127));
        let (values, scales) = random_mxfp8(2, 3, 64, 4);
        let bank = mxfp8_on(&values, &scales, [2, 3, 64], &Device::Cpu)?;
        let expert = bank.dequantize_expert(1)?;
        for (i, value) in expert.iter().enumerate() {
            let flat = 3 * 64 + i;
            assert_eq!(*value, e4m3(values[flat]) * e8m0(scales[flat / 32]));
        }
        Ok(())
    }

    #[test]
    fn iq4_nl_dequantize_matches_ggml_layout() -> Result<()> {
        let mut block = [0u8; 18];
        block[..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        for (j, byte) in block[2..].iter_mut().enumerate() {
            *byte = ((15 - j as u8) << 4) | j as u8;
        }
        let mut values = [0f32; 32];
        dequantize_iq4_nl(&block, &mut values)?;
        for j in 0..16 {
            assert_eq!(values[j], 0.5 * f32::from(IQ4_NL_VALUES[j]));
            assert_eq!(values[j + 16], 0.5 * f32::from(IQ4_NL_VALUES[15 - j]));
        }
        Ok(())
    }

    #[test]
    fn host_forward_matches_dense_products() -> Result<()> {
        let (experts, out_dim, in_dim) = (3usize, 5usize, 128usize);
        let bytes = random_bank(experts, out_dim, in_dim, 3);
        let scales = vec![0.5f32, 2.0, -1.5];
        let bank = bank_on(
            RawFormat::Nvfp4,
            &bytes,
            Some(scales.clone()),
            [experts, out_dim, in_dim],
            &Device::Cpu,
        )?;
        let input = Tensor::from_vec(
            (0..2 * in_dim).map(|i| (i as f32 * 0.37).sin()).collect(),
            (2, 1, in_dim),
            &Device::Cpu,
        )?;
        let ids = Tensor::from_vec(vec![2u32, 0, 1, 2], (2, 2), &Device::Cpu)?;
        let out = bank.forward(&input, &ids)?.to_vec3::<f32>()?;
        for (token, experts_of) in [[2usize, 0], [1, 2]].iter().enumerate() {
            let x = input.get(token)?.squeeze(0)?.to_vec1::<f32>()?;
            for (slot, &expert) in experts_of.iter().enumerate() {
                let weight = bank.dequantize_expert(expert)?;
                for row in 0..out_dim {
                    let expected = scales[expert]
                        * weight[row * in_dim..(row + 1) * in_dim]
                            .iter()
                            .zip(&x)
                            .map(|(w, x)| w * x)
                            .sum::<f32>();
                    let got = out[token][slot][row];
                    assert!(
                        (got - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                        "token {token} expert {expert} row {row}: {got} vs {expected}"
                    );
                }
            }
        }
        Ok(())
    }

    /// The CUDA decode, grouped and tensor-core kernels against the host path.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_kernels_match_host_reference() -> Result<()> {
        let Some(cuda) = crate::test_cuda_device() else {
            return Ok(());
        };
        let (experts, out_dim, in_dim, top_k) = (6usize, 70usize, 192usize, 3usize);
        let scales = (0..experts)
            .map(|e| 0.25 + e as f32 * 0.5)
            .collect::<Vec<_>>();
        let dims = [experts, out_dim, in_dim];
        for (format, with_scales) in [
            (RawFormat::Nvfp4, false),
            (RawFormat::Nvfp4, true),
            (RawFormat::Iq4Nl, false),
            (RawFormat::Mxfp8, false),
        ] {
            let scales = with_scales.then(|| scales.clone());
            let (host, gpu) = match format {
                RawFormat::Mxfp8 => {
                    let (values, block_scales) = random_mxfp8(experts, out_dim, in_dim, 13);
                    (
                        mxfp8_on(&values, &block_scales, dims, &Device::Cpu)?,
                        mxfp8_on(&values, &block_scales, dims, &cuda)?,
                    )
                }
                _ => {
                    let bytes = match format {
                        RawFormat::Nvfp4 => random_bank(experts, out_dim, in_dim, 11),
                        _ => random_iq4_nl_bank(experts, out_dim, in_dim, 12),
                    };
                    (
                        bank_on(format, &bytes, scales.clone(), dims, &Device::Cpu)?,
                        bank_on(format, &bytes, scales, dims, &cuda)?,
                    )
                }
            };
            assert!(gpu.runs_on(&cuda));
            // One decode token, a prompt sharing one input row per token, and
            // a prompt with one input row per route (the down projection).
            // 200 tokens fill route tiles with up to eight 16-route slabs.
            for (tokens, input_rows) in [
                (1usize, 1usize),
                (1, top_k),
                (37, 1),
                (37, top_k),
                (200, 1),
                (200, top_k),
            ] {
                let input = Tensor::from_vec(
                    (0..tokens * input_rows * in_dim)
                        .map(|i| ((i * 7919 % 1013) as f32 / 1013.0 - 0.5) * 2.0)
                        .collect(),
                    (tokens, input_rows, in_dim),
                    &Device::Cpu,
                )?;
                let ids = Tensor::from_vec(
                    (0..tokens * top_k)
                        .map(|i| ((i * 5 + i / top_k) % experts) as u32)
                        .collect(),
                    (tokens, top_k),
                    &Device::Cpu,
                )?;
                let expected = host
                    .forward(&input, &ids)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let got = gpu
                    .forward(&input.to_device(&cuda)?, &ids.to_device(&cuda)?)?
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let scale = expected.iter().fold(1f32, |m, v| m.max(v.abs()));
                let error = expected
                    .iter()
                    .zip(&got)
                    .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                // Prompts round inputs to F16 on tensor cores.
                let tolerance = if tokens > 1 { 2e-3 } else { 1e-5 } * scale;
                assert!(
                    error <= tolerance,
                    "{format:?} tokens {tokens}, input rows {input_rows}, scales {with_scales}: error {error} (scale {scale})"
                );
                // A prompt whose inputs are [gate | up] rows twice as wide:
                // SiTU applied as the down product gathers them.
                if tokens > 1 && input_rows == top_k {
                    let (beta, linear_beta) = (4.0f32, 25.0f32);
                    let gate_up = Tensor::from_vec(
                        (0..tokens * top_k * 2 * in_dim)
                            .map(|i| ((i * 4271 % 977) as f32 / 977.0 - 0.5) * 6.0)
                            .collect(),
                        (tokens, top_k, 2 * in_dim),
                        &Device::Cpu,
                    )?;
                    let gate = gate_up.narrow(2, 0, in_dim)?;
                    let up = gate_up.narrow(2, in_dim, in_dim)?;
                    let activated = (((&gate / beta as f64)?.tanh()? * beta as f64)?
                        * candle_nn::ops::sigmoid(&gate)?)?
                    .mul(&((&up / linear_beta as f64)?.tanh()? * linear_beta as f64)?)?;
                    let expected = host
                        .forward(&activated.contiguous()?, &ids)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let got = gpu
                        .forward_after_situ(
                            &gate_up.to_device(&cuda)?,
                            &ids.to_device(&cuda)?,
                            beta,
                            linear_beta,
                        )?
                        .expect("prompts take the tensor-core path")
                        .to_device(&Device::Cpu)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let scale = expected.iter().fold(1f32, |m, v| m.max(v.abs()));
                    let error = expected
                        .iter()
                        .zip(&got)
                        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                    assert!(
                        got.len() == expected.len() && error <= 2e-3 * scale,
                        "{format:?} SiTU prompt, tokens {tokens}: error {error} (scale {scale})"
                    );
                }
                // One token: SiTU over [gate; up] row pairs in the same kernel.
                if tokens == 1 {
                    let (beta, linear_beta) = (4.0f32, 25.0f32);
                    let half = out_dim / 2;
                    let situ = |g: f32, u: f32| {
                        beta * (g / beta).tanh() / (1.0 + (-g).exp())
                            * linear_beta
                            * (u / linear_beta).tanh()
                    };
                    let expected = expected
                        .chunks(out_dim)
                        .flat_map(|row| (0..half).map(move |i| situ(row[i], row[half + i])))
                        .collect::<Vec<_>>();
                    let got = gpu
                        .forward_situ(
                            &input.to_device(&cuda)?,
                            &ids.to_device(&cuda)?,
                            beta,
                            linear_beta,
                        )?
                        .expect("one token runs the SiTU kernel")
                        .to_device(&Device::Cpu)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let error = expected
                        .iter()
                        .zip(&got)
                        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                    assert!(
                        got.len() == expected.len() && error <= 1e-4 * scale.max(1.0),
                        "{format:?} SiTU, input rows {input_rows}: error {error}"
                    );
                }
            }
        }
        Ok(())
    }

    /// The Metal vector and matrix kernels against the host path.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_kernels_match_host_reference() -> Result<()> {
        let Ok(metal) = Device::new_metal(0) else {
            assert!(
                std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_none(),
                "Metal device unavailable"
            );
            return Ok(());
        };
        let (experts, out_dim, in_dim, top_k) = (6usize, 70usize, 192usize, 3usize);
        let scales = (0..experts)
            .map(|e| 0.25 + e as f32 * 0.5)
            .collect::<Vec<_>>();
        let dims = [experts, out_dim, in_dim];
        for (format, with_scales) in [
            (RawFormat::Nvfp4, false),
            (RawFormat::Nvfp4, true),
            (RawFormat::Iq4Nl, true),
            (RawFormat::Mxfp8, false),
        ] {
            let scales = with_scales.then(|| scales.clone());
            let (host, gpu) = match format {
                RawFormat::Mxfp8 => {
                    let (values, block_scales) = random_mxfp8(experts, out_dim, in_dim, 13);
                    (
                        mxfp8_on(&values, &block_scales, dims, &Device::Cpu)?,
                        mxfp8_on(&values, &block_scales, dims, &metal)?,
                    )
                }
                _ => {
                    let bytes = match format {
                        RawFormat::Nvfp4 => random_bank(experts, out_dim, in_dim, 11),
                        _ => random_iq4_nl_bank(experts, out_dim, in_dim, 12),
                    };
                    (
                        bank_on(format, &bytes, scales.clone(), dims, &Device::Cpu)?,
                        bank_on(format, &bytes, scales, dims, &metal)?,
                    )
                }
            };
            assert!(gpu.runs_on(&metal) && !host.runs_on(&metal));
            // The vector kernel up to eight tokens, the matmul from there.
            for (tokens, input_rows) in [
                (1usize, 1usize),
                (1, top_k),
                (7, top_k),
                (8, 1),
                (9, 1),
                (37, top_k),
                (200, 1),
                (200, top_k),
            ] {
                let input = Tensor::from_vec(
                    (0..tokens * input_rows * in_dim)
                        .map(|i| ((i * 7919 % 1013) as f32 / 1013.0 - 0.5) * 2.0)
                        .collect(),
                    (tokens, input_rows, in_dim),
                    &Device::Cpu,
                )?;
                let ids = Tensor::from_vec(
                    (0..tokens * top_k)
                        .map(|i| ((i * 5 + i / top_k) % experts) as u32)
                        .collect(),
                    (tokens, top_k),
                    &Device::Cpu,
                )?;
                let expected = host
                    .forward(&input, &ids)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let got = gpu
                    .forward(&input.to_device(&metal)?, &ids.to_device(&metal)?)?
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let scale = expected.iter().fold(1f32, |m, v| m.max(v.abs()));
                let error = expected
                    .iter()
                    .zip(&got)
                    .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                // The matmul stages weights in half precision.
                let tolerance = if tokens > 8 { 2e-3 } else { 1e-5 } * scale;
                assert!(
                    error <= tolerance,
                    "{format:?} tokens {tokens}, input rows {input_rows}, scales {with_scales}: error {error} (scale {scale})"
                );
                // The vector kernel's SiTU epilogue against SiTU of the host rows.
                let (beta, linear_beta) = (4.0f32, 25.0f32);
                if let Some(fused) = gpu.forward_situ(
                    &input.to_device(&metal)?,
                    &ids.to_device(&metal)?,
                    beta,
                    linear_beta,
                )? {
                    assert!(tokens <= 8);
                    let fused = fused
                        .to_device(&Device::Cpu)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let half = out_dim / 2;
                    for (route, values) in expected.chunks(out_dim).enumerate() {
                        for i in 0..half {
                            let (g, u) = (values[i], values[half + i]);
                            let want = beta * (g / beta).tanh() / (1.0 + (-g).exp())
                                * linear_beta
                                * (u / linear_beta).tanh();
                            let got = fused[route * half + i];
                            assert!(
                                (got - want).abs() <= 2e-4 * want.abs().max(1.0),
                                "{format:?} SiTU route {route} row {i}: {got} vs {want}"
                            );
                        }
                    }
                } else {
                    assert!(tokens > 8);
                }
            }
        }
        Ok(())
    }

    /// The sorted-route prompt kernels against the host experts, SiTU and
    /// route sum, with a skewed routing that fills every tile queue: two
    /// experts with several full tiles, one for each smaller tile kernel,
    /// and a few with one to four routes for the vector kernel.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_prompt_moe_matches_host_reference() -> Result<()> {
        let Ok(metal) = Device::new_metal(0) else {
            assert!(
                std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_none(),
                "Metal device unavailable"
            );
            return Ok(());
        };
        let (experts, in_dim, ffn, out_dim, top_k, tokens) =
            (11usize, 192usize, 128usize, 68usize, 3usize, 300usize);
        let (beta, linear_beta) = (4.0f32, 25.0f32);
        let scales = (0..experts)
            .map(|e| 0.25 + e as f32 * 0.5)
            .collect::<Vec<_>>();
        let second = |token: usize| -> u32 {
            match token {
                0..100 => 1,
                100..140 => 2,
                140..165 => 3,
                165..177 => 4,
                177..181 => 5,
                181..184 => 6,
                184..186 => 7,
                186 => 10,
                _ => 8,
            }
        };
        let ids = Tensor::from_vec(
            (0..tokens).flat_map(|t| [0, second(t), 9]).collect(),
            (tokens, top_k),
            &Device::Cpu,
        )?;
        // Inputs that are exact in half, which the tile kernels stage them in.
        let x = Tensor::from_vec(
            (0..tokens * in_dim)
                .map(|i| ((i * 7919 % 1013) as f32 - 506.0) / 512.0)
                .collect(),
            (tokens, in_dim),
            &Device::Cpu,
        )?;
        let weights = Tensor::from_vec(
            (0..tokens * top_k)
                .map(|i| 0.1 + (i * 37 % 101) as f32 / 101.0)
                .collect(),
            (tokens, top_k),
            &Device::Cpu,
        )?;
        let up_dims = [experts, 2 * ffn, in_dim];
        let down_dims = [experts, out_dim, ffn];
        for (format, with_scales) in [
            (RawFormat::Nvfp4, true),
            (RawFormat::Nvfp4, false),
            (RawFormat::Mxfp8, false),
        ] {
            let scales = with_scales.then(|| scales.clone());
            let build = |dims: [usize; 3], seed: u32, device: &Device| -> Result<RawExperts> {
                match format {
                    RawFormat::Mxfp8 => {
                        let (values, block_scales) = random_mxfp8(dims[0], dims[1], dims[2], seed);
                        mxfp8_on(&values, &block_scales, dims, device)
                    }
                    _ => bank_on(
                        format,
                        &random_bank(dims[0], dims[1], dims[2], seed),
                        scales.clone(),
                        dims,
                        device,
                    ),
                }
            };
            let (host_up, host_down) = (
                build(up_dims, 21, &Device::Cpu)?,
                build(down_dims, 22, &Device::Cpu)?,
            );
            let (gpu_up, gpu_down) = (build(up_dims, 21, &metal)?, build(down_dims, 22, &metal)?);
            let gate_up = host_up
                .forward(&x.reshape((tokens, 1, in_dim))?, &ids)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let activated = gate_up
                .chunks(2 * ffn)
                .flat_map(|route| {
                    (0..ffn).map(|i| {
                        let (g, u) = (route[i], route[ffn + i]);
                        beta * (g / beta).tanh() / (1.0 + (-g).exp())
                            * linear_beta
                            * (u / linear_beta).tanh()
                    })
                })
                .collect::<Vec<_>>();
            let routed = host_down
                .forward(
                    &Tensor::from_vec(activated, (tokens, top_k, ffn), &Device::Cpu)?,
                    &ids,
                )?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let route_weights = weights.flatten_all()?.to_vec1::<f32>()?;
            let mut expected = vec![0f32; tokens * out_dim];
            for (route, values) in routed.chunks(out_dim).enumerate() {
                for (column, value) in values.iter().enumerate() {
                    expected[route / top_k * out_dim + column] += route_weights[route] * value;
                }
            }
            let run = || -> Result<Option<Vec<f32>>> {
                let Some(moe) = gpu_up.prompt_moe(
                    &gpu_down,
                    &x.to_device(&metal)?,
                    &ids.to_device(&metal)?,
                    &weights.to_device(&metal)?,
                    beta,
                    linear_beta,
                )?
                else {
                    return Ok(None);
                };
                assert_eq!(moe.dims(), [tokens, out_dim]);
                Ok(Some(
                    moe.to_device(&Device::Cpu)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                ))
            };
            let Some(got) = run()? else {
                // No Metal 4 tensor operations on this device.
                return Ok(());
            };
            let scale = expected.iter().fold(1f32, |m, v| m.max(v.abs()));
            let (worst, error) = expected.iter().zip(&got).enumerate().fold(
                (0usize, 0f32),
                |(at, m), (i, (a, b))| {
                    if (a - b).abs() > m {
                        (i, (a - b).abs())
                    } else {
                        (at, m)
                    }
                },
            );
            // The tile kernels stage the SiTU outputs in half precision.
            assert!(
                error <= 3e-3 * scale,
                "{format:?} scales {with_scales}: error {error} (scale {scale}) at token {} column {}",
                worst / out_dim,
                worst % out_dim
            );
            // Routes land in the sorted order in whatever order the plan's
            // threads place them; the sums do not depend on it.
            assert!(
                run()?.is_some_and(|again| again == got),
                "{format:?} scales {with_scales}: a second pass differs"
            );
        }
        Ok(())
    }
}
