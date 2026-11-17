//! TurboQuant KV cache compression (3-bit, 4-bit, and 8-bit profiles).
//!
//! Implements TurboQuant_mse (Algorithm 1) from "TurboQuant: Online Vector Quantization
//! with Near-optimal Distortion Rate" (Zandieh et al., ICLR 2026, arxiv:2504.19874).
//!
//! Randomized Walsh-Hadamard rotation + Lloyd-Max scalar quantization.
//! Model inference accepts only the quality-qualified 8-bit profile; 3/4-bit
//! primitives remain experimental. The CLI uses 8-bit when KV compression is
//! explicitly enabled (the normal inference default remains F16).
//!
//! Metal keeps packed records on the GPU and uses native kernels for head
//! widths up to 1024. Larger heads, CPU, and CUDA use the portable CPU
//! implementation. Stored-size
//! compression does not describe peak memory: attention still needs a dense
//! reconstruction of the active layer.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};

/// Lloyd-Max centroids for standard Gaussian N(0,1) at 3-bit (8 levels).
/// Symmetric around zero. Precomputed via iterative Lloyd-Max algorithm.
const CENTROIDS_3BIT: [f32; 8] = [
    -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
];

/// Lloyd-Max decision boundaries for N(0,1) at 3-bit.
/// boundary[i] = midpoint between centroid[i] and centroid[i+1].
const BOUNDARIES_3BIT: [f32; 7] = [-1.7480, -1.0500, -0.5006, 0.0, 0.5006, 1.0500, 1.7480];

/// Lloyd-Max centroids and boundaries for standard Gaussian N(0,1) at 4-bit.
const CENTROIDS_4BIT: [f32; 16] = [
    -2.69013, -2.04740, -1.60586, -1.24921, -0.93835, -0.65458, -0.38699, -0.12808, 0.12808,
    0.38699, 0.65458, 0.93835, 1.24921, 1.60586, 2.04740, 2.69013,
];
const BOUNDARIES_4BIT: [f32; 15] = [
    -2.36876, -1.82663, -1.42753, -1.09378, -0.79647, -0.52079, -0.25754, 0.0, 0.25754, 0.52079,
    0.79647, 1.09378, 1.42753, 1.82663, 2.36876,
];

/// Zero-preserving 8-bit quantization range for rotated, normalized coordinates.
/// WHT coordinates are approximately N(0, 1), so clipping at four standard
/// deviations is rare while retaining a small quantization step.
const SCALE_8BIT: f32 = 127.0 / 4.0;

/// Only the 8-bit profile has passed the current model-level regression gate.
/// Lower-bit primitives remain available for quantizer research, not serving.
pub(crate) fn validate_inference_bits(bits: u8) -> Result<()> {
    ensure!(
        bits == 8,
        "{bits}-bit KV quantization is not quality-qualified for model inference; use 8-bit KV or the default F16 cache"
    );
    Ok(())
}

/// Configuration for TurboQuant quantization. Shared across layers.
#[derive(Clone)]
pub struct TurboQuantConfig {
    /// Quantization width. Supported values are 3, 4, and 8.
    pub bits: u8,
    /// Number of coordinates in each attention-head vector.
    pub head_dim: usize,
    /// Random ±1 signs for randomized Walsh-Hadamard Transform.
    signs: Vec<f32>,
    /// sqrt(head_dim) — cached for scaling.
    scale: f32,
    /// Packed bytes per quantized vector.
    packed_bytes: usize,
}

impl TurboQuantConfig {
    /// Build a TurboQuant configuration.
    ///
    /// # Panics
    ///
    /// Panics when `bits` is unsupported, `head_dim` is not a power of two of
    /// at least eight, or packed-size arithmetic overflows. This legacy
    /// constructor does not apply the defensive maximum accepted by
    /// [`Self::try_new`].
    pub fn new(bits: u8, head_dim: usize, seed: u64) -> Self {
        assert!(matches!(bits, 3 | 4 | 8), "unsupported KV quantization");
        assert!(
            head_dim.is_power_of_two(),
            "head_dim must be power of 2 for WHT"
        );
        assert!(head_dim >= 8, "head_dim must be >= 8");
        let packed_bytes = head_dim
            .checked_mul(bits as usize)
            .expect("TurboQuant packed size overflow")
            .div_ceil(8);
        Self::build(bits, head_dim, seed, packed_bytes)
    }

    /// Build a validated TurboQuant configuration from fallible input.
    ///
    /// # Errors
    ///
    /// Returns an error when `bits` is not a supported quantization width or
    /// when `head_dim` cannot be transformed safely by this implementation.
    pub fn try_new(bits: u8, head_dim: usize, seed: u64) -> Result<Self> {
        ensure!(matches!(bits, 3 | 4 | 8), "unsupported KV quantization");
        ensure!(
            head_dim.is_power_of_two(),
            "head_dim must be power of 2 for WHT"
        );
        ensure!(head_dim >= 8, "head_dim must be >= 8");
        ensure!(
            head_dim <= 16_384,
            "head_dim must not exceed 16384 for TurboQuant"
        );
        let packed_bytes = head_dim
            .checked_mul(bits as usize)
            .context("TurboQuant packed size overflow")?
            .div_ceil(8);

        Ok(Self::build(bits, head_dim, seed, packed_bytes))
    }

    fn build(bits: u8, head_dim: usize, seed: u64, packed_bytes: usize) -> Self {
        Self {
            bits,
            head_dim,
            signs: generate_signs(head_dim, seed),
            scale: (head_dim as f32).sqrt(),
            packed_bytes,
        }
    }

    /// Check that the public legacy fields still agree with the derived state
    /// captured when this configuration was constructed.
    ///
    /// `bits` and `head_dim` remain public for source compatibility, but
    /// changing either field directly would otherwise leave `signs`, `scale`,
    /// or `packed_bytes` stale and could turn a valid cache update into an
    /// indexing panic. Callers must receive an error before any cache state is
    /// allocated or published when that happens.
    fn validate_consistency(&self) -> Result<()> {
        ensure!(
            matches!(self.bits, 3 | 4 | 8),
            "TurboQuant configuration was mutated after construction: unsupported bit width {}",
            self.bits
        );
        ensure!(
            self.head_dim >= 8 && self.head_dim.is_power_of_two(),
            "TurboQuant configuration was mutated after construction: invalid head dimension {}",
            self.head_dim
        );
        ensure!(
            self.signs.len() == self.head_dim,
            "TurboQuant configuration was mutated after construction: cached signs do not match head dimension"
        );
        let expected_packed_bytes = self
            .head_dim
            .checked_mul(self.bits as usize)
            .context("TurboQuant configuration packed size overflow")?
            .div_ceil(8);
        ensure!(
            self.packed_bytes == expected_packed_bytes,
            "TurboQuant configuration was mutated after construction: cached packed size does not match bits/head dimension"
        );
        let expected_scale = (self.head_dim as f32).sqrt();
        ensure!(
            self.scale.is_finite() && self.scale == expected_scale,
            "TurboQuant configuration was mutated after construction: cached scale does not match head dimension"
        );
        ensure!(
            self.packed_bytes.checked_add(4).is_some(),
            "TurboQuant configuration packed record size overflows"
        );
        Ok(())
    }

    /// Quantization width in bits per transformed coordinate.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Number of coordinates in each attention-head vector.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Bytes per quantized vector (packed indices + f32 norm).
    pub fn bytes_per_vector(&self) -> usize {
        self.packed_bytes + 4
    }

    /// Quantize a single f32 vector of length head_dim.
    fn quantize_into(&self, v: &[f32], out_packed: &mut [u8], out_norm: &mut f32) {
        debug_assert_eq!(v.len(), self.head_dim);

        // Compute L2 norm
        // Accumulate in f64: squaring otherwise-finite f32 inputs can overflow.
        // The cache update validates representability before calling this.
        let norm = v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt() as f32;
        *out_norm = norm;

        if norm < 1e-10 {
            out_packed.iter_mut().for_each(|b| *b = 0);
            return;
        }

        // Scratch buffer on stack for small dims, heap for large
        let mut buf = vec![0.0f32; self.head_dim];

        // Normalize + apply random signs
        let inv_norm = 1.0 / norm;
        for (i, value) in buf.iter_mut().enumerate() {
            *value = v[i] * inv_norm * self.signs[i];
        }

        // Walsh-Hadamard Transform
        wht_inplace(&mut buf);

        // Scale to standard Gaussian range + quantize + pack
        out_packed.iter_mut().for_each(|b| *b = 0);
        for (i, value) in buf.iter().enumerate() {
            let y = value * self.scale;
            let idx = quantize_scalar(y, self.bits);
            let bit_off = i * self.bits as usize;
            let byte_off = bit_off / 8;
            let shift = bit_off % 8;
            out_packed[byte_off] |= idx << shift;
            if shift + self.bits as usize > 8 {
                out_packed[byte_off + 1] |= idx >> (8 - shift);
            }
        }
    }

    /// Dequantize packed indices + norm into an f32 vector.
    fn dequantize_into(&self, packed: &[u8], norm: f32, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.head_dim);

        if norm < 1e-10 {
            out.iter_mut().for_each(|x| *x = 0.0);
            return;
        }

        let inv_scale = 1.0 / self.scale;

        // Unpack + centroid lookup
        for (i, value) in out.iter_mut().enumerate() {
            let bit_off = i * self.bits as usize;
            let byte_off = bit_off / 8;
            let shift = bit_off % 8;
            let mask = scalar_mask(self.bits);
            let mut idx = (packed[byte_off] >> shift) & mask;
            if shift + self.bits as usize > 8 && byte_off + 1 < packed.len() {
                idx |= (packed[byte_off + 1] << (8 - shift)) & mask;
            }
            *value = centroid(self.bits, idx) * inv_scale;
        }

        // Inverse WHT (self-inverse)
        wht_inplace(out);

        // Undo signs + scale by norm
        for (value, sign) in out.iter_mut().zip(&self.signs) {
            *value *= sign * norm;
        }
    }
}

/// Quantized KV cache for a single attention layer (all heads).
pub struct QuantizedKvCache {
    /// Packed scalar indices: [seq_len * num_heads * packed_bytes]
    k_data: Vec<u8>,
    v_data: Vec<u8>,
    /// Norms: [seq_len * num_heads]
    k_norms: Vec<f32>,
    v_norms: Vec<f32>,
    seq_len: usize,
    num_heads: usize,
    config: Arc<TurboQuantConfig>,
    #[cfg(target_os = "macos")]
    metal: Option<MetalKvState>,
}

#[cfg(target_os = "macos")]
struct MetalKvState {
    // One allocation holds both sides, enabling a single quantize-append
    // dispatch. K/V are full-capacity views, never exposed outside the cache.
    backing: Tensor,
    k: Tensor,
    v: Tensor,
    signs: Tensor,
    // Eviction changes only the logical view. Compact into fresh storage only
    // when the append reaches the end, preserving rollback on allocation errors.
    start: usize,
}

/// Capacity policy shared by packed Metal allocation and stage admission.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn metal_cache_capacity(current: usize, required: usize) -> Result<usize> {
    if required == 0 || required <= current {
        return Ok(current);
    }
    crate::architectures::PreallocatedKvCache::next_capacity(current, required)
}

/// Return the packed-cache capacity that a CPU/CUDA cache should reserve for
/// `required` token positions. The portable cache uses the same bounded,
/// aligned growth policy as the native preallocated cache so admission can
/// predict the allocation before quantization starts.
pub(crate) fn cpu_cache_capacity(current: usize, required: usize) -> Result<usize> {
    if required == 0 || required <= current {
        return Ok(current);
    }
    crate::architectures::PreallocatedKvCache::next_capacity(current, required)
}

fn reserve_cpu_vec<T>(vec: &mut Vec<T>, target_len: usize, label: &'static str) -> Result<()> {
    if vec.capacity() < target_len {
        vec.try_reserve_exact(target_len - vec.len())
            .with_context(|| format!("failed to reserve TurboQuant {label} cache"))?;
    }
    ensure!(
        vec.capacity() >= target_len,
        "TurboQuant {label} cache capacity is below the requested target"
    );
    Ok(())
}

impl QuantizedKvCache {
    /// Create an empty cache for `num_heads` attention heads.
    ///
    /// Tensor updates must use the head width recorded by `config`.
    pub fn new(config: Arc<TurboQuantConfig>, num_heads: usize) -> Self {
        Self {
            k_data: Vec::new(),
            v_data: Vec::new(),
            k_norms: Vec::new(),
            v_norms: Vec::new(),
            seq_len: 0,
            num_heads,
            config,
            #[cfg(target_os = "macos")]
            metal: None,
        }
    }

    /// Quantize new K, V tokens and return full dequantized cache as f32 tensors.
    ///
    /// Input k, v: [batch=1, num_heads, new_seq_len, head_dim] (any dtype, will be cast to f32).
    /// Output: [batch=1, num_heads, total_seq_len, head_dim] f32 tensors.
    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        self.update_with_dtype(k, v, DType::F32, false)
    }

    /// Match the dense Metal cache's F16 attention interface. The public
    /// quantizer keeps its F32 reconstruction contract for numerical callers.
    pub(crate) fn update_native(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let output_dtype = if k.device().is_metal() {
            DType::F16
        } else {
            DType::F32
        };
        self.update_with_dtype(k, v, output_dtype, false)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn update_rotated_native(
        &mut self,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        self.update_with_dtype(k, v, DType::F16, true)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn attention_rotation_signs(&self) -> Result<Tensor> {
        let state = self
            .metal
            .as_ref()
            .context("rotated attention requires a Metal cache")?;
        Ok(state.signs.narrow(0, 0, 1)?)
    }

    fn update_with_dtype(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        output_dtype: DType,
        rotated: bool,
    ) -> Result<(Tensor, Tensor)> {
        // Validate before inspecting inputs, reserving memory, or changing
        // the backend/cache state. The legacy public fields can be edited
        // after construction, leaving derived quantizer state stale.
        self.config.validate_consistency()?;
        let device = k.device().clone();
        let (b, h, new_seq, d) = k.dims4()?;
        let (v_b, v_h, v_seq, v_d) = v.dims4()?;
        ensure!(b == 1, "TurboQuant K batch size must be 1, got {b}");
        ensure!(self.num_heads > 0, "TurboQuant cache must contain a head");
        ensure!(
            (v_b, v_h, v_seq, v_d) == (b, h, new_seq, d),
            "TurboQuant K/V shapes must match exactly"
        );
        ensure!(
            k.device().same_device(v.device()),
            "TurboQuant K/V tensors must be on the same device"
        );
        ensure!(
            h == self.num_heads,
            "TurboQuant expected {} heads, got {h}",
            self.num_heads
        );
        ensure!(
            d == self.config.head_dim,
            "TurboQuant expected head dimension {}, got {d}",
            self.config.head_dim
        );
        ensure!(new_seq > 0, "TurboQuant update must contain a token");

        #[cfg(target_os = "macos")]
        {
            if k.device().is_metal()
                && self.config.head_dim <= crate::turboquant_metal::MAX_METAL_HEAD_DIM
            {
                return self.update_metal(k, v, output_dtype, rotated);
            }
            ensure!(
                self.metal.is_none(),
                "cannot change the backend of a populated TurboQuant cache"
            );
        }
        ensure!(
            !rotated,
            "rotated attention requires the native Metal cache"
        );

        let pb = self.config.packed_bytes;

        // Pull new data to CPU f32
        let k_cpu = k.to_dtype(DType::F32)?.squeeze(0)?.to_vec3::<f32>()?;
        let v_cpu = v.to_dtype(DType::F32)?.squeeze(0)?.to_vec3::<f32>()?;
        let norm_limit = if output_dtype == DType::F16 {
            65_504.0 / 8.0
        } else {
            f64::from(f32::MAX) / 8.0
        };
        ensure!(
            self.k_norms
                .iter()
                .chain(&self.v_norms)
                .all(|&norm| norm.is_finite() && f64::from(norm) <= norm_limit),
            "TurboQuant existing cache exceeds the requested reconstruction range"
        );

        // Validate both complete inputs before reserving or appending anything.
        // A bad V vector must not leave its corresponding K (or earlier tokens)
        // committed. Bound the norm so reconstruction cannot overflow either.
        for (name, vectors) in [("K", &k_cpu), ("V", &v_cpu)] {
            for head in vectors {
                for vector in head {
                    ensure!(
                        vector.iter().all(|value| value.is_finite()),
                        "TurboQuant {name} contains a non-finite value"
                    );
                    let norm = vector
                        .iter()
                        .map(|&value| f64::from(value).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    ensure!(
                        norm <= norm_limit,
                        "TurboQuant {name} norm exceeds the safe reconstruction range"
                    );
                }
            }
        }

        // Reserve the next amortized capacity before quantizing. This keeps
        // stage admission in sync with the actual Vec backing allocations and
        // makes a failed partial reserve visible through allocation_tokens().
        let next_seq_len = self
            .seq_len
            .checked_add(new_seq)
            .context("TurboQuant sequence length overflow")?;
        self.reserve_cpu_capacity(next_seq_len)?;

        // Quantize new tokens (token-major order: t0_h0, t0_h1, ..., t1_h0, ...)
        let mut packed = vec![0u8; pb];
        let mut norm = 0.0f32;

        for t in 0..new_seq {
            for hi in 0..h {
                self.config
                    .quantize_into(&k_cpu[hi][t], &mut packed, &mut norm);
                self.k_data.extend_from_slice(&packed);
                self.k_norms.push(norm);

                self.config
                    .quantize_into(&v_cpu[hi][t], &mut packed, &mut norm);
                self.v_data.extend_from_slice(&packed);
                self.v_norms.push(norm);
            }
        }
        let previous_seq_len = self.seq_len;
        self.seq_len = next_seq_len;

        // Dequantize full cache into [num_heads, total_seq, head_dim]. A failed
        // output allocation must not consume positions from the next retry.
        let result = self
            .dequantize_all(&device)
            .and_then(|(k, v)| Ok((k.to_dtype(output_dtype)?, v.to_dtype(output_dtype)?)));
        if result.is_err() {
            self.seq_len = previous_seq_len;
            self.k_data.truncate(previous_seq_len * h * pb);
            self.v_data.truncate(previous_seq_len * h * pb);
            self.k_norms.truncate(previous_seq_len * h);
            self.v_norms.truncate(previous_seq_len * h);
        }
        result
    }

    /// Dequantize all stored data into f32 tensors.
    ///
    /// Uses native Metal for supported geometry and the portable CPU path for
    /// other backends/geometries. CUDA currently uploads CPU reconstructions.
    fn dequantize_all(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        #[cfg(target_os = "macos")]
        if device.is_metal() {
            return self.dequantize_all_metal(device);
        }
        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        if device.is_cuda() {
            return self.dequantize_all_cuda(device);
        }
        self.dequantize_all_cpu(device)
    }

    /// CPU dequantize path — sequential per-vector dequantization.
    fn dequantize_all_cpu(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        let h = self.num_heads;
        let total = self.seq_len;
        let d = self.config.head_dim;
        let pb = self.config.packed_bytes;

        let mut k_out = vec![0.0f32; h * total * d];
        let mut v_out = vec![0.0f32; h * total * d];
        let mut vec_buf = vec![0.0f32; d];

        for t in 0..total {
            for hi in 0..h {
                let idx = t * h + hi;
                let off = idx * pb;

                self.config.dequantize_into(
                    &self.k_data[off..off + pb],
                    self.k_norms[idx],
                    &mut vec_buf,
                );
                let out_off = (hi * total + t) * d;
                k_out[out_off..out_off + d].copy_from_slice(&vec_buf);

                self.config.dequantize_into(
                    &self.v_data[off..off + pb],
                    self.v_norms[idx],
                    &mut vec_buf,
                );
                v_out[out_off..out_off + d].copy_from_slice(&vec_buf);
            }
        }

        let k_tensor = Tensor::from_vec(k_out, (1, h, total, d), device)?;
        let v_tensor = Tensor::from_vec(v_out, (1, h, total, d), device)?;
        Ok((k_tensor, v_tensor))
    }

    /// Reconstruct stored records directly on Metal, including a one-time
    /// upload when this read-only diagnostic is called on a CPU-owned cache.
    #[cfg(target_os = "macos")]
    fn dequantize_all_metal(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        if self.config.head_dim > crate::turboquant_metal::MAX_METAL_HEAD_DIM {
            return self.dequantize_all_cpu(device);
        }
        let uploaded;
        let state = if let Some(state) = &self.metal {
            ensure!(
                state.k.device().same_device(device),
                "TurboQuant cache belongs to another Metal device"
            );
            state
        } else {
            uploaded = self.upload_packed_cpu(device)?;
            &uploaded
        };
        crate::turboquant_metal::dequantize_kv(
            &state.k.narrow(2, state.start, self.seq_len)?,
            &state.v.narrow(2, state.start, self.seq_len)?,
            &state.signs,
            self.config.bits,
            DType::F32,
        )
        .map_err(Into::into)
    }

    #[cfg(target_os = "macos")]
    fn metal_signs(&self, device: &Device) -> Result<Tensor> {
        Ok(Tensor::from_vec(
            self.config.signs.repeat(self.num_heads),
            (self.num_heads, self.config.head_dim),
            device,
        )?)
    }

    #[cfg(target_os = "macos")]
    fn upload_packed_cpu(&self, device: &Device) -> Result<MetalKvState> {
        let record_bytes = self.config.bytes_per_vector();
        let capacity = metal_cache_capacity(0, self.seq_len)?;
        let size = self.metal_backing_size(capacity)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .context("allocating TurboQuant upload")?;
        bytes.resize(size, 0u8);
        for (side, (data, norms)) in [(&self.k_data, &self.k_norms), (&self.v_data, &self.v_norms)]
            .into_iter()
            .enumerate()
        {
            for head in 0..self.num_heads {
                for token in 0..self.seq_len {
                    let index = token * self.num_heads + head;
                    let offset = index * self.config.packed_bytes;
                    let out = ((side * self.num_heads + head) * capacity + token) * record_bytes;
                    bytes[out..out + self.config.packed_bytes]
                        .copy_from_slice(&data[offset..offset + self.config.packed_bytes]);
                    bytes[out + self.config.packed_bytes..out + record_bytes]
                        .copy_from_slice(&norms[index].to_le_bytes());
                }
            }
        }
        let backing = Tensor::from_vec(bytes, (2, self.num_heads, capacity, record_bytes), device)?;
        Ok(MetalKvState {
            k: backing.narrow(0, 0, 1)?,
            v: backing.narrow(0, 1, 1)?,
            backing,
            signs: self.metal_signs(device)?,
            start: 0,
        })
    }

    #[cfg(target_os = "macos")]
    fn metal_backing_size(&self, capacity: usize) -> Result<usize> {
        let size = capacity
            .checked_mul(self.num_heads)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_mul(self.config.bytes_per_vector()))
            .context("TurboQuant packed capacity overflow")?;
        ensure!(
            size <= u32::MAX as usize,
            "TurboQuant packed capacity exceeds Metal index range"
        );
        Ok(size)
    }

    #[cfg(target_os = "macos")]
    fn allocate_metal_backing(&self, capacity: usize, device: &Device) -> Result<Tensor> {
        let size = self.metal_backing_size(capacity)?;
        // Candle's ordinary zeroed allocator rounds up to a power-of-two
        // bucket. The per-vector norm pushes KV8 just over that boundary,
        // often making the packed allocation as large as dense F16. Uploading
        // initialized bytes uses its exact-sized, residency-managed allocator.
        // This happens only on growth/compaction, never on an in-capacity append.
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .context("allocating exact-size TurboQuant backing")?;
        bytes.resize(size, 0u8);
        Ok(Tensor::from_vec(
            bytes,
            (2, self.num_heads, capacity, self.config.bytes_per_vector()),
            device,
        )?)
    }

    #[cfg(target_os = "macos")]
    fn update_metal(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        output_dtype: DType,
        rotated: bool,
    ) -> Result<(Tensor, Tensor)> {
        use crate::turboquant_metal::{dequantize_kv, quantize_kv_into};
        let next_len = self
            .seq_len
            .checked_add(k.dim(2)?)
            .context("TurboQuant sequence length overflow")?;
        let uploaded = if self.metal.is_none() && self.seq_len > 0 {
            Some(self.upload_packed_cpu(k.device())?)
        } else {
            None
        };
        let previous = self.metal.as_ref().or(uploaded.as_ref());
        if let Some(state) = previous {
            ensure!(
                state.k.device().same_device(k.device()),
                "TurboQuant cache belongs to another Metal device"
            );
        }
        let signs = match previous {
            Some(state) => state.signs.clone(),
            None => self.metal_signs(k.device())?,
        };
        let current_capacity = previous.map_or(0, |state| state.k.dims()[2]);
        let capacity = metal_cache_capacity(current_capacity, next_len)?;
        let reusable = previous.filter(|state| {
            state
                .start
                .checked_add(next_len)
                .is_some_and(|end| end <= current_capacity)
        });
        let (backing, packed_k, packed_v, start) = if let Some(state) = reusable {
            (
                state.backing.clone(),
                state.k.clone(),
                state.v.clone(),
                state.start,
            )
        } else {
            let backing = self.allocate_metal_backing(capacity, k.device())?;
            let packed_k = backing.narrow(0, 0, 1)?;
            let packed_v = backing.narrow(0, 1, 1)?;
            if let Some(state) = previous {
                packed_k.slice_set(
                    &state.k.narrow(2, state.start, self.seq_len)?.contiguous()?,
                    2,
                    0,
                )?;
                packed_v.slice_set(
                    &state.v.narrow(2, state.start, self.seq_len)?.contiguous()?,
                    2,
                    0,
                )?;
            }
            (backing, packed_k, packed_v, 0)
        };
        // Writes touch only unused positions. Failed validation or output
        // allocation leaves the retained prefix and all logical state intact.
        let append_start = start
            .checked_add(self.seq_len)
            .context("TurboQuant append offset overflow")?;
        quantize_kv_into(
            &backing,
            k,
            v,
            &signs,
            self.config.bits,
            output_dtype,
            append_start,
        )?;
        let active_k = packed_k.narrow(2, start, next_len)?;
        let active_v = packed_v.narrow(2, start, next_len)?;
        let result = if rotated {
            crate::turboquant_metal::dequantize_kv_rotated(
                &active_k,
                &active_v,
                &signs,
                self.config.bits,
            )?
        } else {
            dequantize_kv(&active_k, &active_v, &signs, self.config.bits, output_dtype)?
        };
        // Public calls validate before publication. A deferred stage pass
        // owns this tentative state and must roll it back if validation fails.
        self.metal = Some(MetalKvState {
            backing,
            k: packed_k,
            v: packed_v,
            signs,
            start,
        });
        self.seq_len = next_len;
        self.k_data = Vec::new();
        self.v_data = Vec::new();
        self.k_norms = Vec::new();
        self.v_norms = Vec::new();
        Ok(result)
    }

    /// CUDA fallback path.
    ///
    /// Upstream Candle does not ship the historical custom TurboQuant PTX
    /// module. Decode on CPU and construct the result on the CUDA device until
    /// that kernel is maintained by this crate. The default 8-bit profile
    /// already used this path; this also keeps explicit 3-bit builds portable.
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    fn dequantize_all_cuda(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        self.dequantize_all_cpu(device)
    }

    /// Number of token positions currently retained in the cache.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn cpu_allocation_tokens(&self) -> usize {
        if self.num_heads == 0 {
            return 0;
        }
        let packed_per_token = self.num_heads.saturating_mul(self.config.packed_bytes);
        if packed_per_token == 0 {
            return 0;
        }
        let packed_capacity = self
            .k_data
            .capacity()
            .div_ceil(packed_per_token)
            .max(self.v_data.capacity().div_ceil(packed_per_token));
        let norm_capacity = self
            .k_norms
            .capacity()
            .div_ceil(self.num_heads)
            .max(self.v_norms.capacity().div_ceil(self.num_heads));
        packed_capacity.max(norm_capacity)
    }

    fn reserve_cpu_capacity(&mut self, required: usize) -> Result<()> {
        let target = cpu_cache_capacity(self.cpu_allocation_tokens(), required)?;
        let packed_per_token = self
            .num_heads
            .checked_mul(self.config.packed_bytes)
            .context("TurboQuant packed cache size overflow")?;
        let packed_len = target
            .checked_mul(packed_per_token)
            .context("TurboQuant packed cache capacity overflow")?;
        let norm_len = target
            .checked_mul(self.num_heads)
            .context("TurboQuant norm cache capacity overflow")?;
        reserve_cpu_vec(&mut self.k_data, packed_len, "K")?;
        reserve_cpu_vec(&mut self.v_data, packed_len, "V")?;
        reserve_cpu_vec(&mut self.k_norms, norm_len, "K norms")?;
        reserve_cpu_vec(&mut self.v_norms, norm_len, "V norms")?;
        Ok(())
    }

    /// Physical slots in the backing allocation, including spare append
    /// capacity and the prefix still owned by a partially evicted Metal view.
    pub(crate) fn allocation_tokens(&self) -> usize {
        #[cfg(target_os = "macos")]
        if let Some(state) = &self.metal {
            return state.k.dims()[2];
        }
        self.cpu_allocation_tokens()
    }

    /// Evict the oldest `count` token positions from the cache.
    ///
    /// Counts greater than the retained length are ignored.
    pub fn evict_oldest(&mut self, count: usize) {
        if count == 0 || count > self.seq_len {
            return;
        }
        #[cfg(target_os = "macos")]
        if let Some(state) = self.metal.as_mut() {
            state.start += count;
            self.seq_len -= count;
            if self.seq_len == 0 {
                self.metal = None;
            }
            return;
        }
        let vecs = count * self.num_heads;
        let bytes = vecs * self.config.packed_bytes;
        self.k_data.drain(..bytes);
        self.v_data.drain(..bytes);
        self.k_norms.drain(..vecs);
        self.v_norms.drain(..vecs);
        self.seq_len -= count;
    }
}

// --- Walsh-Hadamard Transform ---

/// In-place Walsh-Hadamard Transform (self-inverse up to scale, which is baked in).
fn wht_inplace(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
}

// --- Lloyd-Max scalar quantizer ---

fn quantize_scalar(x: f32, bits: u8) -> u8 {
    match bits {
        3 => quantize_3bit(x),
        4 => BOUNDARIES_4BIT.partition_point(|boundary| x >= *boundary) as u8,
        8 => (x.clamp(-4.0, 4.0) * SCALE_8BIT).round() as i8 as u8 ^ 0x80,
        _ => unreachable!("validated TurboQuant bit width"),
    }
}

fn centroid(bits: u8, index: u8) -> f32 {
    match bits {
        3 => CENTROIDS_3BIT[index as usize],
        4 => CENTROIDS_4BIT[index as usize],
        8 => ((index ^ 0x80) as i8) as f32 / SCALE_8BIT,
        _ => unreachable!("validated TurboQuant bit width"),
    }
}

fn scalar_mask(bits: u8) -> u8 {
    if bits == 8 {
        u8::MAX
    } else {
        (1u8 << bits) - 1
    }
}

/// Quantize a scalar to 3-bit (8 bins) using precomputed Lloyd-Max boundaries.
#[inline]
fn quantize_3bit(x: f32) -> u8 {
    // Binary search through 7 boundaries
    if x < BOUNDARIES_3BIT[3] {
        if x < BOUNDARIES_3BIT[1] {
            if x < BOUNDARIES_3BIT[0] { 0 } else { 1 }
        } else if x < BOUNDARIES_3BIT[2] {
            2
        } else {
            3
        }
    } else if x < BOUNDARIES_3BIT[5] {
        if x < BOUNDARIES_3BIT[4] { 4 } else { 5 }
    } else if x < BOUNDARIES_3BIT[6] {
        6
    } else {
        7
    }
}

// --- Deterministic random sign generation ---

fn generate_signs(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1; // ensure nonzero
    let mut signs = Vec::with_capacity(dim);
    for _ in 0..dim {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        signs.push(if state & 1 == 0 { 1.0 } else { -1.0 });
    }
    signs
}

#[cfg(all(test, target_os = "macos"))]
#[path = "turboquant_cache_tests.rs"]
mod metal_capacity_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_invalid_quantization_geometry() {
        assert!(TurboQuantConfig::try_new(2, 128, 42).is_err());
        assert!(TurboQuantConfig::try_new(3, 63, 42).is_err());
        assert!(TurboQuantConfig::try_new(3, 4, 42).is_err());
        assert!(TurboQuantConfig::try_new(3, 32_768, 42).is_err());
        assert!(validate_inference_bits(8).is_ok());
        for bits in [0, 2, 3, 4, 16, 255] {
            assert!(validate_inference_bits(bits).is_err());
        }
    }

    #[test]
    fn packed_capacity_growth_is_bounded_and_preserves_spare_slots() -> Result<()> {
        assert_eq!(metal_cache_capacity(0, 1)?, 64);
        assert_eq!(metal_cache_capacity(64, 63)?, 64);
        assert_eq!(metal_cache_capacity(64, 65)?, 128);
        assert_eq!(metal_cache_capacity(128, 129)?, 192);
        assert!(metal_cache_capacity(0, usize::MAX).is_err());
        assert!(metal_cache_capacity(usize::MAX - 1, usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn test_wht_self_inverse() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x = original.clone();
        wht_inplace(&mut x);
        // After one WHT, values should differ
        assert!((x[0] - original[0]).abs() > 0.01);
        wht_inplace(&mut x);
        // After two WHTs, should recover original
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "WHT not self-inverse: {} != {}", a, b);
        }
    }

    #[test]
    fn test_pack_unpack_3bit_roundtrip() {
        let config = TurboQuantConfig::new(3, 128, 42);
        let indices: Vec<u8> = (0..128).map(|i| (i % 8) as u8).collect();

        // Pack
        let mut packed = vec![0u8; config.packed_bytes];
        for (i, &idx) in indices.iter().enumerate() {
            let bit_off = i * 3;
            let byte_off = bit_off / 8;
            let shift = bit_off % 8;
            packed[byte_off] |= idx << shift;
            if shift > 5 {
                packed[byte_off + 1] |= idx >> (8 - shift);
            }
        }

        // Unpack
        for (i, expected) in indices.iter().enumerate() {
            let bit_off = i * 3;
            let byte_off = bit_off / 8;
            let shift = bit_off % 8;
            let mut val = (packed[byte_off] >> shift) & 0x07;
            if shift > 5 && byte_off + 1 < packed.len() {
                val |= (packed[byte_off + 1] << (8 - shift)) & 0x07;
            }
            assert_eq!(val, *expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_quantize_dequantize_low_error() {
        let config = TurboQuantConfig::new(3, 128, 42);
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();

        let mut packed = vec![0u8; config.packed_bytes];
        let mut norm = 0.0f32;
        config.quantize_into(&v, &mut packed, &mut norm);

        let mut dv = vec![0.0f32; 128];
        config.dequantize_into(&packed, norm, &mut dv);

        // Relative error should be < 25% at 3-bit
        let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let err: f32 = v
            .iter()
            .zip(dv.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let rel = err / v_norm;
        assert!(rel < 0.25, "relative error {rel:.4} too high");
    }

    #[test]
    fn test_4bit_quantize_dequantize_low_error() {
        let config = TurboQuantConfig::new(4, 128, 42);
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut packed = vec![0u8; config.packed_bytes];
        let mut norm = 0.0;
        config.quantize_into(&v, &mut packed, &mut norm);

        let mut reconstructed = vec![0.0; 128];
        config.dequantize_into(&packed, norm, &mut reconstructed);
        let vector_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let error = v
            .iter()
            .zip(&reconstructed)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            error / vector_norm < 0.12,
            "4-bit relative reconstruction error is too high"
        );
    }

    #[test]
    fn test_8bit_quantize_dequantize_near_lossless() {
        let config = TurboQuantConfig::new(8, 128, 42);
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut packed = vec![0u8; config.packed_bytes];
        let mut norm = 0.0;
        config.quantize_into(&v, &mut packed, &mut norm);

        let mut reconstructed = vec![0.0; 128];
        config.dequantize_into(&packed, norm, &mut reconstructed);
        let vector_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let error = v
            .iter()
            .zip(&reconstructed)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            error / vector_norm < 0.02,
            "8-bit relative reconstruction error is too high"
        );
    }

    #[test]
    fn test_8bit_scalar_zero_and_extremes() {
        for value in [-4.0, -1.0, 0.0, 1.0, 4.0] {
            let restored = centroid(8, quantize_scalar(value, 8));
            assert!((restored - value).abs() <= 1.0 / SCALE_8BIT);
        }
    }

    #[test]
    fn test_inner_product_preservation() {
        let config = TurboQuantConfig::new(3, 128, 42);
        // Use correlated vectors so the inner product is non-trivial
        let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..128)
            .map(|i| (i as f32 * 0.1).sin() + 0.3 * (i as f32 * 0.3).cos())
            .collect();

        let mut packed = vec![0u8; config.packed_bytes];
        let mut norm = 0.0f32;
        config.quantize_into(&a, &mut packed, &mut norm);

        let mut da = vec![0.0f32; 128];
        config.dequantize_into(&packed, norm, &mut da);

        let ip_exact: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let ip_approx: f32 = da.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let rel = (ip_exact - ip_approx).abs() / ip_exact.abs().max(1e-6);
        assert!(
            rel < 0.25,
            "inner product error {rel:.4} too high (exact={ip_exact}, approx={ip_approx})"
        );
    }

    #[test]
    fn test_zero_vector() {
        let config = TurboQuantConfig::new(3, 64, 99);
        let v = vec![0.0f32; 64];
        let mut packed = vec![0u8; config.packed_bytes];
        let mut norm = 0.0f32;
        config.quantize_into(&v, &mut packed, &mut norm);

        let mut dv = vec![0.0f32; 64];
        config.dequantize_into(&packed, norm, &mut dv);
        for x in &dv {
            assert_eq!(*x, 0.0);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let config = TurboQuantConfig::new(3, 128, 42);
        let f16_bytes = 128 * 2; // 256
        let quant_bytes = config.bytes_per_vector(); // 48 + 4 = 52
        let ratio = f16_bytes as f32 / quant_bytes as f32;
        assert!(ratio > 4.5, "compression ratio {ratio:.2} too low");
    }

    #[test]
    fn test_4bit_compression_ratio() {
        let config = TurboQuantConfig::new(4, 128, 42);
        let ratio = (128 * 2) as f32 / config.bytes_per_vector() as f32;
        assert!(ratio > 3.7, "compression ratio {ratio:.2} too low");
    }

    #[test]
    fn test_8bit_compression_ratio() {
        let config = TurboQuantConfig::new(8, 128, 42);
        let ratio = (128 * 2) as f32 / config.bytes_per_vector() as f32;
        assert!(ratio > 1.9, "compression ratio {ratio:.2} too low");
    }

    #[test]
    fn test_quantized_kv_cache_update() {
        let config = Arc::new(TurboQuantConfig::new(3, 64, 42));
        let mut cache = QuantizedKvCache::new(config, 4); // 4 heads
        let device = Device::Cpu;

        // Simulate 3 tokens of KV data: [1, 4, 3, 64]
        let k = Tensor::randn(0.0f32, 1.0, (1, 4, 3, 64), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 4, 3, 64), &device).unwrap();

        let (k_out, v_out) = cache.update(&k, &v).unwrap();
        assert_eq!(k_out.dims(), &[1, 4, 3, 64]);
        assert_eq!(v_out.dims(), &[1, 4, 3, 64]);
        assert_eq!(cache.seq_len(), 3);

        // Add 1 more token
        let k2 = Tensor::randn(0.0f32, 1.0, (1, 4, 1, 64), &device).unwrap();
        let v2 = Tensor::randn(0.0f32, 1.0, (1, 4, 1, 64), &device).unwrap();
        let (k_out, v_out) = cache.update(&k2, &v2).unwrap();
        assert_eq!(k_out.dims(), &[1, 4, 4, 64]); // total seq = 4
        assert_eq!(v_out.dims(), &[1, 4, 4, 64]);
        assert_eq!(cache.seq_len(), 4);
    }

    #[test]
    fn quantized_kv_cache_rejects_invalid_shapes_before_mutation() {
        let config = Arc::new(TurboQuantConfig::new(3, 64, 42));
        let mut cache = QuantizedKvCache::new(config, 2);
        let device = Device::Cpu;

        let wrong_batch = Tensor::zeros((2, 2, 1, 64), DType::F32, &device).unwrap();
        assert!(cache.update(&wrong_batch, &wrong_batch).is_err());
        assert_eq!(cache.seq_len(), 0);

        let k = Tensor::zeros((1, 2, 1, 64), DType::F32, &device).unwrap();
        let wrong_v = Tensor::zeros((1, 2, 2, 64), DType::F32, &device).unwrap();
        assert!(cache.update(&k, &wrong_v).is_err());
        assert_eq!(cache.seq_len(), 0);

        let wrong_heads = Tensor::zeros((1, 3, 1, 64), DType::F32, &device).unwrap();
        assert!(cache.update(&wrong_heads, &wrong_heads).is_err());
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn quantized_kv_cache_rejects_nonfinite_inputs_without_poisoning_state() {
        for bits in [3, 4, 8] {
            let config = Arc::new(TurboQuantConfig::new(bits, 64, 42));
            let mut cache = QuantizedKvCache::new(config, 2);
            let good = Tensor::ones((1, 2, 2, 64), DType::F32, &Device::Cpu).unwrap();
            cache.update(&good, &good).unwrap();
            let original_k = cache.k_data.clone();
            let original_v = cache.v_data.clone();
            let original_k_norms = cache.k_norms.clone();
            let original_v_norms = cache.v_norms.clone();
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
                let mut values = vec![1.0f32; 256];
                // Invalid data comes last, after valid heads/tokens.
                values[255] = invalid;
                let bad = Tensor::from_vec(values, (1, 2, 2, 64), &Device::Cpu).unwrap();
                for (k, v) in [(&bad, &good), (&good, &bad)] {
                    assert!(cache.update(k, v).is_err());
                    assert_eq!(cache.seq_len(), 2);
                    assert_eq!(cache.k_data, original_k);
                    assert_eq!(cache.v_data, original_v);
                    assert_eq!(cache.k_norms, original_k_norms);
                    assert_eq!(cache.v_norms, original_v_norms);
                }
            }
            let (k, v) = cache.update(&good, &good).unwrap();
            assert_eq!(cache.seq_len(), 4);
            for tensor in [k, v] {
                assert!(
                    tensor
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                        .iter()
                        .all(|x| x.is_finite())
                );
            }
        }
    }

    #[test]
    fn quantized_kv_cache_handles_large_finite_norms() {
        let config = Arc::new(TurboQuantConfig::new(8, 64, 42));
        let mut cache = QuantizedKvCache::new(config, 1);
        let values: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin() * 1e20).collect();
        let input = Tensor::from_vec(values.clone(), (1, 1, 1, 64), &Device::Cpu).unwrap();
        let (k, _) = cache.update(&input, &input).unwrap();
        let actual = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(actual.iter().all(|value| value.is_finite()));
        let squared_error: f64 = values
            .iter()
            .zip(&actual)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum();
        let squared_norm: f64 = values.iter().map(|&value| f64::from(value).powi(2)).sum();
        assert!((squared_error / squared_norm).sqrt() < 0.02);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rotated_metal_attention_matches_independent_f64_oracle() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for width in [64, 128] {
            let (kv_heads, q_heads, tokens) = (2, 4, 257);
            let k_values: Vec<f32> = (0..kv_heads * tokens * width)
                .map(|i| (i as f32 * 0.017).sin() * 0.7)
                .collect();
            let v_values: Vec<f32> = (0..k_values.len())
                .map(|i| (i as f32 * 0.031).cos() * 0.4)
                .collect();
            let q_values: Vec<f32> = (0..q_heads * width)
                .map(|i| (i as f32 * 0.023).sin() * 0.8)
                .collect();
            let config = Arc::new(TurboQuantConfig::new(8, width, 711));
            let mut cpu = QuantizedKvCache::new(config.clone(), kv_heads);
            let k = Tensor::from_vec(k_values, (1, kv_heads, tokens, width), &Device::Cpu)?;
            let v = Tensor::from_vec(v_values, (1, kv_heads, tokens, width), &Device::Cpu)?;
            let (ref_k, ref_v) = cpu.update(&k, &v)?;
            let ref_k = ref_k.flatten_all()?.to_vec1::<f32>()?;
            let ref_v = ref_v.flatten_all()?.to_vec1::<f32>()?;
            let mut expected = vec![0f64; q_heads * width];
            for head in 0..q_heads {
                let kv_head = head / (q_heads / kv_heads);
                let mut scores = vec![0f64; tokens];
                for (token, score) in scores.iter_mut().enumerate() {
                    for dim in 0..width {
                        *score += f64::from(q_values[head * width + dim])
                            * f64::from(ref_k[(kv_head * tokens + token) * width + dim]);
                    }
                    *score /= (width as f64).sqrt();
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|score| (score - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                for (token, weight) in weights.iter().enumerate() {
                    for dim in 0..width {
                        expected[head * width + dim] += weight / total
                            * f64::from(ref_v[(kv_head * tokens + token) * width + dim]);
                    }
                }
            }
            let mut gpu = QuantizedKvCache::new(config, kv_heads);
            let (rot_k, rot_v) =
                gpu.update_rotated_native(&k.to_device(&device)?, &v.to_device(&device)?)?;
            let signs = gpu.attention_rotation_signs()?;
            let q = Tensor::from_vec(q_values, (1, q_heads, 1, width), &device)?;
            let rotated_q = crate::turboquant_metal::rotate_attention(&q, &signs, false)?;
            let output = crate::architectures::try_fused_f16_sdpa_native(
                &rotated_q,
                &rot_k,
                &rot_v,
                false,
                1.0 / (width as f32).sqrt(),
            )?
            .context("expected native Metal attention")?;
            let output = crate::turboquant_metal::rotate_attention(&output, &signs, true)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let mut error = 0.0;
            let mut norm = 0.0;
            let mut max_abs = 0f64;
            let mut reference_max = 0f64;
            for (&actual, &expected) in output.iter().zip(&expected) {
                assert!(actual.is_finite());
                assert!((f64::from(actual) - expected).abs() < 0.003);
                max_abs = max_abs.max((f64::from(actual) - expected).abs());
                reference_max = reference_max.max(expected.abs());
                error += (f64::from(actual) - expected).powi(2);
                norm += expected.powi(2);
            }
            let baseline_k = Tensor::from_vec(ref_k, (1, kv_heads, tokens, width), &device)?
                .to_dtype(DType::F16)?;
            let baseline_v = Tensor::from_vec(ref_v, (1, kv_heads, tokens, width), &device)?
                .to_dtype(DType::F16)?;
            let baseline = crate::architectures::try_fused_f16_sdpa_native(
                &q,
                &baseline_k,
                &baseline_v,
                false,
                1.0 / (width as f32).sqrt(),
            )?
            .unwrap()
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            let baseline_error: f64 = baseline
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| (f64::from(a) - b).powi(2))
                .sum();
            // This cancellation-heavy fixture has outputs close to zero. A
            // relative-only bound fails even on ordinary fused F16 attention
            // (measured 0.0057 relative L2 vs this F64 oracle). Use an explicit
            // mixed bound for BOTH paths: 3e-5 absolute + 0.3% relative. Keep
            // the independent reference; one GPU path is not the oracle.
            let absolute_l2 = 3e-5 * (output.len() as f64).sqrt();
            assert!(max_abs <= 3e-5 + 0.003 * reference_max);
            assert!(
                error.sqrt() <= absolute_l2 + 0.003 * norm.sqrt(),
                "rotated max_abs={max_abs}, relative_l2={}",
                (error / norm).sqrt()
            );
            assert!(
                baseline_error.sqrt() <= absolute_l2 + 0.003 * norm.sqrt(),
                "baseline relative_l2={}",
                (baseline_error / norm).sqrt()
            );
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn narrowing_an_existing_f32_cache_rejects_overflow_without_mutation() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for width in [64, 2048] {
            let mut cache = QuantizedKvCache::new(Arc::new(TurboQuantConfig::new(8, width, 42)), 1);
            let mut values = vec![0.0f32; width];
            values[0] = 1e20;
            let large = Tensor::from_vec(values, (1, 1, 1, width), &device)?;
            cache.update(&large, &large)?;
            let small = Tensor::ones((1, 1, 1, width), DType::F32, &device)?;
            assert!(cache.update_native(&small, &small).is_err());
            assert_eq!(cache.seq_len(), 1);
            // The F32 cache is still valid after a rejected F16 conversion.
            let (output, _) = cache.update(&small, &small)?;
            assert_eq!(cache.seq_len(), 2);
            assert!(
                output
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|x| x.is_finite())
            );
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_cache_min_max_width_and_tiny_norm_match_cpu() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for width in [8, 1024] {
            for bits in [3, 4, 8] {
                let config = Arc::new(TurboQuantConfig::new(bits, width, 42));
                let mut cpu = QuantizedKvCache::new(config.clone(), 1);
                let mut metal = QuantizedKvCache::new(config, 1);
                let mut values = vec![0.0f32; 3 * width];
                values[width] = 1e-10; // Exactly the nonzero-quantization boundary.
                values[2 * width] = 8188.0; // Safe native-F16 norm boundary.
                let input = Tensor::from_vec(values, (1, 1, 3, width), &Device::Cpu)?;
                let (expected, _) = cpu.update(&input, &input)?;
                let gpu_input = input.to_device(&device)?;
                let (actual, _) = metal.update(&gpu_input, &gpu_input)?;
                let expected = expected.flatten_all()?.to_vec1::<f32>()?;
                let actual = actual.flatten_all()?.to_vec1::<f32>()?;
                for (index, (&a, &b)) in expected.iter().zip(&actual).enumerate() {
                    let tolerance = if index < 2 * width { 1e-15 } else { 0.02 };
                    assert!(
                        b.is_finite() && (a - b).abs() <= tolerance,
                        "bits={bits} width={width} index={index} expected={a} actual={b}"
                    );
                }
                metal.evict_oldest(3);
                assert_eq!(metal.seq_len(), 0);
                assert_eq!(metal.allocation_tokens(), 0);
                let (native, _) = metal.update_native(&gpu_input, &gpu_input)?;
                assert!(
                    native
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .all(|x| x.is_finite())
                );
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_cache_append_evict_and_invalid_inputs_preserve_state() {
        let Some(device) = crate::test_metal_device() else {
            return;
        };
        for bits in [3, 4, 8] {
            let config = Arc::new(TurboQuantConfig::new(bits, 64, 42));
            let mut cpu = QuantizedKvCache::new(config.clone(), 2);
            let mut gpu = QuantizedKvCache::new(config, 2);
            for seq in [3, 1, 7] {
                let values: Vec<f32> = (0..2 * (seq + 2) * 64)
                    .map(|i| (i as f32 * 0.137).sin())
                    .collect();
                // Nonzero offset and non-contiguous head stride.
                let full = Tensor::from_vec(values, (1, 2, seq + 2, 64), &Device::Cpu).unwrap();
                let k = full.narrow(2, 1, seq).unwrap();
                let v = (&k * -0.7).unwrap();
                let (expected_k, expected_v) = cpu.update(&k, &v).unwrap();
                let gpu_k = full.to_device(&device).unwrap().narrow(2, 1, seq).unwrap();
                let gpu_v = (&full * -0.7)
                    .unwrap()
                    .to_device(&device)
                    .unwrap()
                    .narrow(2, 1, seq)
                    .unwrap();
                let (actual_k, actual_v) = gpu.update(&gpu_k, &gpu_v).unwrap();
                for (expected, actual) in [(expected_k, actual_k), (expected_v, actual_v)] {
                    assert_eq!(actual.dtype(), DType::F32);
                    let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    for (&a, &b) in expected.iter().zip(&actual) {
                        assert!(
                            b.is_finite() && (a - b).abs() < 1e-3,
                            "{bits}-bit mismatch: {a} vs {b}"
                        );
                    }
                }
                cpu.evict_oldest(1);
                gpu.evict_oldest(1);
                assert!(gpu.allocation_tokens() > gpu.seq_len());
            }
            let previous_len = gpu.seq_len();
            let good = Tensor::ones((1, 2, 1, 64), DType::F32, &device).unwrap();
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
                let mut values = vec![1.0f32; 128];
                values[127] = invalid;
                let bad = Tensor::from_vec(values, (1, 2, 1, 64), &device).unwrap();
                for (k, v) in [(&good, &bad), (&bad, &good)] {
                    assert!(gpu.update(k, v).is_err(), "accepted {invalid}");
                    assert_eq!(gpu.seq_len(), previous_len);
                }
            }
            let (native_k, native_v) = gpu.update_native(&good, &good).unwrap();
            assert_eq!(native_k.dtype(), DType::F16);
            assert_eq!(native_v.dtype(), DType::F16);
            assert_eq!(gpu.seq_len(), previous_len + 1);
            assert!(gpu.allocation_tokens() >= gpu.seq_len());
            let cpu_good = good.to_device(&Device::Cpu).unwrap();
            let (expected_k, expected_v) = cpu.update(&cpu_good, &cpu_good).unwrap();
            for (expected, actual) in [(expected_k, &native_k), (expected_v, &native_v)] {
                let expected = expected
                    .to_dtype(DType::F16)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let actual = actual
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                for (&a, &b) in expected.iter().zip(&actual) {
                    assert!(b.is_finite() && (a - b).abs() < 2e-3);
                }
            }
            let q = Tensor::ones((1, 2, 1, 64), DType::F32, &device).unwrap();
            assert!(
                crate::architectures::try_fused_f16_sdpa_native(
                    &q, &native_k, &native_v, false, 0.125
                )
                .unwrap()
                .is_some()
            );
        }
    }

    #[test]
    fn test_quantized_evict_oldest() {
        let config = Arc::new(TurboQuantConfig::new(3, 64, 42));
        let mut cache = QuantizedKvCache::new(config, 4); // 4 heads
        let device = Device::Cpu;

        // Add 10 tokens
        let k = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 64), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 64), &device).unwrap();
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.seq_len(), 10);

        // Evict oldest 3
        cache.evict_oldest(3);
        assert_eq!(cache.seq_len(), 7);

        // Dequantize should produce [1, 4, 7, 64]
        let (k_out, v_out) = cache.dequantize_all(&device).unwrap();
        assert_eq!(k_out.dims(), &[1, 4, 7, 64]);
        assert_eq!(v_out.dims(), &[1, 4, 7, 64]);
    }

    #[test]
    fn cpu_allocation_retains_capacity_after_eviction_and_append() {
        let config = Arc::new(TurboQuantConfig::new(8, 64, 42));
        let mut cache = QuantizedKvCache::new(config, 2);
        let device = Device::Cpu;

        let k = Tensor::zeros((1, 2, 4096, 64), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 2, 4096, 64), DType::F32, &device).unwrap();
        cache.update(&k, &v).unwrap();
        let allocated = cache.allocation_tokens();
        assert!(allocated >= 4096);

        cache.evict_oldest(4000);
        assert_eq!(cache.seq_len(), 96);
        assert_eq!(cache.allocation_tokens(), allocated);

        let k = Tensor::zeros((1, 2, 1, 64), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 2, 1, 64), DType::F32, &device).unwrap();
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.seq_len(), 97);
        assert_eq!(cache.allocation_tokens(), allocated);

        cache.evict_oldest(97);
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.allocation_tokens(), allocated);
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.seq_len(), 1);
        assert_eq!(cache.allocation_tokens(), allocated);
    }

    #[test]
    fn cpu_capacity_growth_and_partial_reservations_stay_accounted() -> Result<()> {
        assert_eq!(cpu_cache_capacity(0, 1)?, 64);
        assert_eq!(cpu_cache_capacity(64, 65)?, 128);
        assert_eq!(cpu_cache_capacity(128, 129)?, 192);
        assert!(cpu_cache_capacity(0, usize::MAX).is_err());
        let mut cache = QuantizedKvCache::new(Arc::new(TurboQuantConfig::new(8, 64, 42)), 2);
        // Model a reservation that succeeded for one backing vector before a
        // later allocation failed. Charge the largest retained capacity even
        // though no logical tokens have been published.
        cache.k_data.try_reserve_exact(2 * 64 * 129)?;
        assert_eq!(cache.allocation_tokens(), 129);
        cache.v_norms.try_reserve_exact(2 * 257)?;
        assert_eq!(cache.allocation_tokens(), 257);
        let incoming = Tensor::ones((1, 2, 1, 64), DType::F32, &Device::Cpu)?;
        cache.update(&incoming, &incoming)?;
        assert_eq!(cache.seq_len(), 1);
        assert_eq!(cache.allocation_tokens(), 257);
        Ok(())
    }

    #[test]
    fn test_quantized_kv_cache_accuracy() {
        let config = Arc::new(TurboQuantConfig::new(3, 128, 42));
        let mut cache = QuantizedKvCache::new(config, 2); // 2 heads
        let device = Device::Cpu;

        // Create known data
        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 1, 128), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 2, 1, 128), &device).unwrap();

        let k_orig: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
        let (k_out, _v_out) = cache.update(&k, &v).unwrap();
        let k_recon: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();

        // Check reconstruction error
        let norm: f32 = k_orig.iter().map(|x| x * x).sum::<f32>().sqrt();
        let err: f32 = k_orig
            .iter()
            .zip(k_recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let rel = err / norm;
        assert!(
            rel < 0.25,
            "KV cache reconstruction error {rel:.4} too high"
        );
    }

    /// Verify GPU dequantize (Metal or CUDA) produces same output as CPU path.
    #[test]
    fn test_gpu_dequantize_matches_cpu() {
        let config = Arc::new(TurboQuantConfig::new(3, 128, 42));
        let num_heads = 4;
        let seq_len = 8;

        // Build cache with known data
        let device_cpu = Device::Cpu;
        let mut cache = QuantizedKvCache::new(config, num_heads);
        let k = Tensor::randn(0.0f32, 1.0, (1, num_heads, seq_len, 128), &device_cpu).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, num_heads, seq_len, 128), &device_cpu).unwrap();
        cache.update(&k, &v).unwrap();

        // CPU path
        let (k_cpu, v_cpu) = cache.dequantize_all_cpu(&device_cpu).unwrap();
        let _k_cpu_data: Vec<f32> = k_cpu.flatten_all().unwrap().to_vec1().unwrap();
        let _v_cpu_data: Vec<f32> = v_cpu.flatten_all().unwrap().to_vec1().unwrap();

        // Helper: compare GPU output against CPU reference
        #[allow(unused)]
        fn assert_gpu_matches_cpu(
            k_cpu_data: &[f32],
            v_cpu_data: &[f32],
            k_gpu: Tensor,
            v_gpu: Tensor,
            backend: &str,
        ) {
            let k_gpu_data: Vec<f32> = k_gpu.flatten_all().unwrap().to_vec1().unwrap();
            let v_gpu_data: Vec<f32> = v_gpu.flatten_all().unwrap().to_vec1().unwrap();

            assert_eq!(
                k_cpu_data.len(),
                k_gpu_data.len(),
                "{backend} K length mismatch"
            );
            assert_eq!(
                v_cpu_data.len(),
                v_gpu_data.len(),
                "{backend} V length mismatch"
            );

            let mut max_diff = 0.0f32;
            for (a, b) in k_cpu_data.iter().zip(k_gpu_data.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
            for (a, b) in v_cpu_data.iter().zip(v_gpu_data.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
            assert!(
                max_diff < 1e-3,
                "{backend} dequantize diverges from CPU: max_diff={max_diff}"
            );
        }

        // Metal path (macOS)
        #[cfg(target_os = "macos")]
        {
            let Some(device_metal) = crate::test_metal_device() else {
                return;
            };
            let (k_gpu, v_gpu) = cache
                .dequantize_all_metal(&device_metal)
                .expect("Metal TurboQuant dequantization must succeed once a device is available");
            assert_gpu_matches_cpu(&_k_cpu_data, &_v_cpu_data, k_gpu, v_gpu, "Metal");
        }

        // CUDA path (Linux/Windows)
        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        {
            let Some(device_cuda) = crate::test_cuda_device() else {
                return;
            };
            let (k_gpu, v_gpu) = cache
                .dequantize_all_cuda(&device_cuda)
                .expect("CUDA TurboQuant dequantization must succeed once a device is available");
            assert_gpu_matches_cpu(&_k_cpu_data, &_v_cpu_data, k_gpu, v_gpu, "CUDA");
        }
    }
}
