//! Fixed DeepSeek-Lite normal YaRN CPU policy for the proved quantized/native-cache path.
use crate::qmatmul::QMatMul;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use anyhow::Context;
use anyhow::Result;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use candle_core::DType;
use candle_core::Tensor;
#[path = "native_yarn_compat.rs"]
mod compat;
#[allow(clippy::too_many_arguments)]
pub(super) fn admitted(
    q: &QMatMul,
    kv: &QMatMul,
    dim: usize,
    base: f64,
    factor: f64,
    original: usize,
    native_cache: bool,
) -> bool {
    native_cache
        && dim == 64
        && base == 10000.0
        && factor == 40.0
        && original == 4096
        && super::native_cpu::supported_quant(q)
        && super::native_cpu::supported_quant(kv)
}
pub(super) fn apply(
    x: &Tensor,
    start: usize,
    context: usize,
    enabled: bool,
) -> Result<Option<Tensor>> {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (x, start, context, enabled);
        Ok(None)
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if !enabled || !x.device().is_cpu() || x.dtype() != DType::F32 || x.rank() != 4 {
            return Ok(None);
        }
        let (batch, heads, queries, dim) = x.dims4()?;
        if dim != 64 || !matches!(heads, 1 | 16) || batch == 0 || queries == 0 {
            return Ok(None);
        }
        let end = start
            .checked_add(queries)
            .context("DeepSeek rotary position overflow")?;
        anyhow::ensure!(
            end <= context && end <= i32::MAX as usize,
            "DeepSeek rotary position outside context"
        );
        // Materialize only the input view. Preserve a contiguous view's logical offset.
        let contiguous = x.contiguous()?;
        let (storage, layout) = contiguous.storage_and_layout();
        let candle_core::Storage::Cpu(storage) = &*storage else {
            anyhow::bail!("DeepSeek rotary CPU storage");
        };
        let (begin, finish) = layout
            .contiguous_offsets()
            .context("DeepSeek rotary input span")?;
        let input = storage
            .as_slice::<f32>()?
            .get(begin..finish)
            .context("DeepSeek rotary storage span")?;
        let mut output = vec![0.0f32; input.len()];
        let mut row = vec![0.0f32; heads * 64];
        for b in 0..batch {
            for t in 0..queries {
                for h in 0..heads {
                    let at = ((b * heads + h) * queries + t) * 64;
                    row[h * 64..(h + 1) * 64].copy_from_slice(&input[at..at + 64]);
                }
                let rotated = compat::rotate(&row, heads, start + t)
                    .context("DeepSeek rotary finite input/shape")?;
                for h in 0..heads {
                    let at = ((b * heads + h) * queries + t) * 64;
                    output[at..at + 64].copy_from_slice(&rotated[h * 64..(h + 1) * 64]);
                }
            }
        }
        Ok(Some(Tensor::from_vec(
            output,
            (batch, heads, queries, 64),
            x.device(),
        )?))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    use candle_core::D;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{DType, Device};
    #[test]
    fn admission_declines_dense_other_quant_metadata_and_custom_cache() -> Result<()> {
        let t = Tensor::ones((8, 64), DType::F32, &Device::Cpu)?;
        let q = QMatMul::from_qtensor(QTensor::quantize(&t, GgmlDType::Q8_0)?)?;
        let k = QMatMul::from_qtensor(QTensor::quantize(&t, GgmlDType::Q4_0)?)?;
        assert_eq!(
            admitted(&q, &q, 64, 10000., 40., 4096, true),
            cfg!(all(target_os = "macos", target_arch = "aarch64"))
        );
        assert!(!admitted(&q, &q, 64, 10000., 40., 4096, false));
        assert!(!admitted(&q, &k, 64, 10000., 40., 4096, true));
        assert!(!admitted(
            &QMatMul::Tensor(t),
            &q,
            64,
            10000.,
            40.,
            4096,
            true
        ));
        for (d, b, f, o) in [
            (96, 10000., 40., 4096),
            (64, 1e6, 40., 4096),
            (64, 10000., 1., 4096),
            (64, 10000., 40., 2048),
        ] {
            assert!(!admitted(&q, &q, d, b, f, o, true));
        }
        Ok(())
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn rotary_offset_strided_batch_and_chunks_exact() -> Result<()> {
        let v: Vec<f32> = (0..2 * 16 * 7 * 66)
            .map(|i| ((i * 71 % 997) as f32 - 498.) / 257.)
            .collect();
        let backing = Tensor::from_vec(v, (2, 16, 7, 66), &Device::Cpu)?;
        let x = backing.narrow(2, 1, 5)?.narrow(3, 2, 64)?;
        assert!(x.layout().start_offset() > 0);
        assert!(!x.is_contiguous());
        let whole = apply(&x, 288, 4096, true)?.unwrap();
        for chunk in [1, 2, 3, 5] {
            let mut out = Vec::new();
            let mut at = 0;
            while at < 5 {
                let n = chunk.min(5 - at);
                out.push(apply(&x.narrow(2, at, n)?, 288 + at, 4096, true)?.unwrap());
                at += n;
            }
            let y = Tensor::cat(&out, 2)?;
            assert_eq!(
                whole
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                y.flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        let c = Tensor::ones((2, 1, 64), DType::F32, &Device::Cpu)?
            .narrow(0, 1, 1)?
            .unsqueeze(0)?;
        assert!(c.layout().start_offset() > 0);
        assert!(apply(&c, 0, 4096, true)?.is_some());
        assert!(apply(&x, 4094, 4096, true).is_err());
        assert!(apply(&x, usize::MAX, 4096, true).is_err());
        assert!(apply(&x, 288, 4096, false)?.is_none());
        assert!(apply(&x.to_dtype(DType::F16)?, 288, 4096, true)?.is_none());
        assert!(apply(&x.narrow(D::Minus1, 0, 32)?, 288, 4096, true)?.is_none());
        Ok(())
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn rotary_small_position_f64_and_long_position_norm() {
        let x: Vec<f32> = (0..16 * 64)
            .map(|i| ((i * 43 % 257) as f32 - 128.) / 128.)
            .collect();
        let low = (64.0f64 * (4096.0 / (32.0 * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * 10000f64.ln()))
        .floor()
        .max(0.0);
        let high = (64.0f64 * (4096.0 / (2.0 * std::f64::consts::PI)).ln() / (2.0 * 10000f64.ln()))
            .ceil()
            .min(63.0);
        for pos in [0, 1] {
            let y = compat::rotate(&x, 16, pos).unwrap();
            let mut max = 0f64;
            for h in 0..16 {
                for p in 0..32 {
                    let ramp = 1.0 - ((p as f64 - low) / (high - low)).clamp(0.0, 1.0);
                    let theta = pos as f64 * 10000f64.powf(-2.0 * p as f64 / 64.0);
                    let a = theta / 40.0 * (1.0 - ramp) + theta * ramp;
                    let (s, c) = a.sin_cos();
                    let i = h * 64 + 2 * p;
                    let u = x[i] as f64;
                    let v = x[i + 1] as f64;
                    max = max
                        .max((y[i] as f64 - (u * c - v * s)).abs())
                        .max((y[i + 1] as f64 - (u * s + v * c)).abs());
                }
            }
            assert!(max < 2e-5, "position{pos}, error{max}");
        }
        for pos in [289, 414, 512, 1600, 3266, 4095] {
            let y = compat::rotate(&x, 16, pos).unwrap();
            for (a, b) in x.chunks_exact(2).zip(y.chunks_exact(2)) {
                let before = f64::from(a[0]).powi(2) + f64::from(a[1]).powi(2);
                let after = f64::from(b[0]).powi(2) + f64::from(b[1]).powi(2);
                assert!((after - before).abs() < 2e-5);
            }
        }
        assert!(compat::rotate(&[f32::NAN; 64], 1, 0).is_none());
        assert!(compat::rotate(&[0.; 63], 1, 0).is_none());
    }
}
