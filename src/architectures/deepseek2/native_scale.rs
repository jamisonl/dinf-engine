//! DeepSeek native graph scale order for the fixed macOS ARM CPU contract.
//! Mirrors llama.cpp (MIT), Copyright (c) 2023-2026 The ggml authors.
//! The caller admits only actual macOS ARM CPU quantized weights and native F16 cache.
pub(super) fn for_native_yarn(
    enabled: bool,
    factor: f64,
    stored_log_multiplier: f64,
    qk_dim: usize,
) -> Option<f32> {
    // This is the exact Lite metadata contract captured from the native graph.
    if !enabled || factor != 40.0 || stored_log_multiplier != 0.0707f32 as f64 || qk_dim != 192 {
        return None;
    }
    native_scale(factor as f32, stored_log_multiplier as f32, qk_dim)
}

fn native_scale(factor: f32, stored_log_multiplier: f32, qk_dim: usize) -> Option<f32> {
    if !factor.is_finite()
        || factor <= 1.0
        || !stored_log_multiplier.is_finite()
        || stored_log_multiplier <= 0.0
        || qk_dim == 0
        || qk_dim > 4096
    {
        return None;
    }
    let freq_scale = 1.0f32 / factor;
    let log = (1.0f32 / freq_scale).ln();
    let hparams_log_multiplier = stored_log_multiplier / 0.1f32;
    // DeepSeek's context uses equal mscale and mscale_all_dims, hence ratio1.
    let adjustment = 0.1f32.mul_add(log, 1.0);
    let context_attention_factor = 1.0f32 / adjustment;
    let original_attention_factor = context_attention_factor * adjustment;
    let mscale = original_attention_factor * (0.1f32 * hparams_log_multiplier).mul_add(log, 1.0);
    let value = (mscale * mscale) / (qk_dim as f32).sqrt();
    value.is_finite().then_some(value)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_fixed_deep_metadata_order() {
        assert_eq!(
            native_scale(40.0, 0.0707, 192).unwrap().to_bits(),
            0x3deaf30e
        );
        let old = ((1.0 + (0.0707f32 as f64) * 40.0f64.ln()).powi(2) / 192.0f64.sqrt()) as f32;
        assert_eq!(old.to_bits(), 0x3deaf30c);
    }
    #[test]
    fn unsupported_contract_keeps_old_path() {
        let log = 0.0707f32 as f64;
        assert_eq!(
            for_native_yarn(true, 40.0, log, 192).unwrap().to_bits(),
            0x3deaf30e
        );
        assert!(for_native_yarn(false, 40.0, log, 192).is_none());
        assert!(for_native_yarn(true, 20.0, log, 192).is_none());
        assert!(for_native_yarn(true, 40.0, 0.1, 192).is_none());
        assert!(for_native_yarn(true, 40.0, log, 128).is_none());
        assert!(for_native_yarn(true, f64::NAN, log, 192).is_none());
    }
    #[test]
    fn bad_metadata_declines() {
        for factor in [0.0, 1.0, f32::NAN, f32::INFINITY] {
            assert!(native_scale(factor, 0.0707, 192).is_none());
        }
        assert!(native_scale(40.0, 0.0, 192).is_none());
        assert!(native_scale(40.0, f32::NAN, 192).is_none());
        assert!(native_scale(40.0, 0.0707, 0).is_none());
        assert!(native_scale(40.0, 0.0707, 4097).is_none());
    }
}
