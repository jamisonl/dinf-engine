use anyhow::{Context, Result, ensure};

const MAX_CONV_KERNEL: usize = 8;
#[cfg(feature = "cuda")]
pub(super) const CUDA_WARP_SIZE: usize = 32;
const SUPPORTED_HEAD_DIMS: [usize; 3] = [32, 64, 128];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DeltaNetGeometry {
    pub(super) seq_len: usize,
    pub(super) conv_dim: usize,
    pub(super) kernel_size: usize,
    pub(super) n_k_heads: usize,
    pub(super) n_v_heads: usize,
    pub(super) head_dim: usize,
    pub(super) conv_elements: usize,
    pub(super) conv_state_elements: usize,
    pub(super) recurrent_output_elements: usize,
    pub(super) recurrent_state_elements: usize,
}

impl DeltaNetGeometry {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        seq_len: usize,
        conv_dim: usize,
        kernel_size: usize,
        n_k_heads: usize,
        n_v_heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Result<Self> {
        ensure!(
            seq_len > 0,
            "Qwen3.5 CUDA DeltaNet requires at least one token"
        );
        ensure!(
            (1..=MAX_CONV_KERNEL).contains(&kernel_size),
            "Qwen3.5 CUDA DeltaNet supports convolution widths 1 through {MAX_CONV_KERNEL}"
        );
        ensure!(n_k_heads > 0, "Qwen3.5 CUDA DeltaNet requires key heads");
        ensure!(n_v_heads > 0, "Qwen3.5 CUDA DeltaNet requires value heads");
        ensure!(
            n_v_heads.is_multiple_of(n_k_heads),
            "Qwen3.5 CUDA DeltaNet value heads must divide evenly by key heads"
        );
        ensure!(
            key_dim == value_dim && SUPPORTED_HEAD_DIMS.contains(&key_dim),
            "Qwen3.5 CUDA DeltaNet supports square 32, 64, or 128 element heads"
        );

        let total_key_dim = checked_mul(n_k_heads, key_dim, "total key width")?;
        let total_value_dim = checked_mul(n_v_heads, value_dim, "total value width")?;
        let expected_conv_dim = checked_add(
            checked_mul(total_key_dim, 2, "Q/K convolution width")?,
            total_value_dim,
            "Q/K/V convolution width",
        )?;
        ensure!(
            conv_dim == expected_conv_dim,
            "Qwen3.5 CUDA DeltaNet convolution width is {conv_dim}, expected {expected_conv_dim}"
        );

        let conv_elements = checked_mul(seq_len, conv_dim, "convolution output")?;
        let conv_state_elements = checked_mul(conv_dim, kernel_size, "convolution state")?;
        let recurrent_output_elements = checked_mul(seq_len, total_value_dim, "DeltaNet output")?;
        let recurrent_state_elements = checked_mul(
            checked_mul(n_v_heads, key_dim, "DeltaNet state heads")?,
            value_dim,
            "DeltaNet state",
        )?;
        let conv_packed_elements = checked_add(
            conv_elements,
            conv_state_elements,
            "packed convolution output",
        )?;
        let recurrent_packed_elements = checked_add(
            recurrent_output_elements,
            recurrent_state_elements,
            "packed DeltaNet output",
        )?;

        for (name, value) in [
            ("sequence length", seq_len),
            ("convolution width", conv_dim),
            ("key heads", n_k_heads),
            ("value heads", n_v_heads),
            ("convolution output", conv_elements),
            ("convolution state", conv_state_elements),
            ("DeltaNet output", recurrent_output_elements),
            ("DeltaNet state", recurrent_state_elements),
            ("packed convolution output", conv_packed_elements),
            ("packed DeltaNet output", recurrent_packed_elements),
        ] {
            ensure!(
                value <= i32::MAX as usize,
                "Qwen3.5 CUDA DeltaNet {name} exceeds the kernel index limit"
            );
        }

        Ok(Self {
            seq_len,
            conv_dim,
            kernel_size,
            n_k_heads,
            n_v_heads,
            head_dim: key_dim,
            conv_elements,
            conv_state_elements,
            recurrent_output_elements,
            recurrent_state_elements,
        })
    }

    pub(super) fn conv_packed_elements(self) -> Result<usize> {
        checked_add(
            self.conv_elements,
            self.conv_state_elements,
            "packed convolution output",
        )
    }

    pub(super) fn recurrent_packed_elements(self) -> Result<usize> {
        checked_add(
            self.recurrent_output_elements,
            self.recurrent_state_elements,
            "packed DeltaNet output",
        )
    }

    pub(super) fn scan_kernel_name(self) -> &'static str {
        match self.head_dim {
            32 => "qwen35_deltanet_scan_f32_32",
            64 => "qwen35_deltanet_scan_f32_64",
            128 => "qwen35_deltanet_scan_f32_128",
            _ => unreachable!("head size was checked when the geometry was built"),
        }
    }
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .with_context(|| format!("Qwen3.5 CUDA DeltaNet {label} overflows usize"))
}

fn checked_add(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_add(rhs)
        .with_context(|| format!("Qwen3.5 CUDA DeltaNet {label} overflows usize"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn supports(
    seq_len: usize,
    conv_dim: usize,
    kernel_size: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> bool {
    DeltaNetGeometry::new(
        seq_len,
        conv_dim,
        kernel_size,
        n_k_heads,
        n_v_heads,
        key_dim,
        value_dim,
    )
    .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_qwen35_9b_layout() -> Result<()> {
        let geometry = DeltaNetGeometry::new(257, 8192, 4, 16, 32, 128, 128)?;
        assert_eq!(geometry.conv_elements, 257 * 8192);
        assert_eq!(geometry.conv_state_elements, 8192 * 4);
        assert_eq!(geometry.recurrent_output_elements, 257 * 4096);
        assert_eq!(geometry.recurrent_state_elements, 32 * 128 * 128);
        assert_eq!(geometry.conv_packed_elements()?, 257 * 8192 + 8192 * 4);
        assert_eq!(
            geometry.recurrent_packed_elements()?,
            257 * 4096 + 32 * 128 * 128
        );
        assert_eq!(geometry.scan_kernel_name(), "qwen35_deltanet_scan_f32_128");
        Ok(())
    }

    #[test]
    fn accepts_qwen36_and_qwen38_27b_layout() -> Result<()> {
        let geometry = DeltaNetGeometry::new(257, 10_240, 4, 16, 48, 128, 128)?;
        assert_eq!(geometry.conv_elements, 257 * 10_240);
        assert_eq!(geometry.conv_state_elements, 10_240 * 4);
        assert_eq!(geometry.recurrent_output_elements, 257 * 6_144);
        assert_eq!(geometry.recurrent_state_elements, 48 * 128 * 128);
        assert_eq!(geometry.scan_kernel_name(), "qwen35_deltanet_scan_f32_128");
        Ok(())
    }

    #[test]
    fn accepts_decode_and_smaller_square_heads() {
        for head_dim in SUPPORTED_HEAD_DIMS {
            let conv_dim = 2 * 2 * head_dim + 4 * head_dim;
            assert!(supports(1, conv_dim, 1, 2, 4, head_dim, head_dim));
        }
    }

    #[test]
    fn rejects_unsafe_or_slow_shapes() {
        assert!(!supports(0, 8192, 4, 16, 32, 128, 128));
        assert!(!supports(1, 8192, 9, 16, 32, 128, 128));
        assert!(!supports(1, 8192, 4, 16, 30, 128, 128));
        assert!(!supports(1, 8192, 4, 16, 32, 64, 128));
        assert!(!supports(1, 8192, 4, 16, 32, 256, 256));
        assert!(!supports(1, 8191, 4, 16, 32, 128, 128));
        assert!(!supports(300_000, 8192, 4, 16, 32, 128, 128));
    }
}
