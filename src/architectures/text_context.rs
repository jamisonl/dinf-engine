//! Explicit text-request bounds and the arithmetic used to create its cache.
//!
//! A prompt chunk does not reveal the end of the request. Callers that know
//! the full text prompt can pass this context; older raw APIs keep their
//! existing arithmetic and cannot silently consume a context-bound cache.

use anyhow::{Result, ensure};

/// The complete prompt end, shared by every text chunk and decode call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextRequestContext {
    prompt_end: usize,
}

impl TextRequestContext {
    /// Create bounds for a nonempty full prompt.
    pub fn new(prompt_end: usize) -> Result<Self> {
        ensure!(prompt_end > 0, "text prompt must contain a token");
        Ok(Self { prompt_end })
    }

    /// Return the absolute end of the full prompt.
    pub fn prompt_end(self) -> usize {
        self.prompt_end
    }

    /// Check that a nonempty chunk fits its prefill or decode phase.
    pub fn validate(self, start: usize, rows: usize, prefill: bool) -> Result<()> {
        ensure!(rows > 0, "text chunk must contain a token");
        let end = start
            .checked_add(rows)
            .ok_or_else(|| anyhow::anyhow!("text position overflow"))?;
        if prefill {
            ensure!(
                end <= self.prompt_end,
                "prefill exceeds the complete prompt"
            );
        } else {
            ensure!(start >= self.prompt_end, "decode starts inside the prompt");
        }
        Ok(())
    }
}

/// The row arithmetic that created a causal text cache.
///
/// CPU K projections retain a fixed M4 prefix cutoff and use M1 afterward.
/// Metal QKV retains the complete prompt end to choose its native tail family.
/// Decode must not reinterpret earlier rows as a new prefill batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArithmeticKind {
    PhiCpuK,
    QwenCpuK,
    QwenMetalQkv,
    DeepCpuQ8,
    DeepCpuK,
    QwenCpuBf16,
}

/// The request-wide arithmetic used to create each row of a causal text cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextArithmeticPolicy {
    kind: ArithmeticKind,
    epoch: u32,
    m4_prefix_end: usize,
    prompt_end: usize,
}

impl TextArithmeticPolicy {
    pub(crate) fn phi_k(context: TextRequestContext) -> Self {
        Self {
            kind: ArithmeticKind::PhiCpuK,
            epoch: 1,
            m4_prefix_end: context.prompt_end / 4 * 4,
            prompt_end: context.prompt_end,
        }
    }

    /// Phi K projection families plus native prefill/decode attention families.
    pub(crate) fn phi_k_native_attention(context: TextRequestContext) -> Self {
        Self {
            epoch: 2,
            ..Self::phi_k(context)
        }
    }

    /// Deep Q8 uses native attention families without an M4 projection cutoff.
    pub(crate) fn deep_cpu_q8(context: TextRequestContext) -> Self {
        Self {
            kind: ArithmeticKind::DeepCpuQ8,
            epoch: 1,
            m4_prefix_end: 0,
            prompt_end: context.prompt_end,
        }
    }

    /// Deep CPU K keeps ordinary M4/M1 rows and native attention row families.
    pub(crate) fn deep_cpu_k(context: TextRequestContext) -> Self {
        Self {
            kind: ArithmeticKind::DeepCpuK,
            epoch: 1,
            m4_prefix_end: context.prompt_end / 4 * 4,
            prompt_end: context.prompt_end,
        }
    }

    /// Qwen BF16 CPU uses native attention families without an M4 projection cutoff.
    pub(crate) fn qwen_cpu_bf16(context: TextRequestContext) -> Self {
        Self {
            kind: ArithmeticKind::QwenCpuBf16,
            epoch: 1,
            m4_prefix_end: 0,
            prompt_end: context.prompt_end,
        }
    }

    /// Native split attention for this absolute row, independent of chunk size.
    /// Before row 256 the padded native cache has fewer than 512 keys. After
    /// that point, only decode and a one-row final native prompt batch split.
    /// Older arithmetic epochs and other backends do not choose this policy.
    pub(crate) fn native_q1_split_at(self, position: usize) -> Option<bool> {
        if !matches!(
            (self.kind, self.epoch),
            (ArithmeticKind::PhiCpuK, 2)
                | (ArithmeticKind::QwenCpuK, 2)
                | (ArithmeticKind::DeepCpuQ8, 1)
                | (ArithmeticKind::DeepCpuK, 1)
                | (ArithmeticKind::QwenCpuBf16, 1)
        ) {
            return None;
        }
        Some(
            position >= 256
                && (position >= self.prompt_end
                    || (self.prompt_end % 8 == 1 && position == self.prompt_end - 1)),
        )
    }

    pub(crate) fn qwen_cpu_k(context: TextRequestContext) -> Self {
        Self {
            kind: ArithmeticKind::QwenCpuK,
            ..Self::phi_k(context)
        }
    }

    /// Qwen K projection families plus native prefill/decode attention families.
    pub(crate) fn qwen_cpu_k_native_attention(context: TextRequestContext) -> Self {
        Self {
            epoch: 2,
            ..Self::qwen_cpu_k(context)
        }
    }

    // Only the macOS Metal path constructs this policy; tests cover its rules on all hosts.
    #[cfg(any(target_os = "macos", test))]
    pub(crate) fn qwen_metal_qkv(context: TextRequestContext) -> Self {
        Self {
            kind: ArithmeticKind::QwenMetalQkv,
            // Epoch 2: prompt rows use the ordinary matrix kernels; only
            // decode rows use the native single-row kernels.
            epoch: 2,
            ..Self::phi_k(context)
        }
    }

    /// Native single-row kernels for decode rows. Prompt rows return `None`
    /// and use the ordinary batched kernels, independent of chunk size.
    pub(crate) fn qwen_metal_rows_at(self, position: usize) -> Option<usize> {
        if self.kind != ArithmeticKind::QwenMetalQkv || position < self.prompt_end {
            return None;
        }
        Some(1)
    }

    pub(crate) fn m4_prefix_end(self) -> usize {
        self.m4_prefix_end
    }

    /// Whether every cached causal row used the desired arithmetic.
    /// CPU K policies compare the fixed M4 cutoff and any active attention
    /// family. Deep Q8 and Qwen BF16 compare attention only; Deep K also
    /// compares M4 rows. Metal QKV compares each prefix,
    /// tail and decode segment. A cold cache admits any
    /// policy; legacy and explicit arithmetic cannot share nonempty state.
    pub fn compatible(old: Option<Self>, new: Option<Self>, cached_rows: usize) -> bool {
        if cached_rows == 0 {
            return true;
        }
        match (old, new) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                if a.kind != b.kind || a.epoch != b.epoch {
                    return false;
                }
                if a.kind != ArithmeticKind::QwenMetalQkv {
                    if !matches!(
                        a.kind,
                        ArithmeticKind::DeepCpuQ8 | ArithmeticKind::QwenCpuBf16
                    ) && cached_rows.min(a.m4_prefix_end) != cached_rows.min(b.m4_prefix_end)
                    {
                        return false;
                    }
                    // Each row policy is constant between these boundaries.
                    // Include both request ends and tail starts: an old decode
                    // row must not become a new unsplit prompt row on reuse.
                    return [
                        0,
                        256,
                        a.m4_prefix_end,
                        b.m4_prefix_end,
                        a.prompt_end / 8 * 8,
                        b.prompt_end / 8 * 8,
                        a.prompt_end,
                        b.prompt_end,
                    ]
                    .into_iter()
                    .filter(|&p| p < cached_rows)
                    .all(|p| a.native_q1_split_at(p) == b.native_q1_split_at(p));
                }
                // Both functions are constant between these boundaries. Check
                // each existing segment start, including zero, without scanning
                // the cached prefix.
                [0, a.prompt_end, b.prompt_end]
                    .into_iter()
                    .filter(|&p| p < cached_rows)
                    .all(|p| a.qwen_metal_rows_at(p) == b.qwen_metal_rows_at(p))
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_policy_is_request_wide_for_every_chunk() -> Result<()> {
        for end in 1..=137 {
            let context = TextRequestContext::new(end)?;
            let policy = TextArithmeticPolicy::phi_k(context);
            let whole: Vec<_> = (0..end).map(|p| p < policy.m4_prefix_end()).collect();
            for size in [1, 5, 7, 32, 128, usize::MAX] {
                let mut result = Vec::new();
                let mut start = 0;
                while start < end {
                    let rows = size.min(end - start);
                    context.validate(start, rows, true)?;
                    result.extend((start..start + rows).map(|p| p < policy.m4_prefix_end()));
                    start += rows;
                }
                assert_eq!(result, whole);
                context.validate(end, 1, false)?;
                assert!(end >= policy.m4_prefix_end());
            }
        }
        Ok(())
    }

    #[test]
    fn cutoff_equality_and_causal_prefix_are_exact() -> Result<()> {
        for a in 1..=33 {
            for b in 1..=33 {
                let aa = TextArithmeticPolicy::phi_k(TextRequestContext::new(a)?);
                let bb = TextArithmeticPolicy::phi_k(TextRequestContext::new(b)?);
                for n in 0..=40 {
                    let rows_equal =
                        (0..n).all(|p| (p < aa.m4_prefix_end) == (p < bb.m4_prefix_end));
                    assert_eq!(
                        TextArithmeticPolicy::compatible(Some(aa), Some(bb), n),
                        rows_equal
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn decoded_prefix_and_short_snapshot_keep_original_cutoff() -> Result<()> {
        let old = TextArithmeticPolicy::phi_k(TextRequestContext::new(7)?);
        let extension = TextArithmeticPolicy::phi_k(TextRequestContext::new(12)?);
        assert!(TextArithmeticPolicy::compatible(
            Some(old),
            Some(extension),
            4
        ));
        assert!(!TextArithmeticPolicy::compatible(
            Some(old),
            Some(extension),
            5
        ));
        // Rows7..11 are old decode M1 rows, not a new M4 prefix.
        assert!(!TextArithmeticPolicy::compatible(
            Some(old),
            Some(extension),
            12
        ));
        let long_request = TextArithmeticPolicy::phi_k(TextRequestContext::new(128)?);
        assert!(TextArithmeticPolicy::compatible(
            Some(long_request),
            Some(long_request),
            113
        ));
        let shorter_request = TextArithmeticPolicy::phi_k(TextRequestContext::new(113)?);
        assert!(!TextArithmeticPolicy::compatible(
            Some(long_request),
            Some(shorter_request),
            113
        ));
        assert!(TextArithmeticPolicy::compatible(Some(old), None, 0));
        assert!(!TextArithmeticPolicy::compatible(Some(old), None, 1));
        assert!(!TextArithmeticPolicy::compatible(
            Some(old),
            Some(TextArithmeticPolicy { epoch: 2, ..old }),
            1
        ));
        Ok(())
    }

    #[test]
    fn kinds_cannot_share_nonempty_cache_and_cpu_rule_stays_exact() -> Result<()> {
        for end in 1..=33 {
            let c = TextRequestContext::new(end)?;
            let phi = TextArithmeticPolicy::phi_k(c);
            let cpu = TextArithmeticPolicy::qwen_cpu_k(c);
            let metal = TextArithmeticPolicy::qwen_metal_qkv(c);
            assert!(TextArithmeticPolicy::compatible(Some(phi), Some(cpu), 0));
            assert!(!TextArithmeticPolicy::compatible(Some(phi), Some(cpu), 1));
            assert!(!TextArithmeticPolicy::compatible(Some(cpu), Some(metal), 1));
            assert!(cpu.qwen_metal_rows_at(0).is_none());
            assert_eq!(phi.m4_prefix_end(), cpu.m4_prefix_end());
        }
        Ok(())
    }

    #[test]
    fn metal_prefix_tail_decode_compatibility_matches_every_row() -> Result<()> {
        for a in 1..=40 {
            for b in 1..=40 {
                let aa = TextArithmeticPolicy::qwen_metal_qkv(TextRequestContext::new(a)?);
                let bb = TextArithmeticPolicy::qwen_metal_qkv(TextRequestContext::new(b)?);
                for n in 0..=48 {
                    let exact =
                        (0..n).all(|p| aa.qwen_metal_rows_at(p) == bb.qwen_metal_rows_at(p));
                    assert_eq!(
                        TextArithmeticPolicy::compatible(Some(aa), Some(bb), n),
                        exact
                    );
                }
                assert_eq!(aa.qwen_metal_rows_at(a), Some(1));
            }
        }
        // Prompt rows are shared; a decode row of a shorter prompt is not a
        // prompt row of a longer one.
        let a = TextArithmeticPolicy::qwen_metal_qkv(TextRequestContext::new(9)?);
        let b = TextArithmeticPolicy::qwen_metal_qkv(TextRequestContext::new(10)?);
        assert!(TextArithmeticPolicy::compatible(Some(a), Some(b), 9));
        assert!(!TextArithmeticPolicy::compatible(Some(a), Some(b), 10));
        Ok(())
    }

    #[test]
    fn context_rejects_wrong_phase_end_and_overflow() -> Result<()> {
        assert!(TextRequestContext::new(0).is_err());
        let c = TextRequestContext::new(7)?;
        assert!(c.validate(6, 2, true).is_err());
        assert!(c.validate(6, 1, false).is_err());
        assert!(c.validate(7, 0, false).is_err());
        assert!(c.validate(usize::MAX, 1, false).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod native_attention_tests {
    use super::*;

    // Independent form: native prefill groups contain eight rows except the
    // last group; only a one-row group uses the split decode operation.
    fn expected_split(prompt_end: usize, position: usize) -> bool {
        if position < 256 {
            return false;
        }
        if position >= prompt_end {
            return true;
        }
        let group_start = position / 8 * 8;
        prompt_end - group_start == 1
    }

    #[test]
    fn native_attention_matches_request_batches_and_chunked_calls() -> Result<()> {
        for end in (1..=530).chain([1537, 1793, 2048, 3051, 3267, 3298, 4096]) {
            let c = TextRequestContext::new(end)?;
            for policy in [
                TextArithmeticPolicy::phi_k_native_attention(c),
                TextArithmeticPolicy::qwen_cpu_k_native_attention(c),
                TextArithmeticPolicy::deep_cpu_q8(c),
                TextArithmeticPolicy::deep_cpu_k(c),
                TextArithmeticPolicy::qwen_cpu_bf16(c),
            ] {
                let expected: Vec<_> = (0..end + 16)
                    .map(|p| Some(expected_split(end, p)))
                    .collect();
                for size in [1, 5, 7, 8, 32, 128, usize::MAX] {
                    let mut actual = Vec::new();
                    let mut start = 0;
                    while start < end {
                        let rows = size.min(end - start);
                        c.validate(start, rows, true)?;
                        actual.extend((start..start + rows).map(|p| policy.native_q1_split_at(p)));
                        start += rows;
                    }
                    for p in end..end + 16 {
                        c.validate(p, 1, false)?;
                        actual.push(policy.native_q1_split_at(p));
                    }
                    assert_eq!(actual, expected, "end={end}, chunk={size}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn finite_boundaries_match_every_cached_row_exhaustively() -> Result<()> {
        // Cross the 256-key threshold and all eight possible native tails.
        // For every pair, compare every prefix, including the first decode rows.
        for a in 1..=288 {
            for b in 1..=288 {
                for kind in 0..5 {
                    let make = |e| {
                        let c = TextRequestContext::new(e).unwrap();
                        match kind {
                            0 => TextArithmeticPolicy::deep_cpu_q8(c),
                            1 => TextArithmeticPolicy::phi_k_native_attention(c),
                            2 => TextArithmeticPolicy::qwen_cpu_k_native_attention(c),
                            3 => TextArithmeticPolicy::deep_cpu_k(c),
                            _ => TextArithmeticPolicy::qwen_cpu_bf16(c),
                        }
                    };
                    let (aa, bb) = (make(a), make(b));
                    let mut exact = true;
                    for n in 0..=304 {
                        assert_eq!(
                            TextArithmeticPolicy::compatible(Some(aa), Some(bb), n),
                            exact,
                            "a={a}, b={b}, cached={n}, kind={kind}"
                        );
                        exact &= expected_split(a, n) == expected_split(b, n);
                        if kind != 0 && kind != 4 {
                            exact &= (n < a / 4 * 4) == (n < b / 4 * 4);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn extended_boundary_sets_match_brute_force_and_avoid_overflow() -> Result<()> {
        let ends = [
            255, 256, 257, 258, 263, 264, 265, 511, 512, 513, 1537, 1793, 2048, 3051, 3267, 3298,
            4095, 4096,
        ];
        for a in ends {
            for b in ends {
                for kind in 0..5 {
                    let make = |e| {
                        let c = TextRequestContext::new(e).unwrap();
                        match kind {
                            0 => TextArithmeticPolicy::deep_cpu_q8(c),
                            1 => TextArithmeticPolicy::phi_k_native_attention(c),
                            2 => TextArithmeticPolicy::qwen_cpu_k_native_attention(c),
                            3 => TextArithmeticPolicy::deep_cpu_k(c),
                            _ => TextArithmeticPolicy::qwen_cpu_bf16(c),
                        }
                    };
                    let (aa, bb) = (make(a), make(b));
                    let mut exact = true;
                    for n in 0..=a.max(b) + 16 {
                        assert_eq!(
                            TextArithmeticPolicy::compatible(Some(aa), Some(bb), n),
                            exact
                        );
                        exact &= expected_split(a, n) == expected_split(b, n);
                        if kind != 0 && kind != 4 {
                            exact &= (n < a / 4 * 4) == (n < b / 4 * 4);
                        }
                    }
                }
            }
        }
        for end in [usize::MAX - 7, usize::MAX - 1, usize::MAX] {
            let c = TextRequestContext::new(end)?;
            let p = TextArithmeticPolicy::deep_cpu_q8(c);
            for row in [0, 255, 256, end - 1, end, usize::MAX] {
                assert_eq!(p.native_q1_split_at(row), Some(expected_split(end, row)));
            }
            assert!(TextArithmeticPolicy::compatible(
                Some(p),
                Some(p),
                usize::MAX
            ));
        }
        Ok(())
    }

    #[test]
    fn attention_epochs_kinds_and_prefix_reuse_are_separate() -> Result<()> {
        let c = TextRequestContext::new(257)?;
        let phi_old = TextArithmeticPolicy::phi_k(c);
        let phi = TextArithmeticPolicy::phi_k_native_attention(c);
        let deep = TextArithmeticPolicy::deep_cpu_q8(c);
        for old in [
            phi_old,
            TextArithmeticPolicy::qwen_cpu_k(c),
            TextArithmeticPolicy::qwen_metal_qkv(c),
        ] {
            assert_eq!(old.native_q1_split_at(256), None);
            assert_eq!(old.native_q1_split_at(300), None);
            assert!(!TextArithmeticPolicy::compatible(Some(old), Some(phi), 1));
            assert!(TextArithmeticPolicy::compatible(Some(old), Some(phi), 0));
        }
        let qwen_old = TextArithmeticPolicy::qwen_cpu_k(c);
        let qwen = TextArithmeticPolicy::qwen_cpu_k_native_attention(c);
        assert_eq!(qwen.epoch, 2);
        assert_eq!(qwen.m4_prefix_end(), qwen_old.m4_prefix_end());
        assert_eq!(qwen.native_q1_split_at(256), Some(true));
        for other in [
            phi_old,
            phi,
            deep,
            qwen_old,
            TextArithmeticPolicy::qwen_metal_qkv(c),
        ] {
            assert!(!TextArithmeticPolicy::compatible(
                Some(other),
                Some(qwen),
                1
            ));
            assert!(!TextArithmeticPolicy::compatible(
                Some(qwen),
                Some(other),
                1
            ));
            assert!(TextArithmeticPolicy::compatible(Some(other), Some(qwen), 0));
        }
        assert_eq!(phi.epoch, 2);
        assert_eq!(deep.epoch, 1);
        assert_eq!(phi.m4_prefix_end(), phi_old.m4_prefix_end());
        assert_eq!(deep.m4_prefix_end(), 0);
        assert!(!TextArithmeticPolicy::compatible(Some(phi), Some(deep), 1));
        assert!(!TextArithmeticPolicy::compatible(Some(deep), None, 1));
        assert!(TextArithmeticPolicy::compatible(Some(deep), None, 0));
        let deep_258 = TextArithmeticPolicy::deep_cpu_q8(TextRequestContext::new(258)?);
        assert!(TextArithmeticPolicy::compatible(
            Some(deep),
            Some(deep_258),
            256
        ));
        assert!(!TextArithmeticPolicy::compatible(
            Some(deep),
            Some(deep_258),
            257
        ));
        // Deep ignores the M4 cutoff, but Phi must compare it as well.
        let c = TextRequestContext::new(12)?;
        let d = TextRequestContext::new(16)?;
        assert!(TextArithmeticPolicy::compatible(
            Some(TextArithmeticPolicy::deep_cpu_q8(c)),
            Some(TextArithmeticPolicy::deep_cpu_q8(d)),
            20
        ));
        assert!(!TextArithmeticPolicy::compatible(
            Some(TextArithmeticPolicy::phi_k_native_attention(c)),
            Some(TextArithmeticPolicy::phi_k_native_attention(d)),
            13
        ));
        Ok(())
    }
}

#[cfg(test)]
mod deep_k_tests {
    use super::*;
    #[test]
    fn deep_k_is_distinct_and_uses_both_row_families() -> Result<()> {
        for e in [1, 7, 8, 9, 255, 256, 257, 258, 512, 2048, 4096, usize::MAX] {
            let c = TextRequestContext::new(e)?;
            let k = TextArithmeticPolicy::deep_cpu_k(c);
            assert_eq!(k.epoch, 1);
            assert_eq!(k.m4_prefix_end(), e / 4 * 4);
            for old in [
                TextArithmeticPolicy::deep_cpu_q8(c),
                TextArithmeticPolicy::phi_k_native_attention(c),
                TextArithmeticPolicy::qwen_cpu_k_native_attention(c),
            ] {
                assert!(TextArithmeticPolicy::compatible(Some(k), Some(old), 0));
                assert!(!TextArithmeticPolicy::compatible(Some(k), Some(old), 1));
                for p in [0, 255, 256, e - 1, e, usize::MAX] {
                    assert_eq!(k.native_q1_split_at(p), old.native_q1_split_at(p));
                }
            }
            assert!(TextArithmeticPolicy::compatible(
                Some(k),
                Some(k),
                usize::MAX
            ));
        }
        let a = TextArithmeticPolicy::deep_cpu_k(TextRequestContext::new(12)?);
        let b = TextArithmeticPolicy::deep_cpu_k(TextRequestContext::new(16)?);
        assert!(TextArithmeticPolicy::compatible(Some(a), Some(b), 12));
        assert!(!TextArithmeticPolicy::compatible(Some(a), Some(b), 13));
        Ok(())
    }
}

#[cfg(test)]
mod qwen_bf16_tests {
    use super::*;

    #[test]
    fn bf16_has_attention_only_and_a_separate_cache_kind() -> Result<()> {
        for end in [1, 7, 8, 9, 255, 256, 257, 258, 512, 2048, 4096, usize::MAX] {
            let c = TextRequestContext::new(end)?;
            let bf16 = TextArithmeticPolicy::qwen_cpu_bf16(c);
            assert_eq!(bf16.kind, ArithmeticKind::QwenCpuBf16);
            assert_eq!(bf16.epoch, 1);
            assert_eq!(bf16.m4_prefix_end(), 0);
            for other in [
                TextArithmeticPolicy::phi_k(c),
                TextArithmeticPolicy::phi_k_native_attention(c),
                TextArithmeticPolicy::qwen_cpu_k(c),
                TextArithmeticPolicy::qwen_cpu_k_native_attention(c),
                TextArithmeticPolicy::qwen_metal_qkv(c),
                TextArithmeticPolicy::deep_cpu_q8(c),
                TextArithmeticPolicy::deep_cpu_k(c),
            ] {
                assert!(TextArithmeticPolicy::compatible(Some(bf16), Some(other), 0));
                assert!(!TextArithmeticPolicy::compatible(
                    Some(bf16),
                    Some(other),
                    1
                ));
                assert!(!TextArithmeticPolicy::compatible(
                    Some(other),
                    Some(bf16),
                    1
                ));
            }
            assert!(!TextArithmeticPolicy::compatible(None, Some(bf16), 1));
            for p in [0, 255, 256, end - 1, end, usize::MAX] {
                assert_eq!(bf16.qwen_metal_rows_at(p), None);
                assert_eq!(
                    bf16.native_q1_split_at(p),
                    TextArithmeticPolicy::deep_cpu_q8(c).native_q1_split_at(p)
                );
            }
            assert!(TextArithmeticPolicy::compatible(
                Some(bf16),
                Some(bf16),
                usize::MAX
            ));
        }
        // Distinct K cutoffs do not matter to this attention-only policy.
        let a = TextArithmeticPolicy::qwen_cpu_bf16(TextRequestContext::new(12)?);
        let b = TextArithmeticPolicy::qwen_cpu_bf16(TextRequestContext::new(16)?);
        assert!(TextArithmeticPolicy::compatible(Some(a), Some(b), 20));
        let a = TextArithmeticPolicy::qwen_cpu_bf16(TextRequestContext::new(257)?);
        let b = TextArithmeticPolicy::qwen_cpu_bf16(TextRequestContext::new(258)?);
        assert!(TextArithmeticPolicy::compatible(Some(a), Some(b), 256));
        assert!(!TextArithmeticPolicy::compatible(Some(a), Some(b), 257));
        Ok(())
    }
}
