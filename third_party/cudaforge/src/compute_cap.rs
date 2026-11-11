//! Compute capability detection and management

use crate::error::{Error, Result};
use std::collections::HashMap;
use std::process::Command;

/// GPU architecture specification
///
/// Supports both numeric (80, 90) and string-based (90a, 100a) formats.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuArch {
    /// Base compute capability number (e.g., 90, 100, 120)
    pub base: usize,
    /// Optional suffix for accelerated variants (e.g., "a" for async)
    pub suffix: Option<String>,
}

impl GpuArch {
    /// Create a new GPU architecture from base number
    pub fn new(base: usize) -> Self {
        Self { base, suffix: None }
    }

    /// Create a new GPU architecture with suffix (e.g., 90a, 100a)
    pub fn with_suffix(base: usize, suffix: &str) -> Self {
        Self {
            base,
            suffix: Some(suffix.to_string()),
        }
    }

    /// Parse from string like "90", "90a", "100a", "sm_90a"
    ///
    /// If no suffix is provided (e.g., "90"), auto-suffix is applied for sm_90+.
    /// To explicitly disable the suffix, use the numeric API directly.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().to_lowercase();

        // Strip "sm_" prefix if present
        let s = s.strip_prefix("sm_").unwrap_or(&s);

        // Check for suffix (letters at the end) - support 'a' and 'f' suffixes
        let (num_part, explicit_suffix) = if let Some(stripped) = s.strip_suffix('f') {
            (stripped, Some("f".to_string()))
        } else if let Some(stripped) = s.strip_suffix('a') {
            (stripped, Some("a".to_string()))
        } else {
            (s, None)
        };

        let base = num_part.parse::<usize>().map_err(|_| {
            Error::ComputeCapDetectionFailed(format!("Invalid compute capability: {}", s))
        })?;

        // Normalize (accept both 80 and 8.0 style)
        let base = if base < 20 { base * 10 } else { base };

        // If explicit suffix provided, use it; otherwise auto-suffix for >=90
        if explicit_suffix.is_some() {
            Ok(Self {
                base,
                suffix: explicit_suffix,
            })
        } else {
            Ok(Self::auto_suffix(base))
        }
    }

    /// Create GPU arch with auto-detected suffix for newer architectures
    ///
    /// Suffix selection follows PTX ISA rules:
    /// - SM 120: 'a' suffix — enables arch-specific NVFP4/MXFP4 MMA instructions
    /// - SM 121 (GB10/Spark): 'f' suffix — family-level features only
    /// - SM 90-100/103: 'a' suffix — async/accelerated features
    /// - SM < 90: no suffix
    pub fn auto_suffix(base: usize) -> Self {
        match base {
            120 => Self::with_suffix(base, "a"),
            b if b > 120 => Self::with_suffix(b, "f"),
            b if b >= 90 => Self::with_suffix(b, "a"),
            b => Self::new(b),
        }
    }

    /// Get the nvcc --gpu-architecture string (e.g., "sm_90a", "sm_80")
    pub fn to_nvcc_arch(&self) -> String {
        match &self.suffix {
            Some(s) => format!("sm_{}{}", self.base, s),
            None => format!("sm_{}", self.base),
        }
    }

    /// Get the nvcc -gencode argument (e.g., "-gencode=arch=compute_90a,code=sm_90a")
    ///
    /// This format is preferred for fat binary support and explicit architecture targeting.
    pub fn to_gencode_arg(&self) -> String {
        let compute = self.to_compute_arch();
        let sm = self.to_nvcc_arch();
        format!("-gencode=arch={},code={}", compute, sm)
    }

    /// Get the nvcc virtual architecture string (e.g., "compute_90a", "compute_80")
    pub fn to_compute_arch(&self) -> String {
        match &self.suffix {
            Some(s) => format!("compute_{}{}", self.base, s),
            None => format!("compute_{}", self.base),
        }
    }

    /// Get the nvcc argument that embeds forward-compatible PTX.
    pub fn to_ptx_gencode_arg(&self) -> String {
        let compute = self.to_compute_arch();
        format!("-gencode=arch={0},code={0}", compute)
    }

    /// Get the base compute capability number
    pub fn base(&self) -> usize {
        self.base
    }
}

impl std::fmt::Display for GpuArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.suffix {
            Some(s) => write!(f, "{}{}", self.base, s),
            None => write!(f, "{}", self.base),
        }
    }
}

impl From<usize> for GpuArch {
    fn from(base: usize) -> Self {
        Self::auto_suffix(base)
    }
}

/// Compute capability configuration
#[derive(Debug, Clone, Default)]
pub struct ComputeCapability {
    /// Default compute cap (auto-detected or manually set)
    default_cap: Option<GpuArch>,
    /// Per-file overrides (filename pattern -> compute cap)
    overrides: HashMap<String, GpuArch>,
}

impl ComputeCapability {
    /// Create new compute capability config with auto-detection
    pub fn new() -> Self {
        Self::default()
    }

    /// Set default compute capability (numeric, auto-selects suffix)
    pub fn with_default(mut self, cap: usize) -> Self {
        self.default_cap = Some(GpuArch::auto_suffix(cap));
        self
    }

    /// Set default compute capability with explicit arch string (e.g., "90a", "100a")
    pub fn with_default_arch(mut self, arch: &str) -> Self {
        if let Ok(gpu_arch) = GpuArch::parse(arch) {
            self.default_cap = Some(gpu_arch);
        }
        self
    }

    /// Add compute cap override for files matching pattern (numeric)
    ///
    /// Pattern can be:
    /// - Exact filename: "my_kernel.cu"
    /// - Glob pattern: "sm90_*.cu", "*_hopper.cu"
    pub fn with_override(mut self, pattern: &str, cap: usize) -> Self {
        self.overrides
            .insert(pattern.to_string(), GpuArch::auto_suffix(cap));
        self
    }

    /// Add compute cap override with explicit arch string (e.g., "90a", "100a")
    pub fn with_override_arch(mut self, pattern: &str, arch: &str) -> Self {
        if let Ok(gpu_arch) = GpuArch::parse(arch) {
            self.overrides.insert(pattern.to_string(), gpu_arch);
        }
        self
    }

    /// Get GPU arch for a specific file
    ///
    /// Priority:
    /// 1. Per-file override matching pattern
    /// 2. Default compute cap
    /// 3. Auto-detected from nvidia-smi
    /// 4. CUDA_COMPUTE_CAP environment variable
    pub fn get_for_file(&self, filename: &str) -> Result<GpuArch> {
        self.get_for_file_all(filename)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                Error::ComputeCapDetectionFailed(
                    "No CUDA compute capabilities were configured".to_string(),
                )
            })
    }

    /// Get every GPU architecture for a specific file.
    ///
    /// Explicit per-file and builder defaults remain single-architecture
    /// overrides. Otherwise `CUDA_COMPUTE_CAPS` selects a reproducible
    /// multi-architecture build, with `CUDA_COMPUTE_CAP` and `nvidia-smi`
    /// retained as single-architecture fallbacks.
    pub fn get_for_file_all(&self, filename: &str) -> Result<Vec<GpuArch>> {
        // Check overrides first
        for (pattern, arch) in &self.overrides {
            if matches_pattern(filename, pattern) {
                return Ok(vec![arch.clone()]);
            }
        }

        // Use default if set
        if let Some(arch) = &self.default_cap {
            return Ok(vec![arch.clone()]);
        }

        // Auto-detect
        detect_compute_caps()
    }

    /// Get the default GPU architecture
    pub fn get_default(&self) -> Result<GpuArch> {
        if let Some(arch) = &self.default_cap {
            return Ok(arch.clone());
        }
        detect_compute_cap()
    }

    /// Check if any overrides are configured
    pub fn has_overrides(&self) -> bool {
        !self.overrides.is_empty()
    }
}

/// Detect compute capability from system
///
/// Priority:
/// 1. First architecture in CUDA_COMPUTE_CAPS
/// 2. CUDA_COMPUTE_CAP environment variable (supports "90", "90a", "100a")
/// 3. nvidia-smi query
pub fn detect_compute_cap() -> Result<GpuArch> {
    if let Some(architectures) = configured_compute_caps()? {
        return architectures.into_iter().next().ok_or_else(|| {
            Error::ComputeCapDetectionFailed(
                "CUDA_COMPUTE_CAPS did not contain an architecture".to_string(),
            )
        });
    }

    // Check environment variable first
    if let Ok(cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        return GpuArch::parse(&cap_str);
    }

    // Try nvidia-smi
    detect_from_nvidia_smi()
}

/// Detect one or more compute capabilities for a static CUDA fat binary.
///
/// `CUDA_COMPUTE_CAPS` is a comma-separated list such as `80,89,90a`.
/// Values are normalized in ascending order and duplicates are rejected.
/// Without that variable, the legacy single-architecture detection applies.
pub fn detect_compute_caps() -> Result<Vec<GpuArch>> {
    if let Some(architectures) = configured_compute_caps()? {
        return Ok(architectures);
    }
    Ok(vec![detect_compute_cap()?])
}

fn configured_compute_caps() -> Result<Option<Vec<GpuArch>>> {
    let cuda_value = std::env::var("CUDA_COMPUTE_CAPS").ok();
    let dinf_value = std::env::var("DINF_CUDA_ARCHS").ok();
    parse_configured_compute_caps(cuda_value.as_deref(), dinf_value.as_deref())
}

fn parse_configured_compute_caps(
    cuda_value: Option<&str>,
    dinf_value: Option<&str>,
) -> Result<Option<Vec<GpuArch>>> {
    if let (Some(cuda_value), Some(dinf_value)) = (cuda_value, dinf_value) {
        if cuda_value != dinf_value {
            return Err(Error::ComputeCapDetectionFailed(
                "CUDA_COMPUTE_CAPS and DINF_CUDA_ARCHS disagree".to_string(),
            ));
        }
    }

    cuda_value
        .or(dinf_value)
        .map(parse_compute_caps)
        .transpose()
}

fn parse_compute_caps(value: &str) -> Result<Vec<GpuArch>> {
    let mut architectures = value
        .split(',')
        .map(str::trim)
        .map(|item| {
            if item.is_empty() {
                Err(Error::ComputeCapDetectionFailed(
                    "CUDA_COMPUTE_CAPS contains an empty architecture".to_string(),
                ))
            } else {
                GpuArch::parse(item)
            }
        })
        .collect::<Result<Vec<_>>>()?;

    if architectures.is_empty() {
        return Err(Error::ComputeCapDetectionFailed(
            "CUDA_COMPUTE_CAPS must contain at least one architecture".to_string(),
        ));
    }

    architectures.sort_by(|left, right| {
        (left.base, left.suffix.as_deref()).cmp(&(right.base, right.suffix.as_deref()))
    });
    if architectures.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(Error::ComputeCapDetectionFailed(
            "CUDA_COMPUTE_CAPS contains a duplicate architecture".to_string(),
        ));
    }

    Ok(architectures)
}

/// Stable cache/provenance key for a set of GPU architectures.
pub fn gpu_arch_list_key(architectures: &[GpuArch]) -> String {
    architectures
        .iter()
        .map(GpuArch::to_nvcc_arch)
        .collect::<Vec<_>>()
        .join(",")
}

/// Distinguish single-target fat binaries from upstream's cubin-only cache.
/// Multi-target keys were introduced with PTX fallback and cannot collide
/// with upstream's single-SM keys.
pub(crate) fn fatbin_cache_key(architectures: &[GpuArch]) -> String {
    let key = gpu_arch_list_key(architectures);
    if architectures.len() == 1 {
        format!("{key}+ptx")
    } else {
        key
    }
}

/// Native cubin targets plus a PTX fallback for the oldest architecture.
pub fn fatbin_gencode_args(architectures: &[GpuArch]) -> Vec<String> {
    let mut arguments = architectures
        .iter()
        .map(GpuArch::to_gencode_arg)
        .collect::<Vec<_>>();
    if let Some(oldest) = architectures.first() {
        arguments.push(oldest.to_ptx_gencode_arg());
    }
    arguments
}

/// Detect compute capability using nvidia-smi
fn detect_from_nvidia_smi() -> Result<GpuArch> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv"])
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            parse_nvidia_smi_output(&stdout)
        }
        Ok(output) => Err(Error::ComputeCapDetectionFailed(format!(
            "nvidia-smi failed: {}. \
            If building in Docker, set CUDA_COMPUTE_CAP environment variable (e.g., CUDA_COMPUTE_CAP=90).",
            String::from_utf8_lossy(&output.stderr)
        ))),
        Err(e) => Err(Error::ComputeCapDetectionFailed(format!(
            "Failed to run nvidia-smi: {}. \
            If building in Docker, set CUDA_COMPUTE_CAP environment variable (e.g., CUDA_COMPUTE_CAP=90). \
            GPU is not accessible during 'docker build' - only during 'docker run --gpus all'.",
            e
        ))),
    }
}

/// Parse nvidia-smi output for compute capability
fn parse_nvidia_smi_output(output: &str) -> Result<GpuArch> {
    let line = output.lines().nth(1).ok_or_else(|| {
        Error::ComputeCapDetectionFailed("Unexpected nvidia-smi output".to_string())
    })?;

    let cap = line.trim().parse::<f32>().map_err(|_| {
        Error::ComputeCapDetectionFailed(format!("Failed to parse compute_cap: {}", line))
    })?;

    let base = (cap * 10.0) as usize;
    Ok(GpuArch::auto_suffix(base))
}

/// Match filename against pattern (simple glob matching)
fn matches_pattern(filename: &str, pattern: &str) -> bool {
    // Handle exact match
    if filename == pattern {
        return true;
    }

    // Simple glob matching for * wildcard
    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();

        if parts.len() == 2 {
            let (prefix, suffix) = (parts[0], parts[1]);
            return filename.starts_with(prefix) && filename.ends_with(suffix);
        }

        // Handle single * at start or end
        if let Some(stripped) = pattern.strip_prefix('*') {
            return filename.ends_with(stripped);
        }
        if let Some(stripped) = pattern.strip_suffix('*') {
            return filename.starts_with(stripped);
        }
    }

    false
}

/// Get GPU architecture string for nvcc (e.g., "sm_90a" or "sm_80")
///
/// This is a convenience function. For more control, use GpuArch directly.
pub fn get_gpu_arch_string(compute_cap: usize) -> String {
    GpuArch::auto_suffix(compute_cap).to_nvcc_arch()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_pattern() {
        assert!(matches_pattern("kernel.cu", "kernel.cu"));
        assert!(matches_pattern("sm90_kernel.cu", "sm90_*.cu"));
        assert!(matches_pattern("kernel_hopper.cu", "*_hopper.cu"));
        assert!(matches_pattern("prefix_middle_suffix.cu", "prefix_*.cu"));
        assert!(!matches_pattern("other.cu", "sm90_*.cu"));
    }

    #[test]
    fn test_gpu_arch_string() {
        assert_eq!(get_gpu_arch_string(80), "sm_80");
        assert_eq!(get_gpu_arch_string(90), "sm_90a");
        assert_eq!(get_gpu_arch_string(100), "sm_100a");
        assert_eq!(get_gpu_arch_string(120), "sm_120a");
        assert_eq!(get_gpu_arch_string(121), "sm_121f");
    }

    #[test]
    fn test_gpu_arch_parse() {
        let arch = GpuArch::parse("90a").unwrap();
        assert_eq!(arch.base, 90);
        assert_eq!(arch.suffix, Some("a".to_string()));
        assert_eq!(arch.to_nvcc_arch(), "sm_90a");

        let arch = GpuArch::parse("100a").unwrap();
        assert_eq!(arch.base, 100);
        assert_eq!(arch.to_nvcc_arch(), "sm_100a");

        let arch = GpuArch::parse("sm_120a").unwrap();
        assert_eq!(arch.base, 120);
        assert_eq!(arch.to_nvcc_arch(), "sm_120a");

        let arch = GpuArch::parse("80").unwrap();
        assert_eq!(arch.base, 80);
        assert_eq!(arch.suffix, None);
        assert_eq!(arch.to_nvcc_arch(), "sm_80");
    }

    #[test]
    fn test_gpu_arch_auto_suffix() {
        assert_eq!(GpuArch::auto_suffix(80).to_nvcc_arch(), "sm_80");
        assert_eq!(GpuArch::auto_suffix(89).to_nvcc_arch(), "sm_89");
        assert_eq!(GpuArch::auto_suffix(90).to_nvcc_arch(), "sm_90a");
        assert_eq!(GpuArch::auto_suffix(100).to_nvcc_arch(), "sm_100a");
        assert_eq!(GpuArch::auto_suffix(120).to_nvcc_arch(), "sm_120a");
        assert_eq!(GpuArch::auto_suffix(121).to_nvcc_arch(), "sm_121f");
    }

    #[test]
    fn test_gpu_arch_gencode() {
        // Pre-Hopper architectures (no suffix)
        assert_eq!(
            GpuArch::auto_suffix(75).to_gencode_arg(),
            "-gencode=arch=compute_75,code=sm_75"
        );
        assert_eq!(
            GpuArch::auto_suffix(80).to_gencode_arg(),
            "-gencode=arch=compute_80,code=sm_80"
        );
        assert_eq!(
            GpuArch::auto_suffix(89).to_gencode_arg(),
            "-gencode=arch=compute_89,code=sm_89"
        );

        // Hopper/Blackwell architectures (with 'a' suffix)
        assert_eq!(
            GpuArch::auto_suffix(90).to_gencode_arg(),
            "-gencode=arch=compute_90a,code=sm_90a"
        );
        assert_eq!(
            GpuArch::auto_suffix(100).to_gencode_arg(),
            "-gencode=arch=compute_100a,code=sm_100a"
        );

        // SM120 (RTX 5090/B200) — 'a' suffix for arch-specific NVFP4/MXFP4 MMA
        assert_eq!(
            GpuArch::auto_suffix(120).to_gencode_arg(),
            "-gencode=arch=compute_120a,code=sm_120a"
        );

        // SM121 (GB10/Spark) — 'f' suffix for family features only (no NVFP4 hardware)
        assert_eq!(
            GpuArch::auto_suffix(121).to_gencode_arg(),
            "-gencode=arch=compute_121f,code=sm_121f"
        );
    }

    #[test]
    fn test_compute_cap_list_is_normalized_and_has_a_ptx_fallback() {
        let architectures = parse_compute_caps("90a, 80,89").unwrap();
        assert_eq!(gpu_arch_list_key(&architectures), "sm_80,sm_89,sm_90a");
        assert_eq!(
            fatbin_gencode_args(&architectures),
            [
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_89,code=sm_89",
                "-gencode=arch=compute_90a,code=sm_90a",
                "-gencode=arch=compute_80,code=compute_80",
            ]
        );
    }

    #[test]
    fn test_compute_cap_list_rejects_empty_and_duplicate_values() {
        assert!(parse_compute_caps("").is_err());
        assert!(parse_compute_caps("80,,89").is_err());
        assert!(parse_compute_caps("80,sm_80").is_err());
    }

    #[test]
    fn test_dinf_arch_alias_must_match_lower_level_configuration() {
        let architectures = parse_configured_compute_caps(None, Some("89,80"))
            .unwrap()
            .unwrap();
        assert_eq!(gpu_arch_list_key(&architectures), "sm_80,sm_89");
        assert!(parse_configured_compute_caps(Some("80,89"), Some("80,89")).is_ok());
        assert!(parse_configured_compute_caps(Some("80"), Some("80,89")).is_err());
    }
}
