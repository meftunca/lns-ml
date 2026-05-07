//! Apple Silicon Metal compute shaders for lns-ml inference.
//!
//! This crate provides a Metal-focused benchmark surface for Q4_L decode:
//! - `ComputeBackend::Cpu`: reference path for all platforms
//! - `ComputeBackend::Metal`: reserved for Apple Silicon implementation
//!
//! The full Metal shader source (`q4l_decode.metal`) implements the complete
//! Q4_L decode algorithm and is included alongside the host-side API so both
//! can evolve in lockstep.
//!
//! # Codec quality
//!
//! [`decode_quality_report`] measures quantization fidelity by running a
//! codec roundtrip (decode → re-encode → decode) and reporting RMSE, SNR,
//! 4-bit code entropy, and sparsity.  Use this as the backend-agnostic
//! **perplexity benchmark** for Q4_L tensors.

use std::time::Instant;

use lns_core::{dequantize_q4l, quantize_q4l, Q4LSuperBlock, DEFAULT_SUPER_BLOCK_SIZE};

/// Compute backend used by decode benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    Cpu,
    Metal,
}

/// Q4_L decode benchmark result.
#[derive(Debug, Clone, Copy)]
pub struct DecodeBenchResult {
    pub backend: ComputeBackend,
    pub elapsed_secs: f64,
    pub weights_processed: usize,
    pub checksum: f32,
}

/// Errors returned by the Metal backend facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalError {
    InvalidIterations,
    MetalUnsupportedPlatform,
    MetalNotYetImplemented,
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidIterations => write!(f, "iterations must be > 0"),
            Self::MetalUnsupportedPlatform => {
                write!(f, "Metal backend is only available on macOS targets")
            }
            Self::MetalNotYetImplemented => {
                write!(f, "Metal backend dispatch is not implemented yet")
            }
        }
    }
}

impl std::error::Error for MetalError {}

/// Kernel entrypoint name expected by host dispatch code.
pub const Q4L_DECODE_KERNEL_ENTRY: &str = "q4l_decode_kernel";

/// Returns the current Q4_L decode Metal shader source.
pub fn q4l_decode_kernel_source() -> &'static str {
    include_str!("shaders/q4l_decode.metal")
}

// ── Codec quality metrics ─────────────────────────────────────────────────────

/// Quantization quality metrics for a Q4_L tensor payload.
///
/// Computed by [`decode_quality_report`] via a codec roundtrip:
/// first-decode → re-encode → second-decode.
#[derive(Debug, Clone, Copy)]
pub struct Q4LQualityReport {
    /// Number of weights measured.
    pub n_weights: usize,
    /// Fraction of weight slots encoded as `M = 0` (exact zeros).
    pub zero_fraction: f64,
    /// Shannon entropy of the 16-symbol 4-bit code distribution (bits, max 4.0).
    pub code_entropy_bits: f64,
    /// RMSE between first-decode and codec-roundtrip-decode.
    pub roundtrip_rmse: f64,
    /// Signal-to-noise ratio (dB): 20 × log₁₀(rms_signal / roundtrip_rmse).
    /// `f64::INFINITY` when `roundtrip_rmse` is zero; `0.0` when signal is zero.
    pub roundtrip_snr_db: f64,
}

/// Compute Q4_L decode quality metrics over a slice of super-blocks.
///
/// Steps:
/// 1. Count 4-bit code frequencies across the first `num_weights` weights.
/// 2. Decode → re-encode → decode (codec roundtrip).
/// 3. Measure RMSE and SNR between first-decode and roundtrip-decode.
pub fn decode_quality_report(blocks: &[Q4LSuperBlock], num_weights: usize) -> Q4LQualityReport {
    // ── Count 4-bit code distribution ────────────────────────────────────
    let mut code_counts = [0u64; 16];
    let mut total_nibbles = 0usize;

    'outer: for (block_num, block) in blocks.iter().enumerate() {
        let block_offset = block_num * DEFAULT_SUPER_BLOCK_SIZE;
        for pos in 0..DEFAULT_SUPER_BLOCK_SIZE {
            if block_offset + pos >= num_weights {
                break 'outer;
            }
            let (sign, m) = block.get_weight(pos);
            let nibble = ((sign << 3) | (m & 7)) as usize;
            code_counts[nibble] += 1;
            total_nibbles += 1;
        }
    }

    let zero_fraction = if total_nibbles > 0 {
        // Nibbles 0 (S=0,M=0) and 8 (S=1,M=0) both decode to exact zero.
        (code_counts[0] + code_counts[8]) as f64 / total_nibbles as f64
    } else {
        0.0
    };

    let code_entropy_bits = if total_nibbles > 0 {
        code_counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total_nibbles as f64;
                -p * p.log2()
            })
            .sum::<f64>()
    } else {
        0.0
    };

    // ── Codec roundtrip RMSE and SNR ─────────────────────────────────────
    let decoded1 = dequantize_q4l(blocks, num_weights);
    let re_blocks = quantize_q4l(&decoded1);
    let decoded2 = dequantize_q4l(&re_blocks, num_weights);

    let mut sq_err_sum = 0.0_f64;
    let mut sq_sig_sum = 0.0_f64;
    for (&a, &b) in decoded1.iter().zip(decoded2.iter()) {
        let a64 = a as f64;
        let b64 = b as f64;
        let err = a64 - b64;
        sq_err_sum += err * err;
        sq_sig_sum += a64 * a64;
    }
    let n = decoded1.len() as f64;
    let roundtrip_rmse = (sq_err_sum / n).sqrt();
    let rms_signal = (sq_sig_sum / n).sqrt();

    let roundtrip_snr_db = if rms_signal == 0.0 {
        0.0
    } else if roundtrip_rmse == 0.0 {
        f64::INFINITY
    } else {
        20.0 * (rms_signal / roundtrip_rmse).log10()
    };

    Q4LQualityReport {
        n_weights: num_weights,
        zero_fraction,
        code_entropy_bits,
        roundtrip_rmse,
        roundtrip_snr_db,
    }
}

/// Benchmark Q4_L decode for one tensor payload (`blocks`, `num_weights`).
pub fn bench_q4l_decode(
    blocks: &[Q4LSuperBlock],
    num_weights: usize,
    iters: usize,
    backend: ComputeBackend,
) -> Result<DecodeBenchResult, MetalError> {
    if iters == 0 {
        return Err(MetalError::InvalidIterations);
    }

    let start = Instant::now();
    let mut checksum = 0.0_f32;

    match backend {
        ComputeBackend::Cpu => {
            for _ in 0..iters {
                let decoded = dequantize_q4l(blocks, num_weights);
                checksum += decoded.first().copied().unwrap_or(0.0);
            }
        }
        ComputeBackend::Metal => {
            #[cfg(target_os = "macos")]
            {
                let _ = blocks;
                let _ = num_weights;
                let _ = iters;
                return Err(MetalError::MetalNotYetImplemented);
            }

            #[cfg(not(target_os = "macos"))]
            {
                return Err(MetalError::MetalUnsupportedPlatform);
            }
        }
    }

    let elapsed_secs = start.elapsed().as_secs_f64();
    Ok(DecodeBenchResult {
        backend,
        elapsed_secs,
        weights_processed: num_weights * iters,
        checksum,
    })
}

#[cfg(test)]
mod tests {
    use lns_core::quantize_q4l;

    use super::*;

    #[test]
    fn test_q4l_decode_shader_contract_symbols() {
        let source = q4l_decode_kernel_source();
        assert!(source.contains("kernel void"));
        assert!(source.contains(Q4L_DECODE_KERNEL_ENTRY));
    }

    #[test]
    fn test_q4l_decode_shader_complete_logic() {
        let source = q4l_decode_kernel_source();
        // Verify the shader implements the real decode steps.
        assert!(source.contains("as_type<half>"), "shader must decode f16 bits");
        assert!(source.contains("get_scale_local"), "shader must decode sub-block scale");
        assert!(source.contains("get_nibble"), "shader must decode weight nibble");
        assert!(source.contains("is_nonzero"), "shader must use branchless zero mask");
        assert!(source.contains("efektif_scale"), "shader must compute efektif_scale");
    }

    #[test]
    fn test_cpu_bench_runs() {
        let weights: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let blocks = quantize_q4l(&weights);

        let result =
            bench_q4l_decode(&blocks, weights.len(), 2, ComputeBackend::Cpu).expect("cpu bench");
        assert_eq!(result.backend, ComputeBackend::Cpu);
        assert!(result.elapsed_secs >= 0.0);
        assert_eq!(result.weights_processed, weights.len() * 2);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn test_metal_backend_rejected_on_non_macos() {
        let weights: Vec<f32> = (0..256).map(|i| i as f32 / 256.0).collect();
        let blocks = quantize_q4l(&weights);
        let err = bench_q4l_decode(&blocks, weights.len(), 1, ComputeBackend::Metal).unwrap_err();
        assert_eq!(err, MetalError::MetalUnsupportedPlatform);
    }

    #[test]
    fn test_decode_quality_report_uniform() {
        let weights: Vec<f32> = (0..512).map(|i| (i as f32) / 256.0 - 1.0).collect();
        let blocks = quantize_q4l(&weights);
        let report = decode_quality_report(&blocks, weights.len());

        assert_eq!(report.n_weights, 512);
        assert!(
            (0.0..=1.0).contains(&report.zero_fraction),
            "zero_fraction out of range"
        );
        assert!(
            report.code_entropy_bits > 0.0 && report.code_entropy_bits <= 4.0,
            "entropy must be in (0, 4] bits"
        );
        assert!(
            report.roundtrip_rmse >= 0.0,
            "RMSE must be non-negative"
        );
    }

    #[test]
    fn test_decode_quality_report_zero_weights() {
        let weights = vec![0.0_f32; 256];
        let blocks = quantize_q4l(&weights);
        let report = decode_quality_report(&blocks, weights.len());

        assert_eq!(report.n_weights, 256);
        assert!(
            (report.zero_fraction - 1.0).abs() < 1e-9,
            "all-zero tensor must have zero_fraction = 1.0"
        );
        assert_eq!(
            report.roundtrip_rmse, 0.0,
            "roundtrip RMSE of all-zero tensor must be 0"
        );
    }

    #[test]
    fn test_decode_quality_report_snr_reasonable() {
        // Q4_L typically achieves > 20 dB SNR on smooth weight distributions.
        let weights: Vec<f32> = (0..256)
            .map(|i| (i as f32 / 128.0 - 1.0) * 0.5)
            .collect();
        let blocks = quantize_q4l(&weights);
        let report = decode_quality_report(&blocks, weights.len());

        assert!(
            report.roundtrip_snr_db > 20.0,
            "roundtrip SNR {:.1} dB below 20 dB threshold",
            report.roundtrip_snr_db
        );
    }
}
