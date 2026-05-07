//! Apple Silicon Metal compute shaders for lns-ml inference.
//!
//! This crate provides a Metal-focused benchmark surface for Q4_L decode:
//! - `ComputeBackend::Cpu`: reference path for all platforms
//! - `ComputeBackend::Metal`: reserved for Apple Silicon implementation
//!
//! The first Metal shader skeleton (`q4l_decode.metal`) is included so the
//! host-side API and kernel contract can evolve in lockstep.

use std::time::Instant;

use lns_core::{dequantize_q4l, Q4LSuperBlock};

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
}
