//! Q4_L — 4-bit Logarithmic Number System quantization.
//!
//! # Format
//!
//! Every super-block (default: 256 weights = 8 sub-blocks of 32) stores:
//!
//! ```text
//! scale_global (f16 as u16, 2 bytes)  — absorbs outliers at super-block level
//! scale_local  ([u8; 4], 4 bytes)     — 8 × 4-bit per-sub-block fine-tuning, log-encoded
//! weights      ([u8; 128], 128 bytes) — 256 × 4-bit values, 2 packed per byte
//! ```
//!
//! Total: **134 bytes** for 256 weights ≈ **4.19 bits/weight**.
//!
//! # Weight encoding (4-bit)
//!
//! ```text
//! bit 3       → sign  S  (0 = positive, 1 = negative)
//! bits 2..0   → magnitude M ∈ [0, 7]
//!   M = 0           → exact zero  (sparsity short-circuit)
//!   M ∈ [1, 7]      → ±efektif_scale × 2^(M − 7)
//! ```
//!
//! # Effective scale per sub-block
//!
//! ```text
//! efektif_scale = scale_global × 2^(scale_local − 7)
//! ```
//!
//! > **Note**: `efektif_scale` is the Turkish word for "effective scale" and is
//! > used intentionally throughout this module to match the project's design
//! > specification.
//!
//! # Branchless dequantization kernel
//!
//! ```rust,ignore
//! let is_nonzero = (m != 0) as i32;
//! let exp        = m as i32 - 7;
//! let magnitude  = efektif_scale * 2_f32.powi(exp);
//! let result     = magnitude * sign * is_nonzero as f32;
//! ```

use bytemuck::{Pod, Zeroable};
use half::f16;

// ── Constants ────────────────────────────────────────────────────────────────

/// Number of weights per sub-block.
pub const BLOCK_SIZE: usize = 32;

/// Default number of sub-blocks per super-block.
pub const DEFAULT_SUPER_BLOCK_BLOCKS: usize = 8;

/// Default super-block size in weights (`BLOCK_SIZE × DEFAULT_SUPER_BLOCK_BLOCKS`).
pub const DEFAULT_SUPER_BLOCK_SIZE: usize = BLOCK_SIZE * DEFAULT_SUPER_BLOCK_BLOCKS;

// ── Q4LSuperBlock ─────────────────────────────────────────────────────────────

/// A Q4_L super-block: 256 weights compressed into 134 bytes.
///
/// All fields are packed with no padding (`#[repr(C)]`) so that a slice of
/// these structs can be safely reinterpreted from/to `&[u8]` via `bytemuck`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Q4LSuperBlock {
    /// Global scale stored as the raw bits of an `f16`.
    ///
    /// This equals `max(|w|)` for all weights in the super-block (rounded to
    /// the nearest f16 representable value).  A zero super-block stores `1.0`.
    pub scale_global_bits: u16,

    /// Eight 4-bit local-scale values packed two-per-byte (little-endian nibble
    /// order: nibble 0 in the low 4 bits of byte 0, nibble 1 in the high 4
    /// bits of byte 0, …).
    ///
    /// `efektif_scale[i] = scale_global × 2^(scale_local[i] − 7)`
    pub scale_local: [u8; 4],

    /// 256 weight nibbles packed two-per-byte (little-endian nibble order).
    ///
    /// Each nibble: `[sign(1) | magnitude(3)]`
    pub weights: [u8; 128],
}

impl Q4LSuperBlock {
    /// Return `scale_global` as `f32`.
    #[inline]
    pub fn scale_global(&self) -> f32 {
        f16::from_bits(self.scale_global_bits).to_f32()
    }

    /// Return the 4-bit local scale for sub-block `block_idx` (0..8).
    #[inline]
    pub fn get_scale_local(&self, block_idx: usize) -> u8 {
        let byte = block_idx / 2;
        let shift = (block_idx % 2) * 4;
        (self.scale_local[byte] >> shift) & 0xF
    }

    /// Return `(sign_bit, magnitude)` for weight at position `idx` (0..256).
    #[inline]
    pub fn get_weight(&self, idx: usize) -> (u8, u8) {
        let byte = idx / 2;
        let shift = (idx % 2) * 4;
        let nibble = (self.weights[byte] >> shift) & 0xF;
        let sign = (nibble >> 3) & 1;
        let mag = nibble & 0x7;
        (sign, mag)
    }
}

// ── Encoding ─────────────────────────────────────────────────────────────────

/// Encode exactly `DEFAULT_SUPER_BLOCK_SIZE` (256) `f32` weights into one
/// [`Q4LSuperBlock`].
///
/// # Panics
///
/// Panics if `weights.len() != DEFAULT_SUPER_BLOCK_SIZE`.
pub fn encode_superblock(weights: &[f32]) -> Q4LSuperBlock {
    assert_eq!(
        weights.len(),
        DEFAULT_SUPER_BLOCK_SIZE,
        "encode_superblock: expected {DEFAULT_SUPER_BLOCK_SIZE} weights, got {}",
        weights.len()
    );

    // ── Global scale: maximum absolute value across the whole super-block ──
    let max_abs = weights
        .iter()
        .map(|w| w.abs())
        .fold(0.0_f32, f32::max);

    // Avoid a zero scale (would make all weights unrepresentable).
    let scale_global_f32 = if max_abs == 0.0 { 1.0_f32 } else { max_abs };

    // Round-trip through f16 so decode sees the same value.
    let scale_global_f16 = f16::from_f32(scale_global_f32);
    let scale_global = scale_global_f16.to_f32();

    let mut scale_local_packed = [0u8; 4];
    let mut weight_packed = [0u8; 128];

    for block_idx in 0..DEFAULT_SUPER_BLOCK_BLOCKS {
        let block_start = block_idx * BLOCK_SIZE;
        let block = &weights[block_start..block_start + BLOCK_SIZE];

        // ── Local scale: maximum absolute value in this sub-block ──────────
        let local_max = block
            .iter()
            .map(|w| w.abs())
            .fold(0.0_f32, f32::max);

        // scale_local (4-bit [0,15]):
        //   efektif_scale = scale_global × 2^(sl − 7)
        //   → sl = round(log2(local_max / scale_global) + 7)
        let sl: u8 = if local_max == 0.0 || scale_global == 0.0 {
            0
        } else {
            let log_ratio = (local_max / scale_global).log2() + 7.0;
            log_ratio.round().clamp(0.0, 15.0) as u8
        };

        // Pack into nibble (little-endian: low nibble first).
        let sl_byte = block_idx / 2;
        let sl_shift = (block_idx % 2) * 4;
        scale_local_packed[sl_byte] |= (sl & 0xF) << sl_shift;

        // Effective scale for this sub-block.
        let efektif_scale = scale_global * (2.0_f32).powi(sl as i32 - 7);

        // ── Encode each weight ─────────────────────────────────────────────
        for (w_idx, &w) in block.iter().enumerate() {
            let global_idx = block_start + w_idx;
            let abs_w = w.abs();
            let sign_bit: u8 = u8::from(w < 0.0);

            // Compute magnitude M: round(log2(|w| / efektif_scale) + 7)
            // M = 0 → exact zero; M ∈ [1,7] → non-zero.
            let m: u8 = if abs_w < f32::MIN_POSITIVE || efektif_scale < f32::MIN_POSITIVE {
                0
            } else {
                let log_m = (abs_w / efektif_scale).log2() + 7.0;
                let m_rounded = log_m.round() as i32;
                if m_rounded <= 0 {
                    0
                } else {
                    m_rounded.min(7) as u8
                }
            };

            let nibble = (sign_bit << 3) | (m & 0x7);
            let w_byte = global_idx / 2;
            let w_shift = (global_idx % 2) * 4;
            weight_packed[w_byte] |= nibble << w_shift;
        }
    }

    Q4LSuperBlock {
        scale_global_bits: scale_global_f16.to_bits(),
        scale_local: scale_local_packed,
        weights: weight_packed,
    }
}

// ── Decoding ──────────────────────────────────────────────────────────────────

/// Decode one [`Q4LSuperBlock`] into `DEFAULT_SUPER_BLOCK_SIZE` (256) `f32`
/// values.
///
/// The inner loop is fully branchless: the `M = 0` zero case is handled by
/// multiplying the result by `is_nonzero` (0 or 1) rather than a conditional
/// branch.
pub fn decode_superblock(block: &Q4LSuperBlock) -> [f32; DEFAULT_SUPER_BLOCK_SIZE] {
    let scale_global = block.scale_global();
    let mut result = [0.0_f32; DEFAULT_SUPER_BLOCK_SIZE];

    for block_idx in 0..DEFAULT_SUPER_BLOCK_BLOCKS {
        let sl = block.get_scale_local(block_idx);
        // efektif_scale = scale_global × 2^(sl − 7)
        let efektif_scale = scale_global * (2.0_f32).powi(sl as i32 - 7);

        let block_start = block_idx * BLOCK_SIZE;
        for w_idx in 0..BLOCK_SIZE {
            let global_idx = block_start + w_idx;
            let (sign_bit, m) = block.get_weight(global_idx);

            // Branchless dequantization (M = 0 → zero via is_nonzero mask).
            let is_nonzero = i32::from(m != 0);
            let exp = m as i32 - 7;
            let magnitude = efektif_scale * (2.0_f32).powi(exp);
            let sign = if sign_bit != 0 { -1.0_f32 } else { 1.0_f32 };

            result[global_idx] = magnitude * sign * is_nonzero as f32;
        }
    }

    result
}

// ── Bulk helpers ──────────────────────────────────────────────────────────────

/// Quantize a flat slice of `f32` weights into a `Vec<Q4LSuperBlock>`.
///
/// If `weights.len()` is not a multiple of `DEFAULT_SUPER_BLOCK_SIZE`, the
/// last super-block is zero-padded.
pub fn quantize_q4l(weights: &[f32]) -> Vec<Q4LSuperBlock> {
    let n = weights.len();
    let num_full = n / DEFAULT_SUPER_BLOCK_SIZE;
    let remainder = n % DEFAULT_SUPER_BLOCK_SIZE;

    let capacity = num_full + usize::from(remainder > 0);
    let mut blocks = Vec::with_capacity(capacity);

    for i in 0..num_full {
        let start = i * DEFAULT_SUPER_BLOCK_SIZE;
        blocks.push(encode_superblock(
            &weights[start..start + DEFAULT_SUPER_BLOCK_SIZE],
        ));
    }

    if remainder > 0 {
        let mut padded = [0.0_f32; DEFAULT_SUPER_BLOCK_SIZE];
        padded[..remainder].copy_from_slice(&weights[num_full * DEFAULT_SUPER_BLOCK_SIZE..]);
        blocks.push(encode_superblock(&padded));
    }

    blocks
}

/// Dequantize `Q4_L` super-blocks back into `f32` weights.
///
/// Returns exactly `num_weights` values (truncates any zero-padding from the
/// last super-block).
pub fn dequantize_q4l(blocks: &[Q4LSuperBlock], num_weights: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(num_weights);

    for block in blocks {
        let decoded = decode_superblock(block);
        result.extend_from_slice(&decoded);
        if result.len() >= num_weights {
            break;
        }
    }

    result.truncate(num_weights);
    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Maximum acceptable absolute reconstruction error for Q4_L.
    ///
    /// With 3-bit magnitude (M ∈ [1,7]) the worst-case rounding error is
    /// roughly a factor of √2 − 1 ≈ 41 % of the smallest step, but in
    /// practice the sub-block local scale tightens this considerably.  We
    /// allow a generous 50 % relative error here as a smoke test.
    fn max_relative_error(original: f32, decoded: f32) -> f32 {
        if original == 0.0 {
            decoded.abs()
        } else {
            ((original - decoded) / original).abs()
        }
    }

    #[test]
    fn test_zero_superblock() {
        let weights = [0.0_f32; DEFAULT_SUPER_BLOCK_SIZE];
        let block = encode_superblock(&weights);
        let decoded = decode_superblock(&block);
        for &v in &decoded {
            assert_eq!(v, 0.0, "zero input must decode to zero");
        }
    }

    #[test]
    fn test_uniform_positive() {
        // All weights equal 1.0 → should encode and decode with negligible error.
        let weights = [1.0_f32; DEFAULT_SUPER_BLOCK_SIZE];
        let block = encode_superblock(&weights);
        let decoded = decode_superblock(&block);
        for &v in &decoded {
            let err = (v - 1.0).abs();
            assert!(err < 0.01, "uniform 1.0 decode error {err} exceeds tolerance");
        }
    }

    #[test]
    fn test_sign_preservation() {
        // Mix of positive and negative weights.
        let weights: Vec<f32> = (0..DEFAULT_SUPER_BLOCK_SIZE)
            .map(|i| if i % 2 == 0 { 0.5_f32 } else { -0.5_f32 })
            .collect();
        let block = encode_superblock(&weights);
        let decoded = decode_superblock(&block);
        for (i, (&orig, &dec)) in weights.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                orig.signum(),
                dec.signum(),
                "sign mismatch at index {i}: orig={orig} dec={dec}"
            );
        }
    }

    #[test]
    fn test_roundtrip_log_normal_distribution() {
        // Simulate a log-normal-ish weight distribution (dense near zero).
        let weights: Vec<f32> = (0..DEFAULT_SUPER_BLOCK_SIZE)
            .map(|i| {
                let t = i as f32 / DEFAULT_SUPER_BLOCK_SIZE as f32;
                // Range [-0.5, 0.5], dense near zero via tanh-like shape.
                (t - 0.5) * 0.4
            })
            .collect();

        let block = encode_superblock(&weights);
        let decoded = decode_superblock(&block);

        let mut max_rel = 0.0_f32;
        for (&orig, &dec) in weights.iter().zip(decoded.iter()) {
            if orig.abs() > 1e-4 {
                max_rel = max_rel.max(max_relative_error(orig, dec));
            }
        }
        assert!(
            max_rel < 0.5,
            "max relative error {max_rel:.3} exceeds 50 % threshold"
        );
    }

    #[test]
    fn test_quantize_dequantize_bulk() {
        // Non-multiple-of-256 length to exercise zero-padding.
        let n = 512 + 100;
        let weights: Vec<f32> = (0..n)
            .map(|i| (i as f32 - (n as f32 / 2.0)) / (n as f32))
            .collect();

        let blocks = quantize_q4l(&weights);
        let decoded = dequantize_q4l(&blocks, n);

        assert_eq!(decoded.len(), n, "decoded length must equal input length");
    }

    #[test]
    fn test_superblock_byte_size() {
        assert_eq!(
            std::mem::size_of::<Q4LSuperBlock>(),
            134,
            "Q4LSuperBlock must be exactly 134 bytes"
        );
    }

    #[test]
    fn test_pod_roundtrip_via_bytes() {
        // Verify bytemuck cast works correctly.
        let weights: Vec<f32> = (0..DEFAULT_SUPER_BLOCK_SIZE)
            .map(|i| (i as f32) / 128.0 - 1.0)
            .collect();
        let block = encode_superblock(&weights);

        // Cast to bytes and back.
        let bytes: &[u8] = bytemuck::bytes_of(&block);
        assert_eq!(bytes.len(), 134);
        let block2: Q4LSuperBlock = *bytemuck::from_bytes(bytes);
        assert_eq!(block, block2);
    }

    #[test]
    fn test_max_magnitude_roundtrip() {
        // The maximum magnitude weight should decode back to scale_global (M=7).
        let mut weights = [0.0_f32; DEFAULT_SUPER_BLOCK_SIZE];
        weights[0] = 1.0;
        let block = encode_superblock(&weights);
        let decoded = decode_superblock(&block);
        let err = (decoded[0] - 1.0).abs();
        assert!(
            err < 0.01,
            "max-magnitude weight decoded with error {err}"
        );
    }
}
