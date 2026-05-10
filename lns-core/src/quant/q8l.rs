//! Q8_L — 8-bit Logarithmic Number System quantization.
//!
//! # Format
//!
//! Every super-block (default: 256 weights) stores:
//! - scale_global (2 bytes)
//! - scale_local  (4 bytes)
//! - weights      (256 bytes, 1 per byte)
//!
//! Total: 262 bytes for 256 weights ≈ 8.19 bits/weight.

use crate::quant::q4l::{BLOCK_SIZE, DEFAULT_SUPER_BLOCK_BLOCKS, DEFAULT_SUPER_BLOCK_SIZE};
use bytemuck::{Pod, Zeroable};
use half::f16;
use rayon::prelude::*;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Q8LSuperBlock {
    pub scale_global_bits: u16,
    pub scale_local: [u8; 4],
    pub weights: [u8; 256],
}

impl Q8LSuperBlock {
    pub fn scale_global(&self) -> f32 {
        f16::from_bits(self.scale_global_bits).to_f32()
    }

    pub fn get_scale_local(&self, block_idx: usize) -> u8 {
        let byte = block_idx / 2;
        let shift = (block_idx % 2) * 4;
        (self.scale_local[byte] >> shift) & 0xF
    }

    pub fn get_weight(&self, idx: usize) -> (u8, u8) {
        let val = self.weights[idx];
        let sign = (val >> 7) & 1;
        let mag = val & 0x7F;
        (sign, mag)
    }
}

pub fn encode_superblock(weights: &[f32]) -> Q8LSuperBlock {
    assert_eq!(weights.len(), DEFAULT_SUPER_BLOCK_SIZE);

    let max_abs = weights.iter().map(|w| w.abs()).fold(0.0_f32, f32::max);
    let scale_global_f32 = if max_abs == 0.0 { 1.0_f32 } else { max_abs };
    let scale_global_f16 = f16::from_f32(scale_global_f32);
    let scale_global = scale_global_f16.to_f32();

    let mut scale_local_packed = [0u8; 4];
    let mut weight_packed = [0u8; 256];

    for block_idx in 0..DEFAULT_SUPER_BLOCK_BLOCKS {
        let block_start = block_idx * BLOCK_SIZE;
        let block = &weights[block_start..block_start + BLOCK_SIZE];

        let local_max = block.iter().map(|w| w.abs()).fold(0.0_f32, f32::max);
        let sl: u8 = if local_max == 0.0 || scale_global == 0.0 {
            0
        } else {
            let log_ratio = (local_max / scale_global).log2() + 7.0;
            log_ratio.round().clamp(0.0, 15.0) as u8
        };

        let sl_byte = block_idx / 2;
        let sl_shift = (block_idx % 2) * 4;
        scale_local_packed[sl_byte] |= (sl & 0xF) << sl_shift;

        let efektif_scale = scale_global * (2.0_f32).powi(sl as i32 - 7);

        for (w_idx, &w) in block.iter().enumerate() {
            let global_idx = block_start + w_idx;
            let abs_w = w.abs();
            let sign_bit: u8 = u8::from(w < 0.0);

            let m: u8 = if abs_w < f32::MIN_POSITIVE || efektif_scale < f32::MIN_POSITIVE {
                0
            } else {
                // Multiplier 0.125, Offset 15.0
                let log_m = (abs_w / efektif_scale).log2() / 0.125 + 15.0 / 0.125;
                let m_rounded = log_m.round() as i32;
                if m_rounded <= 0 {
                    0
                } else {
                    m_rounded.min(127) as u8
                }
            };

            weight_packed[global_idx] = (sign_bit << 7) | (m & 0x7F);
        }
    }

    Q8LSuperBlock {
        scale_global_bits: scale_global_f16.to_bits(),
        scale_local: scale_local_packed,
        weights: weight_packed,
    }
}

pub fn decode_superblock(block: &Q8LSuperBlock) -> [f32; DEFAULT_SUPER_BLOCK_SIZE] {
    let scale_global = block.scale_global();
    let mut result = [0.0_f32; DEFAULT_SUPER_BLOCK_SIZE];

    for block_idx in 0..DEFAULT_SUPER_BLOCK_BLOCKS {
        let sl = block.get_scale_local(block_idx);
        let efektif_scale = scale_global * (2.0_f32).powi(sl as i32 - 7);

        let block_start = block_idx * BLOCK_SIZE;
        for w_idx in 0..BLOCK_SIZE {
            let global_idx = block_start + w_idx;
            let (sign_bit, m) = block.get_weight(global_idx);

            let is_nonzero = f32::from(m != 0);
            let exp = (m as f32) * 0.125 - 15.0;
            let magnitude = efektif_scale * (2.0_f32).powf(exp);
            let sign = if sign_bit != 0 { -1.0_f32 } else { 1.0_f32 };

            result[global_idx] = magnitude * sign * is_nonzero;
        }
    }
    result
}

pub fn quantize_q8l(weights: &[f32]) -> Vec<Q8LSuperBlock> {
    let n = weights.len();
    let num_full = n / DEFAULT_SUPER_BLOCK_SIZE;
    let remainder = n % DEFAULT_SUPER_BLOCK_SIZE;

    let mut blocks: Vec<Q8LSuperBlock> = weights[..num_full * DEFAULT_SUPER_BLOCK_SIZE]
        .par_chunks(DEFAULT_SUPER_BLOCK_SIZE)
        .map(|chunk| {
            let mut padded = [0.0_f32; DEFAULT_SUPER_BLOCK_SIZE];
            padded[..chunk.len()].copy_from_slice(chunk);
            encode_superblock(&padded)
        })
        .collect();

    if remainder > 0 {
        let mut padded = [0.0_f32; DEFAULT_SUPER_BLOCK_SIZE];
        padded[..remainder].copy_from_slice(&weights[num_full * DEFAULT_SUPER_BLOCK_SIZE..]);
        blocks.push(encode_superblock(&padded));
    }
    blocks
}

pub fn dequantize_q8l(blocks: &[Q8LSuperBlock], num_weights: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(num_weights);
    for block in blocks {
        result.extend_from_slice(&decode_superblock(block));
        if result.len() >= num_weights {
            break;
        }
    }
    result.truncate(num_weights);
    result
}
