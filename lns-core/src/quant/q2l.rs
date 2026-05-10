//! Q2_L — 2-bit Logarithmic Number System quantization.
//!
//! # Format
//!
//! Every super-block (default: 256 weights) stores:
//! - scale_global (2 bytes)
//! - scale_local  (4 bytes)
//! - weights      (64 bytes, 4 packed per byte)
//!
//! Total: 70 bytes for 256 weights ≈ 2.19 bits/weight.

use crate::quant::q4l::{BLOCK_SIZE, DEFAULT_SUPER_BLOCK_BLOCKS, DEFAULT_SUPER_BLOCK_SIZE};
use bytemuck::{Pod, Zeroable};
use half::f16;
use rayon::prelude::*;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Q2LSuperBlock {
    pub scale_global_bits: u16,
    pub scale_local: [u8; 4],
    pub weights: [u8; 64],
}

impl Q2LSuperBlock {
    pub fn scale_global(&self) -> f32 {
        f16::from_bits(self.scale_global_bits).to_f32()
    }

    pub fn get_scale_local(&self, block_idx: usize) -> u8 {
        let byte = block_idx / 2;
        let shift = (block_idx % 2) * 4;
        (self.scale_local[byte] >> shift) & 0xF
    }

    pub fn get_weight(&self, idx: usize) -> (u8, u8) {
        let byte = idx / 4;
        let shift = (idx % 4) * 2;
        let val = (self.weights[byte] >> shift) & 0x3;
        let sign = (val >> 1) & 1;
        let mag = val & 0x1;
        (sign, mag)
    }
}

pub fn encode_superblock(weights: &[f32]) -> Q2LSuperBlock {
    assert_eq!(weights.len(), DEFAULT_SUPER_BLOCK_SIZE);

    let max_abs = weights.iter().map(|w| w.abs()).fold(0.0_f32, f32::max);
    let scale_global_f32 = if max_abs == 0.0 { 1.0_f32 } else { max_abs };
    let scale_global_f16 = f16::from_f32(scale_global_f32);
    let scale_global = scale_global_f16.to_f32();

    let mut scale_local_packed = [0u8; 4];
    let mut weight_packed = [0u8; 64];

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

            // Q2L: M=0 if zero, M=1 if non-zero.
            let m: u8 = if abs_w < efektif_scale * 0.5 { 0 } else { 1 };

            let val = (sign_bit << 1) | m;
            let w_byte = global_idx / 4;
            let w_shift = (global_idx % 4) * 2;
            weight_packed[w_byte] |= val << w_shift;
        }
    }

    Q2LSuperBlock {
        scale_global_bits: scale_global_f16.to_bits(),
        scale_local: scale_local_packed,
        weights: weight_packed,
    }
}

pub fn decode_superblock(block: &Q2LSuperBlock) -> [f32; DEFAULT_SUPER_BLOCK_SIZE] {
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
            let sign = if sign_bit != 0 { -1.0_f32 } else { 1.0_f32 };

            result[global_idx] = efektif_scale * sign * is_nonzero;
        }
    }
    result
}

pub fn quantize_q2l(weights: &[f32]) -> Vec<Q2LSuperBlock> {
    let n = weights.len();
    let num_full = n / DEFAULT_SUPER_BLOCK_SIZE;
    let remainder = n % DEFAULT_SUPER_BLOCK_SIZE;

    let mut blocks: Vec<Q2LSuperBlock> = weights[..num_full * DEFAULT_SUPER_BLOCK_SIZE]
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

pub fn dequantize_q2l(blocks: &[Q2LSuperBlock], num_weights: usize) -> Vec<f32> {
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
