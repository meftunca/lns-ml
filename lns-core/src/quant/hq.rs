//! HQ tensor payload headers and validation.

use bytemuck::{Pod, Zeroable};
use half::f16;
use rayon::prelude::*;

use crate::quant::q4l::{BLOCK_SIZE, DEFAULT_SUPER_BLOCK_BLOCKS, DEFAULT_SUPER_BLOCK_SIZE};
use crate::{LnsError, QuantType};

pub const HQ_MAGIC: [u8; 4] = *b"LHQ4";
pub const HQ_HEADER_BYTES: u16 = HqTensorPayloadHeader::SERIALIZED_LEN as u16;

pub const FLAG_AWQ_INV_ALPHA: u32 = 1 << 0;
pub const FLAG_SEGMENT_TABLE: u32 = 1 << 1;
pub const FLAG_CUSTOM_CODEBOOK: u32 = 1 << 2;
pub const VALID_HQ_FLAGS: u32 = FLAG_AWQ_INV_ALPHA | FLAG_SEGMENT_TABLE | FLAG_CUSTOM_CODEBOOK;

pub const Q2HQ_32_BLOCK_BYTES: u16 = 80;
pub const Q2HQ_16_BLOCK_BYTES: u16 = 96;
pub const Q4HQ_BLOCK_BYTES: u16 = 144;
pub const Q8HQ_BLOCK_BYTES: u16 = 272;
pub const HQ_SEGMENT_ENTRY_BYTES: usize = 24;
pub const HQ_CANONICAL_BLOCK_DATA_OFF: u64 = 80;

pub const Q4HQ_CODEBOOK_ID_NF4_Z: u8 = 0;
pub const Q4HQ_ZERO_CODE: u8 = 7;
pub const Q4HQ_NF4_Z_CODEBOOK: [f32; 16] = [
    -1.0000000, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0.0000000,
    0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0000000,
];
const Q4HQ_NF4_Z_RMS: f32 = 0.35355338;
const Q4HQ_SCALE_SEARCH_STEPS: usize = 32;
const Q4HQ_REFINE_CANDIDATES: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Q4HQSuperBlock {
    pub scale_bits: [u16; DEFAULT_SUPER_BLOCK_BLOCKS],
    pub qs: [u8; 128],
}

impl Q4HQSuperBlock {
    pub const SERIALIZED_LEN: usize = Q4HQ_BLOCK_BYTES as usize;

    #[inline]
    pub fn scale(&self, block_idx: usize) -> f32 {
        f16::from_bits(self.scale_bits[block_idx]).to_f32()
    }

    #[inline]
    pub fn get_code(&self, idx: usize) -> u8 {
        let byte = idx / 2;
        let shift = (idx % 2) * 4;
        (self.qs[byte] >> shift) & 0x0f
    }

    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_LEN] {
        let mut bytes = [0u8; Self::SERIALIZED_LEN];
        for (scale_index, scale_bits) in self.scale_bits.iter().enumerate() {
            let start = scale_index * 2;
            bytes[start..start + 2].copy_from_slice(&scale_bits.to_le_bytes());
        }
        bytes[16..].copy_from_slice(&self.qs);
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LnsError> {
        if bytes.len() != Self::SERIALIZED_LEN {
            return Err(LnsError::Validation(format!(
                "Q4HQ block size mismatch: expected {}, got {}",
                Self::SERIALIZED_LEN,
                bytes.len()
            )));
        }

        let mut scale_bits = [0u16; DEFAULT_SUPER_BLOCK_BLOCKS];
        for (scale_index, scale_slot) in scale_bits.iter_mut().enumerate() {
            let start = scale_index * 2;
            *scale_slot = u16::from_le_bytes(
                bytes[start..start + 2]
                    .try_into()
                    .expect("scale field is present"),
            );
        }

        let mut qs = [0u8; 128];
        qs.copy_from_slice(&bytes[16..]);
        Ok(Self { scale_bits, qs })
    }
}

pub fn encode_q4hq_superblock(weights: &[f32]) -> Q4HQSuperBlock {
    assert_eq!(
        weights.len(),
        DEFAULT_SUPER_BLOCK_SIZE,
        "encode_q4hq_superblock: expected {DEFAULT_SUPER_BLOCK_SIZE} weights, got {}",
        weights.len()
    );

    let mut scale_bits = [0u16; DEFAULT_SUPER_BLOCK_BLOCKS];
    let mut qs = [0u8; 128];

    for block_index in 0..DEFAULT_SUPER_BLOCK_BLOCKS {
        let block_start = block_index * BLOCK_SIZE;
        let block = &weights[block_start..block_start + BLOCK_SIZE];
        let scale = choose_q4hq_scale(block);
        scale_bits[block_index] = f16::from_f32(scale).to_bits();
        let rounded_scale = f16::from_bits(scale_bits[block_index]).to_f32();

        for (local_index, &weight) in block.iter().enumerate() {
            let global_index = block_start + local_index;
            let code = nearest_q4hq_code(weight, rounded_scale);
            let byte_index = global_index / 2;
            let shift = (global_index % 2) * 4;
            qs[byte_index] |= (code & 0x0f) << shift;
        }
    }

    Q4HQSuperBlock { scale_bits, qs }
}

pub fn decode_q4hq_superblock(block: &Q4HQSuperBlock) -> [f32; DEFAULT_SUPER_BLOCK_SIZE] {
    let mut result = [0.0_f32; DEFAULT_SUPER_BLOCK_SIZE];

    for block_index in 0..DEFAULT_SUPER_BLOCK_BLOCKS {
        let scale = block.scale(block_index);
        let block_start = block_index * BLOCK_SIZE;
        for local_index in 0..BLOCK_SIZE {
            let global_index = block_start + local_index;
            let code = block.get_code(global_index) as usize;
            result[global_index] = scale * Q4HQ_NF4_Z_CODEBOOK[code];
        }
    }

    result
}

pub fn quantize_q4hq(weights: &[f32]) -> Vec<Q4HQSuperBlock> {
    let full_len = weights.len() / DEFAULT_SUPER_BLOCK_SIZE * DEFAULT_SUPER_BLOCK_SIZE;
    let remainder = weights.len() % DEFAULT_SUPER_BLOCK_SIZE;

    let mut blocks: Vec<Q4HQSuperBlock> = weights[..full_len]
        .par_chunks(DEFAULT_SUPER_BLOCK_SIZE)
        .map(encode_q4hq_superblock)
        .collect();

    if remainder > 0 {
        let mut padded = [0.0_f32; DEFAULT_SUPER_BLOCK_SIZE];
        padded[..remainder].copy_from_slice(&weights[full_len..]);
        blocks.push(encode_q4hq_superblock(&padded));
    }

    blocks
}

pub fn dequantize_q4hq(blocks: &[Q4HQSuperBlock], num_weights: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(num_weights);
    for block in blocks {
        result.extend_from_slice(&decode_q4hq_superblock(block));
        if result.len() >= num_weights {
            break;
        }
    }
    result.truncate(num_weights);
    result
}

pub fn q4hq_blocks_to_payload(blocks: &[Q4HQSuperBlock]) -> Vec<u8> {
    let block_data_bytes = blocks.len() * Q4HQSuperBlock::SERIALIZED_LEN;
    let header = HqTensorPayloadHeader {
        codebook_id: Q4HQ_CODEBOOK_ID_NF4_Z,
        ..HqTensorPayloadHeader::canonical(
            Q4HQ_BLOCK_BYTES,
            HQ_CANONICAL_BLOCK_DATA_OFF,
            block_data_bytes as u64,
        )
    };
    let mut payload = Vec::with_capacity(HQ_CANONICAL_BLOCK_DATA_OFF as usize + block_data_bytes);
    payload.extend_from_slice(&header.to_bytes());
    payload.resize(HQ_CANONICAL_BLOCK_DATA_OFF as usize, 0);
    for block in blocks {
        payload.extend_from_slice(&block.to_bytes());
    }
    payload
}

pub fn q4hq_blocks_from_payload(payload: &[u8]) -> Result<Vec<Q4HQSuperBlock>, LnsError> {
    let header = parse_hq_payload_header(payload, QuantType::Q4HQ)?;
    if header.codebook_id != Q4HQ_CODEBOOK_ID_NF4_Z {
        return Err(LnsError::Validation(format!(
            "unsupported Q4HQ codebook id {}",
            header.codebook_id
        )));
    }

    let start = header.block_data_off as usize;
    let end = start + header.block_data_bytes as usize;
    payload[start..end]
        .chunks_exact(Q4HQSuperBlock::SERIALIZED_LEN)
        .map(Q4HQSuperBlock::from_bytes)
        .collect()
}

pub fn dequantize_q4hq_payload(payload: &[u8], num_weights: usize) -> Result<Vec<f32>, LnsError> {
    let blocks = q4hq_blocks_from_payload(payload)?;
    Ok(dequantize_q4hq(&blocks, num_weights))
}

fn choose_q4hq_scale(block: &[f32]) -> f32 {
    let max_abs = block
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    if max_abs == 0.0 {
        return 1.0;
    }

    let mean_square = block.iter().map(|value| value * value).sum::<f32>() / block.len() as f32;
    let rms_seed = mean_square.sqrt() / Q4HQ_NF4_Z_RMS;

    let lower_log = (max_abs / 4.0).max(f32::MIN_POSITIVE).log2();
    let upper_log = (max_abs * 4.0).max(f32::MIN_POSITIVE).log2();
    let mut best_scale = refine_q4hq_scale(block, max_abs.max(rms_seed));
    let mut best_error = q4hq_block_error(block, best_scale);
    let mut coarse_candidates = [(f32::INFINITY, 1.0_f32); Q4HQ_REFINE_CANDIDATES];

    for candidate_index in 0..=Q4HQ_SCALE_SEARCH_STEPS {
        let fraction = candidate_index as f32 / Q4HQ_SCALE_SEARCH_STEPS as f32;
        let candidate =
            round_positive_f16(2.0_f32.powf(lower_log + (upper_log - lower_log) * fraction));
        let error = q4hq_block_error(block, candidate);
        retain_q4hq_scale_candidate(&mut coarse_candidates, error, candidate);
    }

    for &(coarse_error, candidate) in &coarse_candidates {
        if !coarse_error.is_finite() {
            continue;
        }
        let refined = refine_q4hq_scale(block, candidate);
        let error = q4hq_block_error(block, refined);
        if error < best_error {
            best_error = error;
            best_scale = refined;
        }
    }

    best_scale
}

fn retain_q4hq_scale_candidate(
    candidates: &mut [(f32, f32); Q4HQ_REFINE_CANDIDATES],
    error: f32,
    scale: f32,
) {
    if candidates
        .iter()
        .any(|&(_, existing_scale)| existing_scale == scale)
    {
        return;
    }

    if let Some((insert_index, _)) = candidates
        .iter()
        .enumerate()
        .find(|&(_, &(candidate_error, _))| error < candidate_error)
    {
        for move_index in (insert_index + 1..Q4HQ_REFINE_CANDIDATES).rev() {
            candidates[move_index] = candidates[move_index - 1];
        }
        candidates[insert_index] = (error, scale);
    }
}

fn refine_q4hq_scale(block: &[f32], initial_scale: f32) -> f32 {
    let mut scale = round_positive_f16(initial_scale);
    for _ in 0..2 {
        let mut numerator = 0.0_f64;
        let mut denominator = 0.0_f64;
        for &weight in block {
            let code = nearest_q4hq_code(weight, scale) as usize;
            let codebook_value = Q4HQ_NF4_Z_CODEBOOK[code] as f64;
            numerator += codebook_value * weight as f64;
            denominator += codebook_value * codebook_value;
        }
        if denominator <= f64::EPSILON {
            return 1.0;
        }
        scale = round_positive_f16((numerator / denominator).max(f64::MIN_POSITIVE) as f32);
    }
    scale
}

fn q4hq_block_error(block: &[f32], scale: f32) -> f32 {
    block
        .iter()
        .map(|&weight| {
            let code = nearest_q4hq_code(weight, scale) as usize;
            let decoded = scale * Q4HQ_NF4_Z_CODEBOOK[code];
            let delta = weight - decoded;
            delta * delta
        })
        .sum()
}

#[inline]
fn nearest_q4hq_code(weight: f32, scale: f32) -> u8 {
    if !weight.is_finite() || scale <= f32::MIN_POSITIVE {
        return Q4HQ_ZERO_CODE;
    }
    let normalized = weight / scale;
    let mut best_code = Q4HQ_ZERO_CODE;
    let mut best_error = normalized * normalized;
    for (code, &codebook_value) in Q4HQ_NF4_Z_CODEBOOK.iter().enumerate() {
        let delta = normalized - codebook_value;
        let error = delta * delta;
        if error < best_error {
            best_error = error;
            best_code = code as u8;
        }
    }
    best_code
}

#[inline]
fn round_positive_f16(value: f32) -> f32 {
    f16::from_f32(value.max(f32::MIN_POSITIVE))
        .to_f32()
        .max(f32::MIN_POSITIVE)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HqTensorPayloadHeader {
    pub magic: [u8; 4],
    pub header_bytes: u16,
    pub block_bytes: u16,
    pub flags: u32,
    pub codebook_id: u8,
    pub reserved0: [u8; 7],
    pub inv_alpha_count: u32,
    pub segment_count: u32,
    pub custom_codebook_off: u64,
    pub inv_alpha_off: u64,
    pub segment_table_off: u64,
    pub block_data_off: u64,
    pub block_data_bytes: u64,
}

impl HqTensorPayloadHeader {
    pub const SERIALIZED_LEN: usize = 68;

    pub fn canonical(block_bytes: u16, block_data_off: u64, block_data_bytes: u64) -> Self {
        Self {
            magic: HQ_MAGIC,
            header_bytes: HQ_HEADER_BYTES,
            block_bytes,
            flags: 0,
            codebook_id: 0,
            reserved0: [0; 7],
            inv_alpha_count: 0,
            segment_count: 0,
            custom_codebook_off: 0,
            inv_alpha_off: 0,
            segment_table_off: 0,
            block_data_off,
            block_data_bytes,
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LnsError> {
        if bytes.len() < Self::SERIALIZED_LEN {
            return Err(LnsError::Validation(format!(
                "HQ payload header too short: {} < {}",
                bytes.len(),
                Self::SERIALIZED_LEN
            )));
        }

        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);
        let mut reserved0 = [0u8; 7];
        reserved0.copy_from_slice(&bytes[13..20]);

        Ok(Self {
            magic,
            header_bytes: read_u16(bytes, 4),
            block_bytes: read_u16(bytes, 6),
            flags: read_u32(bytes, 8),
            codebook_id: bytes[12],
            reserved0,
            inv_alpha_count: read_u32(bytes, 20),
            segment_count: read_u32(bytes, 24),
            custom_codebook_off: read_u64(bytes, 28),
            inv_alpha_off: read_u64(bytes, 36),
            segment_table_off: read_u64(bytes, 44),
            block_data_off: read_u64(bytes, 52),
            block_data_bytes: read_u64(bytes, 60),
        })
    }

    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_LEN] {
        let mut bytes = [0u8; Self::SERIALIZED_LEN];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..6].copy_from_slice(&self.header_bytes.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.block_bytes.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.flags.to_le_bytes());
        bytes[12] = self.codebook_id;
        bytes[13..20].copy_from_slice(&self.reserved0);
        bytes[20..24].copy_from_slice(&self.inv_alpha_count.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.segment_count.to_le_bytes());
        bytes[28..36].copy_from_slice(&self.custom_codebook_off.to_le_bytes());
        bytes[36..44].copy_from_slice(&self.inv_alpha_off.to_le_bytes());
        bytes[44..52].copy_from_slice(&self.segment_table_off.to_le_bytes());
        bytes[52..60].copy_from_slice(&self.block_data_off.to_le_bytes());
        bytes[60..68].copy_from_slice(&self.block_data_bytes.to_le_bytes());
        bytes
    }

    pub fn validate_for_quant_type(
        &self,
        quant_type: QuantType,
        payload_len: usize,
    ) -> Result<(), LnsError> {
        if !is_hq_quant_type(quant_type) {
            return Err(LnsError::UnsupportedQuantType(quant_type.as_u8()));
        }
        if self.magic != HQ_MAGIC {
            return Err(LnsError::Validation("invalid HQ payload magic".to_string()));
        }
        if self.header_bytes < HQ_HEADER_BYTES {
            return Err(LnsError::Validation(format!(
                "HQ header_bytes {} is smaller than minimum {}",
                self.header_bytes, HQ_HEADER_BYTES
            )));
        }
        if self.header_bytes as usize > payload_len {
            return Err(LnsError::Validation(
                "HQ header extends beyond payload".to_string(),
            ));
        }
        if self.flags & !VALID_HQ_FLAGS != 0 {
            return Err(LnsError::Validation(format!(
                "unknown HQ payload flags: 0x{:08x}",
                self.flags
            )));
        }
        if self.reserved0.iter().any(|&value| value != 0) {
            return Err(LnsError::Validation(
                "HQ reserved bytes must be zero".to_string(),
            ));
        }
        if !valid_block_bytes(quant_type, self.block_bytes) {
            return Err(LnsError::Validation(format!(
                "invalid HQ block size {} for {:?}",
                self.block_bytes, quant_type
            )));
        }

        let mut ranges = Vec::with_capacity(4);

        if self.flags & FLAG_CUSTOM_CODEBOOK != 0 {
            if quant_type == QuantType::Q8HQ {
                return Err(LnsError::Validation(
                    "Q8HQ does not use a custom codebook".to_string(),
                ));
            }
            let codebook_bytes = custom_codebook_bytes(quant_type)?;
            ranges.push(required_range(
                "custom codebook",
                self.custom_codebook_off,
                codebook_bytes,
                payload_len,
                4,
            )?);
        } else if self.custom_codebook_off != 0 {
            return Err(LnsError::Validation(
                "custom codebook offset set without flag".to_string(),
            ));
        }

        if self.flags & FLAG_AWQ_INV_ALPHA != 0 {
            if self.inv_alpha_count == 0 {
                return Err(LnsError::Validation(
                    "AWQ flag set with zero inv_alpha_count".to_string(),
                ));
            }
            let inv_alpha_bytes =
                checked_mul_usize(self.inv_alpha_count as usize, 2, "inv_alpha bytes")?;
            ranges.push(required_range(
                "inv_alpha",
                self.inv_alpha_off,
                inv_alpha_bytes,
                payload_len,
                2,
            )?);
        } else if self.inv_alpha_count != 0 || self.inv_alpha_off != 0 {
            return Err(LnsError::Validation(
                "AWQ metadata set without AWQ flag".to_string(),
            ));
        }

        if self.flags & FLAG_SEGMENT_TABLE != 0 {
            if self.segment_count == 0 {
                return Err(LnsError::Validation(
                    "segment flag set with zero segment_count".to_string(),
                ));
            }
            let segment_bytes = checked_mul_usize(
                self.segment_count as usize,
                HQ_SEGMENT_ENTRY_BYTES,
                "segment table bytes",
            )?;
            ranges.push(required_range(
                "segment table",
                self.segment_table_off,
                segment_bytes,
                payload_len,
                8,
            )?);
        } else if self.segment_count != 0 || self.segment_table_off != 0 {
            return Err(LnsError::Validation(
                "segment metadata set without segment flag".to_string(),
            ));
        }

        if self.block_data_bytes == 0 {
            return Err(LnsError::Validation("HQ block data is empty".to_string()));
        }
        if self.block_data_bytes % u64::from(self.block_bytes) != 0 {
            return Err(LnsError::Validation(
                "HQ block data length is not a whole number of blocks".to_string(),
            ));
        }
        ranges.push(required_range(
            "block data",
            self.block_data_off,
            self.block_data_bytes as usize,
            payload_len,
            16,
        )?);

        reject_overlaps(&ranges)?;
        Ok(())
    }
}

pub fn parse_hq_payload_header(
    payload: &[u8],
    quant_type: QuantType,
) -> Result<HqTensorPayloadHeader, LnsError> {
    let header = HqTensorPayloadHeader::from_bytes(payload)?;
    header.validate_for_quant_type(quant_type, payload.len())?;
    if header.flags & FLAG_CUSTOM_CODEBOOK != 0 {
        validate_custom_codebook(payload, quant_type, &header)?;
    }
    Ok(header)
}

pub fn is_hq_quant_type(quant_type: QuantType) -> bool {
    matches!(
        quant_type,
        QuantType::Q4HQ | QuantType::Q8HQ | QuantType::Q4HQM | QuantType::Q2HQ | QuantType::Q2HQM
    )
}

pub fn valid_block_bytes(quant_type: QuantType, block_bytes: u16) -> bool {
    match quant_type {
        QuantType::Q2HQ | QuantType::Q2HQM => {
            block_bytes == Q2HQ_32_BLOCK_BYTES || block_bytes == Q2HQ_16_BLOCK_BYTES
        }
        QuantType::Q4HQ | QuantType::Q4HQM => block_bytes == Q4HQ_BLOCK_BYTES,
        QuantType::Q8HQ => block_bytes == Q8HQ_BLOCK_BYTES,
        _ => false,
    }
}

fn custom_codebook_bytes(quant_type: QuantType) -> Result<usize, LnsError> {
    match quant_type {
        QuantType::Q2HQ | QuantType::Q2HQM => Ok(4 * std::mem::size_of::<f32>()),
        QuantType::Q4HQ | QuantType::Q4HQM => Ok(16 * std::mem::size_of::<f32>()),
        QuantType::Q8HQ => Err(LnsError::Validation(
            "Q8HQ has no custom codebook".to_string(),
        )),
        _ => Err(LnsError::UnsupportedQuantType(quant_type.as_u8())),
    }
}

fn validate_custom_codebook(
    payload: &[u8],
    quant_type: QuantType,
    header: &HqTensorPayloadHeader,
) -> Result<(), LnsError> {
    let codebook_bytes = custom_codebook_bytes(quant_type)?;
    let offset = header.custom_codebook_off as usize;
    let bytes = &payload[offset..offset + codebook_bytes];
    let mut previous = f32::NEG_INFINITY;
    let mut has_non_zero = false;

    for value_bytes in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes(value_bytes.try_into().expect("chunk size is four"));
        if !value.is_finite() {
            return Err(LnsError::Validation(
                "HQ custom codebook contains a non-finite value".to_string(),
            ));
        }
        if value < previous {
            return Err(LnsError::Validation(
                "HQ custom codebook must be monotonic".to_string(),
            ));
        }
        has_non_zero |= value != 0.0;
        previous = value;
    }

    if !has_non_zero {
        return Err(LnsError::Validation(
            "HQ custom codebook must not be all zero".to_string(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct ByteRange {
    name: &'static str,
    start: usize,
    end: usize,
}

fn required_range(
    name: &'static str,
    offset: u64,
    len: usize,
    payload_len: usize,
    alignment: usize,
) -> Result<ByteRange, LnsError> {
    if offset == 0 {
        return Err(LnsError::Validation(format!("HQ {name} offset is zero")));
    }
    let start = usize::try_from(offset)
        .map_err(|_| LnsError::Validation(format!("HQ {name} offset does not fit usize")))?;
    if start < HQ_HEADER_BYTES as usize {
        return Err(LnsError::Validation(format!(
            "HQ {name} starts inside the header"
        )));
    }
    if start % alignment != 0 {
        return Err(LnsError::Validation(format!(
            "HQ {name} offset is not {alignment}-byte aligned"
        )));
    }
    let end = start
        .checked_add(len)
        .ok_or_else(|| LnsError::Validation(format!("HQ {name} range overflows")))?;
    if end > payload_len {
        return Err(LnsError::Validation(format!(
            "HQ {name} range exceeds payload length"
        )));
    }
    Ok(ByteRange { name, start, end })
}

fn reject_overlaps(ranges: &[ByteRange]) -> Result<(), LnsError> {
    for left_index in 0..ranges.len() {
        for right_index in (left_index + 1)..ranges.len() {
            let left = ranges[left_index];
            let right = ranges[right_index];
            if left.start < right.end && right.start < left.end {
                return Err(LnsError::Validation(format!(
                    "HQ ranges overlap: {} and {}",
                    left.name, right.name
                )));
            }
        }
    }
    Ok(())
}

fn checked_mul_usize(left: usize, right: usize, label: &'static str) -> Result<usize, LnsError> {
    left.checked_mul(right)
        .ok_or_else(|| LnsError::Validation(format!("HQ {label} overflow")))
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .expect("u16 field is present"),
    )
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("u32 field is present"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("u64 field is present"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rmse(left: &[f32], right: &[f32]) -> f32 {
        let sum = left
            .iter()
            .zip(right.iter())
            .map(|(left_value, right_value)| {
                let delta = left_value - right_value;
                delta * delta
            })
            .sum::<f32>();
        (sum / left.len() as f32).sqrt()
    }

    fn payload_with_header(header: &HqTensorPayloadHeader, payload_len: usize) -> Vec<u8> {
        let mut payload = vec![0u8; payload_len];
        payload[..HqTensorPayloadHeader::SERIALIZED_LEN].copy_from_slice(&header.to_bytes());
        payload
    }

    #[test]
    fn q4hq_header_roundtrips_and_validates() {
        let header =
            HqTensorPayloadHeader::canonical(Q4HQ_BLOCK_BYTES, 80, u64::from(Q4HQ_BLOCK_BYTES));
        let payload = payload_with_header(&header, 80 + Q4HQ_BLOCK_BYTES as usize);

        let parsed =
            parse_hq_payload_header(&payload, QuantType::Q4HQ).expect("valid Q4HQ payload");

        assert_eq!(parsed, header);
    }

    #[test]
    fn q4hq_superblock_has_canonical_size() {
        assert_eq!(
            std::mem::size_of::<Q4HQSuperBlock>(),
            Q4HQ_BLOCK_BYTES as usize
        );
        assert_eq!(Q4HQSuperBlock::SERIALIZED_LEN, Q4HQ_BLOCK_BYTES as usize);
    }

    #[test]
    fn q4hq_zero_superblock_decodes_exact_zero() {
        let weights = [0.0_f32; DEFAULT_SUPER_BLOCK_SIZE];
        let block = encode_q4hq_superblock(&weights);
        let decoded = decode_q4hq_superblock(&block);

        assert!(decoded.iter().all(|value| *value == 0.0));
        assert!((0..DEFAULT_SUPER_BLOCK_SIZE).all(|index| block.get_code(index) == Q4HQ_ZERO_CODE));
    }

    #[test]
    fn q4hq_roundtrip_has_reasonable_error() {
        let weights: Vec<f32> = (0..DEFAULT_SUPER_BLOCK_SIZE)
            .map(|index| {
                let wave = (index as f32 * 0.073).sin() * 0.75;
                let drift = (index as f32 % 17.0 - 8.0) * 0.015;
                wave + drift
            })
            .collect();

        let block = encode_q4hq_superblock(&weights);
        let decoded = decode_q4hq_superblock(&block);

        assert!(rmse(&weights, &decoded) < 0.06);
    }

    #[test]
    fn q4hq_payload_wraps_and_decodes_blocks() {
        let weights: Vec<f32> = (0..384)
            .map(|index| (index as f32 * 0.047).cos() * 0.5)
            .collect();
        let blocks = quantize_q4hq(&weights);
        let payload = q4hq_blocks_to_payload(&blocks);

        let header = parse_hq_payload_header(&payload, QuantType::Q4HQ).expect("valid payload");
        assert_eq!(header.block_data_off, HQ_CANONICAL_BLOCK_DATA_OFF);
        assert_eq!(header.block_bytes, Q4HQ_BLOCK_BYTES);

        let decoded = dequantize_q4hq_payload(&payload, weights.len()).expect("decode payload");
        assert_eq!(decoded.len(), weights.len());
        assert!(rmse(&weights, &decoded) < 0.05);
    }

    #[test]
    fn q2hq_allows_both_scale_granularities() {
        for block_bytes in [Q2HQ_32_BLOCK_BYTES, Q2HQ_16_BLOCK_BYTES] {
            let header = HqTensorPayloadHeader::canonical(block_bytes, 80, u64::from(block_bytes));
            let payload = payload_with_header(&header, 80 + block_bytes as usize);

            parse_hq_payload_header(&payload, QuantType::Q2HQ).expect("valid Q2HQ payload");
        }
    }

    #[test]
    fn rejects_unknown_flags() {
        let mut header =
            HqTensorPayloadHeader::canonical(Q4HQ_BLOCK_BYTES, 80, u64::from(Q4HQ_BLOCK_BYTES));
        header.flags = 1 << 31;
        let payload = payload_with_header(&header, 80 + Q4HQ_BLOCK_BYTES as usize);

        assert!(parse_hq_payload_header(&payload, QuantType::Q4HQ).is_err());
    }

    #[test]
    fn rejects_invalid_block_size_for_quant_type() {
        let header =
            HqTensorPayloadHeader::canonical(Q4HQ_BLOCK_BYTES, 80, u64::from(Q4HQ_BLOCK_BYTES));
        let payload = payload_with_header(&header, 80 + Q4HQ_BLOCK_BYTES as usize);

        assert!(parse_hq_payload_header(&payload, QuantType::Q2HQ).is_err());
    }

    #[test]
    fn rejects_metadata_without_matching_flag() {
        let mut header =
            HqTensorPayloadHeader::canonical(Q4HQ_BLOCK_BYTES, 80, u64::from(Q4HQ_BLOCK_BYTES));
        header.inv_alpha_count = 16;
        header.inv_alpha_off = 224;
        let payload = payload_with_header(&header, 256);

        assert!(parse_hq_payload_header(&payload, QuantType::Q4HQ).is_err());
    }

    #[test]
    fn rejects_overlapping_ranges() {
        let mut header =
            HqTensorPayloadHeader::canonical(Q4HQ_BLOCK_BYTES, 96, u64::from(Q4HQ_BLOCK_BYTES));
        header.flags = FLAG_CUSTOM_CODEBOOK;
        header.custom_codebook_off = 80;
        let mut payload = payload_with_header(&header, 256);
        let codebook_values: [f32; 16] = [
            -1.0, -0.8, -0.6, -0.4, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0,
        ];
        for (value_index, value) in codebook_values.iter().enumerate() {
            let start = 80 + value_index * 4;
            payload[start..start + 4].copy_from_slice(&value.to_le_bytes());
        }

        assert!(parse_hq_payload_header(&payload, QuantType::Q4HQ).is_err());
    }

    #[test]
    fn rejects_bad_custom_codebook_values() {
        let mut header = HqTensorPayloadHeader::canonical(
            Q2HQ_32_BLOCK_BYTES,
            112,
            u64::from(Q2HQ_32_BLOCK_BYTES),
        );
        header.flags = FLAG_CUSTOM_CODEBOOK;
        header.custom_codebook_off = 80;
        let mut payload = payload_with_header(&header, 112 + Q2HQ_32_BLOCK_BYTES as usize);
        let codebook_values = [-1.0f32, f32::NAN, 0.5, 1.0];
        for (value_index, value) in codebook_values.iter().enumerate() {
            let start = 80 + value_index * 4;
            payload[start..start + 4].copy_from_slice(&value.to_le_bytes());
        }

        assert!(parse_hq_payload_header(&payload, QuantType::Q2HQ).is_err());
    }
}
