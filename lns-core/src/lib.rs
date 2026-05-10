//! lns-core — LNS quantization format, encoding/decoding, and rkyv model schema.
//!
//! This crate provides:
//! - [`quant::q4l`]: Q4_L 4-bit logarithmic quantization encode/decode
//! - [`format`]: rkyv-based zero-copy model format types
//! - [`error`]: unified error type

pub mod error;
pub mod format;
pub mod quant;

pub use error::LnsError;
pub use format::{
    model_contains_hq_tensors, validate_model_format_version, LnsModel, LnsTensor, QuantType,
    FORMAT_VERSION, HQ_FORMAT_VERSION,
};
pub use quant::hq::{
    decode_q4hq_superblock, dequantize_q4hq, dequantize_q4hq_payload, encode_q4hq_superblock,
    parse_hq_payload_header, q4hq_blocks_from_payload, q4hq_blocks_to_payload, quantize_q4hq,
    valid_block_bytes, HqTensorPayloadHeader, Q4HQSuperBlock, FLAG_AWQ_INV_ALPHA,
    FLAG_CUSTOM_CODEBOOK, FLAG_SEGMENT_TABLE, HQ_CANONICAL_BLOCK_DATA_OFF, HQ_HEADER_BYTES,
    HQ_MAGIC, Q2HQ_16_BLOCK_BYTES, Q2HQ_32_BLOCK_BYTES, Q4HQ_BLOCK_BYTES, Q4HQ_CODEBOOK_ID_NF4_Z,
    Q4HQ_NF4_Z_CODEBOOK, Q4HQ_ZERO_CODE, Q8HQ_BLOCK_BYTES,
};
pub use quant::q2l::{
    decode_superblock as decode_q2l_sb, dequantize_q2l, encode_superblock as encode_q2l_sb,
    quantize_q2l, Q2LSuperBlock,
};
pub use quant::q4l::{
    decode_superblock as decode_q4l_sb, dequantize_q4l, encode_superblock as encode_q4l_sb,
    quantize_q4l, Q4LSuperBlock, BLOCK_SIZE, DEFAULT_SUPER_BLOCK_BLOCKS, DEFAULT_SUPER_BLOCK_SIZE,
};
pub use quant::q8l::{
    decode_superblock as decode_q8l_sb, dequantize_q8l, encode_superblock as encode_q8l_sb,
    quantize_q8l, Q8LSuperBlock,
};
