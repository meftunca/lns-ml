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
pub use format::{LnsModel, LnsTensor, QuantType, FORMAT_VERSION};
pub use quant::q4l::{
    decode_superblock, dequantize_q4l, encode_superblock, quantize_q4l, Q4LSuperBlock,
    BLOCK_SIZE, DEFAULT_SUPER_BLOCK_BLOCKS, DEFAULT_SUPER_BLOCK_SIZE,
};
