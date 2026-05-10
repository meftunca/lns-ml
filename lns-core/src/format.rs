//! rkyv-based zero-copy model format for lns-ml.
//!
//! # File structure
//!
//! An `.lns` model file is the raw output of rkyv serialisation of a
//! [`LnsModel`].  rkyv places the archive at the **end** of the buffer,
//! so the file can be memory-mapped and accessed directly via
//! [`archived_model`] / [`check_archived_model`] without any parsing or
//! allocation overhead.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │  rkyv serialised LnsModel (variable length, archive at end)          │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Quantisation type codes
//!
//! | Code | [`QuantType`]  | Description                       |
//! |------|----------------|-----------------------------------|
//! | 0    | `F32`          | 32-bit float (unquantised)        |
//! | 1    | `F16`          | 16-bit float (stored as u16 bits) |
//! | 2    | `Q4L`          | 4-bit LNS (see [`quant::q4l`])    |

use rkyv::{Archive, Deserialize, Serialize};
use std::io::Write;

use crate::{LnsError, Q4LSuperBlock, DEFAULT_SUPER_BLOCK_SIZE};

// ── Version ───────────────────────────────────────────────────────────────────

/// Current `.lns` model format version for newly written archives.
pub const FORMAT_VERSION: u32 = 4;

/// Minimum `.lns` model format version for archives that contain HQ tensors.
pub const HQ_FORMAT_VERSION: u32 = 4;

// ── QuantType ─────────────────────────────────────────────────────────────────

/// Quantisation type for a tensor.
///
/// Stored as a raw `u8` in the archive so that new variants can be added
/// without requiring a rkyv schema migration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum QuantType {
    /// 32-bit float (no quantisation).
    F32 = 0,
    /// 16-bit float (stored as raw `u16` bits, IEEE 754 half-precision).
    F16 = 1,
    /// 4-bit logarithmic quantisation ([`crate::quant::q4l`]).
    Q4L = 2,
    /// 2-bit logarithmic quantisation.
    Q2L = 3,
    /// 6-bit logarithmic quantisation.
    Q6L = 4,
    /// 8-bit logarithmic quantisation.
    Q8L = 5,
    /// 4-bit high-quality direct codebook quantisation.
    Q4HQ = 6,
    /// 8-bit high-quality direct linear quantisation.
    Q8HQ = 7,
    /// Mixed Q4HQ with promoted contiguous runs.
    Q4HQM = 8,
    /// 2-bit high-quality direct codebook quantisation.
    Q2HQ = 9,
    /// Mixed Q2HQ with Q4HQ/Q8HQ rescue runs.
    Q2HQM = 10,
}

impl QuantType {
    /// Convert from a raw `u8` code as stored in [`LnsTensor::quant_type`].
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4L),
            3 => Some(Self::Q2L),
            4 => Some(Self::Q6L),
            5 => Some(Self::Q8L),
            6 => Some(Self::Q4HQ),
            7 => Some(Self::Q8HQ),
            8 => Some(Self::Q4HQM),
            9 => Some(Self::Q2HQ),
            10 => Some(Self::Q2HQM),
            _ => None,
        }
    }

    /// Return the raw `u8` code.
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

// ── LnsTensor ─────────────────────────────────────────────────────────────────

/// A single quantised tensor inside an [`LnsModel`].
#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
#[archive_attr(derive(Debug))]
pub struct LnsTensor {
    /// Tensor name, e.g. `"model.layers.0.self_attn.q_proj.weight"`.
    pub name: String,

    /// Tensor shape (all dimensions as `u64` for forward compatibility).
    pub shape: Vec<u64>,

    /// Quantisation type code — see [`QuantType`].
    pub quant_type: u8,

    /// Raw quantised bytes.
    ///
    /// - `F32`: packed little-endian `f32` values.
    /// - `F16`: packed `u16` f16 bit-patterns.
    /// - `Q4L`: packed [`Q4LSuperBlock`] structs (use
    ///   `bytemuck::cast_slice` to reinterpret).
    pub data: Vec<u8>,
}

impl LnsTensor {
    /// Number of elements described by [`Self::shape`].
    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Decode a `Q4L` tensor to `f32`.
    ///
    /// Returns `Err` if [`Self::quant_type`] is not `Q4L` or the data length
    /// is inconsistent.
    pub fn decode_q4l(&self) -> Result<Vec<f32>, LnsError> {
        if self.quant_type != QuantType::Q4L.as_u8() {
            return Err(LnsError::UnsupportedQuantType(self.quant_type));
        }
        if self.data.len() % std::mem::size_of::<crate::Q4LSuperBlock>() != 0 {
            return Err(LnsError::ShapeMismatch {
                expected: 0,
                actual: self.data.len(),
            });
        }
        let blocks: &[crate::Q4LSuperBlock] = bytemuck::cast_slice(&self.data);
        Ok(crate::dequantize_q4l(blocks, self.num_elements()))
    }

    /// Decode a `Q2L` tensor to `f32`.
    pub fn decode_q2l(&self) -> Result<Vec<f32>, LnsError> {
        if self.quant_type != QuantType::Q2L.as_u8() {
            return Err(LnsError::UnsupportedQuantType(self.quant_type));
        }
        if self.data.len() % std::mem::size_of::<crate::Q2LSuperBlock>() != 0 {
            return Err(LnsError::ShapeMismatch {
                expected: 0,
                actual: self.data.len(),
            });
        }
        let blocks: &[crate::Q2LSuperBlock] = bytemuck::cast_slice(&self.data);
        Ok(crate::dequantize_q2l(blocks, self.num_elements()))
    }

    /// Decode a `Q8L` tensor to `f32`.
    pub fn decode_q8l(&self) -> Result<Vec<f32>, LnsError> {
        if self.quant_type != QuantType::Q8L.as_u8() {
            return Err(LnsError::UnsupportedQuantType(self.quant_type));
        }
        if self.data.len() % std::mem::size_of::<crate::Q8LSuperBlock>() != 0 {
            return Err(LnsError::ShapeMismatch {
                expected: 0,
                actual: self.data.len(),
            });
        }
        let blocks: &[crate::Q8LSuperBlock] = bytemuck::cast_slice(&self.data);
        Ok(crate::dequantize_q8l(blocks, self.num_elements()))
    }

    /// Decode an `F32` tensor.
    pub fn decode_f32(&self) -> Result<Vec<f32>, LnsError> {
        if self.quant_type != QuantType::F32.as_u8() {
            return Err(LnsError::UnsupportedQuantType(self.quant_type));
        }
        Ok(bytemuck::cast_slice::<u8, f32>(&self.data).to_vec())
    }

    /// Decode an `F16` tensor to `f32`.
    pub fn decode_f16(&self) -> Result<Vec<f32>, LnsError> {
        if self.quant_type != QuantType::F16.as_u8() {
            return Err(LnsError::UnsupportedQuantType(self.quant_type));
        }
        let u16s: &[u16] = bytemuck::cast_slice(&self.data);
        Ok(u16s
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect())
    }

    pub fn to_f32(&self) -> Result<Vec<f32>, LnsError> {
        match QuantType::from_u8(self.quant_type) {
            Some(QuantType::F32) => self.decode_f32(),
            Some(QuantType::F16) => self.decode_f16(),
            Some(QuantType::Q4L) => self.decode_q4l(),
            Some(QuantType::Q2L) => self.decode_q2l(),
            Some(QuantType::Q8L) => self.decode_q8l(),
            Some(QuantType::Q6L) => Err(LnsError::Serialization("Q6L not implemented".into())),
            Some(QuantType::Q4HQ) => {
                crate::quant::hq::dequantize_q4hq_payload(&self.data, self.num_elements())
            }
            Some(QuantType::Q8HQ | QuantType::Q4HQM | QuantType::Q2HQ | QuantType::Q2HQM) => {
                Err(LnsError::UnsupportedQuantType(self.quant_type))
            }
            None => Err(LnsError::UnsupportedQuantType(self.quant_type)),
        }
    }
}

impl ArchivedLnsTensor {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    pub fn decode_q4l(&self) -> Result<Vec<f32>, LnsError> {
        let blocks: &[crate::Q4LSuperBlock] = bytemuck::cast_slice(self.data.as_slice());
        Ok(crate::dequantize_q4l(blocks, self.num_elements()))
    }

    pub fn decode_q2l(&self) -> Result<Vec<f32>, LnsError> {
        let blocks: &[crate::Q2LSuperBlock] = bytemuck::cast_slice(self.data.as_slice());
        Ok(crate::dequantize_q2l(blocks, self.num_elements()))
    }

    pub fn decode_q8l(&self) -> Result<Vec<f32>, LnsError> {
        let blocks: &[crate::Q8LSuperBlock] = bytemuck::cast_slice(self.data.as_slice());
        Ok(crate::dequantize_q8l(blocks, self.num_elements()))
    }

    pub fn decode_f32(&self) -> Result<Vec<f32>, LnsError> {
        Ok(bytemuck::cast_slice::<u8, f32>(self.data.as_slice()).to_vec())
    }

    pub fn decode_f16(&self) -> Result<Vec<f32>, LnsError> {
        let u16s: &[u16] = bytemuck::cast_slice(self.data.as_slice());
        Ok(u16s
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect())
    }

    pub fn to_f32(&self) -> Result<Vec<f32>, LnsError> {
        match QuantType::from_u8(self.quant_type) {
            Some(QuantType::F32) => self.decode_f32(),
            Some(QuantType::F16) => self.decode_f16(),
            Some(QuantType::Q4L) => self.decode_q4l(),
            Some(QuantType::Q2L) => self.decode_q2l(),
            Some(QuantType::Q8L) => self.decode_q8l(),
            Some(QuantType::Q4HQ) => {
                crate::quant::hq::dequantize_q4hq_payload(self.data.as_slice(), self.num_elements())
            }
            Some(QuantType::Q8HQ | QuantType::Q4HQM | QuantType::Q2HQ | QuantType::Q2HQM) => {
                Err(LnsError::UnsupportedQuantType(self.quant_type))
            }
            _ => Err(LnsError::UnsupportedQuantType(self.quant_type)),
        }
    }
}

// ── LnsModel ──────────────────────────────────────────────────────────────────

/// Top-level container for an lns-ml model.
#[derive(Archive, Serialize, Deserialize, Clone, Debug)]
#[archive(check_bytes)]
#[archive_attr(derive(Debug))]
pub struct LnsModel {
    /// Format version — always [`FORMAT_VERSION`] for files written by this
    /// version of lns-core.
    pub version: u32,
    /// All tensors in the model, in the order they were added by the converter.
    pub tensors: Vec<LnsTensor>,
}

impl LnsModel {
    /// Look up a tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&LnsTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Total number of quantised bytes across all tensors.
    pub fn total_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.data.len()).sum()
    }

    /// Total number of weight elements across all tensors.
    pub fn total_elements(&self) -> usize {
        self.tensors.iter().map(|t| t.num_elements()).sum()
    }
}

pub fn model_contains_hq_tensors(model: &LnsModel) -> bool {
    model
        .tensors
        .iter()
        .filter_map(|tensor| QuantType::from_u8(tensor.quant_type))
        .any(crate::quant::hq::is_hq_quant_type)
}

pub fn validate_model_format_version(model: &LnsModel) -> Result<(), LnsError> {
    if model_contains_hq_tensors(model) && model.version < HQ_FORMAT_VERSION {
        return Err(LnsError::Validation(format!(
            "HQ tensors require format version >= {HQ_FORMAT_VERSION}, got {}",
            model.version
        )));
    }
    Ok(())
}

// ── Serialisation helpers ─────────────────────────────────────────────────────

/// Serialise an [`LnsModel`] into a byte buffer ready to be written to disk.
///
/// The output is a raw rkyv archive.  Use [`archived_model`] or
/// [`check_archived_model`] to read it back zero-copy.
pub fn serialize_model(model: &LnsModel) -> Result<Vec<u8>, LnsError> {
    use rkyv::ser::{serializers::AllocSerializer, Serializer};

    validate_model_format_version(model)?;

    let mut serializer = AllocSerializer::<4096>::default();
    serializer
        .serialize_value(model)
        .map_err(|e| LnsError::Serialization(e.to_string()))?;

    Ok(serializer.into_serializer().into_inner().to_vec())
}

/// Serialise an [`LnsModel`] directly to a writer.
///
/// This avoids holding both tensor data and a second full archive buffer in
/// memory during conversion of multi-GB checkpoints.
pub fn serialize_model_to_writer<W: Write>(model: &LnsModel, writer: W) -> Result<(), LnsError> {
    use rkyv::ser::{
        serializers::{
            AllocScratch, CompositeSerializer, FallbackScratch, HeapScratch, SharedSerializeMap,
            WriteSerializer,
        },
        Serializer,
    };

    validate_model_format_version(model)?;

    let mut serializer = CompositeSerializer::new(
        WriteSerializer::new(writer),
        FallbackScratch::new(HeapScratch::<4096>::new(), AllocScratch::new()),
        SharedSerializeMap::new(),
    );
    serializer
        .serialize_value(model)
        .map_err(|e| LnsError::Serialization(e.to_string()))?;

    Ok(())
}

/// Get a zero-copy reference to an archived [`LnsModel`] from raw bytes.
///
/// The `bytes` slice must have been produced by [`serialize_model`] (or
/// written by the converter).
///
/// # Safety
///
/// The caller must guarantee that `bytes` contains a validly-formed rkyv
/// archive of an [`LnsModel`].  Use [`check_archived_model`] for a safe
/// (bounds-checked) variant at the cost of a one-time validation pass.
pub unsafe fn archived_model(bytes: &[u8]) -> &ArchivedLnsModel {
    rkyv::archived_root::<LnsModel>(bytes)
}

/// Get a validated zero-copy reference to an archived [`LnsModel`].
///
/// Performs a full bounds- and type-check over the archive before returning.
/// More expensive than [`archived_model`] but safe against corrupt files.
pub fn check_archived_model(bytes: &[u8]) -> Result<&ArchivedLnsModel, LnsError> {
    rkyv::check_archived_root::<LnsModel>(bytes).map_err(|e| LnsError::Validation(format!("{e:?}")))
}

/// A helper to compute the disk size of one Q4_L super-block.
///
/// Exposed so that the converter can compute expected output sizes.
pub fn q4l_superblock_size() -> usize {
    std::mem::size_of::<Q4LSuperBlock>()
}

/// Number of super-blocks needed to store `num_weights` Q4_L values.
pub fn q4l_block_count(num_weights: usize) -> usize {
    num_weights.div_ceil(DEFAULT_SUPER_BLOCK_SIZE)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{quantize_q4l, DEFAULT_SUPER_BLOCK_SIZE};

    fn make_test_model() -> LnsModel {
        let weights: Vec<f32> = (0..DEFAULT_SUPER_BLOCK_SIZE)
            .map(|i| (i as f32) / 128.0 - 1.0)
            .collect();
        let blocks = quantize_q4l(&weights);
        let data: Vec<u8> = bytemuck::cast_slice::<Q4LSuperBlock, u8>(&blocks).to_vec();

        LnsModel {
            version: FORMAT_VERSION,
            tensors: vec![LnsTensor {
                name: "test.weight".to_string(),
                shape: vec![16, 16],
                quant_type: QuantType::Q4L.as_u8(),
                data,
            }],
        }
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let model = make_test_model();
        let bytes = serialize_model(&model).expect("serialise");

        let archived = check_archived_model(&bytes).expect("validate");
        assert_eq!(archived.version, FORMAT_VERSION);
        assert_eq!(archived.tensors.len(), 1);
        assert_eq!(archived.tensors[0].name, "test.weight");
    }

    #[test]
    fn test_tensor_lookup() {
        let model = make_test_model();
        assert!(model.get_tensor("test.weight").is_some());
        assert!(model.get_tensor("does.not.exist").is_none());
    }

    #[test]
    fn test_decode_q4l_tensor() {
        let model = make_test_model();
        let tensor = model.get_tensor("test.weight").unwrap();
        let decoded = tensor.decode_q4l().expect("decode");
        assert_eq!(decoded.len(), DEFAULT_SUPER_BLOCK_SIZE);
    }

    #[test]
    fn test_q4l_block_count() {
        assert_eq!(q4l_block_count(256), 1);
        assert_eq!(q4l_block_count(257), 2);
        assert_eq!(q4l_block_count(512), 2);
        assert_eq!(q4l_block_count(0), 0);
    }

    #[test]
    fn test_v4_hq_quant_type_ids_are_stable() {
        assert_eq!(FORMAT_VERSION, 4);
        assert_eq!(HQ_FORMAT_VERSION, 4);
        assert_eq!(QuantType::from_u8(6), Some(QuantType::Q4HQ));
        assert_eq!(QuantType::from_u8(7), Some(QuantType::Q8HQ));
        assert_eq!(QuantType::from_u8(8), Some(QuantType::Q4HQM));
        assert_eq!(QuantType::from_u8(9), Some(QuantType::Q2HQ));
        assert_eq!(QuantType::from_u8(10), Some(QuantType::Q2HQM));
        assert_eq!(QuantType::Q4HQ.as_u8(), 6);
        assert_eq!(QuantType::Q2HQM.as_u8(), 10);
    }

    #[test]
    fn test_hq_tensors_require_hq_format_version() {
        let model = LnsModel {
            version: 2,
            tensors: vec![LnsTensor {
                name: "hq.weight".to_string(),
                shape: vec![1, 256],
                quant_type: QuantType::Q4HQ.as_u8(),
                data: Vec::new(),
            }],
        };

        assert!(validate_model_format_version(&model).is_err());

        let mut hq_model = model;
        hq_model.version = HQ_FORMAT_VERSION;
        validate_model_format_version(&hq_model).expect("HQ format version accepts HQ tensors");
    }
}
