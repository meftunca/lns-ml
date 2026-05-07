use thiserror::Error;

/// Unified error type for lns-core operations.
#[derive(Debug, Error)]
pub enum LnsError {
    /// Serialization failure (rkyv).
    #[error("serialization error: {0}")]
    Serialization(String),

    /// The byte slice does not contain a valid archived model.
    #[error("archive validation error: {0}")]
    Validation(String),

    /// A tensor's data length is inconsistent with its shape.
    #[error("tensor shape mismatch: expected {expected} elements but data holds {actual}")]
    ShapeMismatch { expected: usize, actual: usize },

    /// A tensor uses an unsupported quantization type.
    #[error("unsupported quantization type: {0}")]
    UnsupportedQuantType(u8),

    /// I/O error wrapper.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
