//! lns-convert — convert safetensors models to the `.lns` format.
//!
//! # Usage
//!
//! ```text
//! lns-convert --input  model.safetensors \
//!             --output model.lns         \
//!             --quant  Q4_L              \
//!             --super-block 256
//! ```

use std::{
    fmt,
    fs,
    path::PathBuf,
    time::Instant,
};

use anyhow::{bail, Context, Result};
use bytemuck::cast_slice;
use clap::Parser;
use half::f16;
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};

use lns_core::{
    format::{serialize_model, LnsModel, LnsTensor, QuantType, FORMAT_VERSION},
    quantize_q4l, Q4LSuperBlock, DEFAULT_SUPER_BLOCK_SIZE,
};

// ── CLI ───────────────────────────────────────────────────────────────────────

/// Supported quantisation formats.
#[derive(Clone, Debug, PartialEq, Eq)]
enum QuantFormat {
    Q4L,
    F16,
    F32,
}

impl fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Q4L => write!(f, "Q4_L"),
            Self::F16 => write!(f, "F16"),
            Self::F32 => write!(f, "F32"),
        }
    }
}

impl std::str::FromStr for QuantFormat {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "Q4_L" | "Q4L" => Ok(Self::Q4L),
            "F16" => Ok(Self::F16),
            "F32" => Ok(Self::F32),
            other => bail!("unknown quantisation format '{other}' (valid: Q4_L, F16, F32)"),
        }
    }
}

/// Convert a safetensors model to the lns-ml `.lns` format.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input safetensors file.
    #[arg(short, long)]
    input: PathBuf,

    /// Output `.lns` file.
    #[arg(short, long)]
    output: PathBuf,

    /// Quantisation format for weight tensors (Q4_L, F16, F32).
    #[arg(short, long, default_value = "Q4_L")]
    quant: QuantFormat,

    /// Super-block size for Q4_L (must be a multiple of 32; default 256).
    /// Currently informational only — the super-block size is fixed at 256
    /// in lns-core v0.1.
    #[arg(long, default_value_t = DEFAULT_SUPER_BLOCK_SIZE)]
    super_block: usize,

    /// Minimum number of elements a tensor must have to be weight-quantised.
    /// Tensors smaller than this threshold are stored as F32.
    #[arg(long, default_value_t = DEFAULT_SUPER_BLOCK_SIZE)]
    min_quant_elements: usize,

    /// Suppress progress output.
    #[arg(long)]
    quiet: bool,
}

// ── Conversion helpers ────────────────────────────────────────────────────────

/// Return `true` if a tensor should be quantised with the requested format
/// rather than stored as F32.
///
/// Rules:
/// - Tensor must be at least 2-D (weight matrices, embedding tables).
/// - Tensor must contain more than `min_elements` values.
/// - 1-D tensors (bias, layer-norm gamma/beta) are always F32.
fn should_quantize(shape: &[usize], min_elements: usize) -> bool {
    if shape.len() < 2 {
        return false;
    }
    let n: usize = shape.iter().product();
    n >= min_elements
}

/// Convert a raw byte slice from a safetensors tensor to a `Vec<f32>`.
///
/// Uses element-wise byte reads so the input slice need not be aligned.
fn to_f32_vec(dtype: Dtype, raw: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        Dtype::F32 => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),

        Dtype::F16 => raw
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),

        Dtype::BF16 => {
            // BF16 = top 16 bits of an IEEE-754 f32.
            raw.chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        }

        Dtype::I8 => {
            // Dequantise int8 assuming symmetric scale = 1/127.
            raw.iter().map(|&v| (v as i8) as f32 / 127.0).collect()
        }

        other => bail!("unsupported dtype {other:?}"),
    })
}

/// Convert `f32` values to packed F16 bytes (u16 bit-patterns).
fn f32_to_f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
    }
    out
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate super-block size.
    if args.super_block % 32 != 0 || args.super_block < 64 || args.super_block > 512 {
        bail!(
            "--super-block must be a multiple of 32 in [64, 512], got {}",
            args.super_block
        );
    }
    if args.super_block != DEFAULT_SUPER_BLOCK_SIZE && !args.quiet {
        eprintln!(
            "warning: --super-block {} requested but lns-core v0.1 uses a fixed size of {}; \
             the value is recorded in the archive but has no effect on encoding.",
            args.super_block, DEFAULT_SUPER_BLOCK_SIZE
        );
    }

    let t0 = Instant::now();

    // ── Open and mmap the safetensors file ───────────────────────────────────
    if !args.quiet {
        println!("Reading {:?} …", args.input);
    }
    let input_file = fs::File::open(&args.input)
        .with_context(|| format!("cannot open input file '{}'", args.input.display()))?;
    let mmap = unsafe {
        MmapOptions::new()
            .map(&input_file)
            .with_context(|| format!("cannot mmap '{}'", args.input.display()))?
    };

    let safetensors = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("failed to parse safetensors file '{}'", args.input.display()))?;

    // ── Convert tensors ──────────────────────────────────────────────────────
    let mut lns_tensors: Vec<LnsTensor> = Vec::new();
    let mut q4l_count = 0usize;
    let mut f32_count = 0usize;
    let mut f16_count = 0usize;
    let mut skipped = 0usize;

    let tensor_list: Vec<_> = safetensors.tensors();

    for (name, tensor_view) in &tensor_list {
        let shape_usize: Vec<usize> = tensor_view.shape().to_vec();
        let shape: Vec<u64> = shape_usize.iter().map(|&s| s as u64).collect();
        let raw = tensor_view.data();
        let dtype = tensor_view.dtype();

        let f32_values = match to_f32_vec(dtype, raw) {
            Ok(v) => v,
            Err(e) => {
                if !args.quiet {
                    eprintln!("  skip {name}: {e}");
                }
                skipped += 1;
                continue;
            }
        };

        let (quant_type, data) =
            if args.quant == QuantFormat::Q4L && should_quantize(&shape_usize, args.min_quant_elements) {
                let blocks = quantize_q4l(&f32_values);
                let bytes = cast_slice::<Q4LSuperBlock, u8>(&blocks).to_vec();
                q4l_count += 1;
                (QuantType::Q4L.as_u8(), bytes)
            } else if args.quant == QuantFormat::F16 && should_quantize(&shape_usize, args.min_quant_elements) {
                let bytes = f32_to_f16_bytes(&f32_values);
                f16_count += 1;
                (QuantType::F16.as_u8(), bytes)
            } else {
                let bytes = cast_slice::<f32, u8>(&f32_values).to_vec();
                f32_count += 1;
                (QuantType::F32.as_u8(), bytes)
            };

        if !args.quiet {
            let orig_bytes = raw.len();
            let quant_bytes = data.len();
            let ratio = orig_bytes as f64 / quant_bytes.max(1) as f64;
            let qt_label = QuantType::from_u8(quant_type)
                .map(|q| format!("{q:?}"))
                .unwrap_or_else(|| format!("unknown({quant_type})"));
            println!(
                "  {name:60} {shape_usize:?} → {qt_label} \
                 ({orig_bytes} → {quant_bytes} bytes, {ratio:.2}×)"
            );
        }

        lns_tensors.push(LnsTensor {
            name: name.to_string(),
            shape,
            quant_type,
            data,
        });
    }

    // ── Assemble and write model ─────────────────────────────────────────────
    let model = LnsModel {
        version: FORMAT_VERSION,
        tensors: lns_tensors,
    };

    let total_orig: usize = tensor_list
        .iter()
        .map(|(_, tv)| tv.data().len())
        .sum();
    let total_quant = model.total_bytes();

    if !args.quiet {
        println!(
            "\nTensors: {} Q4_L  {} F16  {} F32  {} skipped",
            q4l_count, f16_count, f32_count, skipped
        );
        println!(
            "Size:    {:.2} MB → {:.2} MB  ({:.2}× compression)",
            total_orig as f64 / 1e6,
            total_quant as f64 / 1e6,
            total_orig as f64 / total_quant.max(1) as f64,
        );
    }

    let serialized =
        serialize_model(&model).context("failed to serialise model")?;

    fs::write(&args.output, &serialized)
        .with_context(|| format!("cannot write output file '{}'", args.output.display()))?;

    if !args.quiet {
        println!(
            "Written {:?}  ({:.2} MB)  in {:.2}s",
            args.output,
            serialized.len() as f64 / 1e6,
            t0.elapsed().as_secs_f64(),
        );
    }

    Ok(())
}
