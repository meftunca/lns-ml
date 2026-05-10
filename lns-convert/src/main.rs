//! lns-convert — convert safetensors models to the `.lns` format.

mod input;

use std::{fmt, fs, io::BufWriter, path::PathBuf, time::Instant};

use anyhow::{bail, Result};
use bytemuck::cast_slice;
use clap::Parser;
use half::f16;
use safetensors::Dtype;

use lns_core::{
    format::{serialize_model_to_writer, LnsModel, LnsTensor, QuantType, FORMAT_VERSION},
    DEFAULT_SUPER_BLOCK_SIZE,
};

use crate::input::{input_tensor_names, visit_input_tensors};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
enum QuantFormat {
    Q2L,
    Q4L,
    Q4HQ,
    Q8L,
    F16,
    F32,
}

impl fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Q2L => write!(f, "Q2_L"),
            Self::Q4L => write!(f, "Q4_L"),
            Self::Q4HQ => write!(f, "Q4_HQ"),
            Self::Q8L => write!(f, "Q8_L"),
            Self::F16 => write!(f, "F16"),
            Self::F32 => write!(f, "F32"),
        }
    }
}

impl std::str::FromStr for QuantFormat {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "Q2_L" | "Q2L" => Ok(Self::Q2L),
            "Q4_L" | "Q4L" => Ok(Self::Q4L),
            "Q4_HQ" | "Q4HQ" => Ok(Self::Q4HQ),
            "Q8_L" | "Q8L" => Ok(Self::Q8L),
            "F16" => Ok(Self::F16),
            "F32" => Ok(Self::F32),
            other => bail!(
                "unknown quantisation format '{other}' (valid: Q2_L, Q4_L, Q4_HQ, Q8_L, F16, F32)"
            ),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long)]
    output: PathBuf,

    #[arg(short, long, default_value = "Q4_L")]
    quant: QuantFormat,

    #[arg(long, default_value_t = DEFAULT_SUPER_BLOCK_SIZE)]
    super_block: usize,

    #[arg(long, default_value_t = DEFAULT_SUPER_BLOCK_SIZE)]
    min_quant_elements: usize,

    #[arg(long)]
    quiet: bool,

    #[arg(
        long,
        help = "Skip known non-text branches such as Qwen vision/MTP tensors during conversion"
    )]
    text_only: bool,

    /// Copy config.json, tokenizer.json and other metadata files from the input
    /// directory into the same directory as the output .lns file, producing a
    /// self-contained runnable bundle.
    #[arg(long)]
    bundle: bool,

    /// Spec v3.3 sensitivity policy: when the base quant is Q4_L, promote the
    /// PPL-sensitive output projections (`o_proj`, `down_proj`) to Q8_L.
    /// Roughly +0.4 bits/weight average, ~30-50% PPL reduction. No effect for
    /// non-Q4_L base quants.
    #[arg(long)]
    mixed_precision: bool,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn to_f32_vec(dtype: Dtype, raw: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => Ok(cast_slice::<u8, f32>(raw).to_vec()),
        Dtype::F16 => {
            let f16s = cast_slice::<u8, f16>(raw);
            Ok(f16s.iter().map(|&v| v.to_f32()).collect())
        }
        Dtype::BF16 => {
            let bf16s = cast_slice::<u8, half::bf16>(raw);
            Ok(bf16s.iter().map(|&v| v.to_f32()).collect())
        }
        _ => bail!("unsupported dtype {:?}", dtype),
    }
}

fn should_quantize(shape: &[usize], min_elements: usize) -> bool {
    let n: usize = shape.iter().product();
    n >= min_elements && n % 32 == 0
}

fn keep_high_precision(name: &str) -> bool {
    name.contains("embed_tokens")
        || name.contains("lm_head")
        || name.ends_with("norm.weight")
        || name.ends_with("layernorm.weight")
        || name.ends_with("linear_attn.conv1d.weight")
        || name.ends_with("linear_attn.conv1d.bias")
        || name.ends_with("linear_attn.A_log")
        || name.ends_with("linear_attn.dt_bias")
        || name.ends_with(".gate.weight") // MoE router — keep F16 for routing precision
}

fn skip_for_text_only_runtime(name: &str) -> bool {
    name.starts_with("model.visual.") || name.starts_with("mtp.")
}

/// Spec v3.3 sensitivity policy. Returns `true` for output-projection tensors
/// that are empirically 3-5× more sensitive to Q4 quantisation error than
/// other linear weights, and so should be promoted to Q8_L when the
/// `--mixed-precision` flag is on.
///
/// The classifier is purely name-based and matches the standard LLaMA /
/// Qwen / Mistral naming conventions:
///   * `o_proj`     — attention output projection
///   * `down_proj`  — MLP down projection (post-SwiGLU)
///   * `out_proj`   — alt name (some Qwen variants, linear-attn)
fn is_sensitive_for_q8_promotion(name: &str) -> bool {
    name.ends_with(".o_proj.weight")
        || name.ends_with(".down_proj.weight")
        || name.ends_with(".out_proj.weight")
}

fn is_embedding_weight(name: &str) -> bool {
    name.ends_with("embed_tokens.weight") || name == "tok_embeddings.weight"
}

fn has_explicit_lm_head(names: &[String]) -> bool {
    names
        .iter()
        .any(|name| name == "lm_head.weight" || name == "model.language_model.lm_head.weight")
}

fn f32_to_f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
    }
    out
}

fn raw_to_f16_bytes(dtype: Dtype, raw: Vec<u8>) -> Result<Vec<u8>> {
    match dtype {
        Dtype::F16 => Ok(raw),
        Dtype::BF16 => {
            let bf16s = cast_slice::<u8, half::bf16>(&raw);
            let mut out = Vec::with_capacity(bf16s.len() * 2);
            for &v in bf16s {
                out.extend_from_slice(&f16::from_f32(v.to_f32()).to_bits().to_le_bytes());
            }
            Ok(out)
        }
        Dtype::F32 => {
            let f32s = cast_slice::<u8, f32>(&raw);
            let mut out = Vec::with_capacity(f32s.len() * 2);
            for &v in f32s {
                out.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
            }
            Ok(out)
        }
        _ => bail!("unsupported dtype {:?}", dtype),
    }
}

fn convert_tensor(
    name: &str,
    shape_usize: &[usize],
    dtype: Dtype,
    raw: Vec<u8>,
    args: &Args,
) -> Result<LnsTensor> {
    let shape: Vec<u64> = shape_usize.iter().map(|&s| s as u64).collect();
    let keep_fp = keep_high_precision(name);
    let can_quantize = should_quantize(shape_usize, args.min_quant_elements);
    let promote_to_q8 = args.mixed_precision
        && args.quant == QuantFormat::Q4L
        && !keep_fp
        && is_sensitive_for_q8_promotion(name)
        && can_quantize;

    let (quant_type, data) = if keep_fp {
        (QuantType::F16.as_u8(), raw_to_f16_bytes(dtype, raw)?)
    } else {
        let f32_values = to_f32_vec(dtype, &raw)?;
        if promote_to_q8 {
            (
                QuantType::Q8L.as_u8(),
                cast_slice::<lns_core::Q8LSuperBlock, u8>(&lns_core::quantize_q8l(&f32_values))
                    .to_vec(),
            )
        } else if args.quant == QuantFormat::Q2L && can_quantize {
            (
                QuantType::Q2L.as_u8(),
                cast_slice::<lns_core::Q2LSuperBlock, u8>(&lns_core::quantize_q2l(&f32_values))
                    .to_vec(),
            )
        } else if args.quant == QuantFormat::Q4L && can_quantize {
            (
                QuantType::Q4L.as_u8(),
                cast_slice::<lns_core::Q4LSuperBlock, u8>(&lns_core::quantize_q4l(&f32_values))
                    .to_vec(),
            )
        } else if args.quant == QuantFormat::Q4HQ && can_quantize {
            let blocks = lns_core::quantize_q4hq(&f32_values);
            (
                QuantType::Q4HQ.as_u8(),
                lns_core::q4hq_blocks_to_payload(&blocks),
            )
        } else if args.quant == QuantFormat::Q8L && can_quantize {
            (
                QuantType::Q8L.as_u8(),
                cast_slice::<lns_core::Q8LSuperBlock, u8>(&lns_core::quantize_q8l(&f32_values))
                    .to_vec(),
            )
        } else if args.quant == QuantFormat::F16 && can_quantize {
            (QuantType::F16.as_u8(), f32_to_f16_bytes(&f32_values))
        } else {
            (
                QuantType::F32.as_u8(),
                cast_slice::<f32, u8>(&f32_values).to_vec(),
            )
        }
    };

    Ok(LnsTensor {
        name: name.to_string(),
        quant_type,
        shape,
        data,
    })
}

fn convert_tied_lm_head_duplicate(
    shape_usize: &[usize],
    dtype: Dtype,
    raw: &[u8],
    args: &Args,
) -> Result<Option<LnsTensor>> {
    if !should_quantize(shape_usize, args.min_quant_elements) {
        return Ok(None);
    }

    let f32_values = to_f32_vec(dtype, raw)?;
    let (quant_type, data) = match args.quant {
        QuantFormat::Q4HQ => {
            let blocks = lns_core::quantize_q4hq(&f32_values);
            (
                QuantType::Q4HQ.as_u8(),
                lns_core::q4hq_blocks_to_payload(&blocks),
            )
        }
        QuantFormat::Q4L | QuantFormat::Q8L | QuantFormat::Q2L => (
            QuantType::Q4L.as_u8(),
            cast_slice::<lns_core::Q4LSuperBlock, u8>(&lns_core::quantize_q4l(&f32_values))
                .to_vec(),
        ),
        QuantFormat::F16 | QuantFormat::F32 => return Ok(None),
    };

    Ok(Some(LnsTensor {
        name: "lm_head.weight".to_string(),
        quant_type,
        shape: shape_usize.iter().map(|&s| s as u64).collect(),
        data,
    }))
}

/// Files that must be present for the bundle to be usable.
const BUNDLE_REQUIRED: &[&str] = &["config.json", "tokenizer.json"];

/// Files that are copied when present but are not mandatory.
const BUNDLE_OPTIONAL: &[&str] = &[
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
];

/// Copy metadata files from `src_dir` into `dest_dir` so the bundle is
/// self-contained.  Returns an error only if a *required* file is missing or
/// a copy fails; optional files are silently skipped when absent.
fn copy_bundle_files(
    src_dir: &std::path::Path,
    dest_dir: &std::path::Path,
    quiet: bool,
) -> Result<()> {
    for &name in BUNDLE_REQUIRED {
        let src = src_dir.join(name);
        if !src.exists() {
            bail!(
                "--bundle: required file '{}' not found in {:?}",
                name,
                src_dir
            );
        }
        let dst = dest_dir.join(name);
        fs::copy(&src, &dst)?;
        if !quiet {
            println!("  bundled: {name}");
        }
    }
    for &name in BUNDLE_OPTIONAL {
        let src = src_dir.join(name);
        if src.exists() {
            let dst = dest_dir.join(name);
            fs::copy(&src, &dst)?;
            if !quiet {
                println!("  bundled: {name}");
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();

    // ── Phase 1: I/O + quantization — one tensor at a time ──────────────────
    if !args.quiet {
        println!("Reading {:?} …", args.input);
    }

    let mut lns_tensors = Vec::new();
    let mut skipped_count = 0usize;
    let mut seen_count = 0usize;
    let mut q2l_count = 0usize;
    let mut q4l_count = 0usize;
    let mut q4hq_count = 0usize;
    let mut q8l_count = 0usize;
    let mut f32_count = 0usize;
    let mut f16_count = 0usize;
    let tensor_names = input_tensor_names(&args.input)?;
    let should_duplicate_tied_output = !has_explicit_lm_head(&tensor_names);

    visit_input_tensors(
        &args.input,
        |name| {
            let keep = !(args.text_only && skip_for_text_only_runtime(name));
            if !keep {
                skipped_count += 1;
            }
            keep
        },
        |name, dtype, shape, data| {
            let tied_output_duplicate = if should_duplicate_tied_output && is_embedding_weight(name)
            {
                convert_tied_lm_head_duplicate(shape, dtype, &data, &args)?
            } else {
                None
            };
            let tensor = convert_tensor(name, shape, dtype, data, &args)?;
            match tensor.quant_type {
                x if x == QuantType::Q2L.as_u8() => q2l_count += 1,
                x if x == QuantType::Q4L.as_u8() => q4l_count += 1,
                x if x == QuantType::Q4HQ.as_u8() => q4hq_count += 1,
                x if x == QuantType::Q8L.as_u8() => q8l_count += 1,
                x if x == QuantType::F16.as_u8() => f16_count += 1,
                _ => f32_count += 1,
            }
            lns_tensors.push(tensor);
            seen_count += 1;

            if let Some(tensor) = tied_output_duplicate {
                match tensor.quant_type {
                    x if x == QuantType::Q4L.as_u8() => q4l_count += 1,
                    x if x == QuantType::Q4HQ.as_u8() => q4hq_count += 1,
                    _ => f32_count += 1,
                }
                lns_tensors.push(tensor);
                seen_count += 1;
            }

            if !args.quiet && seen_count % 10 == 0 {
                eprintln!("  [{seen_count}] {name}");
            }
            Ok(())
        },
    )?;

    if !args.quiet {
        println!(
            "Converted {seen_count} tensors in {:.2?} — serializing …",
            t0.elapsed(),
        );
    }

    let model = LnsModel {
        version: FORMAT_VERSION,
        tensors: lns_tensors,
    };

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let output = fs::File::create(&args.output)?;
    serialize_model_to_writer(&model, BufWriter::new(output))?;

    if args.bundle {
        // Resolve the input directory (handle file-path inputs too).
        let src_dir = if args.input.is_dir() {
            args.input.clone()
        } else {
            args.input
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."))
        };
        let dest_dir = args
            .output
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        if !args.quiet {
            println!("Bundling metadata …");
        }
        copy_bundle_files(&src_dir, &dest_dir, args.quiet)?;
    }

    if !args.quiet {
        println!("Conversion finished in {:?}", t0.elapsed());
        println!("Summary:");
        println!("  Q2_L: {}", q2l_count);
        println!("  Q4_L: {}", q4l_count);
        println!("  Q4_HQ: {}", q4hq_count);
        println!("  Q8_L: {}", q8l_count);
        println!("  F16:  {}", f16_count);
        println!("  F32:  {}", f32_count);
        println!("  Skipped: {}", skipped_count);
        println!("Output saved to {:?}", args.output);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{keep_high_precision, skip_for_text_only_runtime};

    #[test]
    fn keeps_qwen35_cpu_only_linear_attention_tensors_high_precision() {
        assert!(keep_high_precision(
            "model.language_model.layers.0.linear_attn.conv1d.weight"
        ));
        assert!(keep_high_precision(
            "model.language_model.layers.0.linear_attn.A_log"
        ));
        assert!(keep_high_precision(
            "model.language_model.layers.0.linear_attn.dt_bias"
        ));
        assert!(keep_high_precision(
            "model.language_model.layers.0.linear_attn.norm.weight"
        ));
    }

    #[test]
    fn still_allows_quantizing_large_projection_weights() {
        assert!(!keep_high_precision(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        ));
        assert!(!keep_high_precision(
            "model.language_model.layers.0.linear_attn.out_proj.weight"
        ));
    }

    #[test]
    fn text_only_mode_skips_qwen_non_text_branches() {
        assert!(skip_for_text_only_runtime(
            "model.visual.blocks.0.attn.qkv.weight"
        ));
        assert!(skip_for_text_only_runtime(
            "mtp.layers.0.self_attn.q_proj.weight"
        ));
        assert!(!skip_for_text_only_runtime(
            "model.language_model.layers.0.self_attn.q_proj.weight"
        ));
    }

    #[test]
    fn bundle_copies_required_and_optional_files() {
        use super::copy_bundle_files;
        use std::fs;

        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();

        // Write required files.
        fs::write(src.path().join("config.json"), b"{}").unwrap();
        fs::write(src.path().join("tokenizer.json"), b"{}").unwrap();
        // Write one optional file.
        fs::write(src.path().join("chat_template.jinja"), b"tmpl").unwrap();

        copy_bundle_files(src.path(), dst.path(), true).unwrap();

        assert!(dst.path().join("config.json").exists());
        assert!(dst.path().join("tokenizer.json").exists());
        assert!(dst.path().join("chat_template.jinja").exists());
        // Non-existent optional file must NOT be created.
        assert!(!dst.path().join("merges.txt").exists());
    }

    #[test]
    fn bundle_errors_when_required_file_is_missing() {
        use super::copy_bundle_files;

        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        // Deliberately omit config.json / tokenizer.json.

        let result = copy_bundle_files(src.path(), dst.path(), true);
        assert!(result.is_err());
    }
}
