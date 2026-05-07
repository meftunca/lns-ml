//! lns-cli — inspect, benchmark, and run inference with lns-ml models.
//!
//! # Commands
//!
//! ```text
//! lns-cli inspect --model model.lns
//!     Print model metadata and per-tensor statistics.
//!
//! lns-cli bench --model model.lns [--tensor NAME] [--iters N]
//!     Measure Q4_L decode throughput.
//! ```

use std::{fs, path::PathBuf, time::Instant};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use memmap2::MmapOptions;

use lns_core::{
    format::{check_archived_model, QuantType},
    DEFAULT_SUPER_BLOCK_SIZE,
};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "lns-ml — LNS quantised LLM inference engine",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Print model metadata and per-tensor statistics.
    Inspect {
        /// Path to the `.lns` model file.
        #[arg(short, long)]
        model: PathBuf,

        /// Show detailed per-tensor byte sizes.
        #[arg(short, long)]
        verbose: bool,
    },

    /// Measure Q4_L decode throughput (tokens/s proxy).
    Bench {
        /// Path to the `.lns` model file.
        #[arg(short, long)]
        model: PathBuf,

        /// Restrict benchmark to a single tensor (substring match on name).
        #[arg(short, long)]
        tensor: Option<String>,

        /// Number of decode iterations.
        #[arg(short, long, default_value_t = 10)]
        iters: usize,
    },
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn human_bytes(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2} GB", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2} MB", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.2} KB", n as f64 / 1e3)
    } else {
        format!("{n} B")
    }
}

fn quant_label(code: u8) -> &'static str {
    match QuantType::from_u8(code) {
        Some(QuantType::F32) => "F32",
        Some(QuantType::F16) => "F16",
        Some(QuantType::Q4L) => "Q4_L",
        None => "unknown",
    }
}

// ── Sub-commands ──────────────────────────────────────────────────────────────

fn cmd_inspect(model_path: &PathBuf, verbose: bool) -> Result<()> {
    let file =
        fs::File::open(model_path).with_context(|| format!("cannot open '{}'", model_path.display()))?;
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .with_context(|| format!("cannot mmap '{}'", model_path.display()))?
    };

    let archived = check_archived_model(&mmap)
        .with_context(|| format!("'{}' is not a valid .lns model", model_path.display()))?;

    println!("Model:   {}", model_path.display());
    println!("Version: {}", archived.version);
    println!("Tensors: {}", archived.tensors.len());

    let total_bytes: usize = archived.tensors.iter().map(|t| t.data.len()).sum::<usize>();
    let total_elements: usize = archived
        .tensors
        .iter()
        .map(|t| t.shape.iter().map(|&d| d as usize).product::<usize>())
        .sum();
    println!("Total:   {} elements / {}", total_elements, human_bytes(total_bytes));

    // Per-type breakdown.
    let mut q4l_bytes = 0usize;
    let mut f16_bytes = 0usize;
    let mut f32_bytes = 0usize;
    for t in archived.tensors.iter() {
        match QuantType::from_u8(t.quant_type) {
            Some(QuantType::Q4L) => q4l_bytes += t.data.len(),
            Some(QuantType::F16) => f16_bytes += t.data.len(),
            _ => f32_bytes += t.data.len(),
        }
    }
    if q4l_bytes > 0 {
        println!("  Q4_L: {}", human_bytes(q4l_bytes));
    }
    if f16_bytes > 0 {
        println!("  F16:  {}", human_bytes(f16_bytes));
    }
    if f32_bytes > 0 {
        println!("  F32:  {}", human_bytes(f32_bytes));
    }

    if verbose {
        println!("\n{:<60}  {:>8}  {:>10}  {:>12}", "name", "type", "elements", "bytes");
        println!("{}", "-".repeat(96));
        for t in archived.tensors.iter() {
            let shape: Vec<usize> = t.shape.iter().map(|&d| d as usize).collect();
            let n_elements: usize = shape.iter().product();
            println!(
                "{:<60}  {:>8}  {:>10}  {:>12}",
                t.name.as_str(),
                quant_label(t.quant_type),
                n_elements,
                human_bytes(t.data.len()),
            );
        }
    }

    Ok(())
}

fn cmd_bench(model_path: &PathBuf, tensor_filter: Option<&str>, iters: usize) -> Result<()> {
    if iters == 0 {
        bail!("--iters must be > 0");
    }

    let file =
        fs::File::open(model_path).with_context(|| format!("cannot open '{}'", model_path.display()))?;
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .with_context(|| format!("cannot mmap '{}'", model_path.display()))?
    };

    let archived = check_archived_model(&mmap)
        .with_context(|| format!("'{}' is not a valid .lns model", model_path.display()))?;

    // Collect Q4_L tensors matching the optional filter.
    let q4l_tensors: Vec<_> = archived
        .tensors
        .iter()
        .filter(|t| t.quant_type == QuantType::Q4L.as_u8())
        .filter(|t| {
            tensor_filter
                .map(|f| t.name.as_str().contains(f))
                .unwrap_or(true)
        })
        .collect();

    if q4l_tensors.is_empty() {
        bail!("no Q4_L tensors found matching the filter");
    }

    let total_blocks: usize = q4l_tensors
        .iter()
        .map(|t| t.data.len() / std::mem::size_of::<lns_core::Q4LSuperBlock>())
        .sum();
    let total_weights = total_blocks * DEFAULT_SUPER_BLOCK_SIZE;

    println!(
        "Benchmarking {} Q4_L tensor(s)  ({} super-blocks, {} weights)  ×{} iters",
        q4l_tensors.len(),
        total_blocks,
        total_weights,
        iters,
    );

    let t_start = Instant::now();
    let mut checksum = 0.0_f32; // prevent dead-code elimination

    for _ in 0..iters {
        for t in &q4l_tensors {
            let n_elements: usize = t.shape.iter().map(|&d| d as usize).product();
            let blocks: &[lns_core::Q4LSuperBlock] = bytemuck::cast_slice(&t.data);
            let decoded = lns_core::dequantize_q4l(blocks, n_elements);
            // Accumulate into checksum to prevent the compiler from
            // eliminating the decode entirely.
            checksum += decoded.first().copied().unwrap_or(0.0);
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    let weights_per_sec = (total_weights * iters) as f64 / elapsed;

    println!(
        "Elapsed: {elapsed:.3}s  |  {:.2} M weights/s  (checksum: {checksum:.6})",
        weights_per_sec / 1e6,
    );

    Ok(())
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Command::Inspect { model, verbose } => cmd_inspect(model, *verbose),
        Command::Bench {
            model,
            tensor,
            iters,
        } => cmd_bench(model, tensor.as_deref(), *iters),
    }
}
