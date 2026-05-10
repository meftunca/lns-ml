//! lns-cli — inspect, benchmark, and run inference with lns-ml models.

use std::{
    collections::{HashMap, VecDeque},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use memmap2::MmapOptions;

use lns_core::{
    format::{check_archived_model, QuantType},
    DEFAULT_SUPER_BLOCK_SIZE,
};
use lns_inference::Tokenizer;
use lns_metal::{bench_q4l_decode, decode_quality_report, ComputeBackend};

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

        /// Decode backend to benchmark.
        #[arg(long, value_enum, default_value_t = BenchBackend::Cpu)]
        backend: BenchBackend,
    },

    /// Codec-roundtrip quality benchmark for Q4_L tensors.
    Perplexity {
        /// Path to the `.lns` model file.
        #[arg(short, long)]
        model: PathBuf,

        /// Restrict analysis to tensors whose name contains this substring.
        #[arg(short, long)]
        tensor: Option<String>,
    },

    /// Start an interactive chat session with a model.
    Chat {
        /// Path to the `.lns` model file.
        #[arg(short, long)]
        model: PathBuf,

        /// Optional one-shot prompt. If provided, chat runs once and exits.
        #[arg(long)]
        prompt: Option<String>,

        /// Temperature for sampling.
        #[arg(short, long, default_value_t = 0.8)]
        temp: f32,

        /// Nucleus sampling threshold.
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,

        /// Minimum probability threshold relative to the highest-probability token.
        /// When > 0, Min-P sampling takes precedence over top-p.
        #[arg(long, default_value_t = 0.0)]
        min_p: f32,

        /// Multiplicative repetition penalty (>1 discourages repeats).
        #[arg(long, default_value_t = 1.15)]
        repeat_penalty: f32,

        /// Number of recent tokens considered for repetition penalties.
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Additive penalty applied once to any token that already appeared in the recent window.
        #[arg(long, default_value_t = 0.0)]
        presence_penalty: f32,

        /// Additive penalty scaled by token frequency in the recent window.
        #[arg(long, default_value_t = 0.0)]
        frequency_penalty: f32,

        /// Maximum number of new tokens to generate.
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
    },

    /// Diagnose whether a model matches the currently supported runtime surface.
    Doctor {
        /// Path to a model directory or `.lns` model file.
        #[arg(short, long)]
        model: PathBuf,

        /// Show the full missing-tensor list when unsupported.
        #[arg(short, long)]
        verbose: bool,
    },

    /// Print prompt-time top-k logits and per-layer RMS trace for debugging.
    DebugPrompt {
        /// Path to the `.lns` model file.
        #[arg(short, long)]
        model: PathBuf,

        /// Prompt to evaluate.
        #[arg(long)]
        prompt: String,

        /// Number of top logits to print.
        #[arg(long, default_value_t = 10)]
        top_k: usize,

        /// Force CPU GEMV paths for quantized matmuls during the diagnostic run.
        #[arg(long, default_value_t = false)]
        cpu_gemv: bool,

        /// Override prompt template for the diagnostic run.
        #[arg(long)]
        template: Option<ChatTemplate>,

        /// Evaluate the prompt exactly as provided, without applying a chat template.
        #[arg(long, default_value_t = false)]
        raw_prompt: bool,

        /// Leave Qwen thinking open instead of appending a closed thinking block.
        #[arg(long, default_value_t = false)]
        qwen_thinking: bool,
    },

    /// Measure true token-level perplexity (NLL) on a text corpus.
    ///
    /// Tokenises the input, runs a sliding context forward pass,
    /// computes mean negative log-likelihood, and reports exp(NLL) as PPL.
    Ppl {
        /// Path to the `.lns` model file.
        #[arg(short, long)]
        model: PathBuf,

        /// UTF-8 plain-text file to evaluate.  Defaults to a built-in
        /// 500-word calibration passage when omitted.
        #[arg(long)]
        text: Option<PathBuf>,

        /// Maximum context window for each forward pass (tokens).
        #[arg(long, default_value_t = 512)]
        context: usize,

        /// Stride between windows (0 = non-overlapping; ≤ context = sliding window).
        /// Smaller stride → more accurate but slower.
        #[arg(long, default_value_t = 256)]
        stride: usize,

        /// Cap the corpus to the first N tokens (after BOS). 0 = no cap.
        /// Useful for quick smoke runs on big files like wikitext.test.
        #[arg(long, default_value_t = 0)]
        max_tokens: usize,
    },

    /// End-to-end benchmark: TTFT, tokens/s, peak memory, and PPL summary.
    Eval {
        /// Path to the `.lns` model file.
        #[arg(short, long)]
        model: PathBuf,

        /// Number of new tokens to generate per prompt.
        #[arg(long, default_value_t = 64)]
        gen_tokens: usize,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BenchBackend {
    Cpu,
    Metal,
}

#[derive(Debug, Clone, Copy)]
struct SamplingConfig {
    temp: f32,
    top_p: f32,
    min_p: f32,
    repeat_penalty: f32,
    repeat_last_n: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
}

impl SamplingConfig {
    fn sanitized(self) -> Self {
        Self {
            temp: self.temp,
            top_p: self.top_p.clamp(0.0, 1.0).max(1e-6),
            min_p: self.min_p.clamp(0.0, 1.0),
            repeat_penalty: if self.repeat_penalty > 0.0 {
                self.repeat_penalty
            } else {
                1.0
            },
            repeat_last_n: self.repeat_last_n,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
        }
    }
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
        Some(QuantType::Q2L) => "Q2_L",
        Some(QuantType::Q6L) => "Q6_L",
        Some(QuantType::Q8L) => "Q8_L",
        Some(QuantType::Q4HQ) => "Q4_HQ",
        Some(QuantType::Q8HQ) => "Q8_HQ",
        Some(QuantType::Q4HQM) => "Q4_HQM",
        Some(QuantType::Q2HQ) => "Q2_HQ",
        Some(QuantType::Q2HQM) => "Q2_HQM",
        None => "unknown",
    }
}

fn apply_sampling_penalties(
    logits: &mut [f32],
    recent_tokens: &VecDeque<u32>,
    repeat_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
) {
    let mut counts = HashMap::<u32, usize>::new();
    for &token in recent_tokens.iter() {
        *counts.entry(token).or_insert(0) += 1;
    }

    if presence_penalty != 0.0 || frequency_penalty != 0.0 {
        for (&token, &count) in &counts {
            let idx = token as usize;
            if idx >= logits.len() {
                continue;
            }
            let penalty = presence_penalty + frequency_penalty * count as f32;
            logits[idx] -= penalty;
        }
    }

    if (repeat_penalty - 1.0).abs() > f32::EPSILON {
        for &token in recent_tokens.iter() {
            let idx = token as usize;
            if idx >= logits.len() {
                continue;
            }
            let value = logits[idx];
            logits[idx] = if value > 0.0 {
                value / repeat_penalty
            } else {
                value * repeat_penalty
            };
        }
    }
}

fn sample_next_token(logits: &[f32], sampling: SamplingConfig) -> u32 {
    if sampling.min_p > 0.0 {
        lns_inference::sampler::sample_min_p(logits, sampling.temp, sampling.min_p)
    } else {
        lns_inference::sampler::sample(logits, sampling.temp, sampling.top_p)
    }
}

fn can_use_gpu_greedy(sampling: SamplingConfig) -> bool {
    sampling.temp <= 0.0
        && sampling.min_p == 0.0
        && (sampling.repeat_penalty - 1.0).abs() <= f32::EPSILON
        && sampling.presence_penalty == 0.0
        && sampling.frequency_penalty == 0.0
}

// ── Sub-commands ──────────────────────────────────────────────────────────────

fn cmd_inspect(model_path: &PathBuf, verbose: bool) -> Result<()> {
    let file = fs::File::open(model_path)
        .with_context(|| format!("cannot open '{}'", model_path.display()))?;
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

    let total_bytes: usize = archived.tensors.iter().map(|t| t.data.len()).sum();
    let total_elements: usize = archived
        .tensors
        .iter()
        .map(|t| t.shape.iter().map(|&d| d as usize).product::<usize>())
        .sum();
    println!(
        "Total:   {} elements / {}",
        total_elements,
        human_bytes(total_bytes)
    );

    let tensor_names_storage: Vec<String> = archived
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str().to_string())
        .collect();
    let tensor_names: Vec<&str> = tensor_names_storage.iter().map(String::as_str).collect();
    println!();
    print_support_summary(model_path, &tensor_names);

    if verbose {
        println!(
            "\n{:<60}  {:>8}  {:>10}  {:>12}",
            "name", "type", "elements", "bytes"
        );
        println!("{}", "-".repeat(96));
        for t in archived.tensors.iter() {
            let shape: Vec<usize> = t.shape.iter().map(|&d| d as usize).collect();
            let n_elements: usize = shape.iter().product();
            println!(
                "{:<60}  {:>8}  {:>10}  {:>12}",
                t.name.as_str(),
                quant_label(t.quant_type),
                n_elements,
                human_bytes(t.data.len())
            );
        }
    }
    Ok(())
}

fn cmd_bench(
    model_path: &PathBuf,
    tensor_filter: Option<&str>,
    iters: usize,
    backend: BenchBackend,
) -> Result<()> {
    if iters == 0 {
        bail!("--iters must be > 0");
    }
    let file = fs::File::open(model_path)
        .with_context(|| format!("cannot open '{}'", model_path.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let archived = check_archived_model(&mmap)?;

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
        bail!("no Q4_L tensors found");
    }

    let q4l_block_bytes = std::mem::size_of::<lns_core::Q4LSuperBlock>();
    let total_blocks: usize = q4l_tensors
        .iter()
        .map(|t| t.data.len() / q4l_block_bytes)
        .sum();
    let total_weights = total_blocks * DEFAULT_SUPER_BLOCK_SIZE;

    println!(
        "Benchmarking {} Q4_L tensor(s) ({} weights) ×{} iters [backend={:?}]",
        q4l_tensors.len(),
        total_weights,
        iters,
        backend
    );

    let compute_backend = match backend {
        BenchBackend::Cpu => ComputeBackend::Cpu,
        BenchBackend::Metal => ComputeBackend::Metal,
    };
    let mut elapsed = 0.0_f64;
    let mut processed = 0usize;
    let mut checksum = 0.0_f32;

    for t in &q4l_tensors {
        let n_elements: usize = t.shape.iter().map(|&d| d as usize).product();
        let blocks: &[lns_core::Q4LSuperBlock] = bytemuck::cast_slice(&t.data);
        let result = bench_q4l_decode(blocks, n_elements, iters, compute_backend)
            .map_err(anyhow::Error::msg)?;
        elapsed += result.elapsed_secs;
        processed += result.weights_processed;
        checksum += result.checksum;
    }

    let weights_per_sec = if elapsed > 0.0 {
        processed as f64 / elapsed
    } else {
        0.0
    };
    println!(
        "Elapsed: {elapsed:.3}s | {:.2} M weights/s (checksum: {checksum:.6})",
        weights_per_sec / 1e6
    );
    Ok(())
}

fn cmd_perplexity(model_path: &PathBuf, tensor_filter: Option<&str>) -> Result<()> {
    let file = fs::File::open(model_path)
        .with_context(|| format!("cannot open '{}'", model_path.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let archived = check_archived_model(&mmap)?;

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

    println!(
        "{:<60}  {:>10}  {:>8}  {:>8}  {:>10}  {:>8}",
        "name", "weights", "zero%", "entropy", "rmse(m)", "snr(dB)"
    );
    println!("{}", "-".repeat(114));

    for t in &q4l_tensors {
        let n_elements: usize = t.shape.iter().map(|&d| d as usize).product();
        let blocks: &[lns_core::Q4LSuperBlock] = bytemuck::cast_slice(&t.data);
        let r = decode_quality_report(blocks, n_elements);
        println!(
            "{:<60}  {:>10}  {:>7.2}%  {:>8.3}  {:>10.4}  {:>8.1}",
            t.name.as_str(),
            r.n_weights,
            r.zero_fraction * 100.0,
            r.code_entropy_bits,
            r.roundtrip_rmse * 1e3,
            r.roundtrip_snr_db
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ChatTemplate {
    Llama2,
    Llama3,
    ChatML,
    Plain,
}

fn apply_template(
    template: ChatTemplate,
    family: lns_inference::ModelFamily,
    user_input: &str,
    system_prompt: Option<&str>,
    enable_qwen_thinking: bool,
) -> String {
    match template {
        ChatTemplate::Llama2 => {
            if let Some(sys) = system_prompt {
                format!(
                    "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
                    sys, user_input
                )
            } else {
                format!("[INST] {} [/INST]", user_input)
            }
        }
        ChatTemplate::Llama3 => {
            let mut out = String::from("<|begin_of_text|>");
            if let Some(sys) = system_prompt {
                out.push_str(&format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
                    sys
                ));
            }
            out.push_str(&format!("<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", user_input));
            out
        }
        ChatTemplate::ChatML => {
            let mut out = String::new();
            if let Some(sys) = system_prompt {
                out.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys));
            }
            out.push_str(&format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                user_input
            ));
            if family == lns_inference::ModelFamily::Qwen35Text {
                if enable_qwen_thinking {
                    out.push_str("<think>\n");
                } else {
                    out.push_str("<think>\n\n</think>\n\n");
                }
            }
            out
        }
        ChatTemplate::Plain => format!("Question: {}\nAnswer (one short line):", user_input),
    }
}

fn resolve_model_dir(model_path: &PathBuf) -> Result<&Path> {
    if model_path.is_dir() {
        Ok(model_path.as_path())
    } else {
        model_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("model path has no parent directory"))
    }
}

fn load_model_json(model_path: &PathBuf) -> Result<serde_json::Value> {
    let model_dir = resolve_model_dir(model_path)?;
    let cfg_path = model_dir.join("config.json");
    let cfg_raw = fs::read_to_string(&cfg_path)
        .with_context(|| format!("cannot read '{}'", cfg_path.display()))?;
    serde_json::from_str(&cfg_raw)
        .with_context(|| format!("invalid json in '{}'", cfg_path.display()))
}

fn load_directory_tensor_names(model_dir: &Path) -> Result<Vec<String>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.is_file() {
        return Ok(Vec::new());
    }

    let raw = fs::read_to_string(&index_path)
        .with_context(|| format!("cannot read '{}'", index_path.display()))?;
    let root: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("invalid json in '{}'", index_path.display()))?;
    let mut names = root
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .map(|map| map.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    names.sort();
    Ok(names)
}

fn load_normalized_model_config(
    model_path: &PathBuf,
) -> Result<lns_inference::NormalizedModelConfig> {
    let root = load_model_json(model_path)?;
    Ok(lns_inference::parse_model_config(&root)?)
}

fn detect_template(tokenizer: &Tokenizer) -> ChatTemplate {
    if tokenizer.has_token("<|im_start|>") && tokenizer.has_token("<|im_end|>") {
        ChatTemplate::ChatML
    } else if tokenizer.has_token("<|start_header_id|>") && tokenizer.has_token("<|eot_id|>") {
        ChatTemplate::Llama3
    } else if tokenizer.has_token("[INST]") && tokenizer.has_token("[/INST]") {
        ChatTemplate::Llama2
    } else {
        ChatTemplate::Plain
    }
}

fn collect_tokenizer_markers(tokenizer: &Tokenizer) -> Vec<&'static str> {
    let mut tokenizer_tokens = Vec::new();
    for token in [
        "<|im_start|>",
        "<|im_end|>",
        "<|start_header_id|>",
        "<|eot_id|>",
        "[INST]",
        "[/INST]",
    ] {
        if tokenizer.has_token(token) {
            tokenizer_tokens.push(token);
        }
    }
    tokenizer_tokens
}

fn stop_token_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    let mut ids = vec![tokenizer.eos()];
    for marker in ["<|im_end|>", "<|im_start|>", "<|eot_id|>", "</s>"] {
        if let Some(id) = tokenizer.token_id(marker) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}

fn support_label(tier: lns_inference::SupportTier) -> &'static str {
    match tier {
        lns_inference::SupportTier::Supported => "supported",
        lns_inference::SupportTier::Unsupported => "unsupported",
    }
}

fn template_label(template: lns_inference::TemplateRecommendation) -> &'static str {
    match template {
        lns_inference::TemplateRecommendation::Llama2 => "llama2",
        lns_inference::TemplateRecommendation::Llama3 => "llama3",
        lns_inference::TemplateRecommendation::ChatML => "chatml",
        lns_inference::TemplateRecommendation::Plain => "plain",
    }
}

fn build_support_report(
    model_path: &PathBuf,
    tensor_names: &[&str],
) -> Result<lns_inference::CompatibilityReport> {
    let root = load_model_json(model_path)?;
    let tokenizer_path = resolve_model_dir(model_path)?.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let tokenizer_tokens = collect_tokenizer_markers(&tokenizer);
    Ok(lns_inference::inspect_compatibility(
        &root,
        tensor_names,
        &tokenizer_tokens,
    ))
}

fn print_support_summary(model_path: &PathBuf, tensor_names: &[&str]) {
    match build_support_report(model_path, tensor_names) {
        Ok(report) => {
            println!("Compatibility:");
            println!("  Support:  {}", support_label(report.tier));
            println!("  Template: {}", template_label(report.template));
            if let Some(config) = report.config {
                println!("  Family:   {:?}", config.family);
            }
            if report.issues.is_empty() {
                println!("  Issues:   none");
            } else {
                println!("  Issues:   {}", report.issues.join(" | "));
            }
            if !report.missing_tensors.is_empty() {
                println!(
                    "  Weights:  {} missing tensor(s)",
                    report.missing_tensors.len()
                );
            }
        }
        Err(err) => {
            println!("Compatibility:");
            println!("  Status:   unavailable ({})", err);
        }
    }
}

fn cmd_doctor(model_path: &PathBuf, verbose: bool) -> Result<()> {
    use lns_inference::{inspect_compatibility, Tokenizer};

    let root = load_model_json(model_path)?;
    let model_dir = resolve_model_dir(model_path)?;
    let tensor_names_storage: Vec<String> = if model_path.is_dir() {
        load_directory_tensor_names(model_dir)?
    } else {
        let file = fs::File::open(model_path)
            .with_context(|| format!("cannot open '{}'", model_path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let archived = check_archived_model(&mmap)
            .with_context(|| format!("'{}' is not a valid .lns model", model_path.display()))?;
        archived
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str().to_string())
            .collect()
    };
    let tensor_names: Vec<&str> = tensor_names_storage.iter().map(String::as_str).collect();

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let tokenizer_tokens = collect_tokenizer_markers(&tokenizer);

    let report = inspect_compatibility(&root, &tensor_names, &tokenizer_tokens);

    println!("Model:      {}", model_path.display());
    println!("Config dir: {}", model_dir.display());
    println!("Support:    {}", support_label(report.tier));
    println!(
        "Archive:     {}",
        if model_path.is_dir() {
            "not provided"
        } else {
            "provided"
        }
    );
    println!("Template:   {}", template_label(report.template));

    if let Some(config) = &report.config {
        println!("Family:     {:?}", config.family);
        println!(
            "Config:     dim={} layers={} heads={} kv_heads={} hidden={} vocab={} rope_theta={}",
            config.transformer.dim,
            config.transformer.n_layers,
            config.transformer.n_heads,
            config.transformer.n_kv_heads,
            config.transformer.hidden_dim,
            config.transformer.vocab_size,
            config.transformer.rope_theta,
        );
    }

    if report.issues.is_empty() {
        println!("Issues:     none");
    } else {
        for (idx, issue) in report.issues.iter().enumerate() {
            if idx == 0 {
                println!("Issues:     {}", issue);
            } else {
                println!("            {}", issue);
            }
        }
    }

    if !report.details.is_empty() {
        for (idx, detail) in report.details.iter().enumerate() {
            if idx == 0 {
                println!("Details:    {}", detail);
            } else {
                println!("            {}", detail);
            }
        }
    }

    if report.missing_tensors.is_empty() {
        println!("Weights:    tensor mapping looks complete");
    } else {
        println!(
            "Weights:    {} required tensor(s) missing",
            report.missing_tensors.len()
        );
        if verbose {
            for name in &report.missing_tensors {
                println!("            - {}", name);
            }
        }
    }

    Ok(())
}

fn run_chat_turn(
    engine: &mut lns_inference::InferenceEngine,
    tokenizer: &Tokenizer,
    template: ChatTemplate,
    family: lns_inference::ModelFamily,
    input: &str,
    sampling: SamplingConfig,
    max_new_tokens: usize,
) -> Result<()> {
    use std::io::{self, Write};

    let sampling = sampling.sanitized();

    let system_prompt = "You are a helpful and intelligent AI assistant.";
    let system_for_template = if template == ChatTemplate::Plain {
        None
    } else {
        Some(system_prompt)
    };
    let formatted_input = apply_template(template, family, input, system_for_template, false);
    let mut tokens = tokenizer.encode(&formatted_input);
    if let Some(bos) = tokenizer.bos() {
        if tokens.first().copied() != Some(bos) {
            tokens.insert(0, bos);
        }
    }

    engine.clear_kv_cache();

    let (mut last_logits, pos_after_prefill) = match engine.forward_prefill(&tokens, 0) {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("[ERR] prefill failed: {}", err);
            return Ok(());
        }
    };
    let mut pos = pos_after_prefill;

    if last_logits.is_empty() {
        return Ok(());
    }

    print!("Assistant: ");
    io::stdout().flush()?;

    let mut recent_tokens: VecDeque<u32> = VecDeque::with_capacity(sampling.repeat_last_n.max(1));
    let mut emitted_token_ids: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut rendered_text = String::new();
    let stop_ids = stop_token_ids(tokenizer);
    let sample_temp = if sampling.temp <= 0.15 {
        0.0
    } else {
        sampling.temp
    };
    let effective_sampling = SamplingConfig {
        temp: sample_temp,
        ..sampling
    };

    let use_gpu_greedy = can_use_gpu_greedy(effective_sampling);
    let mut pending_greedy_token = None;

    for _ in 0..max_new_tokens {
        let next_token = if let Some(token) = pending_greedy_token.take() {
            token
        } else {
            let mut penalized = last_logits.clone();
            apply_sampling_penalties(
                &mut penalized,
                &recent_tokens,
                effective_sampling.repeat_penalty,
                effective_sampling.presence_penalty,
                effective_sampling.frequency_penalty,
            );
            sample_next_token(&penalized, effective_sampling)
        };
        if stop_ids.contains(&next_token) {
            break;
        }

        emitted_token_ids.push(next_token);
        let decoded = tokenizer.decode_tokens(&emitted_token_ids);
        let suffix = decoded.strip_prefix(&rendered_text).unwrap_or(&decoded);
        print!("{}", suffix);
        io::stdout().flush()?;
        rendered_text = decoded.clone();

        if effective_sampling.repeat_last_n > 0 {
            recent_tokens.push_back(next_token);
            if recent_tokens.len() > effective_sampling.repeat_last_n {
                recent_tokens.pop_front();
            }
        }

        if emitted_token_ids.len() >= max_new_tokens {
            break;
        }

        let forward_result = if use_gpu_greedy {
            engine.forward_greedy(next_token, pos).map(|token| {
                pending_greedy_token = Some(token);
                Vec::new()
            })
        } else {
            engine.forward(next_token, pos)
        };
        match forward_result {
            Ok(logits) => {
                if !use_gpu_greedy {
                    last_logits = logits;
                }
            }
            Err(err) => {
                eprintln!("\n[ERR] gen failed: {}", err);
                break;
            }
        }
        pos += 1;

        if decoded.contains("<|eot_id|>")
            || decoded.contains("<|im_end|>")
            || decoded.contains("[/INST]")
        {
            break;
        }
        if template == ChatTemplate::Plain
            && (decoded.contains('\n')
                || decoded.contains("User:")
                || decoded.contains("Assistant:")
                || decoded.contains("Question:"))
        {
            break;
        }
    }

    println!("\n");
    Ok(())
}

fn cmd_chat(
    model_path: &PathBuf,
    sampling: SamplingConfig,
    prompt: Option<&str>,
    max_new_tokens: usize,
) -> Result<()> {
    use lns_inference::{engine::InferenceEngine, inspect_compatibility, SupportTier, Tokenizer};
    use std::io::{self, BufRead, Write};

    let file = fs::File::open(model_path)
        .with_context(|| format!("cannot open '{}'", model_path.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let archived = check_archived_model(&mmap).map_err(|e| anyhow::anyhow!("{:?}", e))?;

    let root = load_model_json(model_path)?;

    let normalized_config = load_normalized_model_config(model_path)?;

    let tokenizer_path = resolve_model_dir(model_path)?.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let tensor_names_storage: Vec<String> = archived
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str().to_string())
        .collect();
    let tensor_names: Vec<&str> = tensor_names_storage.iter().map(String::as_str).collect();
    let tokenizer_tokens = collect_tokenizer_markers(&tokenizer);
    let report = inspect_compatibility(&root, &tensor_names, &tokenizer_tokens);
    if report.tier == SupportTier::Unsupported {
        anyhow::bail!(report.issues.join(" "));
    }

    let template = detect_template(&tokenizer);
    let family = normalized_config.family;

    println!("Loading engine [Template: {:?}]...", template);
    let mut engine =
        InferenceEngine::new(archived, normalized_config).map_err(|e| anyhow::anyhow!(e))?;

    if let Some(prompt) = prompt {
        println!("User: {}", prompt);
        return run_chat_turn(
            &mut engine,
            &tokenizer,
            template,
            family,
            prompt,
            sampling,
            max_new_tokens,
        );
    }

    println!("\nLNS-ML Chat (type 'exit' to quit)");
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        print!("User: ");
        io::stdout().flush()?;
        let input = match lines.next() {
            None => break,
            Some(Err(e)) => return Err(e.into()),
            Some(Ok(l)) => l,
        };
        let input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }
        if input == "exit" || input == "quit" {
            break;
        }

        run_chat_turn(
            &mut engine,
            &tokenizer,
            template,
            family,
            &input,
            sampling,
            max_new_tokens,
        )?;
    }
    Ok(())
}

// ── built-in calibration corpus (used when --text is omitted for ppl/eval) ──
const CALIBRATION_TEXT: &str = "\
The transformer architecture has become the dominant framework for natural language processing. \
It relies on self-attention mechanisms that allow the model to relate positions in a sequence \
regardless of their distance, overcoming limitations of recurrent models. \
At its core, each layer computes queries, keys, and values from the input embeddings, \
then uses scaled dot-product attention to produce a weighted sum of values. \
Positional encodings are added to preserve word-order information. \
Feed-forward sub-layers apply two linear projections with a non-linearity between them. \
Residual connections and layer normalisation stabilise training. \
In modern large language models, the feed-forward blocks often use SwiGLU activations \
which combine a sigmoid-gated linear unit with a plain linear path for better gradient flow. \
Mixture-of-experts variants route each token to a small subset of expert networks, \
scaling model capacity without proportionally increasing inference cost. \
Quantisation compresses weight matrices from 32-bit floats to 4-bit representations, \
dramatically reducing memory bandwidth requirements at the cost of minor accuracy degradation. \
The logarithmic number system offers an alternative representation where multiplication \
becomes addition, potentially accelerating neural-network inference on specialised hardware. \
Context windows have grown from 512 tokens in early BERT models to over one million tokens \
in recent architectures, thanks to advances like rotary positional encodings and efficient \
attention kernels. Inference engines must balance memory, compute, and latency to serve \
users in real time. Speculative decoding, continuous batching, and paged attention are \
among the techniques used to improve throughput in production deployments.";

/// Load the engine and tokenizer from a `.lns` model path (shared helper).
fn load_engine_and_tokenizer(
    model_path: &PathBuf,
) -> Result<(
    memmap2::Mmap,
    lns_inference::Tokenizer,
    lns_inference::model_config::NormalizedModelConfig,
)> {
    use lns_inference::Tokenizer;
    let file = fs::File::open(model_path)
        .with_context(|| format!("cannot open '{}'", model_path.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    check_archived_model(&mmap).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let tokenizer_path = resolve_model_dir(model_path)?.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let normalized_config = load_normalized_model_config(model_path)?;
    Ok((mmap, tokenizer, normalized_config))
}

/// Compute log-softmax and return log P(target_id).
fn log_prob(logits: &[f32], target_id: u32) -> f32 {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|l| (l - max_l).exp()).sum();
    let log_z = max_l + sum_exp.ln();
    logits[target_id as usize] - log_z
}

fn cmd_ppl(
    model_path: &PathBuf,
    text_path: Option<&PathBuf>,
    context: usize,
    stride: usize,
    max_tokens: usize,
) -> Result<()> {
    use lns_inference::engine::InferenceEngine;
    use std::time::Instant;

    let (mmap, tokenizer, normalized_config) = load_engine_and_tokenizer(model_path)?;
    let archived = check_archived_model(&mmap).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let corpus = if let Some(p) = text_path {
        fs::read_to_string(p).with_context(|| format!("cannot read '{}'", p.display()))?
    } else {
        CALIBRATION_TEXT.to_string()
    };

    let mut all_tokens = tokenizer.encode(&corpus);
    if let Some(bos) = tokenizer.bos() {
        if all_tokens.first().copied() != Some(bos) {
            all_tokens.insert(0, bos);
        }
    }
    if max_tokens > 0 && all_tokens.len() > max_tokens {
        all_tokens.truncate(max_tokens);
    }
    let n = all_tokens.len();
    if n < 2 {
        anyhow::bail!("corpus too short (< 2 tokens), cannot compute PPL");
    }

    println!("Corpus: {} tokens", n);
    println!("Context: {context}  Stride: {stride}");
    println!("Loading model…");

    let mut engine =
        InferenceEngine::new(archived, normalized_config).map_err(|e| anyhow::anyhow!(e))?;
    let stride = stride.max(1).min(context);

    let mut total_nll = 0.0f64;
    let mut total_scored = 0usize;
    let mut window_count = 0usize;
    let t_start = Instant::now();

    let mut window_start = 0usize;
    loop {
        let window_end = (window_start + context).min(n);
        let window = &all_tokens[window_start..window_end];

        // Each window of `context` tokens overlaps the previous by
        // `context - stride` tokens. Those overlap tokens were already scored
        // when they were the *new* tail of the prior window, so we only score
        // positions [overlap, context) here. For the first window, we score
        // from position 1 (can't predict the implicit BOS).
        let overlap = context.saturating_sub(stride);
        let score_from = if window_start == 0 { 1 } else { overlap };

        engine.clear_kv_cache();
        let mut pos = 0usize;
        for (i, &tok) in window.iter().enumerate() {
            let logits = engine.forward(tok, pos).map_err(|e| anyhow::anyhow!(e))?;
            pos += 1;

            // Score the *next* token if it falls within the scoring range for this window.
            if i >= score_from.saturating_sub(1) && i + 1 < window.len() {
                let next_tok = window[i + 1];
                total_nll -= log_prob(&logits, next_tok) as f64;
                total_scored += 1;
            }
        }

        window_count += 1;
        let elapsed = t_start.elapsed().as_secs_f64();
        let ppl_so_far = if total_scored > 0 {
            (total_nll / total_scored as f64).exp()
        } else {
            f64::NAN
        };
        print!("\rWindow {window_count}: scored {total_scored} tokens  |  PPL so far = {ppl_so_far:.2}  ({elapsed:.1}s)   ");
        use std::io::Write;
        std::io::stdout().flush().ok();

        if window_end >= n {
            break;
        }
        window_start += stride;
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    let mean_nll = total_nll / total_scored as f64;
    let ppl = mean_nll.exp();

    println!();
    println!("─────────────────────────────────────────");
    println!("  Tokens scored : {total_scored}");
    println!("  Mean NLL      : {mean_nll:.6} nats");
    println!("  Perplexity    : {ppl:.4}");
    println!("  Elapsed       : {elapsed:.2}s");
    println!("─────────────────────────────────────────");

    Ok(())
}

fn cmd_eval(model_path: &PathBuf, gen_tokens: usize) -> Result<()> {
    use lns_inference::engine::InferenceEngine;
    use std::time::Instant;

    let (mmap, tokenizer, normalized_config) = load_engine_and_tokenizer(model_path)?;
    let archived = check_archived_model(&mmap).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let family = normalized_config.family;
    let config = &normalized_config.transformer;
    println!("Model      : {}", model_path.display());
    println!("Family     : {:?}", family);
    println!(
        "Params     : dim={} layers={} heads={} kv_heads={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads
    );
    if let Some(moe) = &config.moe {
        println!(
            "MoE        : {} experts, {} active, moe_hidden={}",
            moe.n_experts, moe.n_experts_per_tok, moe.moe_hidden_dim
        );
    }
    println!();

    // ── Define eval prompts ────────────────────────────────────────────────
    let prompts: &[(&str, usize)] = &[
        ("Say only: hello", 8),                         // short factual
        ("What is the capital of France?", gen_tokens), // factual
        ("Write a haiku about the ocean.", gen_tokens), // creative
        (
            "Explain the transformer architecture in two sentences.",
            gen_tokens,
        ), // technical
    ];

    let template = detect_template(&tokenizer);
    let system_prompt = "You are a helpful and intelligent AI assistant.";
    let system_for_template: Option<&str> = if template == ChatTemplate::Plain {
        None
    } else {
        Some(system_prompt)
    };

    let mut engine =
        InferenceEngine::new(archived, normalized_config).map_err(|e| anyhow::anyhow!(e))?;

    println!(
        "{:<55}  {:>7}  {:>8}  {:>8}  {:>6}",
        "Prompt", "TTFT(ms)", "tok/s", "tokens", "result"
    );
    println!("{}", "─".repeat(95));

    let mut total_tokens_generated = 0usize;
    let mut total_gen_secs = 0.0f64;

    for (prompt_text, max_new) in prompts {
        let formatted = apply_template(template, family, prompt_text, system_for_template, false);
        let mut tokens = tokenizer.encode(&formatted);
        if let Some(bos) = tokenizer.bos() {
            if tokens.first().copied() != Some(bos) {
                tokens.insert(0, bos);
            }
        }

        engine.clear_kv_cache();

        // Prefill (TTFT = time to first token after full prompt)
        let t_prefill = Instant::now();
        let (mut logits, mut pos) = engine
            .forward_prefill(&tokens, 0)
            .map_err(|e| anyhow::anyhow!(e))?;
        let ttft_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

        // Decode
        let stop_ids = stop_token_ids(&tokenizer);
        let sampling = SamplingConfig {
            temp: 0.0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        };
        let mut recent_tokens: VecDeque<u32> = VecDeque::with_capacity(64);
        let mut generated = 0usize;
        let mut output_text = String::new();
        let t_gen = Instant::now();
        let use_gpu_greedy = can_use_gpu_greedy(sampling);
        let mut pending_greedy_token = None;

        loop {
            if generated >= *max_new {
                break;
            }
            let next_id = if let Some(token) = pending_greedy_token.take() {
                token
            } else {
                let mut penalized = logits.clone();
                apply_sampling_penalties(
                    &mut penalized,
                    &recent_tokens,
                    sampling.repeat_penalty,
                    sampling.presence_penalty,
                    sampling.frequency_penalty,
                );
                sample_next_token(&penalized, sampling)
            };
            if stop_ids.contains(&next_id) {
                break;
            }
            output_text.push_str(&tokenizer.decode(next_id));
            recent_tokens.push_back(next_id);
            if recent_tokens.len() > sampling.repeat_last_n {
                recent_tokens.pop_front();
            }
            generated += 1;
            if generated >= *max_new {
                break;
            }

            if use_gpu_greedy {
                pending_greedy_token = Some(
                    engine
                        .forward_greedy(next_id, pos)
                        .map_err(|e| anyhow::anyhow!(e))?,
                );
            } else {
                logits = engine
                    .forward(next_id, pos)
                    .map_err(|e| anyhow::anyhow!(e))?;
            }
            pos += 1;
        }

        let gen_secs = t_gen.elapsed().as_secs_f64();
        let tok_per_s = if gen_secs > 0.0 {
            generated as f64 / gen_secs
        } else {
            0.0
        };
        total_tokens_generated += generated;
        total_gen_secs += gen_secs;

        // Truncate output for display
        let display: String = output_text.chars().take(30).collect();
        let display = if output_text.chars().count() > 30 {
            format!("{display}…")
        } else {
            display
        };
        println!(
            "{:<55}  {:>7.1}  {:>8.1}  {:>8}  {:?}",
            &prompt_text[..prompt_text.len().min(54)],
            ttft_ms,
            tok_per_s,
            generated,
            display,
        );
    }

    println!("{}", "─".repeat(95));
    let avg_tok_per_s = if total_gen_secs > 0.0 {
        total_tokens_generated as f64 / total_gen_secs
    } else {
        0.0
    };
    println!("Average decode speed : {avg_tok_per_s:.1} tok/s  ({total_tokens_generated} tokens in {total_gen_secs:.2}s)");

    // ── Quick PPL on calibration text ────────────────────────────────────
    println!();
    println!("Running PPL on calibration corpus…");
    let all_ppl_tokens = {
        let mut t = tokenizer.encode(CALIBRATION_TEXT);
        if let Some(bos) = tokenizer.bos() {
            if t.first().copied() != Some(bos) {
                t.insert(0, bos);
            }
        }
        t
    };
    let context = 512.min(all_ppl_tokens.len());
    let stride = context / 2;
    let mut total_nll = 0.0f64;
    let mut total_scored = 0usize;

    engine.clear_kv_cache();
    let mut pos = 0usize;
    for (i, &tok) in all_ppl_tokens[..context].iter().enumerate() {
        let logits = engine.forward(tok, pos).map_err(|e| anyhow::anyhow!(e))?;
        pos += 1;
        if i >= stride.saturating_sub(1) && i + 1 < context {
            let next_tok = all_ppl_tokens[i + 1];
            total_nll -= log_prob(&logits, next_tok) as f64;
            total_scored += 1;
        }
    }

    let ppl = if total_scored > 0 {
        (total_nll / total_scored as f64).exp()
    } else {
        f64::NAN
    };
    println!("Calibration PPL (stride={stride}, {total_scored} tokens) : {ppl:.2}");

    println!();
    println!("Eval complete.");

    Ok(())
}

fn cmd_debug_prompt(
    model_path: &PathBuf,
    prompt: &str,
    top_k: usize,
    cpu_gemv: bool,
    template_override: Option<ChatTemplate>,
    raw_prompt: bool,
    qwen_thinking: bool,
) -> Result<()> {
    use lns_inference::{engine::InferenceEngine, inspect_compatibility, SupportTier, Tokenizer};

    let file = fs::File::open(model_path)
        .with_context(|| format!("cannot open '{}'", model_path.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let archived = check_archived_model(&mmap).map_err(|e| anyhow::anyhow!("{:?}", e))?;

    let root = load_model_json(model_path)?;
    let normalized_config = load_normalized_model_config(model_path)?;
    let tokenizer_path = resolve_model_dir(model_path)?.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let tensor_names_storage: Vec<String> = archived
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str().to_string())
        .collect();
    let tensor_names: Vec<&str> = tensor_names_storage.iter().map(String::as_str).collect();
    let tokenizer_tokens = collect_tokenizer_markers(&tokenizer);
    let report = inspect_compatibility(&root, &tensor_names, &tokenizer_tokens);
    if report.tier == SupportTier::Unsupported {
        anyhow::bail!(report.issues.join(" "));
    }

    let template = template_override.unwrap_or_else(|| detect_template(&tokenizer));
    let system_prompt = "You are a helpful and intelligent AI assistant.";
    let system_for_template = if template == ChatTemplate::Plain {
        None
    } else {
        Some(system_prompt)
    };
    let formatted_input = if raw_prompt {
        prompt.to_string()
    } else {
        apply_template(
            template,
            normalized_config.family,
            prompt,
            system_for_template,
            qwen_thinking,
        )
    };
    let mut tokens = tokenizer.encode(&formatted_input);
    if let Some(bos) = tokenizer.bos() {
        if tokens.first().copied() != Some(bos) {
            tokens.insert(0, bos);
        }
    }

    println!("Template: {:?}", template);
    println!("Prompt text: {:?}", formatted_input);
    println!("Prompt tokens: {}", tokens.len());
    println!("CPU GEMV: {}", cpu_gemv);

    let mut engine =
        InferenceEngine::new(archived, normalized_config).map_err(|e| anyhow::anyhow!(e))?;
    if cpu_gemv {
        engine.force_cpu_weights().map_err(|e| anyhow::anyhow!(e))?;
    }
    engine.clear_kv_cache();

    let mut pos = 0usize;
    let mut last_logits = Vec::new();
    let mut last_trace = None;

    for &token in &tokens {
        let (logits, trace) = engine
            .forward_with_trace(token, pos)
            .map_err(|e| anyhow::anyhow!(e))?;
        last_logits = logits;
        last_trace = Some(trace);
        pos += 1;
    }

    let trace = last_trace.ok_or_else(|| anyhow::anyhow!("no trace captured"))?;
    println!("Embedding RMS: {:.6}", trace.embedding_rms);
    for layer in &trace.layers {
        println!(
            "Layer {:>2}: in={:.6} attn={:.6} post_attn={:.6} mlp={:.6} post_mlp={:.6}",
            layer.layer_idx,
            layer.input_rms,
            layer.attention_rms,
            layer.post_attention_rms,
            layer.mlp_rms,
            layer.post_mlp_rms,
        );
    }
    println!("Final norm RMS: {:.6}", trace.final_norm_rms);

    let mut ranked: Vec<(usize, f32)> = last_logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("Top-{} logits:", top_k);
    for (rank, (token_id, logit)) in ranked.into_iter().take(top_k).enumerate() {
        println!(
            "  {:>2}. id={:<8} logit={:>12.6} piece={:?}",
            rank + 1,
            token_id,
            logit,
            tokenizer.decode(token_id as u32),
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Command::Inspect { model, verbose } => cmd_inspect(model, *verbose),
        Command::Bench {
            model,
            tensor,
            iters,
            backend,
        } => cmd_bench(model, tensor.as_deref(), *iters, *backend),
        Command::Perplexity { model, tensor } => cmd_perplexity(model, tensor.as_deref()),
        Command::Chat {
            model,
            prompt,
            temp,
            top_p,
            min_p,
            repeat_penalty,
            repeat_last_n,
            presence_penalty,
            frequency_penalty,
            max_new_tokens,
        } => cmd_chat(
            model,
            SamplingConfig {
                temp: *temp,
                top_p: *top_p,
                min_p: *min_p,
                repeat_penalty: *repeat_penalty,
                repeat_last_n: *repeat_last_n,
                presence_penalty: *presence_penalty,
                frequency_penalty: *frequency_penalty,
            },
            prompt.as_deref(),
            *max_new_tokens,
        ),
        Command::Doctor { model, verbose } => cmd_doctor(model, *verbose),
        Command::DebugPrompt {
            model,
            prompt,
            top_k,
            cpu_gemv,
            template,
            raw_prompt,
            qwen_thinking,
        } => cmd_debug_prompt(
            model,
            prompt,
            *top_k,
            *cpu_gemv,
            *template,
            *raw_prompt,
            *qwen_thinking,
        ),
        Command::Ppl {
            model,
            text,
            context,
            stride,
            max_tokens,
        } => cmd_ppl(model, text.as_ref(), *context, *stride, *max_tokens),
        Command::Eval { model, gen_tokens } => cmd_eval(model, *gen_tokens),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::{apply_sampling_penalties, apply_template, ChatTemplate};

    #[test]
    fn qwen35_chatml_generation_prompt_disables_visible_thinking_by_default() {
        let rendered = apply_template(
            ChatTemplate::ChatML,
            lns_inference::ModelFamily::Qwen35Text,
            "Say only: merhaba",
            Some("You are a helpful and intelligent AI assistant."),
            false,
        );

        assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn qwen35_chatml_generation_prompt_can_keep_think_open() {
        let rendered = apply_template(
            ChatTemplate::ChatML,
            lns_inference::ModelFamily::Qwen35Text,
            "Say only: merhaba",
            Some("You are a helpful and intelligent AI assistant."),
            true,
        );

        assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n"));
        assert!(!rendered.contains("</think>"));
    }

    #[test]
    fn sampling_penalties_apply_presence_frequency_and_repeat() {
        let mut logits = vec![1.0, 2.0, -1.0, 0.5];
        let recent = VecDeque::from(vec![1u32, 1u32, 2u32]);

        apply_sampling_penalties(&mut logits, &recent, 2.0, 0.2, 0.1);

        assert!((logits[1] - 0.4).abs() < 1e-6);
        assert!((logits[2] + 2.6).abs() < 1e-6);
        assert!((logits[0] - 1.0).abs() < 1e-6);
    }
}
