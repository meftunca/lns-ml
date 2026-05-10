use crate::{
    model_config::{ArchitectureSpec, NormalizedModelConfig, TextLayerKind},
    transformer::{
        HybridAttentionWeights, HybridTransformerWeights, LinearAttentionConfig, MoeMlpWeights,
        RopeScalingMode, TransformerConfig,
    },
};
use half::f16;
use lns_core::format::{ArchivedLnsModel, QuantType};
use lns_metal::MetalContext;
use metal::{Buffer, MTLResourceOptions};
use std::collections::{HashMap, VecDeque};

const MAX_SEQ_LEN: usize = 8192; // Flash attention — no fixed scores buffer, supports any length
/// Number of token slots per KV cache page.  Must match `PAGED_PAGE_SZ` in the Metal shader.
const PAGE_SIZE: usize = 16;
/// Maximum number of logical pages for a sequence of length MAX_SEQ_LEN.
const MAX_LOGICAL_PAGES: usize = MAX_SEQ_LEN / PAGE_SIZE;

/// CPU-side pool of fixed-size KV-cache pages.  Maintains a free list of physical
/// page indices.  Pages are allocated just-in-time and reclaimed when evicted by SWA.
pub struct PagePool {
    capacity: usize,
    free: VecDeque<u32>,
}

impl PagePool {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            free: (0..capacity as u32).collect(),
        }
    }
    fn alloc(&mut self) -> Option<u32> {
        self.free.pop_front()
    }
    fn free_page(&mut self, idx: u32) {
        self.free.push_back(idx);
    }
    fn reset(&mut self) {
        self.free.clear();
        self.free.extend(0..self.capacity as u32);
    }
}

pub struct InferenceEngine<'a> {
    pub model: &'a ArchivedLnsModel,
    pub config: TransformerConfig,
    pub architecture: ArchitectureSpec,
    pub weights: HybridTransformerWeights,
    pub output_weight_name: String,
    pub metal_ctx: &'static MetalContext,
    pub tensor_map: HashMap<String, &'a lns_core::format::ArchivedLnsTensor>,
    pub metal_weights: HashMap<String, lns_metal::MetalBuffer>,
    pub metal_f16_weights: HashMap<String, Buffer>,
    pub metal_f32_weights: HashMap<String, Buffer>,
    pub cpu_weights: HashMap<String, Vec<f32>>,

    // Paged Q8A KV cache.  Physical storage is split into PAGE_SIZE-token pages.
    // The CPU PagePool maintains a free list; kv_block_tables maps logical→physical.
    // SWA models reclaim pages as they slide out of the attention window.
    pub paged_kv_data: Vec<(Buffer, Buffer)>, // (K_i8, V_i8) [num_pages*PAGE_SIZE*kv_dim]
    pub paged_kv_scales: Vec<(Buffer, Buffer)>, // (K_sc, V_sc) [num_pages*PAGE_SIZE*n_kv_heads]
    pub kv_page_pools: Vec<PagePool>,
    pub kv_block_tables: Vec<Vec<u32>>, // logical_page → physical_page (u32::MAX = evicted)
    pub kv_block_table_bufs: Vec<Buffer>, // GPU copy [MAX_LOGICAL_PAGES * 4 bytes]

    // Precomputed RoPE inverse frequencies (length = rope_dim / 2).
    // Supports Default, Linear, LLaMA3-extended, and YaRN scaling modes.
    pub rope_inv_freqs_buf: Buffer,

    // Scratch buffers
    pub x_cur: Vec<f32>,
    pub x_norm: Vec<f32>,
    pub q_vec: Vec<f32>,
    pub k_vec: Vec<f32>,
    pub v_vec: Vec<f32>,
    pub attn_o: Vec<f32>,
    pub h1_vec: Vec<f32>,
    pub h3_vec: Vec<f32>,
    pub res_vec: Vec<f32>,
    pub gate_vec: Vec<f32>,
    pub tmp_vec: Vec<f32>,
    pub tmp2_vec: Vec<f32>,
    pub linear_states: Vec<Option<LinearAttentionState>>,
    pub linear_conv_state_buffers: Vec<Option<Buffer>>,
    pub linear_recurrent_state_buffers: Vec<Option<Buffer>>,

    // GPU buffers
    pub x_buf: Buffer,
    pub x_norm_buf: Buffer,
    pub q_buf: Buffer,
    pub k_buf: Buffer,
    pub v_buf: Buffer,
    pub attn_out_buf: Buffer,
    pub h1_buf: Buffer,
    pub h3_buf: Buffer,
    pub linear_conv_buf: Buffer,
    pub logits_buf: Buffer,
    pub zero_buf: Buffer,

    /// When true (env `LNS_PROFILE_STAGES=1`), forward_single_cb commits a
    /// separate command buffer per stage to collect wall-clock latencies.
    /// Has no effect on the normal hot path.
    pub profile_stages: bool,
}

pub struct LinearAttentionState {
    pub conv_state: Vec<f32>,
    pub recurrent_state: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LayerTrace {
    pub layer_idx: usize,
    pub input_rms: f32,
    pub attention_rms: f32,
    pub post_attention_rms: f32,
    pub mlp_rms: f32,
    pub post_mlp_rms: f32,
}

#[derive(Debug, Clone)]
pub struct ForwardTrace {
    pub embedding_rms: f32,
    pub layers: Vec<LayerTrace>,
    pub final_norm_rms: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidualResidency {
    Cpu,
    GpuQBuf,
}

/// Compute per-dimension inverse frequencies for RoPE.
/// Returns `rotary_dim / 2` values where `angle[i] = pos * inv_freqs[i]`.
/// Handles Default, Linear, LLaMA3, and YaRN scaling modes.
fn compute_rope_inv_freqs(
    rotary_dim: usize,
    theta: f32,
    scaling_factor: f32,
    mode: &RopeScalingMode,
) -> Vec<f32> {
    let half = rotary_dim / 2;
    let mut inv_freqs = Vec::with_capacity(half);
    for i in 0..half {
        let base = theta.powf(-(2.0 * i as f32) / rotary_dim as f32);
        let scaled = match mode {
            RopeScalingMode::Default => base / scaling_factor.max(1.0),
            RopeScalingMode::Linear { factor } => base / factor.max(1.0),
            RopeScalingMode::Llama3 {
                factor,
                original_max_pos,
                low_freq_factor,
                high_freq_factor,
            } => {
                let wavelen = std::f32::consts::TAU / base;
                let low_w = *original_max_pos as f32 / low_freq_factor;
                let high_w = *original_max_pos as f32 / high_freq_factor;
                if wavelen < high_w {
                    base
                } else if wavelen > low_w {
                    base / factor
                } else {
                    let smooth = (*original_max_pos as f32 / wavelen - low_freq_factor)
                        / (high_freq_factor - low_freq_factor);
                    base * ((1.0 - smooth) / factor + smooth)
                }
            }
            RopeScalingMode::Yarn {
                factor,
                original_max_pos,
                beta_fast,
                beta_slow,
                ..
            } => {
                let wavelen = std::f32::consts::TAU / base;
                let low_w = *original_max_pos as f32 / beta_slow;
                let high_w = *original_max_pos as f32 / beta_fast;
                if wavelen < high_w {
                    base
                } else if wavelen > low_w {
                    base / factor
                } else {
                    let smooth =
                        (*original_max_pos as f32 / wavelen - beta_slow) / (beta_fast - beta_slow);
                    base * ((1.0 - smooth) / factor + smooth)
                }
            }
        };
        inv_freqs.push(scaled);
    }
    inv_freqs
}

/// Apply RoPE using precomputed inverse frequencies.
/// `inv_freqs[i]` must have length `rotary_dim / 2`.
#[inline]
fn apply_cpu_rope_with_freqs(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    q_heads: usize,
    k_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freqs: &[f32],
) {
    let rotary_dim = rotary_dim.min(head_dim).max(2) & !1;
    let half = rotary_dim / 2;
    let apply = |values: &mut [f32], heads: usize| {
        for head in 0..heads {
            let head_slice = &mut values[head * head_dim..(head + 1) * head_dim];
            let (rotary, _) = head_slice.split_at_mut(rotary_dim);
            let (first_half, second_half) = rotary.split_at_mut(half);
            for i in 0..half.min(inv_freqs.len()) {
                let angle = pos as f32 * inv_freqs[i];
                let (sin, cos) = angle.sin_cos();
                let x0 = first_half[i];
                let x1 = second_half[i];
                first_half[i] = x0 * cos - x1 * sin;
                second_half[i] = x0 * sin + x1 * cos;
            }
        }
    };
    apply(q, q_heads);
    apply(k, k_heads);
}

#[inline]
fn cpu_gemv(w: &[f32], x: &[f32], out: &mut [f32], out_dim: usize, in_dim: usize) {
    for i in 0..out_dim {
        let mut s = 0.0f32;
        let row_start = i * in_dim;
        for j in 0..in_dim {
            s += w[row_start + j] * x[j];
        }
        out[i] = s;
    }
}

#[inline]
fn cpu_rmsnorm_weighted(x: &[f32], w: &[f32], out: &mut [f32], eps: f32, zero_centered: bool) {
    let dim = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let inv_rms = (ss + eps).sqrt().recip();
    for i in 0..dim {
        let scale = if zero_centered { 1.0 + w[i] } else { w[i] };
        out[i] = x[i] * inv_rms * scale;
    }
}

#[inline]
fn cpu_rmsnorm_rows_inplace(
    x: &mut [f32],
    w: &[f32],
    row_dim: usize,
    eps: f32,
    zero_centered: bool,
) {
    for row in x.chunks_exact_mut(row_dim) {
        let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / row_dim as f32;
        let inv_rms = (ss + eps).sqrt().recip();
        for i in 0..row_dim {
            let scale = if zero_centered { 1.0 + w[i] } else { w[i] };
            row[i] = row[i] * inv_rms * scale;
        }
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
fn split_gated_query_interleaved(
    input: &[f32],
    q_out: &mut [f32],
    gate_out: &mut [f32],
    head_dim: usize,
) {
    debug_assert_eq!(input.len(), q_out.len() * 2);
    debug_assert_eq!(q_out.len(), gate_out.len());
    debug_assert_eq!(q_out.len() % head_dim, 0);

    for (head_idx, (q_row, gate_row)) in q_out
        .chunks_exact_mut(head_dim)
        .zip(gate_out.chunks_exact_mut(head_dim))
        .enumerate()
    {
        let base = head_idx * head_dim * 2;
        q_row.copy_from_slice(&input[base..base + head_dim]);
        for i in 0..head_dim {
            gate_row[i] = 1.0 / (1.0 + (-input[base + head_dim + i]).exp());
        }
    }
}

#[inline]
fn l2norm_rows_inplace(x: &mut [f32], row_dim: usize, eps: f32) {
    for row in x.chunks_exact_mut(row_dim) {
        let norm_sq: f32 = row.iter().map(|v| v * v).sum();
        let inv = (norm_sq + eps).sqrt().recip();
        for value in row {
            *value *= inv;
        }
    }
}

#[cfg(test)]
fn apply_cpu_rope_inplace(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    q_heads: usize,
    k_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    rope_scaling_factor: f32,
) {
    let rotary_dim = rotary_dim.min(head_dim).max(2) & !1;
    let half_rotary_dim = rotary_dim / 2;
    let apply = |values: &mut [f32], heads: usize| {
        for head in 0..heads {
            let head_slice = &mut values[head * head_dim..(head + 1) * head_dim];
            let (rotary, _) = head_slice.split_at_mut(rotary_dim);
            let (first_half, second_half) = rotary.split_at_mut(half_rotary_dim);
            for i in 0..half_rotary_dim {
                let freq = 1.0 / rope_theta.powf((2 * i) as f32 / rotary_dim as f32);
                let angle = (pos as f32 / rope_scaling_factor.max(1.0)) * freq;
                let (sin, cos) = angle.sin_cos();
                let x0 = first_half[i];
                let x1 = second_half[i];
                first_half[i] = x0 * cos - x1 * sin;
                second_half[i] = x0 * sin + x1 * cos;
            }
        }
    };
    apply(q, q_heads);
    apply(k, k_heads);
}

#[inline]
fn upload_to_buffer(cpu: &[f32], buf: &Buffer, n: usize) {
    unsafe {
        std::ptr::copy_nonoverlapping(cpu.as_ptr(), buf.contents() as *mut f32, n);
    }
}

#[inline]
fn readback_from_buffer(buf: &Buffer, out: &mut [f32], n: usize) {
    unsafe {
        std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), n);
    }
}

#[inline]
fn ensure_buffer_readback(buf: &Buffer, out: &mut [f32], n: usize, valid: &mut bool) {
    if !*valid {
        readback_from_buffer(buf, out, n);
        *valid = true;
    }
}

#[inline]
fn rms(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean_sq: f32 = values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32;
    mean_sq.sqrt()
}

fn gemv_dispatch(
    metal_ctx: &MetalContext,
    metal_weights: &HashMap<String, lns_metal::MetalBuffer>,
    metal_f16_weights: &HashMap<String, Buffer>,
    cpu_weights: &HashMap<String, Vec<f32>>,
    name: &str,
    x_buf: &Buffer,
    y_buf: &Buffer,
    x_cpu: &[f32],
    y_cpu: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<(), String> {
    if let Some(gw) = metal_weights.get(name) {
        metal_ctx.dispatch_gemv_cached(gw, x_buf, y_buf, out_dim, in_dim, gw.1)?;
        readback_from_buffer(y_buf, y_cpu, out_dim);
    } else if let Some(gw_f16) = metal_f16_weights.get(name) {
        // v3.3 path: sensitive tensors stored as F16 dense on GPU.
        metal_ctx.dispatch_f16_gemv(gw_f16, x_buf, y_buf, out_dim, in_dim)?;
        readback_from_buffer(y_buf, y_cpu, out_dim);
    } else if let Some(cw) = cpu_weights.get(name) {
        cpu_gemv(cw, x_cpu, y_cpu, out_dim, in_dim);
        upload_to_buffer(y_cpu, y_buf, out_dim);
    } else {
        return Err(format!("Weight not found: {name}"));
    }
    Ok(())
}

fn gemv_dispatch_gpu_only(
    metal_ctx: &MetalContext,
    metal_weights: &HashMap<String, lns_metal::MetalBuffer>,
    metal_f16_weights: &HashMap<String, Buffer>,
    cpu_weights: &HashMap<String, Vec<f32>>,
    name: &str,
    x_buf: &Buffer,
    y_buf: &Buffer,
    x_cpu: &[f32],
    scratch_cpu: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<(), String> {
    if let Some(gw) = metal_weights.get(name) {
        metal_ctx.dispatch_gemv_cached(gw, x_buf, y_buf, out_dim, in_dim, gw.1)?;
    } else if let Some(gw_f16) = metal_f16_weights.get(name) {
        metal_ctx.dispatch_f16_gemv(gw_f16, x_buf, y_buf, out_dim, in_dim)?;
    } else if let Some(cw) = cpu_weights.get(name) {
        cpu_gemv(cw, x_cpu, scratch_cpu, out_dim, in_dim);
        upload_to_buffer(scratch_cpu, y_buf, out_dim);
    } else {
        return Err(format!("Weight not found: {name}"));
    }
    Ok(())
}

fn rmsnorm_dispatch(
    metal_ctx: &MetalContext,
    metal_f32_weights: &HashMap<String, Buffer>,
    cpu_weights: &HashMap<String, Vec<f32>>,
    name: &str,
    x_buf: &Buffer,
    x_cpu: &mut [f32],
    x_cpu_valid: &mut bool,
    out_buf: &Buffer,
    out_cpu: &mut [f32],
    out_cpu_valid: &mut bool,
    eps: f32,
    zero_centered: bool,
) -> Result<(), String> {
    if let Some(weight_buf) = metal_f32_weights.get(name) {
        metal_ctx.dispatch_rmsnorm_buffer(
            x_buf,
            weight_buf,
            out_buf,
            x_cpu.len(),
            eps,
            zero_centered,
        )?;
        *out_cpu_valid = false;
    } else if let Some(weight_cpu) = cpu_weights.get(name) {
        ensure_buffer_readback(x_buf, x_cpu, x_cpu.len(), x_cpu_valid);
        cpu_rmsnorm_weighted(x_cpu, weight_cpu, out_cpu, eps, zero_centered);
        upload_to_buffer(out_cpu, out_buf, x_cpu.len());
        *out_cpu_valid = true;
    } else {
        return Err(format!("Weight not found: {name}"));
    }
    Ok(())
}

fn load_tensor_row(
    tensor: &lns_core::format::ArchivedLnsTensor,
    row_idx: usize,
    row_dim: usize,
    out: &mut [f32],
) -> Result<(), String> {
    if out.len() != row_dim {
        return Err("output row buffer size mismatch".to_string());
    }
    if tensor.shape.len() < 2 {
        return Err(format!(
            "tensor '{}' is not row-addressable",
            tensor.name.as_str()
        ));
    }

    let n_rows = tensor.shape[0] as usize;
    if row_idx >= n_rows {
        return Err(format!(
            "row index {row_idx} out of bounds for tensor '{}'",
            tensor.name.as_str()
        ));
    }

    match QuantType::from_u8(tensor.quant_type) {
        Some(QuantType::F32) => {
            let values: &[f32] = bytemuck::cast_slice(tensor.data.as_slice());
            let start = row_idx * row_dim;
            out.copy_from_slice(&values[start..start + row_dim]);
        }
        Some(QuantType::F16) => {
            let bytes = tensor.data.as_slice();
            let start = row_idx * row_dim * 2;
            for (idx, value) in out.iter_mut().enumerate() {
                let offset = start + idx * 2;
                *value =
                    f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32();
            }
        }
        Some(QuantType::Q4L) => {
            let blocks_per_row = row_dim.div_ceil(256);
            let block_size = std::mem::size_of::<lns_core::Q4LSuperBlock>();
            let start_block = row_idx * blocks_per_row;
            let end_block = start_block + blocks_per_row;
            let raw = &tensor.data.as_slice()[start_block * block_size..end_block * block_size];
            let blocks: &[lns_core::Q4LSuperBlock] = bytemuck::cast_slice(raw);
            let row = lns_core::dequantize_q4l(blocks, row_dim);
            out.copy_from_slice(&row);
        }
        Some(QuantType::Q2L) => {
            let blocks_per_row = row_dim.div_ceil(256);
            let block_size = std::mem::size_of::<lns_core::Q2LSuperBlock>();
            let start_block = row_idx * blocks_per_row;
            let end_block = start_block + blocks_per_row;
            let raw = &tensor.data.as_slice()[start_block * block_size..end_block * block_size];
            let blocks: &[lns_core::Q2LSuperBlock] = bytemuck::cast_slice(raw);
            let row = lns_core::dequantize_q2l(blocks, row_dim);
            out.copy_from_slice(&row);
        }
        Some(QuantType::Q8L) => {
            let blocks_per_row = row_dim.div_ceil(256);
            let block_size = std::mem::size_of::<lns_core::Q8LSuperBlock>();
            let start_block = row_idx * blocks_per_row;
            let end_block = start_block + blocks_per_row;
            let raw = &tensor.data.as_slice()[start_block * block_size..end_block * block_size];
            let blocks: &[lns_core::Q8LSuperBlock] = bytemuck::cast_slice(raw);
            let row = lns_core::dequantize_q8l(blocks, row_dim);
            out.copy_from_slice(&row);
        }
        Some(QuantType::Q6L) => {
            return Err("Q6L tensor row decode is not implemented".to_string());
        }
        Some(
            QuantType::Q4HQ
            | QuantType::Q8HQ
            | QuantType::Q4HQM
            | QuantType::Q2HQ
            | QuantType::Q2HQM,
        ) => {
            return Err(format!(
                "HQ tensor row decode is not implemented for quant type {}",
                tensor.quant_type
            ));
        }
        None => {
            return Err(format!(
                "unsupported embedding quant type {}",
                tensor.quant_type
            ));
        }
    }

    Ok(())
}

impl<'a> InferenceEngine<'a> {
    pub fn new(
        model: &'a ArchivedLnsModel,
        normalized: NormalizedModelConfig,
    ) -> Result<Self, String> {
        let metal_ctx = lns_metal::get_metal_context().ok_or("Metal not available")?;
        let config = normalized.transformer;
        let tensor_names: Vec<&str> = model
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect();
        let mut tensor_map = HashMap::new();
        for tensor in model.tensors.iter() {
            tensor_map.insert(tensor.name.as_str().to_string(), tensor);
        }

        let weights =
            HybridTransformerWeights::from_tensor_names(&tensor_names, &normalized.architecture);
        let force_tied_output =
            config.tie_word_embeddings && std::env::var_os("LNS_FORCE_TIED_OUTPUT").is_some();
        let output_weight_name = if force_tied_output {
            weights.token_embedding.clone()
        } else if config.tie_word_embeddings {
            if weights.output != weights.token_embedding && tensor_map.contains_key(&weights.output)
            {
                weights.output.clone()
            } else {
                weights.token_embedding.clone()
            }
        } else {
            weights.output.clone()
        };

        if !tensor_map.contains_key(&weights.token_embedding) {
            return Err(format!("Missing: {}", weights.token_embedding));
        }
        if !tensor_map.contains_key(&output_weight_name) {
            return Err(format!("Missing: {output_weight_name}"));
        }

        let mut metal_weights = HashMap::new();
        let mut metal_f16_weights = HashMap::new();
        let mut metal_f32_weights = HashMap::new();
        let mut cpu_weights = HashMap::new();

        for tensor in model.tensors.iter() {
            let name = tensor.name.as_str();
            if tensor.quant_type == QuantType::Q4L.as_u8()
                || tensor.quant_type == QuantType::Q4HQ.as_u8()
            {
                let rows = tensor.shape[0] as usize;
                let cols = if tensor.shape.len() > 1 {
                    tensor.shape[1] as usize
                } else {
                    1
                };
                let quant_type = QuantType::from_u8(tensor.quant_type)
                    .ok_or_else(|| format!("unsupported quant type {}", tensor.quant_type))?;
                let buf = metal_ctx.prepare_tensor(&tensor.data, rows, cols, quant_type)?;
                metal_weights.insert(name.to_string(), buf);
            } else {
                let is_embedding = name == weights.token_embedding;
                let is_output = name == output_weight_name;
                let is_unused_tied_lm_head = config.tie_word_embeddings
                    && name == weights.output
                    && name != output_weight_name;
                let should_prepare_f16_dense = tensor.quant_type == QuantType::F16.as_u8()
                    && tensor.shape.len() > 1
                    && (is_embedding || is_output);

                if should_prepare_f16_dense {
                    let buf = metal_ctx.prepare_f16_dense_tensor(tensor.data.as_slice());
                    metal_f16_weights.insert(name.to_string(), buf);
                }

                if is_unused_tied_lm_head {
                    continue;
                }

                let skip_cpu_decode = is_embedding || (is_output && should_prepare_f16_dense);
                if skip_cpu_decode {
                    continue;
                }

                let decoded = tensor.to_f32().map_err(|e| e.to_string())?;

                // Spec v3.3 mixed precision support: any non-Q4_L 2-D linear
                // weight (Q8_L promote, F16 sensitive, etc.) is mirrored to a
                // GPU F16 dense buffer so GEMV stays on Metal. Without this
                // mirror, promoted tensors fall back to `cpu_gemv` and the
                // GPU sits idle.
                let is_2d_linear = tensor.shape.len() == 2
                    && tensor.shape.iter().all(|&d| d > 0)
                    && tensor.shape[0] as usize * tensor.shape[1] as usize >= 4096;
                let already_f16_dense = is_embedding
                    || (is_output && should_prepare_f16_dense)
                    || is_unused_tied_lm_head;
                if is_2d_linear && !already_f16_dense {
                    let mut f16_bytes = Vec::with_capacity(decoded.len() * 2);
                    for &v in &decoded {
                        f16_bytes.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
                    }
                    let buf = metal_ctx.prepare_f16_dense_tensor(&f16_bytes);
                    metal_f16_weights.insert(name.to_string(), buf);
                }

                if name.contains("norm")
                    || name.contains("linear_attn.conv1d.weight")
                    || name.contains("linear_attn.conv1d.bias")
                    || name.contains("linear_attn.A_log")
                    || name.contains("linear_attn.dt_bias")
                {
                    let buf = metal_ctx.prepare_f32_dense_tensor(&decoded);
                    metal_f32_weights.insert(name.to_string(), buf);
                }
                cpu_weights.insert(name.to_string(), decoded);
            }
        }

        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let head_dim = config.attention_head_dim;
        let attn_dim = config.n_heads * head_dim;
        let kv_dim = config.n_kv_heads * head_dim;
        let linear_value_dim = config
            .linear_attention
            .as_ref()
            .map(LinearAttentionConfig::value_dim)
            .unwrap_or(0);
        let moe_hidden_dim = config.moe.as_ref().map(|m| m.moe_hidden_dim).unwrap_or(0);
        let max_ffn_hidden = hidden_dim.max(moe_hidden_dim);
        // When MoE is present, the down_proj output is `dim` sized, so attn_out_buf must hold `dim`.
        let max_attn_out_dim =
            attn_dim
                .max(linear_value_dim)
                .max(if moe_hidden_dim > 0 { dim } else { 0 });
        let max_tmp_dim = hidden_dim.max(attn_dim * 2).max(
            config
                .linear_attention
                .as_ref()
                .map(LinearAttentionConfig::conv_dim)
                .unwrap_or(0),
        );
        let vocab_size = config.vocab_size;

        // Paged Q8A KV Cache: non-contiguous page-based memory for scalable KV management.
        // Physical pool per layer holds `num_pages` pages of PAGE_SIZE token slots each.
        // CPU PagePool tracks available pages; block tables map logical→physical.
        // SWA models reclaim pages that slide out of the attention window automatically.
        let mut paged_kv_data = Vec::new();
        let mut paged_kv_scales = Vec::new();
        let mut kv_page_pools = Vec::new();
        let mut kv_block_tables = Vec::new();
        let mut kv_block_table_bufs = Vec::new();
        let kv_cache_len = config
            .sliding_window
            .unwrap_or(MAX_SEQ_LEN)
            .min(MAX_SEQ_LEN);
        let num_pages = (kv_cache_len + PAGE_SIZE - 1) / PAGE_SIZE;
        // int8 KV data: num_pages * PAGE_SIZE * kv_dim bytes
        let kv_paged_i8_bytes = (num_pages * PAGE_SIZE * kv_dim) as u64;
        // f32 scale per (physical slot, kv head): num_pages * PAGE_SIZE * n_kv_heads * 4 bytes
        let kv_paged_scale_bytes = (num_pages * PAGE_SIZE * config.n_kv_heads * 4) as u64;
        // Block table GPU buffer: MAX_LOGICAL_PAGES u32 entries = 2 KB fixed
        let bt_buf_bytes = (MAX_LOGICAL_PAGES * 4) as u64;

        for l_idx in 0..config.n_layers {
            let is_full_attention = normalized
                .architecture
                .text_layers
                .get(l_idx)
                .map(|kind| matches!(kind, TextLayerKind::FullAttention))
                .unwrap_or(true);
            if is_full_attention {
                let k_buf = metal_ctx
                    .device
                    .new_buffer(kv_paged_i8_bytes, MTLResourceOptions::StorageModeShared);
                let v_buf = metal_ctx
                    .device
                    .new_buffer(kv_paged_i8_bytes, MTLResourceOptions::StorageModeShared);
                let ks_buf = metal_ctx
                    .device
                    .new_buffer(kv_paged_scale_bytes, MTLResourceOptions::StorageModeShared);
                let vs_buf = metal_ctx
                    .device
                    .new_buffer(kv_paged_scale_bytes, MTLResourceOptions::StorageModeShared);
                let bt_buf = metal_ctx
                    .device
                    .new_buffer(bt_buf_bytes, MTLResourceOptions::StorageModeShared);
                // All block table entries start as u32::MAX (no page allocated)
                unsafe {
                    std::ptr::write_bytes(
                        bt_buf.contents() as *mut u8,
                        0xff,
                        bt_buf_bytes as usize,
                    );
                }
                paged_kv_data.push((k_buf, v_buf));
                paged_kv_scales.push((ks_buf, vs_buf));
                kv_page_pools.push(PagePool::new(num_pages));
                kv_block_tables.push(Vec::new());
                kv_block_table_bufs.push(bt_buf);
            } else {
                // Stub entries for non-attention layers — never accessed by attention path
                let stub = || {
                    metal_ctx
                        .device
                        .new_buffer(4, MTLResourceOptions::StorageModeShared)
                };
                paged_kv_data.push((stub(), stub()));
                paged_kv_scales.push((stub(), stub()));
                kv_page_pools.push(PagePool::new(0));
                kv_block_tables.push(Vec::new());
                kv_block_table_bufs.push(
                    metal_ctx
                        .device
                        .new_buffer(4, MTLResourceOptions::StorageModeShared),
                );
            }
        }

        let mut linear_states = Vec::with_capacity(normalized.architecture.text_layers.len());
        let mut linear_conv_state_buffers =
            Vec::with_capacity(normalized.architecture.text_layers.len());
        let mut linear_recurrent_state_buffers =
            Vec::with_capacity(normalized.architecture.text_layers.len());
        for layer_kind in &normalized.architecture.text_layers {
            if matches!(layer_kind, TextLayerKind::LinearAttention) {
                let linear = config.linear_attention.as_ref().ok_or_else(|| {
                    "linear-attention layer present without linear-attention config".to_string()
                })?;
                let conv_state_len = linear.conv_dim() * linear.conv_kernel_size;
                let recurrent_state_len =
                    linear.num_value_heads * linear.key_head_dim * linear.value_head_dim;
                linear_states.push(Some(LinearAttentionState {
                    conv_state: vec![0.0; conv_state_len],
                    recurrent_state: vec![0.0; recurrent_state_len],
                }));
                linear_conv_state_buffers.push(Some(metal_ctx.device.new_buffer(
                    (conv_state_len * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )));
                linear_recurrent_state_buffers.push(Some(metal_ctx.device.new_buffer(
                    (recurrent_state_len * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )));
            } else {
                linear_states.push(None);
                linear_conv_state_buffers.push(None);
                linear_recurrent_state_buffers.push(None);
            }
        }

        // Precompute RoPE inverse frequencies for this model's head_dim, theta, and scaling mode.
        let rope_dim = {
            let rot = ((head_dim as f32) * config.partial_rotary_factor).round() as usize;
            if rot >= 2 && rot < head_dim {
                rot
            } else {
                head_dim
            }
        };
        let inv_freqs = compute_rope_inv_freqs(
            rope_dim,
            config.rope_theta,
            config.rope_scaling_factor,
            &config.rope_mode,
        );
        let rope_inv_freqs_buf = metal_ctx.device.new_buffer_with_data(
            inv_freqs.as_ptr() as *const _,
            (inv_freqs.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            model,
            config,
            architecture: normalized.architecture,
            weights,
            output_weight_name,
            metal_ctx,
            tensor_map,
            metal_weights,
            metal_f16_weights,
            metal_f32_weights,
            cpu_weights,
            paged_kv_data,
            paged_kv_scales,
            kv_page_pools,
            kv_block_tables,
            kv_block_table_bufs,
            rope_inv_freqs_buf,

            x_cur: vec![0.0; dim],
            x_norm: vec![0.0; dim],
            q_vec: vec![0.0; attn_dim],
            k_vec: vec![0.0; kv_dim],
            v_vec: vec![0.0; kv_dim],
            attn_o: vec![0.0; max_attn_out_dim],
            h1_vec: vec![0.0; max_ffn_hidden],
            h3_vec: vec![0.0; max_ffn_hidden],
            res_vec: vec![0.0; dim],
            gate_vec: vec![0.0; attn_dim],
            tmp_vec: vec![0.0; max_tmp_dim],
            tmp2_vec: vec![0.0; max_tmp_dim],
            linear_states,
            linear_conv_state_buffers,
            linear_recurrent_state_buffers,

            x_buf: metal_ctx
                .device
                .new_buffer((dim * 4) as u64, MTLResourceOptions::StorageModeShared),
            x_norm_buf: metal_ctx
                .device
                .new_buffer((dim * 4) as u64, MTLResourceOptions::StorageModeShared),
            q_buf: metal_ctx
                .device
                .new_buffer((attn_dim * 4) as u64, MTLResourceOptions::StorageModeShared),
            k_buf: metal_ctx
                .device
                .new_buffer((kv_dim * 4) as u64, MTLResourceOptions::StorageModeShared),
            v_buf: metal_ctx
                .device
                .new_buffer((kv_dim * 4) as u64, MTLResourceOptions::StorageModeShared),
            attn_out_buf: metal_ctx.device.new_buffer(
                (max_attn_out_dim * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            h1_buf: metal_ctx.device.new_buffer(
                (max_ffn_hidden * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            h3_buf: metal_ctx.device.new_buffer(
                (max_ffn_hidden * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            linear_conv_buf: metal_ctx.device.new_buffer(
                (max_tmp_dim * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            logits_buf: metal_ctx.device.new_buffer(
                (vocab_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            zero_buf: metal_ctx.device.new_buffer(
                (max_tmp_dim * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            profile_stages: std::env::var("LNS_PROFILE_STAGES").map(|v| v == "1").unwrap_or(false),
        })
    }

    fn run_linear_attention_gpu(
        &mut self,
        layer_idx: usize,
        attention: &HybridAttentionWeights,
        input_cpu_valid: &mut bool,
    ) -> Result<Option<ResidualResidency>, String> {
        let HybridAttentionWeights::LinearAttention {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            out_proj,
            conv1d_weight,
            conv1d_bias,
            norm,
            a_log,
            dt_bias,
        } = attention
        else {
            return Ok(None);
        };

        let linear = match self.config.linear_attention.as_ref() {
            Some(linear) => linear,
            None => return Ok(None),
        };

        let Some(conv_w_buf) = self.metal_f32_weights.get(conv1d_weight) else {
            return Ok(None);
        };
        let Some(a_log_buf) = self.metal_f32_weights.get(a_log) else {
            return Ok(None);
        };
        let Some(dt_bias_buf) = self.metal_f32_weights.get(dt_bias) else {
            return Ok(None);
        };
        let Some(norm_buf) = self.metal_f32_weights.get(norm) else {
            return Ok(None);
        };
        let Some(conv_state_buf) = self.linear_conv_state_buffers[layer_idx].as_ref() else {
            return Ok(None);
        };
        let Some(recurrent_state_buf) = self.linear_recurrent_state_buffers[layer_idx].as_ref()
        else {
            return Ok(None);
        };

        let conv_dim = linear.conv_dim();
        let value_dim = linear.value_dim();
        let bias_buf = conv1d_bias
            .as_ref()
            .and_then(|name| self.metal_f32_weights.get(name))
            .unwrap_or(&self.zero_buf);

        if let (Some(qkv_w), Some(z_w), Some(b_w), Some(a_w), Some(out_w)) = (
            self.metal_weights.get(in_proj_qkv),
            self.metal_weights.get(in_proj_z),
            self.metal_weights.get(in_proj_b),
            self.metal_weights.get(in_proj_a),
            self.metal_weights.get(out_proj),
        ) {
            let jobs = [
                lns_metal::GemvBatchJob {
                    weights: qkv_w,
                    y: &self.h1_buf,
                    out_dim: conv_dim,
                    in_dim: self.config.dim,
                    qt: QuantType::Q4L,
                },
                lns_metal::GemvBatchJob {
                    weights: z_w,
                    y: &self.h3_buf,
                    out_dim: value_dim,
                    in_dim: self.config.dim,
                    qt: QuantType::Q4L,
                },
                lns_metal::GemvBatchJob {
                    weights: b_w,
                    y: &self.q_buf,
                    out_dim: linear.num_value_heads,
                    in_dim: self.config.dim,
                    qt: QuantType::Q4L,
                },
                lns_metal::GemvBatchJob {
                    weights: a_w,
                    y: &self.k_buf,
                    out_dim: linear.num_value_heads,
                    in_dim: self.config.dim,
                    qt: QuantType::Q4L,
                },
            ];
            let cb = self.metal_ctx.queue.new_command_buffer();
            self.metal_ctx
                .encode_gemv_batch_cached(&cb, &self.x_norm_buf, &jobs)?;
            self.metal_ctx.encode_linear_attn_block(
                &cb,
                &self.h1_buf,
                conv_state_buf,
                conv_w_buf,
                bias_buf,
                &self.linear_conv_buf,
                &self.h3_buf,
                &self.q_buf,
                &self.k_buf,
                a_log_buf,
                dt_bias_buf,
                norm_buf,
                recurrent_state_buf,
                &self.attn_out_buf,
                linear.conv_kernel_size,
                conv_dim,
                linear.num_key_heads,
                linear.num_value_heads,
                linear.key_head_dim,
                linear.value_head_dim,
                self.config.eps,
            )?;
            self.metal_ctx.encode_gemv_accumulate_cached(
                &cb,
                out_w,
                &self.attn_out_buf,
                &self.x_buf,
                self.config.dim,
                value_dim,
            )?;
            cb.commit();
            cb.wait_until_completed();
            return Ok(Some(ResidualResidency::GpuQBuf));
        }

        if let (Some(qkv_w), Some(z_w), Some(b_w), Some(a_w)) = (
            self.metal_weights.get(in_proj_qkv),
            self.metal_weights.get(in_proj_z),
            self.metal_weights.get(in_proj_b),
            self.metal_weights.get(in_proj_a),
        ) {
            let jobs = [
                lns_metal::GemvBatchJob {
                    weights: qkv_w,
                    y: &self.h1_buf,
                    out_dim: conv_dim,
                    in_dim: self.config.dim,
                    qt: QuantType::Q4L,
                },
                lns_metal::GemvBatchJob {
                    weights: z_w,
                    y: &self.h3_buf,
                    out_dim: value_dim,
                    in_dim: self.config.dim,
                    qt: QuantType::Q4L,
                },
                lns_metal::GemvBatchJob {
                    weights: b_w,
                    y: &self.q_buf,
                    out_dim: linear.num_value_heads,
                    in_dim: self.config.dim,
                    qt: QuantType::Q4L,
                },
                lns_metal::GemvBatchJob {
                    weights: a_w,
                    y: &self.k_buf,
                    out_dim: linear.num_value_heads,
                    in_dim: self.config.dim,
                    qt: QuantType::Q4L,
                },
            ];
            self.metal_ctx
                .dispatch_gemv_batch_cached(&self.x_norm_buf, &jobs)?;
        } else {
            ensure_buffer_readback(
                &self.x_norm_buf,
                &mut self.x_norm,
                self.config.dim,
                input_cpu_valid,
            );
            gemv_dispatch_gpu_only(
                self.metal_ctx,
                &self.metal_weights,
                &self.metal_f16_weights,
                &self.cpu_weights,
                in_proj_qkv,
                &self.x_norm_buf,
                &self.h1_buf,
                &self.x_norm,
                &mut self.tmp_vec[..conv_dim],
                conv_dim,
                self.config.dim,
            )?;
            gemv_dispatch_gpu_only(
                self.metal_ctx,
                &self.metal_weights,
                &self.metal_f16_weights,
                &self.cpu_weights,
                in_proj_z,
                &self.x_norm_buf,
                &self.h3_buf,
                &self.x_norm,
                &mut self.tmp2_vec[..value_dim],
                value_dim,
                self.config.dim,
            )?;
            gemv_dispatch_gpu_only(
                self.metal_ctx,
                &self.metal_weights,
                &self.metal_f16_weights,
                &self.cpu_weights,
                in_proj_b,
                &self.x_norm_buf,
                &self.q_buf,
                &self.x_norm,
                &mut self.attn_o[..linear.num_value_heads],
                linear.num_value_heads,
                self.config.dim,
            )?;
            gemv_dispatch_gpu_only(
                self.metal_ctx,
                &self.metal_weights,
                &self.metal_f16_weights,
                &self.cpu_weights,
                in_proj_a,
                &self.x_norm_buf,
                &self.k_buf,
                &self.x_norm,
                &mut self.k_vec[..linear.num_value_heads],
                linear.num_value_heads,
                self.config.dim,
            )?;
        }

        self.metal_ctx.dispatch_linear_attn_block(
            &self.h1_buf,
            conv_state_buf,
            conv_w_buf,
            bias_buf,
            &self.linear_conv_buf,
            &self.h3_buf,
            &self.q_buf,
            &self.k_buf,
            a_log_buf,
            dt_bias_buf,
            norm_buf,
            recurrent_state_buf,
            &self.attn_out_buf,
            linear.conv_kernel_size,
            conv_dim,
            linear.num_key_heads,
            linear.num_value_heads,
            linear.key_head_dim,
            linear.value_head_dim,
            self.config.eps,
        )?;
        if let Some(gw) = self.metal_weights.get(out_proj) {
            self.metal_ctx.dispatch_gemv_accumulate_cached(
                gw,
                &self.attn_out_buf,
                &self.x_buf,
                self.config.dim,
                value_dim,
            )?;
            Ok(Some(ResidualResidency::GpuQBuf))
        } else {
            readback_from_buffer(
                &self.attn_out_buf,
                &mut self.tmp2_vec[..value_dim],
                value_dim,
            );
            let weights = self
                .cpu_weights
                .get(out_proj)
                .ok_or_else(|| format!("Weight not found: {out_proj}"))?;
            cpu_gemv(
                weights,
                &self.tmp2_vec[..value_dim],
                &mut self.res_vec,
                self.config.dim,
                value_dim,
            );
            Ok(Some(ResidualResidency::Cpu))
        }
    }

    fn run_linear_attention(
        &mut self,
        layer_idx: usize,
        attention: &HybridAttentionWeights,
        input_cpu_valid: &mut bool,
    ) -> Result<ResidualResidency, String> {
        if let Some(residency) =
            self.run_linear_attention_gpu(layer_idx, attention, input_cpu_valid)?
        {
            return Ok(residency);
        }

        let HybridAttentionWeights::LinearAttention {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            out_proj,
            conv1d_weight,
            conv1d_bias,
            norm,
            a_log,
            dt_bias,
        } = attention
        else {
            return Err("linear-attention weights missing".to_string());
        };

        let linear = self
            .config
            .linear_attention
            .as_ref()
            .ok_or_else(|| "missing linear-attention config".to_string())?;
        let conv_dim = linear.conv_dim();
        let key_dim = linear.key_dim();
        let value_dim = linear.value_dim();
        let rep = linear.num_value_heads / linear.num_key_heads;
        ensure_buffer_readback(
            &self.x_norm_buf,
            &mut self.x_norm,
            self.config.dim,
            input_cpu_valid,
        );
        gemv_dispatch(
            self.metal_ctx,
            &self.metal_weights,
            &self.metal_f16_weights,
            &self.cpu_weights,
            in_proj_qkv,
            &self.x_norm_buf,
            &self.h1_buf,
            &self.x_norm,
            &mut self.tmp_vec[..conv_dim],
            conv_dim,
            self.config.dim,
        )?;
        gemv_dispatch(
            self.metal_ctx,
            &self.metal_weights,
            &self.metal_f16_weights,
            &self.cpu_weights,
            in_proj_z,
            &self.x_norm_buf,
            &self.h3_buf,
            &self.x_norm,
            &mut self.tmp2_vec[..value_dim],
            value_dim,
            self.config.dim,
        )?;
        gemv_dispatch(
            self.metal_ctx,
            &self.metal_weights,
            &self.metal_f16_weights,
            &self.cpu_weights,
            in_proj_b,
            &self.x_norm_buf,
            &self.q_buf,
            &self.x_norm,
            &mut self.attn_o[..linear.num_value_heads],
            linear.num_value_heads,
            self.config.dim,
        )?;
        gemv_dispatch(
            self.metal_ctx,
            &self.metal_weights,
            &self.metal_f16_weights,
            &self.cpu_weights,
            in_proj_a,
            &self.x_norm_buf,
            &self.k_buf,
            &self.x_norm,
            &mut self.k_vec[..linear.num_value_heads],
            linear.num_value_heads,
            self.config.dim,
        )?;

        let conv_w = self
            .cpu_weights
            .get(conv1d_weight)
            .cloned()
            .ok_or_else(|| format!("Weight not found or not loaded on CPU: {conv1d_weight}"))?;
        let conv_b = match conv1d_bias {
            Some(name) => Some(
                self.cpu_weights
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("Weight not found or not loaded on CPU: {name}"))?,
            ),
            None => None,
        };
        let a_log_w = self
            .cpu_weights
            .get(a_log)
            .cloned()
            .ok_or_else(|| format!("Weight not found or not loaded on CPU: {a_log}"))?;
        let dt_bias_w = self
            .cpu_weights
            .get(dt_bias)
            .cloned()
            .ok_or_else(|| format!("Weight not found or not loaded on CPU: {dt_bias}"))?;
        let norm_w = self
            .cpu_weights
            .get(norm)
            .cloned()
            .ok_or_else(|| format!("Weight not found or not loaded on CPU: {norm}"))?;
        let state = self.linear_states[layer_idx]
            .as_mut()
            .ok_or_else(|| format!("missing linear-attention state for layer {layer_idx}"))?;

        for channel in 0..conv_dim {
            let state_row = &mut state.conv_state
                [channel * linear.conv_kernel_size..(channel + 1) * linear.conv_kernel_size];
            state_row.rotate_left(1);
            state_row[linear.conv_kernel_size - 1] = self.tmp_vec[channel];
            let weight_row =
                &conv_w[channel * linear.conv_kernel_size..(channel + 1) * linear.conv_kernel_size];
            let mut acc = 0.0f32;
            for offset in 0..linear.conv_kernel_size {
                acc += state_row[offset] * weight_row[offset];
            }
            if let Some(conv_b) = &conv_b {
                acc += conv_b[channel];
            }
            self.tmp_vec[channel] = silu(acc);
        }

        let (query, rest) = self.tmp_vec[..conv_dim].split_at_mut(key_dim);
        let (key, value) = rest.split_at_mut(key_dim);
        let z = self.tmp2_vec[..value_dim].to_vec();
        let beta = &mut self.attn_o[..linear.num_value_heads];
        let a = &self.k_vec[..linear.num_value_heads];

        l2norm_rows_inplace(query, linear.key_head_dim, 1e-6);
        l2norm_rows_inplace(key, linear.key_head_dim, 1e-6);
        let scale = (linear.key_head_dim as f32).sqrt().recip();
        for value in query.iter_mut() {
            *value *= scale;
        }

        for head in 0..linear.num_value_heads {
            beta[head] = 1.0 / (1.0 + (-beta[head]).exp());
        }

        let mut core = vec![0.0f32; value_dim];
        for v_head in 0..linear.num_value_heads {
            let k_head = v_head / rep;
            let q_row = &query[k_head * linear.key_head_dim..(k_head + 1) * linear.key_head_dim];
            let k_row = &key[k_head * linear.key_head_dim..(k_head + 1) * linear.key_head_dim];
            let v_row =
                &value[v_head * linear.value_head_dim..(v_head + 1) * linear.value_head_dim];
            let state_offset = v_head * linear.key_head_dim * linear.value_head_dim;
            let state_slice = &mut state.recurrent_state
                [state_offset..state_offset + linear.key_head_dim * linear.value_head_dim];
            let decay = (-a_log_w[v_head].exp() * softplus(a[v_head] + dt_bias_w[v_head])).exp();

            for cell in state_slice.iter_mut() {
                *cell *= decay;
            }

            let mut kv_mem = vec![0.0f32; linear.value_head_dim];
            for k_idx in 0..linear.key_head_dim {
                let state_row = &state_slice
                    [k_idx * linear.value_head_dim..(k_idx + 1) * linear.value_head_dim];
                for v_idx in 0..linear.value_head_dim {
                    kv_mem[v_idx] += state_row[v_idx] * k_row[k_idx];
                }
            }

            let mut delta = vec![0.0f32; linear.value_head_dim];
            for v_idx in 0..linear.value_head_dim {
                delta[v_idx] = (v_row[v_idx] - kv_mem[v_idx]) * beta[v_head];
            }

            for k_idx in 0..linear.key_head_dim {
                let state_row = &mut state_slice
                    [k_idx * linear.value_head_dim..(k_idx + 1) * linear.value_head_dim];
                for v_idx in 0..linear.value_head_dim {
                    state_row[v_idx] += k_row[k_idx] * delta[v_idx];
                    core[v_head * linear.value_head_dim + v_idx] += state_row[v_idx] * q_row[k_idx];
                }
            }
        }

        for chunk_idx in 0..linear.num_value_heads {
            let start = chunk_idx * linear.value_head_dim;
            let end = start + linear.value_head_dim;
            let chunk = &mut core[start..end];
            let gate = &z[start..end];
            cpu_rmsnorm_weighted(
                chunk,
                &norm_w,
                &mut self.tmp2_vec[start..end],
                self.config.eps,
                false,
            );
            for i in 0..linear.value_head_dim {
                self.tmp2_vec[start + i] *= silu(gate[i]);
            }
        }

        upload_to_buffer(&self.tmp2_vec[..value_dim], &self.attn_out_buf, value_dim);
        gemv_dispatch(
            self.metal_ctx,
            &self.metal_weights,
            &self.metal_f16_weights,
            &self.cpu_weights,
            out_proj,
            &self.attn_out_buf,
            &self.q_buf,
            &self.tmp2_vec[..value_dim],
            &mut self.res_vec,
            self.config.dim,
            value_dim,
        )?;
        Ok(ResidualResidency::Cpu)
    }

    fn forward_internal(
        &mut self,
        token_id: u32,
        pos: usize,
        mut trace: Option<&mut ForwardTrace>,
        need_logits: bool,
    ) -> Result<Vec<f32>, String> {
        // Reject vision-placeholder tokens early — the vision encoder is not implemented.
        // These tokens appear in multimodal prompts where image/video patch embeddings would
        // normally be injected by the vision branch.  Running the text path on them produces
        // meaningless output; returning a clear error is safer.
        if let Some(img_id) = self.architecture.image_token_id {
            if token_id == img_id {
                return Err(format!(
                    "Vision token (image_token_id={img_id}) fed to text-only inference engine; \
                     vision encoding is not supported — use --text-only when converting"
                ));
            }
        }
        if let Some(vid_id) = self.architecture.video_token_id {
            if token_id == vid_id {
                return Err(format!(
                    "Vision token (video_token_id={vid_id}) fed to text-only inference engine; \
                     vision encoding is not supported — use --text-only when converting"
                ));
            }
        }

        // Fast path: encode the ENTIRE forward pass (all layers + logits) into a single
        // Metal command buffer. Eliminates ~(2 * n_layers) commit/wait round trips per token,
        // giving 10-30× higher tok/s on Apple Silicon.
        if trace.is_none() && self.all_layers_gpu_capable() {
            return self.forward_single_cb(token_id, pos, need_logits, true);
        }

        let dim = self.config.dim;
        let head_dim = self.config.attention_head_dim;
        let attn_dim = self.config.n_heads * head_dim;
        let kv_dim = self.config.n_kv_heads * head_dim;
        let hidden_dim = self.config.hidden_dim;
        let trace_enabled = trace.is_some();

        let embedding_tensor = self
            .tensor_map
            .get(&self.weights.token_embedding)
            .ok_or_else(|| format!("Missing embedding tensor: {}", self.weights.token_embedding))?;
        load_tensor_row(embedding_tensor, token_id as usize, dim, &mut self.x_cur)?;
        upload_to_buffer(&self.x_cur, &self.x_buf, dim);
        let mut x_cpu_valid = true;
        let mut x_norm_cpu_valid = false;
        if let Some(trace) = trace.as_deref_mut() {
            trace.embedding_rms = rms(&self.x_cur);
        }

        for l_idx in 0..self.weights.layer_weights.len() {
            let layer = self.weights.layer_weights[l_idx].clone();
            let layer_attention = layer.attention.clone();
            let layer_kind = self
                .architecture
                .text_layers
                .get(l_idx)
                .cloned()
                .unwrap_or(TextLayerKind::FullAttention);
            if trace_enabled {
                ensure_buffer_readback(&self.x_buf, &mut self.x_cur, dim, &mut x_cpu_valid);
            }
            let input_rms = if trace_enabled { rms(&self.x_cur) } else { 0.0 };

            let attention_norm_can_chain = match (&layer_kind, &layer_attention) {
                (
                    TextLayerKind::FullAttention,
                    HybridAttentionWeights::FullAttention {
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj,
                        q_norm,
                        k_norm,
                    },
                ) => {
                    self.metal_f32_weights.contains_key(&layer.attention_norm)
                        && self.metal_weights.contains_key(q_proj)
                        && self.metal_weights.contains_key(k_proj)
                        && self.metal_weights.contains_key(v_proj)
                        && self.metal_weights.contains_key(o_proj)
                        && q_norm
                            .as_ref()
                            .map(|name| self.metal_f32_weights.contains_key(name))
                            .unwrap_or(true)
                        && k_norm
                            .as_ref()
                            .map(|name| self.metal_f32_weights.contains_key(name))
                            .unwrap_or(true)
                }
                _ => false,
            };

            if !attention_norm_can_chain {
                rmsnorm_dispatch(
                    self.metal_ctx,
                    &self.metal_f32_weights,
                    &self.cpu_weights,
                    &layer.attention_norm,
                    &self.x_buf,
                    &mut self.x_cur,
                    &mut x_cpu_valid,
                    &self.x_norm_buf,
                    &mut self.x_norm,
                    &mut x_norm_cpu_valid,
                    self.config.eps,
                    self.config.zero_centered_rmsnorm,
                )?;
            } else {
                x_norm_cpu_valid = false;
            }

            let mut attention_rms = 0.0f32;
            let mut post_attention_rms = if trace_enabled { input_rms } else { 0.0 };

            match layer_kind {
                TextLayerKind::FullAttention => {
                    let (q_proj, k_proj, v_proj, o_proj, q_norm, k_norm) = match &layer_attention {
                        HybridAttentionWeights::FullAttention {
                            q_proj,
                            k_proj,
                            v_proj,
                            o_proj,
                            q_norm,
                            k_norm,
                        } => (
                            q_proj,
                            k_proj,
                            v_proj,
                            o_proj,
                            q_norm.as_deref(),
                            k_norm.as_deref(),
                        ),
                        _ => return Err(format!("layer {l_idx} expected full-attention weights")),
                    };

                    let rotary_dim =
                        ((head_dim as f32) * self.config.partial_rotary_factor).round() as usize;
                    let rope_dim = if rotary_dim >= 2 && rotary_dim < head_dim {
                        rotary_dim
                    } else {
                        head_dim
                    };
                    let q_norm_gpu_ready = q_norm
                        .map(|name| self.metal_f32_weights.contains_key(name))
                        .unwrap_or(true);
                    let k_norm_gpu_ready = k_norm
                        .map(|name| self.metal_f32_weights.contains_key(name))
                        .unwrap_or(true);
                    let mut q_ready_on_cpu = false;
                    let mut k_ready_on_cpu = false;
                    let mut gate_ready_on_cpu = false;
                    let mut attention_done = false;

                    // Allocate the page for this token, evict SWA-expired pages, upload block table.
                    self.ensure_kv_page(l_idx, pos)?;
                    let _window = self.config.sliding_window.unwrap_or(0);
                    self.evict_kv_pages(l_idx, pos, _window);
                    self.upload_block_table(l_idx);
                    let kv_p_start = self.kv_p_start(pos);

                    if self.config.gated_query_attention {
                        if let (Some(qw), Some(kw), Some(vw), Some(ow)) = (
                            self.metal_weights.get(q_proj),
                            self.metal_weights.get(k_proj),
                            self.metal_weights.get(v_proj),
                            self.metal_weights.get(o_proj),
                        ) {
                            if q_norm_gpu_ready && k_norm_gpu_ready {
                                let attention_norm_w = self
                                    .metal_f32_weights
                                    .get(&layer.attention_norm)
                                    .ok_or_else(|| {
                                        format!("Weight not found: {}", layer.attention_norm)
                                    })?;
                                let jobs = [
                                    lns_metal::GemvBatchJob {
                                        weights: qw,
                                        y: &self.h1_buf,
                                        out_dim: attn_dim * 2,
                                        in_dim: dim,
                                        qt: QuantType::Q4L,
                                    },
                                    lns_metal::GemvBatchJob {
                                        weights: kw,
                                        y: &self.k_buf,
                                        out_dim: kv_dim,
                                        in_dim: dim,
                                        qt: QuantType::Q4L,
                                    },
                                    lns_metal::GemvBatchJob {
                                        weights: vw,
                                        y: &self.v_buf,
                                        out_dim: kv_dim,
                                        in_dim: dim,
                                        qt: QuantType::Q4L,
                                    },
                                ];
                                let (k_data, v_data) = &self.paged_kv_data[l_idx];
                                let (ks_data, vs_data) = &self.paged_kv_scales[l_idx];
                                let bt_buf = &self.kv_block_table_bufs[l_idx];
                                let cb = self.metal_ctx.queue.new_command_buffer();
                                self.metal_ctx.encode_rmsnorm_buffer(
                                    &cb,
                                    &self.x_buf,
                                    attention_norm_w,
                                    &self.x_norm_buf,
                                    dim,
                                    self.config.eps,
                                    self.config.zero_centered_rmsnorm,
                                )?;
                                self.metal_ctx.encode_gemv_batch_cached(
                                    &cb,
                                    &self.x_norm_buf,
                                    &jobs,
                                )?;
                                self.metal_ctx.encode_split_gated_query(
                                    &cb,
                                    &self.h1_buf,
                                    &self.q_buf,
                                    &self.h3_buf,
                                    attn_dim,
                                    head_dim,
                                )?;
                                if let Some(q_norm) = q_norm {
                                    let q_norm_w = self
                                        .metal_f32_weights
                                        .get(q_norm)
                                        .ok_or_else(|| format!("Weight not found: {q_norm}"))?;
                                    self.metal_ctx.encode_rmsnorm_rows_buffer(
                                        &cb,
                                        &self.q_buf,
                                        q_norm_w,
                                        head_dim,
                                        self.config.n_heads,
                                        self.config.eps,
                                        self.config.zero_centered_rmsnorm,
                                    )?;
                                }
                                if let Some(k_norm) = k_norm {
                                    let k_norm_w = self
                                        .metal_f32_weights
                                        .get(k_norm)
                                        .ok_or_else(|| format!("Weight not found: {k_norm}"))?;
                                    self.metal_ctx.encode_rmsnorm_rows_buffer(
                                        &cb,
                                        &self.k_buf,
                                        k_norm_w,
                                        head_dim,
                                        self.config.n_kv_heads,
                                        self.config.eps,
                                        self.config.zero_centered_rmsnorm,
                                    )?;
                                }
                                self.metal_ctx.encode_rope_with_freqs(
                                    &cb,
                                    &self.q_buf,
                                    &self.k_buf,
                                    &self.rope_inv_freqs_buf,
                                    pos,
                                    head_dim,
                                    rope_dim,
                                )?;
                                self.metal_ctx.encode_attention_paged_q8a(
                                    &cb,
                                    &self.q_buf,
                                    &self.k_buf,
                                    &self.v_buf,
                                    k_data,
                                    v_data,
                                    ks_data,
                                    vs_data,
                                    bt_buf,
                                    &self.attn_out_buf,
                                    pos,
                                    head_dim,
                                    self.config.n_heads,
                                    self.config.n_kv_heads,
                                    kv_p_start,
                                )?;
                                self.metal_ctx.encode_mul_inplace(
                                    &cb,
                                    &self.attn_out_buf,
                                    &self.h3_buf,
                                    attn_dim,
                                )?;
                                self.metal_ctx.encode_gemv_accumulate_cached(
                                    &cb,
                                    ow,
                                    &self.attn_out_buf,
                                    &self.x_buf,
                                    dim,
                                    attn_dim,
                                )?;
                                cb.commit();
                                cb.wait_until_completed();
                                if trace_enabled {
                                    readback_from_buffer(
                                        &self.attn_out_buf,
                                        &mut self.attn_o,
                                        attn_dim,
                                    );
                                    attention_rms = rms(&self.attn_o[..attn_dim]);
                                }
                                x_cpu_valid = false;
                                attention_done = true;
                            }
                        }
                        if !attention_done {
                            if let (Some(qw), Some(kw), Some(vw)) = (
                                self.metal_weights.get(q_proj),
                                self.metal_weights.get(k_proj),
                                self.metal_weights.get(v_proj),
                            ) {
                                let jobs = [
                                    lns_metal::GemvBatchJob {
                                        weights: qw,
                                        y: &self.h1_buf,
                                        out_dim: attn_dim * 2,
                                        in_dim: dim,
                                        qt: QuantType::Q4L,
                                    },
                                    lns_metal::GemvBatchJob {
                                        weights: kw,
                                        y: &self.k_buf,
                                        out_dim: kv_dim,
                                        in_dim: dim,
                                        qt: QuantType::Q4L,
                                    },
                                    lns_metal::GemvBatchJob {
                                        weights: vw,
                                        y: &self.v_buf,
                                        out_dim: kv_dim,
                                        in_dim: dim,
                                        qt: QuantType::Q4L,
                                    },
                                ];
                                self.metal_ctx
                                    .dispatch_gemv_batch_cached(&self.x_norm_buf, &jobs)?;
                                self.metal_ctx.dispatch_split_gated_query(
                                    &self.h1_buf,
                                    &self.q_buf,
                                    &self.h3_buf,
                                    attn_dim,
                                    head_dim,
                                )?;
                            } else {
                                ensure_buffer_readback(
                                    &self.x_norm_buf,
                                    &mut self.x_norm,
                                    dim,
                                    &mut x_norm_cpu_valid,
                                );
                                gemv_dispatch(
                                    self.metal_ctx,
                                    &self.metal_weights,
                                    &self.metal_f16_weights,
                                    &self.cpu_weights,
                                    q_proj,
                                    &self.x_norm_buf,
                                    &self.h1_buf,
                                    &self.x_norm,
                                    &mut self.tmp_vec[..attn_dim * 2],
                                    attn_dim * 2,
                                    dim,
                                )?;
                                split_gated_query_interleaved(
                                    &self.tmp_vec[..attn_dim * 2],
                                    &mut self.q_vec[..attn_dim],
                                    &mut self.gate_vec[..attn_dim],
                                    head_dim,
                                );
                                gemv_dispatch(
                                    self.metal_ctx,
                                    &self.metal_weights,
                                    &self.metal_f16_weights,
                                    &self.cpu_weights,
                                    k_proj,
                                    &self.x_norm_buf,
                                    &self.k_buf,
                                    &self.x_norm,
                                    &mut self.k_vec,
                                    kv_dim,
                                    dim,
                                )?;
                                gemv_dispatch(
                                    self.metal_ctx,
                                    &self.metal_weights,
                                    &self.metal_f16_weights,
                                    &self.cpu_weights,
                                    v_proj,
                                    &self.x_norm_buf,
                                    &self.v_buf,
                                    &self.x_norm,
                                    &mut self.v_vec,
                                    kv_dim,
                                    dim,
                                )?;
                                q_ready_on_cpu = true;
                                k_ready_on_cpu = true;
                                gate_ready_on_cpu = true;
                            }
                        }
                    } else {
                        if let (Some(qw), Some(kw), Some(vw), Some(ow)) = (
                            self.metal_weights.get(q_proj),
                            self.metal_weights.get(k_proj),
                            self.metal_weights.get(v_proj),
                            self.metal_weights.get(o_proj),
                        ) {
                            if q_norm_gpu_ready && k_norm_gpu_ready {
                                let attention_norm_w = self
                                    .metal_f32_weights
                                    .get(&layer.attention_norm)
                                    .ok_or_else(|| {
                                        format!("Weight not found: {}", layer.attention_norm)
                                    })?;
                                let jobs = [
                                    lns_metal::GemvBatchJob {
                                        weights: qw,
                                        y: &self.q_buf,
                                        out_dim: attn_dim,
                                        in_dim: dim,
                                        qt: QuantType::Q4L,
                                    },
                                    lns_metal::GemvBatchJob {
                                        weights: kw,
                                        y: &self.k_buf,
                                        out_dim: kv_dim,
                                        in_dim: dim,
                                        qt: QuantType::Q4L,
                                    },
                                    lns_metal::GemvBatchJob {
                                        weights: vw,
                                        y: &self.v_buf,
                                        out_dim: kv_dim,
                                        in_dim: dim,
                                        qt: QuantType::Q4L,
                                    },
                                ];
                                let (k_data, v_data) = &self.paged_kv_data[l_idx];
                                let (ks_data, vs_data) = &self.paged_kv_scales[l_idx];
                                let bt_buf = &self.kv_block_table_bufs[l_idx];
                                let cb = self.metal_ctx.queue.new_command_buffer();
                                self.metal_ctx.encode_rmsnorm_buffer(
                                    &cb,
                                    &self.x_buf,
                                    attention_norm_w,
                                    &self.x_norm_buf,
                                    dim,
                                    self.config.eps,
                                    self.config.zero_centered_rmsnorm,
                                )?;
                                self.metal_ctx.encode_gemv_batch_cached(
                                    &cb,
                                    &self.x_norm_buf,
                                    &jobs,
                                )?;
                                if let Some(q_norm) = q_norm {
                                    let q_norm_w = self
                                        .metal_f32_weights
                                        .get(q_norm)
                                        .ok_or_else(|| format!("Weight not found: {q_norm}"))?;
                                    self.metal_ctx.encode_rmsnorm_rows_buffer(
                                        &cb,
                                        &self.q_buf,
                                        q_norm_w,
                                        head_dim,
                                        self.config.n_heads,
                                        self.config.eps,
                                        self.config.zero_centered_rmsnorm,
                                    )?;
                                }
                                if let Some(k_norm) = k_norm {
                                    let k_norm_w = self
                                        .metal_f32_weights
                                        .get(k_norm)
                                        .ok_or_else(|| format!("Weight not found: {k_norm}"))?;
                                    self.metal_ctx.encode_rmsnorm_rows_buffer(
                                        &cb,
                                        &self.k_buf,
                                        k_norm_w,
                                        head_dim,
                                        self.config.n_kv_heads,
                                        self.config.eps,
                                        self.config.zero_centered_rmsnorm,
                                    )?;
                                }
                                self.metal_ctx.encode_rope_with_freqs(
                                    &cb,
                                    &self.q_buf,
                                    &self.k_buf,
                                    &self.rope_inv_freqs_buf,
                                    pos,
                                    head_dim,
                                    rope_dim,
                                )?;
                                self.metal_ctx.encode_attention_paged_q8a(
                                    &cb,
                                    &self.q_buf,
                                    &self.k_buf,
                                    &self.v_buf,
                                    k_data,
                                    v_data,
                                    ks_data,
                                    vs_data,
                                    bt_buf,
                                    &self.attn_out_buf,
                                    pos,
                                    head_dim,
                                    self.config.n_heads,
                                    self.config.n_kv_heads,
                                    kv_p_start,
                                )?;
                                self.metal_ctx.encode_gemv_accumulate_cached(
                                    &cb,
                                    ow,
                                    &self.attn_out_buf,
                                    &self.x_buf,
                                    dim,
                                    attn_dim,
                                )?;
                                cb.commit();
                                cb.wait_until_completed();
                                if trace_enabled {
                                    readback_from_buffer(
                                        &self.attn_out_buf,
                                        &mut self.attn_o,
                                        attn_dim,
                                    );
                                    attention_rms = rms(&self.attn_o[..attn_dim]);
                                }
                                x_cpu_valid = false;
                                attention_done = true;
                            }
                        }
                        if !attention_done {
                            if let (Some(qw), Some(kw), Some(vw)) = (
                                self.metal_weights.get(q_proj),
                                self.metal_weights.get(k_proj),
                                self.metal_weights.get(v_proj),
                            ) {
                                let jobs = [
                                    lns_metal::GemvBatchJob {
                                        weights: qw,
                                        y: &self.q_buf,
                                        out_dim: attn_dim,
                                        in_dim: dim,
                                        qt: QuantType::Q4L,
                                    },
                                    lns_metal::GemvBatchJob {
                                        weights: kw,
                                        y: &self.k_buf,
                                        out_dim: kv_dim,
                                        in_dim: dim,
                                        qt: QuantType::Q4L,
                                    },
                                    lns_metal::GemvBatchJob {
                                        weights: vw,
                                        y: &self.v_buf,
                                        out_dim: kv_dim,
                                        in_dim: dim,
                                        qt: QuantType::Q4L,
                                    },
                                ];
                                self.metal_ctx
                                    .dispatch_gemv_batch_cached(&self.x_norm_buf, &jobs)?;
                            } else {
                                ensure_buffer_readback(
                                    &self.x_norm_buf,
                                    &mut self.x_norm,
                                    dim,
                                    &mut x_norm_cpu_valid,
                                );
                                gemv_dispatch(
                                    self.metal_ctx,
                                    &self.metal_weights,
                                    &self.metal_f16_weights,
                                    &self.cpu_weights,
                                    q_proj,
                                    &self.x_norm_buf,
                                    &self.q_buf,
                                    &self.x_norm,
                                    &mut self.q_vec,
                                    attn_dim,
                                    dim,
                                )?;
                                gemv_dispatch(
                                    self.metal_ctx,
                                    &self.metal_weights,
                                    &self.metal_f16_weights,
                                    &self.cpu_weights,
                                    k_proj,
                                    &self.x_norm_buf,
                                    &self.k_buf,
                                    &self.x_norm,
                                    &mut self.k_vec,
                                    kv_dim,
                                    dim,
                                )?;
                                gemv_dispatch(
                                    self.metal_ctx,
                                    &self.metal_weights,
                                    &self.metal_f16_weights,
                                    &self.cpu_weights,
                                    v_proj,
                                    &self.x_norm_buf,
                                    &self.v_buf,
                                    &self.x_norm,
                                    &mut self.v_vec,
                                    kv_dim,
                                    dim,
                                )?;
                                q_ready_on_cpu = true;
                                k_ready_on_cpu = true;
                            }
                        }
                    }

                    if !attention_done {
                        if let Some(q_norm) = q_norm {
                            if !q_ready_on_cpu {
                                if let Some(q_norm_w) = self.metal_f32_weights.get(q_norm) {
                                    self.metal_ctx.dispatch_rmsnorm_rows_buffer(
                                        &self.q_buf,
                                        q_norm_w,
                                        head_dim,
                                        self.config.n_heads,
                                        self.config.eps,
                                        self.config.zero_centered_rmsnorm,
                                    )?;
                                } else {
                                    readback_from_buffer(&self.q_buf, &mut self.q_vec, attn_dim);
                                    q_ready_on_cpu = true;
                                }
                            }
                            if q_ready_on_cpu {
                                let q_norm_w = self.cpu_weights.get(q_norm).ok_or_else(|| {
                                    format!("Weight not found or not loaded on CPU: {q_norm}")
                                })?;
                                cpu_rmsnorm_rows_inplace(
                                    &mut self.q_vec,
                                    q_norm_w,
                                    head_dim,
                                    self.config.eps,
                                    self.config.zero_centered_rmsnorm,
                                );
                                upload_to_buffer(&self.q_vec, &self.q_buf, attn_dim);
                            }
                        }
                        if let Some(k_norm) = k_norm {
                            if !k_ready_on_cpu {
                                if let Some(k_norm_w) = self.metal_f32_weights.get(k_norm) {
                                    self.metal_ctx.dispatch_rmsnorm_rows_buffer(
                                        &self.k_buf,
                                        k_norm_w,
                                        head_dim,
                                        self.config.n_kv_heads,
                                        self.config.eps,
                                        self.config.zero_centered_rmsnorm,
                                    )?;
                                } else {
                                    readback_from_buffer(&self.k_buf, &mut self.k_vec, kv_dim);
                                    k_ready_on_cpu = true;
                                }
                            }
                            if k_ready_on_cpu {
                                let k_norm_w = self.cpu_weights.get(k_norm).ok_or_else(|| {
                                    format!("Weight not found or not loaded on CPU: {k_norm}")
                                })?;
                                cpu_rmsnorm_rows_inplace(
                                    &mut self.k_vec,
                                    k_norm_w,
                                    head_dim,
                                    self.config.eps,
                                    self.config.zero_centered_rmsnorm,
                                );
                                upload_to_buffer(&self.k_vec, &self.k_buf, kv_dim);
                            }
                        }

                        if q_ready_on_cpu || k_ready_on_cpu {
                            if !q_ready_on_cpu {
                                readback_from_buffer(&self.q_buf, &mut self.q_vec, attn_dim);
                            }
                            if !k_ready_on_cpu {
                                readback_from_buffer(&self.k_buf, &mut self.k_vec, kv_dim);
                            }
                            apply_cpu_rope_with_freqs(
                                &mut self.q_vec,
                                &mut self.k_vec,
                                pos,
                                self.config.n_heads,
                                self.config.n_kv_heads,
                                head_dim,
                                rope_dim,
                                unsafe {
                                    std::slice::from_raw_parts(
                                        self.rope_inv_freqs_buf.contents() as *const f32,
                                        self.rope_inv_freqs_buf.length() as usize / 4,
                                    )
                                },
                            );
                            upload_to_buffer(&self.q_vec, &self.q_buf, attn_dim);
                            upload_to_buffer(&self.k_vec, &self.k_buf, kv_dim);
                        } else {
                            self.metal_ctx.dispatch_rope_with_freqs(
                                &self.q_buf,
                                &self.k_buf,
                                &self.rope_inv_freqs_buf,
                                pos,
                                head_dim,
                                rope_dim,
                            )?;
                        }

                        let (k_data, v_data) = &self.paged_kv_data[l_idx];
                        let (ks_data, vs_data) = &self.paged_kv_scales[l_idx];
                        let bt_buf = &self.kv_block_table_bufs[l_idx];
                        self.metal_ctx.dispatch_attention_paged_q8a(
                            &self.q_buf,
                            &self.k_buf,
                            &self.v_buf,
                            k_data,
                            v_data,
                            ks_data,
                            vs_data,
                            bt_buf,
                            &self.attn_out_buf,
                            pos,
                            head_dim,
                            self.config.n_heads,
                            self.config.n_kv_heads,
                            kv_p_start,
                        )?;

                        let mut attn_cpu_ready = false;
                        if self.config.gated_query_attention {
                            if gate_ready_on_cpu {
                                readback_from_buffer(
                                    &self.attn_out_buf,
                                    &mut self.attn_o,
                                    attn_dim,
                                );
                                attn_cpu_ready = true;
                                for i in 0..attn_dim {
                                    self.attn_o[i] *= 1.0 / (1.0 + (-self.gate_vec[i]).exp());
                                }
                                upload_to_buffer(
                                    &self.attn_o[..attn_dim],
                                    &self.attn_out_buf,
                                    attn_dim,
                                );
                                if trace_enabled {
                                    attention_rms = rms(&self.attn_o[..attn_dim]);
                                }
                            } else {
                                self.metal_ctx.dispatch_mul_inplace(
                                    &self.attn_out_buf,
                                    &self.h3_buf,
                                    attn_dim,
                                )?;
                                if trace_enabled {
                                    readback_from_buffer(
                                        &self.attn_out_buf,
                                        &mut self.attn_o,
                                        attn_dim,
                                    );
                                    attention_rms = rms(&self.attn_o[..attn_dim]);
                                    attn_cpu_ready = true;
                                }
                            }
                        } else if trace_enabled {
                            readback_from_buffer(&self.attn_out_buf, &mut self.attn_o, attn_dim);
                            attention_rms = rms(&self.attn_o[..attn_dim]);
                            attn_cpu_ready = true;
                        }

                        if let Some(gw) = self.metal_weights.get(o_proj) {
                            self.metal_ctx.dispatch_gemv_accumulate_cached(
                                gw,
                                &self.attn_out_buf,
                                &self.x_buf,
                                dim,
                                attn_dim,
                            )?;
                            x_cpu_valid = false;
                        } else {
                            if !attn_cpu_ready {
                                readback_from_buffer(
                                    &self.attn_out_buf,
                                    &mut self.attn_o,
                                    attn_dim,
                                );
                            }
                            let weights = self
                                .cpu_weights
                                .get(o_proj)
                                .ok_or_else(|| format!("Weight not found: {o_proj}"))?;
                            cpu_gemv(
                                weights,
                                &self.attn_o[..attn_dim],
                                &mut self.res_vec,
                                dim,
                                attn_dim,
                            );
                            ensure_buffer_readback(
                                &self.x_buf,
                                &mut self.x_cur,
                                dim,
                                &mut x_cpu_valid,
                            );
                            for i in 0..dim {
                                self.x_cur[i] += self.res_vec[i];
                            }
                            upload_to_buffer(&self.x_cur, &self.x_buf, dim);
                        }
                    }
                    if trace_enabled {
                        ensure_buffer_readback(&self.x_buf, &mut self.x_cur, dim, &mut x_cpu_valid);
                        post_attention_rms = rms(&self.x_cur);
                    }
                }
                TextLayerKind::MlpOnly => {}
                TextLayerKind::LinearAttention => {
                    let linear_residency =
                        self.run_linear_attention(l_idx, &layer_attention, &mut x_norm_cpu_valid)?;
                    attention_rms = if trace_enabled {
                        rms(&self.res_vec)
                    } else {
                        0.0
                    };
                    match linear_residency {
                        ResidualResidency::GpuQBuf => {
                            x_cpu_valid = false;
                        }
                        ResidualResidency::Cpu => {
                            ensure_buffer_readback(
                                &self.x_buf,
                                &mut self.x_cur,
                                dim,
                                &mut x_cpu_valid,
                            );
                            for i in 0..dim {
                                self.x_cur[i] += self.res_vec[i];
                            }
                            upload_to_buffer(&self.x_cur, &self.x_buf, dim);
                        }
                    }
                    if trace_enabled {
                        ensure_buffer_readback(&self.x_buf, &mut self.x_cur, dim, &mut x_cpu_valid);
                        post_attention_rms = rms(&self.x_cur);
                    }
                }
                TextLayerKind::Unknown(kind) => {
                    return Err(format!(
                        "Layer {l_idx} uses unsupported attention kind '{kind}'"
                    ));
                }
            }

            let mlp_rms = if let Some(moe_w) = &layer.moe {
                // ── MoE FFN path ────────────────────────────────────────────────────
                let moe_cfg = self
                    .config
                    .moe
                    .as_ref()
                    .ok_or("MoE layer present but TransformerConfig.moe is None")?;
                let n_experts = moe_cfg.n_experts;
                let n_tok = moe_cfg.n_experts_per_tok;
                let moe_hidden = moe_cfg.moe_hidden_dim;

                // 1. FFN norm → x_norm_buf / x_norm
                rmsnorm_dispatch(
                    self.metal_ctx,
                    &self.metal_f32_weights,
                    &self.cpu_weights,
                    &layer.ffn_norm,
                    &self.x_buf,
                    &mut self.x_cur,
                    &mut x_cpu_valid,
                    &self.x_norm_buf,
                    &mut self.x_norm,
                    &mut x_norm_cpu_valid,
                    self.config.eps,
                    self.config.zero_centered_rmsnorm,
                )?;
                ensure_buffer_readback(
                    &self.x_norm_buf,
                    &mut self.x_norm,
                    dim,
                    &mut x_norm_cpu_valid,
                );

                // 2. Router logits: gate.weight [n_experts × dim] @ x_norm → router_logits
                let gate_w = self
                    .cpu_weights
                    .get(&moe_w.gate)
                    .ok_or_else(|| format!("MoE gate weight not found: {}", moe_w.gate))?;
                let mut router_logits = vec![0.0f32; n_experts];
                cpu_gemv(gate_w, &self.x_norm, &mut router_logits, n_experts, dim);

                // 3. Softmax + top-k selection, then re-normalise weights
                let max_logit = router_logits
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = router_logits
                    .iter()
                    .map(|x| (x - max_logit).exp())
                    .collect();
                let sum_exp: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

                let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
                indexed.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut selected: Vec<(usize, f32)> = indexed.into_iter().take(n_tok).collect();
                let sum_sel: f32 = selected.iter().map(|(_, w)| *w).sum::<f32>().max(1e-9);
                for (_, w) in &mut selected {
                    *w /= sum_sel;
                }

                // 4. Accumulate expert outputs
                ensure_buffer_readback(&self.x_buf, &mut self.x_cur, dim, &mut x_cpu_valid);
                self.res_vec.iter_mut().for_each(|v| *v = 0.0);
                // Keep x_norm_buf valid on GPU for all expert projections
                upload_to_buffer(&self.x_norm, &self.x_norm_buf, dim);

                // Snapshot moe_w references needed to avoid re-borrow issues
                let selected_idxs: Vec<(usize, f32)> = selected;
                let moe_w: &MoeMlpWeights = moe_w; // rebind lifetime

                for (expert_idx, expert_weight) in &selected_idxs {
                    let expert = &moe_w.experts[*expert_idx];

                    gemv_dispatch(
                        self.metal_ctx,
                        &self.metal_weights,
                        &self.metal_f16_weights,
                        &self.cpu_weights,
                        &expert.gate_proj,
                        &self.x_norm_buf,
                        &self.h1_buf,
                        &self.x_norm,
                        &mut self.h1_vec,
                        moe_hidden,
                        dim,
                    )?;
                    gemv_dispatch(
                        self.metal_ctx,
                        &self.metal_weights,
                        &self.metal_f16_weights,
                        &self.cpu_weights,
                        &expert.up_proj,
                        &self.x_norm_buf,
                        &self.h3_buf,
                        &self.x_norm,
                        &mut self.h3_vec,
                        moe_hidden,
                        dim,
                    )?;

                    // SwiGLU on CPU: h1 = silu(h1) * h3
                    for i in 0..moe_hidden {
                        self.h1_vec[i] = silu(self.h1_vec[i]) * self.h3_vec[i];
                    }
                    upload_to_buffer(&self.h1_vec, &self.h1_buf, moe_hidden);

                    // down_proj: h1_buf [moe_hidden] → attn_out_buf [dim]
                    gemv_dispatch(
                        self.metal_ctx,
                        &self.metal_weights,
                        &self.metal_f16_weights,
                        &self.cpu_weights,
                        &expert.down_proj,
                        &self.h1_buf,
                        &self.attn_out_buf,
                        &self.h1_vec,
                        &mut self.attn_o,
                        dim,
                        moe_hidden,
                    )?;

                    let w = expert_weight;
                    for i in 0..dim {
                        self.res_vec[i] += w * self.attn_o[i];
                    }
                }

                // Shared expert (Qwen2/3-MoE) — always applied with weight 1.0
                if let Some(shared) = &moe_w.shared_expert {
                    gemv_dispatch(
                        self.metal_ctx,
                        &self.metal_weights,
                        &self.metal_f16_weights,
                        &self.cpu_weights,
                        &shared.gate_proj,
                        &self.x_norm_buf,
                        &self.h1_buf,
                        &self.x_norm,
                        &mut self.h1_vec,
                        moe_hidden,
                        dim,
                    )?;
                    gemv_dispatch(
                        self.metal_ctx,
                        &self.metal_weights,
                        &self.metal_f16_weights,
                        &self.cpu_weights,
                        &shared.up_proj,
                        &self.x_norm_buf,
                        &self.h3_buf,
                        &self.x_norm,
                        &mut self.h3_vec,
                        moe_hidden,
                        dim,
                    )?;
                    for i in 0..moe_hidden {
                        self.h1_vec[i] = silu(self.h1_vec[i]) * self.h3_vec[i];
                    }
                    upload_to_buffer(&self.h1_vec, &self.h1_buf, moe_hidden);
                    gemv_dispatch(
                        self.metal_ctx,
                        &self.metal_weights,
                        &self.metal_f16_weights,
                        &self.cpu_weights,
                        &shared.down_proj,
                        &self.h1_buf,
                        &self.attn_out_buf,
                        &self.h1_vec,
                        &mut self.attn_o,
                        dim,
                        moe_hidden,
                    )?;
                    for i in 0..dim {
                        self.res_vec[i] += self.attn_o[i];
                    }
                }

                // Add MoE output to residual and upload
                for i in 0..dim {
                    self.x_cur[i] += self.res_vec[i];
                }
                upload_to_buffer(&self.x_cur, &self.x_buf, dim);
                x_cpu_valid = true;

                if trace_enabled {
                    rms(&self.res_vec)
                } else {
                    0.0
                }
            } else {
                // ── Dense FFN path ───────────────────────────────────────────────────
                let mlp_norm_can_chain = self.metal_f32_weights.contains_key(&layer.ffn_norm)
                    && self.metal_weights.contains_key(&layer.feed_forward_w1)
                    && self.metal_weights.contains_key(&layer.feed_forward_w2)
                    && self.metal_weights.contains_key(&layer.feed_forward_w3);

                if !mlp_norm_can_chain {
                    rmsnorm_dispatch(
                        self.metal_ctx,
                        &self.metal_f32_weights,
                        &self.cpu_weights,
                        &layer.ffn_norm,
                        &self.x_buf,
                        &mut self.x_cur,
                        &mut x_cpu_valid,
                        &self.x_norm_buf,
                        &mut self.x_norm,
                        &mut x_norm_cpu_valid,
                        self.config.eps,
                        self.config.zero_centered_rmsnorm,
                    )?;
                } else {
                    x_norm_cpu_valid = false;
                }

                let mut h1_cpu_ready = false;
                let mlp_rms;

                if let (Some(w1), Some(w3), Some(w2)) = (
                    self.metal_weights.get(&layer.feed_forward_w1),
                    self.metal_weights.get(&layer.feed_forward_w3),
                    self.metal_weights.get(&layer.feed_forward_w2),
                ) {
                    let ffn_norm_w = self
                        .metal_f32_weights
                        .get(&layer.ffn_norm)
                        .ok_or_else(|| format!("Weight not found: {}", layer.ffn_norm))?;
                    let jobs = [
                        lns_metal::GemvBatchJob {
                            weights: w1,
                            y: &self.h1_buf,
                            out_dim: hidden_dim,
                            in_dim: dim,
                            qt: QuantType::Q4L,
                        },
                        lns_metal::GemvBatchJob {
                            weights: w3,
                            y: &self.h3_buf,
                            out_dim: hidden_dim,
                            in_dim: dim,
                            qt: QuantType::Q4L,
                        },
                    ];
                    let cb = self.metal_ctx.queue.new_command_buffer();
                    self.metal_ctx.encode_rmsnorm_buffer(
                        &cb,
                        &self.x_buf,
                        ffn_norm_w,
                        &self.x_norm_buf,
                        dim,
                        self.config.eps,
                        self.config.zero_centered_rmsnorm,
                    )?;
                    self.metal_ctx
                        .encode_gemv_batch_cached(&cb, &self.x_norm_buf, &jobs)?;
                    self.metal_ctx
                        .encode_swiglu(&cb, &self.h1_buf, &self.h3_buf, hidden_dim)?;
                    self.metal_ctx.encode_gemv_accumulate_cached(
                        &cb,
                        w2,
                        &self.h1_buf,
                        &self.x_buf,
                        dim,
                        hidden_dim,
                    )?;
                    cb.commit();
                    cb.wait_until_completed();
                    x_cpu_valid = false;
                    if trace_enabled {
                        readback_from_buffer(&self.h1_buf, &mut self.h1_vec, hidden_dim);
                    }
                    mlp_rms = if trace_enabled {
                        rms(&self.h1_vec)
                    } else {
                        0.0
                    };
                } else {
                    if let (Some(w1), Some(w3)) = (
                        self.metal_weights.get(&layer.feed_forward_w1),
                        self.metal_weights.get(&layer.feed_forward_w3),
                    ) {
                        let jobs = [
                            lns_metal::GemvBatchJob {
                                weights: w1,
                                y: &self.h1_buf,
                                out_dim: hidden_dim,
                                in_dim: dim,
                                qt: QuantType::Q4L,
                            },
                            lns_metal::GemvBatchJob {
                                weights: w3,
                                y: &self.h3_buf,
                                out_dim: hidden_dim,
                                in_dim: dim,
                                qt: QuantType::Q4L,
                            },
                        ];
                        self.metal_ctx
                            .dispatch_gemv_batch_cached(&self.x_norm_buf, &jobs)?;
                    } else {
                        ensure_buffer_readback(
                            &self.x_norm_buf,
                            &mut self.x_norm,
                            dim,
                            &mut x_norm_cpu_valid,
                        );
                        gemv_dispatch(
                            self.metal_ctx,
                            &self.metal_weights,
                            &self.metal_f16_weights,
                            &self.cpu_weights,
                            &layer.feed_forward_w1,
                            &self.x_norm_buf,
                            &self.h1_buf,
                            &self.x_norm,
                            &mut self.h1_vec,
                            hidden_dim,
                            dim,
                        )?;
                        gemv_dispatch(
                            self.metal_ctx,
                            &self.metal_weights,
                            &self.metal_f16_weights,
                            &self.cpu_weights,
                            &layer.feed_forward_w3,
                            &self.x_norm_buf,
                            &self.h3_buf,
                            &self.x_norm,
                            &mut self.h3_vec,
                            hidden_dim,
                            dim,
                        )?;
                    }

                    self.metal_ctx
                        .dispatch_swiglu(&self.h1_buf, &self.h3_buf, hidden_dim)?;

                    let w2_on_gpu = self.metal_weights.contains_key(&layer.feed_forward_w2)
                        || self.metal_f16_weights.contains_key(&layer.feed_forward_w2);
                    let needs_h1_cpu = trace_enabled || !w2_on_gpu;
                    if needs_h1_cpu {
                        readback_from_buffer(&self.h1_buf, &mut self.h1_vec, hidden_dim);
                        h1_cpu_ready = true;
                    }
                    mlp_rms = if trace_enabled {
                        rms(&self.h1_vec)
                    } else {
                        0.0
                    };

                    if let Some(gw) = self.metal_weights.get(&layer.feed_forward_w2) {
                        self.metal_ctx.dispatch_gemv_accumulate_cached(
                            gw,
                            &self.h1_buf,
                            &self.x_buf,
                            dim,
                            hidden_dim,
                        )?;
                        x_cpu_valid = false;
                    } else if let Some(gw_f16) = self.metal_f16_weights.get(&layer.feed_forward_w2)
                    {
                        // v3.3: down_proj promoted to Q8_L is mirrored as F16 on GPU.
                        // No native F16 GEMV-accumulate kernel; emulate with
                        // F16 GEMV → scratch (attn_out_buf), then add_inplace.
                        self.metal_ctx.dispatch_f16_gemv(
                            gw_f16,
                            &self.h1_buf,
                            &self.attn_out_buf,
                            dim,
                            hidden_dim,
                        )?;
                        self.metal_ctx.dispatch_add_inplace(
                            &self.attn_out_buf,
                            &self.x_buf,
                            dim,
                        )?;
                        x_cpu_valid = false;
                    } else {
                        if !h1_cpu_ready {
                            readback_from_buffer(&self.h1_buf, &mut self.h1_vec, hidden_dim);
                        }
                        let weights =
                            self.cpu_weights
                                .get(&layer.feed_forward_w2)
                                .ok_or_else(|| {
                                    format!(
                                        "Weight not found or not loaded on CPU: {}",
                                        layer.feed_forward_w2
                                    )
                                })?;
                        cpu_gemv(weights, &self.h1_vec, &mut self.res_vec, dim, hidden_dim);
                        ensure_buffer_readback(&self.x_buf, &mut self.x_cur, dim, &mut x_cpu_valid);
                        for i in 0..dim {
                            self.x_cur[i] += self.res_vec[i];
                        }
                        upload_to_buffer(&self.x_cur, &self.x_buf, dim);
                    }
                }
                mlp_rms
            }; // end dense FFN else branch

            if let Some(trace) = trace.as_deref_mut() {
                ensure_buffer_readback(&self.x_buf, &mut self.x_cur, dim, &mut x_cpu_valid);
                trace.layers.push(LayerTrace {
                    layer_idx: l_idx,
                    input_rms,
                    attention_rms,
                    post_attention_rms,
                    mlp_rms,
                    post_mlp_rms: rms(&self.x_cur),
                });
            }
        }

        rmsnorm_dispatch(
            self.metal_ctx,
            &self.metal_f32_weights,
            &self.cpu_weights,
            &self.weights.norm_final,
            &self.x_buf,
            &mut self.x_cur,
            &mut x_cpu_valid,
            &self.x_norm_buf,
            &mut self.x_norm,
            &mut x_norm_cpu_valid,
            self.config.eps,
            self.config.zero_centered_rmsnorm,
        )?;
        if let Some(trace) = trace.as_deref_mut() {
            ensure_buffer_readback(
                &self.x_norm_buf,
                &mut self.x_norm,
                dim,
                &mut x_norm_cpu_valid,
            );
            trace.final_norm_rms = rms(&self.x_norm);
        }

        if !need_logits {
            return Ok(vec![]);
        }

        let mut logits = vec![0.0f32; self.config.vocab_size];
        if let Some(gw) = self.metal_weights.get(&self.output_weight_name) {
            self.metal_ctx.dispatch_gemv_cached(
                gw,
                &self.x_norm_buf,
                &self.logits_buf,
                self.config.vocab_size,
                dim,
                QuantType::Q4L,
            )?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.logits_buf.contents() as *const f32,
                    logits.as_mut_ptr(),
                    self.config.vocab_size,
                );
            }
        } else if let Some(gw) = self.metal_f16_weights.get(&self.output_weight_name) {
            self.metal_ctx.dispatch_f16_gemv(
                gw,
                &self.x_norm_buf,
                &self.logits_buf,
                self.config.vocab_size,
                dim,
            )?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.logits_buf.contents() as *const f32,
                    logits.as_mut_ptr(),
                    self.config.vocab_size,
                );
            }
        } else {
            ensure_buffer_readback(
                &self.x_norm_buf,
                &mut self.x_norm,
                dim,
                &mut x_norm_cpu_valid,
            );
            cpu_gemv(
                self.cpu_weights
                    .get(&self.output_weight_name)
                    .ok_or("Missing output")?,
                &self.x_norm,
                &mut logits,
                self.config.vocab_size,
                dim,
            );
        }

        Ok(logits)
    }

    pub fn forward(&mut self, token_id: u32, pos: usize) -> Result<Vec<f32>, String> {
        self.forward_internal(token_id, pos, None, true)
    }

    pub fn forward_greedy(&mut self, token_id: u32, pos: usize) -> Result<u32, String> {
        if self.all_layers_gpu_capable()
            && self.metal_ctx.argmax_pass1_pipeline.is_some()
            && self.metal_ctx.argmax_pass2_pipeline.is_some()
        {
            self.forward_single_cb(token_id, pos, true, false)?;
            return Ok(self.metal_ctx.read_argmax_result());
        }

        let logits = self.forward_internal(token_id, pos, None, true)?;
        let (max_index, _) = logits
            .iter()
            .enumerate()
            .max_by(|left, right| {
                left.1
                    .partial_cmp(right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((0, &0.0));
        Ok(max_index as u32)
    }

    /// Process all prompt tokens, updating the KV cache for each.
    /// The lm-head projection is skipped for every token except the last,
    /// which is the main cost saving compared to calling `forward()` in a loop.
    /// Returns `(logits_for_last_token, next_pos)`.
    pub fn forward_prefill(
        &mut self,
        tokens: &[u32],
        pos_start: usize,
    ) -> Result<(Vec<f32>, usize), String> {
        if tokens.is_empty() {
            return Ok((vec![], pos_start));
        }
        let last_idx = tokens.len() - 1;
        let mut pos = pos_start;
        let mut last_logits = vec![];
        for (i, &token_id) in tokens.iter().enumerate() {
            let is_last = i == last_idx;
            let logits = self.forward_internal(token_id, pos, None, is_last)?;
            if is_last {
                last_logits = logits;
            }
            pos += 1;
        }
        Ok((last_logits, pos))
    }

    pub fn force_cpu_weights(&mut self) -> Result<(), String> {
        // Guard against OOM: decoding Q4L → f32 expands ~8×.
        // A 2 GB packed model would need ~16 GB of heap — guaranteed SIGKILL.
        let packed_bytes: u64 = self.metal_weights.values().map(|b| b.0.length()).sum();
        let estimated_f32_gb = (packed_bytes * 8) / (1 << 30);
        if estimated_f32_gb > 2 {
            return Err(format!(
                "--cpu-gemv refused: decoding {estimated_f32_gb} GB of Q4L weights to f32 would OOM. \
                 Large models must use the Metal GPU path. Drop --cpu-gemv."
            ));
        }
        let names: Vec<String> = self.metal_weights.keys().cloned().collect();
        for name in names {
            if self.cpu_weights.contains_key(&name) {
                continue;
            }
            let tensor = self
                .tensor_map
                .get(&name)
                .ok_or_else(|| format!("missing archived tensor for {name}"))?;
            let decoded = tensor.to_f32().map_err(|e| e.to_string())?;
            self.cpu_weights.insert(name, decoded);
        }
        for name in [&self.output_weight_name] {
            if self.cpu_weights.contains_key(name.as_str()) {
                continue;
            }
            let tensor = self
                .tensor_map
                .get(name.as_str())
                .ok_or_else(|| format!("missing archived tensor for {name}"))?;
            let decoded = tensor.to_f32().map_err(|e| e.to_string())?;
            self.cpu_weights.insert(name.clone(), decoded);
        }
        self.metal_weights.clear();
        self.metal_f16_weights.clear();
        self.metal_f32_weights.clear();
        Ok(())
    }

    pub fn forward_with_trace(
        &mut self,
        token_id: u32,
        pos: usize,
    ) -> Result<(Vec<f32>, ForwardTrace), String> {
        let mut trace = ForwardTrace {
            embedding_rms: 0.0,
            layers: Vec::with_capacity(self.weights.layer_weights.len()),
            final_norm_rms: 0.0,
        };
        let logits = self.forward_internal(token_id, pos, Some(&mut trace), true)?;
        Ok((logits, trace))
    }

    pub fn clear_kv_cache(&mut self) {
        for (k, v) in &self.paged_kv_data {
            unsafe {
                std::ptr::write_bytes(k.contents() as *mut u8, 0, k.length() as usize);
                std::ptr::write_bytes(v.contents() as *mut u8, 0, v.length() as usize);
            }
        }
        for (ks, vs) in &self.paged_kv_scales {
            unsafe {
                std::ptr::write_bytes(ks.contents() as *mut u8, 0, ks.length() as usize);
                std::ptr::write_bytes(vs.contents() as *mut u8, 0, vs.length() as usize);
            }
        }
        for pool in &mut self.kv_page_pools {
            pool.reset();
        }
        for bt in &mut self.kv_block_tables {
            bt.clear();
        }
        for bt_buf in &self.kv_block_table_bufs {
            unsafe {
                std::ptr::write_bytes(bt_buf.contents() as *mut u8, 0xff, bt_buf.length() as usize);
            }
        }
        for buffer in self.linear_conv_state_buffers.iter().flatten() {
            unsafe {
                std::ptr::write_bytes(buffer.contents() as *mut u8, 0, buffer.length() as usize);
            }
        }
        for buffer in self.linear_recurrent_state_buffers.iter().flatten() {
            unsafe {
                std::ptr::write_bytes(buffer.contents() as *mut u8, 0, buffer.length() as usize);
            }
        }
        unsafe {
            std::ptr::write_bytes(
                self.zero_buf.contents() as *mut u8,
                0,
                self.zero_buf.length() as usize,
            );
        }
        for state in self.linear_states.iter_mut().flatten() {
            state.conv_state.fill(0.0);
            state.recurrent_state.fill(0.0);
        }
    }

    // ── Paged KV cache helpers ────────────────────────────────────────────

    /// Ensure the physical page covering `pos` is allocated for layer `l_idx`.
    fn ensure_kv_page(&mut self, l_idx: usize, pos: usize) -> Result<(), String> {
        let lp = pos / PAGE_SIZE;
        if lp < self.kv_block_tables[l_idx].len() && self.kv_block_tables[l_idx][lp] != u32::MAX {
            return Ok(());
        }
        while self.kv_block_tables[l_idx].len() <= lp {
            self.kv_block_tables[l_idx].push(u32::MAX);
        }
        let phys = self.kv_page_pools[l_idx]
            .alloc()
            .ok_or_else(|| format!("KV page pool exhausted at layer {l_idx} (pos={pos})"))?;
        self.kv_block_tables[l_idx][lp] = phys;
        Ok(())
    }

    /// Reclaim logical pages whose every token is strictly before `p_start = pos - window + 1`.
    fn evict_kv_pages(&mut self, l_idx: usize, pos: usize, window_size: usize) {
        if window_size == 0 {
            return;
        }
        let p_start = pos.saturating_sub(window_size - 1);
        let evict_before = p_start / PAGE_SIZE; // pages [0, evict_before) are fully outside window
        for lp in 0..evict_before.min(self.kv_block_tables[l_idx].len()) {
            let phys = self.kv_block_tables[l_idx][lp];
            if phys != u32::MAX {
                self.kv_page_pools[l_idx].free_page(phys);
                self.kv_block_tables[l_idx][lp] = u32::MAX;
            }
        }
    }

    /// Upload the CPU block table for `l_idx` to its GPU buffer.
    fn upload_block_table(&self, l_idx: usize) {
        let bt = &self.kv_block_tables[l_idx];
        if bt.is_empty() {
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                bt.as_ptr() as *const u8,
                self.kv_block_table_bufs[l_idx].contents() as *mut u8,
                bt.len() * 4,
            );
        }
    }

    /// Compute `p_start` for the attention scan window.
    fn kv_p_start(&self, pos: usize) -> usize {
        match self.config.sliding_window {
            Some(w) if w > 0 => pos.saturating_sub(w - 1),
            _ => 0,
        }
    }

    /// Returns true when every layer can be encoded into a single GPU command buffer.
    /// Supports dense (full-attention), hybrid (linear-attention), and any mix,
    /// as long as all weights are on the GPU and no MoE routing is required.
    fn all_layers_gpu_capable(&self) -> bool {
        if self.config.moe.is_some() {
            return false;
        }
        for (l_idx, layer) in self.weights.layer_weights.iter().enumerate() {
            if layer.moe.is_some() {
                return false;
            }
            if !self.metal_f32_weights.contains_key(&layer.attention_norm) {
                return false;
            }
            if !self.metal_weights.contains_key(&layer.feed_forward_w1) {
                return false;
            }
            if !self.metal_weights.contains_key(&layer.feed_forward_w2) {
                return false;
            }
            if !self.metal_weights.contains_key(&layer.feed_forward_w3) {
                return false;
            }
            if !self.metal_f32_weights.contains_key(&layer.ffn_norm) {
                return false;
            }

            match self
                .architecture
                .text_layers
                .get(l_idx)
                .unwrap_or(&TextLayerKind::FullAttention)
            {
                TextLayerKind::FullAttention => {
                    let (q_proj, k_proj, v_proj, o_proj, q_norm, k_norm) = match &layer.attention {
                        HybridAttentionWeights::FullAttention {
                            q_proj,
                            k_proj,
                            v_proj,
                            o_proj,
                            q_norm,
                            k_norm,
                        } => (
                            q_proj.as_str(),
                            k_proj.as_str(),
                            v_proj.as_str(),
                            o_proj.as_str(),
                            q_norm.as_deref(),
                            k_norm.as_deref(),
                        ),
                        _ => return false,
                    };
                    if !self.metal_weights.contains_key(q_proj) {
                        return false;
                    }
                    if !self.metal_weights.contains_key(k_proj) {
                        return false;
                    }
                    if !self.metal_weights.contains_key(v_proj) {
                        return false;
                    }
                    if !self.metal_weights.contains_key(o_proj) {
                        return false;
                    }
                    if let Some(n) = q_norm {
                        if !self.metal_f32_weights.contains_key(n) {
                            return false;
                        }
                    }
                    if let Some(n) = k_norm {
                        if !self.metal_f32_weights.contains_key(n) {
                            return false;
                        }
                    }
                }
                TextLayerKind::LinearAttention => {
                    if self.config.linear_attention.is_none() {
                        return false;
                    }
                    let (
                        in_proj_qkv,
                        in_proj_z,
                        in_proj_b,
                        in_proj_a,
                        out_proj,
                        conv1d_weight,
                        conv1d_bias,
                        a_log,
                        dt_bias,
                        norm,
                    ) = match &layer.attention {
                        HybridAttentionWeights::LinearAttention {
                            in_proj_qkv,
                            in_proj_z,
                            in_proj_b,
                            in_proj_a,
                            out_proj,
                            conv1d_weight,
                            conv1d_bias,
                            a_log,
                            dt_bias,
                            norm,
                        } => (
                            in_proj_qkv.as_str(),
                            in_proj_z.as_str(),
                            in_proj_b.as_str(),
                            in_proj_a.as_str(),
                            out_proj.as_str(),
                            conv1d_weight.as_str(),
                            conv1d_bias.as_deref(),
                            a_log.as_str(),
                            dt_bias.as_str(),
                            norm.as_str(),
                        ),
                        _ => return false,
                    };
                    if !self.metal_weights.contains_key(in_proj_qkv) {
                        return false;
                    }
                    if !self.metal_weights.contains_key(in_proj_z) {
                        return false;
                    }
                    if !self.metal_weights.contains_key(in_proj_b) {
                        return false;
                    }
                    if !self.metal_weights.contains_key(in_proj_a) {
                        return false;
                    }
                    if !self.metal_weights.contains_key(out_proj) {
                        return false;
                    }
                    if !self.metal_f32_weights.contains_key(conv1d_weight) {
                        return false;
                    }
                    if !self.metal_f32_weights.contains_key(a_log) {
                        return false;
                    }
                    if !self.metal_f32_weights.contains_key(dt_bias) {
                        return false;
                    }
                    if !self.metal_f32_weights.contains_key(norm) {
                        return false;
                    }
                    if let Some(b) = conv1d_bias {
                        if !self.metal_f32_weights.contains_key(b) {
                            return false;
                        }
                    }
                    if self.linear_conv_state_buffers[l_idx].is_none() {
                        return false;
                    }
                    if self.linear_recurrent_state_buffers[l_idx].is_none() {
                        return false;
                    }
                }
                TextLayerKind::MlpOnly => {}
                TextLayerKind::Unknown(_) => return false,
            }
        }
        self.metal_weights.contains_key(&self.output_weight_name)
            || self
                .metal_f16_weights
                .contains_key(&self.output_weight_name)
    }

    /// Fast path: encode the entire forward pass into a single Metal command buffer.
    /// Eliminates per-layer commit/wait overhead (~1 ms × n_layers per token).
    /// Handles dense full-attention, hybrid linear-attention, and any mix.
    /// Only called when `all_layers_gpu_capable()` is true and trace is not needed.
    fn forward_single_cb(
        &mut self,
        token_id: u32,
        pos: usize,
        need_logits: bool,
        read_logits: bool,
    ) -> Result<Vec<f32>, String> {
        if self.profile_stages {
            return self.forward_profiled(token_id, pos, need_logits, read_logits);
        }
        let dim = self.config.dim;
        let head_dim = self.config.attention_head_dim;
        let attn_dim = self.config.n_heads * head_dim;
        let kv_dim = self.config.n_kv_heads * head_dim;
        let hidden_dim = self.config.hidden_dim;
        let gated = self.config.gated_query_attention;

        // 1. Load embedding → x_buf (CPU→shared memory, no GPU yet)
        let embedding_tensor = self
            .tensor_map
            .get(&self.weights.token_embedding)
            .ok_or_else(|| format!("Missing embedding: {}", self.weights.token_embedding))?;
        load_tensor_row(embedding_tensor, token_id as usize, dim, &mut self.x_cur)?;
        upload_to_buffer(&self.x_cur, &self.x_buf, dim);

        let rope_dim = {
            let rot = ((head_dim as f32) * self.config.partial_rotary_factor).round() as usize;
            if rot >= 2 && rot < head_dim {
                rot
            } else {
                head_dim
            }
        };

        // 2. CPU: pre-allocate KV pages for every layer at this position.
        //    All writes go to StorageModeShared buffers. They're visible to the GPU
        //    after commit() since the CB hasn't been submitted yet.
        let n_layers = self.weights.layer_weights.len();
        let window = self.config.sliding_window.unwrap_or(0);
        for l_idx in 0..n_layers {
            let kind = self
                .architecture
                .text_layers
                .get(l_idx)
                .cloned()
                .unwrap_or(TextLayerKind::FullAttention);
            if matches!(kind, TextLayerKind::FullAttention) {
                self.ensure_kv_page(l_idx, pos)?;
                self.evict_kv_pages(l_idx, pos, window);
                self.upload_block_table(l_idx);
            }
        }

        let kv_p_start = self.kv_p_start(pos);

        // 3. One command buffer + one persistent compute encoder for the full forward pass.
        //    Eliminates ~198 encoder alloc/end pairs per token vs. the old per-kernel pattern.
        let cb = self.metal_ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();

        for l_idx in 0..n_layers {
            let layer = &self.weights.layer_weights[l_idx];
            let layer_kind = self
                .architecture
                .text_layers
                .get(l_idx)
                .cloned()
                .unwrap_or(TextLayerKind::FullAttention);

            // ── Attention norm (shared for all layer kinds) ──────────────────────
            let attn_norm_w = self
                .metal_f32_weights
                .get(&layer.attention_norm)
                .ok_or_else(|| format!("Missing attn norm: {}", layer.attention_norm))?;
            self.metal_ctx.rmsnorm_enc(
                enc,
                &self.x_buf,
                attn_norm_w,
                &self.x_norm_buf,
                dim,
                self.config.eps,
                self.config.zero_centered_rmsnorm,
            )?;

            match layer_kind {
                TextLayerKind::FullAttention => {
                    let (q_proj, k_proj, v_proj, o_proj, q_norm_name, k_norm_name) = match &layer
                        .attention
                    {
                        HybridAttentionWeights::FullAttention {
                            q_proj,
                            k_proj,
                            v_proj,
                            o_proj,
                            q_norm,
                            k_norm,
                        } => (q_proj, k_proj, v_proj, o_proj, q_norm.as_ref(), k_norm.as_ref()),
                        _ => return Err(format!("Layer {l_idx}: expected full-attention weights")),
                    };

                    {
                        let qw = self
                            .metal_weights
                            .get(q_proj)
                            .ok_or_else(|| format!("Missing {q_proj}"))?;
                        let kw = self
                            .metal_weights
                            .get(k_proj)
                            .ok_or_else(|| format!("Missing {k_proj}"))?;
                        let vw = self
                            .metal_weights
                            .get(v_proj)
                            .ok_or_else(|| format!("Missing {v_proj}"))?;
                        if gated {
                            let jobs = [
                                lns_metal::GemvBatchJob {
                                    weights: qw,
                                    y: &self.h1_buf,
                                    out_dim: attn_dim * 2,
                                    in_dim: dim,
                                    qt: QuantType::Q4L,
                                },
                                lns_metal::GemvBatchJob {
                                    weights: kw,
                                    y: &self.k_buf,
                                    out_dim: kv_dim,
                                    in_dim: dim,
                                    qt: QuantType::Q4L,
                                },
                                lns_metal::GemvBatchJob {
                                    weights: vw,
                                    y: &self.v_buf,
                                    out_dim: kv_dim,
                                    in_dim: dim,
                                    qt: QuantType::Q4L,
                                },
                            ];
                            self.metal_ctx
                                .gemv_batch_multirow_enc(enc, &self.x_norm_buf, &jobs)?;
                            self.metal_ctx.split_gated_query_enc(
                                enc,
                                &self.h1_buf,
                                &self.q_buf,
                                &self.h3_buf,
                                attn_dim,
                                head_dim,
                            )?;
                        } else {
                            let jobs = [
                                lns_metal::GemvBatchJob {
                                    weights: qw,
                                    y: &self.q_buf,
                                    out_dim: attn_dim,
                                    in_dim: dim,
                                    qt: QuantType::Q4L,
                                },
                                lns_metal::GemvBatchJob {
                                    weights: kw,
                                    y: &self.k_buf,
                                    out_dim: kv_dim,
                                    in_dim: dim,
                                    qt: QuantType::Q4L,
                                },
                                lns_metal::GemvBatchJob {
                                    weights: vw,
                                    y: &self.v_buf,
                                    out_dim: kv_dim,
                                    in_dim: dim,
                                    qt: QuantType::Q4L,
                                },
                            ];
                            self.metal_ctx
                                .gemv_batch_multirow_enc(enc, &self.x_norm_buf, &jobs)?;
                        }
                    }

                    // Optional per-head QK norms (Qwen2-style)
                    if let Some(name) = q_norm_name {
                        let w = self
                            .metal_f32_weights
                            .get(name)
                            .ok_or_else(|| format!("Missing q_norm: {name}"))?;
                        self.metal_ctx.rmsnorm_rows_enc(
                            enc,
                            &self.q_buf,
                            w,
                            head_dim,
                            self.config.n_heads,
                            self.config.eps,
                            self.config.zero_centered_rmsnorm,
                        )?;
                    }
                    if let Some(name) = k_norm_name {
                        let w = self
                            .metal_f32_weights
                            .get(name)
                            .ok_or_else(|| format!("Missing k_norm: {name}"))?;
                        self.metal_ctx.rmsnorm_rows_enc(
                            enc,
                            &self.k_buf,
                            w,
                            head_dim,
                            self.config.n_kv_heads,
                            self.config.eps,
                            self.config.zero_centered_rmsnorm,
                        )?;
                    }
                    self.metal_ctx.rope_with_freqs_enc(
                        enc,
                        &self.q_buf,
                        &self.k_buf,
                        &self.rope_inv_freqs_buf,
                        pos,
                        head_dim,
                        rope_dim,
                    )?;

                    {
                        let (k_data, v_data) = &self.paged_kv_data[l_idx];
                        let (ks_data, vs_data) = &self.paged_kv_scales[l_idx];
                        let bt_buf = &self.kv_block_table_bufs[l_idx];
                        self.metal_ctx.attention_paged_q8a_enc(
                            enc,
                            &self.q_buf,
                            &self.k_buf,
                            &self.v_buf,
                            k_data,
                            v_data,
                            ks_data,
                            vs_data,
                            bt_buf,
                            &self.attn_out_buf,
                            pos,
                            head_dim,
                            self.config.n_heads,
                            self.config.n_kv_heads,
                            kv_p_start,
                        )?;
                    }

                    if gated {
                        self.metal_ctx.mul_inplace_enc(
                            enc,
                            &self.attn_out_buf,
                            &self.h3_buf,
                            attn_dim,
                        )?;
                    }

                    {
                        let ow = self
                            .metal_weights
                            .get(o_proj)
                            .ok_or_else(|| format!("Missing {o_proj}"))?;
                        self.metal_ctx.gemv_accumulate_multirow_enc(
                            enc,
                            ow,
                            &self.attn_out_buf,
                            &self.x_buf,
                            dim,
                            attn_dim,
                        )?;
                    }
                }

                TextLayerKind::LinearAttention => {
                    let linear = self
                        .config
                        .linear_attention
                        .as_ref()
                        .ok_or("Linear attention config missing for linear-attention layer")?;
                    let conv_dim = linear.conv_dim();
                    let value_dim = linear.value_dim();
                    let num_key_heads = linear.num_key_heads;
                    let num_value_heads = linear.num_value_heads;
                    let key_head_dim = linear.key_head_dim;
                    let value_head_dim = linear.value_head_dim;
                    let conv_kernel_size = linear.conv_kernel_size;

                    let (
                        in_proj_qkv,
                        in_proj_z,
                        in_proj_b,
                        in_proj_a,
                        out_proj,
                        conv1d_weight,
                        conv1d_bias,
                        a_log,
                        dt_bias,
                        norm,
                    ) = match &layer.attention {
                        HybridAttentionWeights::LinearAttention {
                            in_proj_qkv,
                            in_proj_z,
                            in_proj_b,
                            in_proj_a,
                            out_proj,
                            conv1d_weight,
                            conv1d_bias,
                            a_log,
                            dt_bias,
                            norm,
                        } => (
                            in_proj_qkv,
                            in_proj_z,
                            in_proj_b,
                            in_proj_a,
                            out_proj,
                            conv1d_weight,
                            conv1d_bias.as_ref(),
                            a_log,
                            dt_bias,
                            norm,
                        ),
                        _ => {
                            return Err(format!("Layer {l_idx}: expected linear-attention weights"))
                        }
                    };

                    {
                        let qkv_w = self
                            .metal_weights
                            .get(in_proj_qkv)
                            .ok_or_else(|| format!("Missing {in_proj_qkv}"))?;
                        let z_w = self
                            .metal_weights
                            .get(in_proj_z)
                            .ok_or_else(|| format!("Missing {in_proj_z}"))?;
                        let b_w = self
                            .metal_weights
                            .get(in_proj_b)
                            .ok_or_else(|| format!("Missing {in_proj_b}"))?;
                        let a_w = self
                            .metal_weights
                            .get(in_proj_a)
                            .ok_or_else(|| format!("Missing {in_proj_a}"))?;
                        let jobs = [
                            lns_metal::GemvBatchJob {
                                weights: qkv_w,
                                y: &self.h1_buf,
                                out_dim: conv_dim,
                                in_dim: dim,
                                qt: QuantType::Q4L,
                            },
                            lns_metal::GemvBatchJob {
                                weights: z_w,
                                y: &self.h3_buf,
                                out_dim: value_dim,
                                in_dim: dim,
                                qt: QuantType::Q4L,
                            },
                            lns_metal::GemvBatchJob {
                                weights: b_w,
                                y: &self.q_buf,
                                out_dim: num_value_heads,
                                in_dim: dim,
                                qt: QuantType::Q4L,
                            },
                            lns_metal::GemvBatchJob {
                                weights: a_w,
                                y: &self.k_buf,
                                out_dim: num_value_heads,
                                in_dim: dim,
                                qt: QuantType::Q4L,
                            },
                        ];
                        self.metal_ctx
                            .gemv_batch_multirow_enc(enc, &self.x_norm_buf, &jobs)?;
                    }

                    {
                        let conv_w_buf = self
                            .metal_f32_weights
                            .get(conv1d_weight)
                            .ok_or_else(|| format!("Missing {conv1d_weight}"))?;
                        let a_log_buf = self
                            .metal_f32_weights
                            .get(a_log)
                            .ok_or_else(|| format!("Missing {a_log}"))?;
                        let dt_bias_buf = self
                            .metal_f32_weights
                            .get(dt_bias)
                            .ok_or_else(|| format!("Missing {dt_bias}"))?;
                        let norm_buf = self
                            .metal_f32_weights
                            .get(norm)
                            .ok_or_else(|| format!("Missing {norm}"))?;
                        let bias_buf = conv1d_bias
                            .and_then(|n| self.metal_f32_weights.get(n))
                            .unwrap_or(&self.zero_buf);
                        let conv_state_buf = self.linear_conv_state_buffers[l_idx]
                            .as_ref()
                            .ok_or_else(|| format!("Missing conv state for layer {l_idx}"))?;
                        let recurrent_state_buf = self.linear_recurrent_state_buffers[l_idx]
                            .as_ref()
                            .ok_or_else(|| format!("Missing recurrent state for layer {l_idx}"))?;
                        self.metal_ctx.linear_attn_block_enc(
                            enc,
                            &self.h1_buf,
                            conv_state_buf,
                            conv_w_buf,
                            bias_buf,
                            &self.linear_conv_buf,
                            &self.h3_buf,
                            &self.q_buf,
                            &self.k_buf,
                            a_log_buf,
                            dt_bias_buf,
                            norm_buf,
                            recurrent_state_buf,
                            &self.attn_out_buf,
                            conv_kernel_size,
                            conv_dim,
                            num_key_heads,
                            num_value_heads,
                            key_head_dim,
                            value_head_dim,
                            self.config.eps,
                        )?;
                    }

                    {
                        let out_w = self
                            .metal_weights
                            .get(out_proj)
                            .ok_or_else(|| format!("Missing {out_proj}"))?;
                        self.metal_ctx.gemv_accumulate_multirow_enc(
                            enc,
                            out_w,
                            &self.attn_out_buf,
                            &self.x_buf,
                            dim,
                            value_dim,
                        )?;
                    }
                }

                TextLayerKind::MlpOnly => {
                    // No attention phase; x_norm_buf written by attn_norm_enc above is unused.
                    // x_buf carries residual from previous layer; barrier already set by prev layer's FFN.
                }

                TextLayerKind::Unknown(kind) => {
                    enc.end_encoding();
                    return Err(format!(
                        "Layer {l_idx}: unsupported layer kind '{kind}' in single-CB path"
                    ));
                }
            }

            // ── FFN (all layer kinds with dense MLP) ─────────────────────────────
            let ffn_norm_w = self
                .metal_f32_weights
                .get(&layer.ffn_norm)
                .ok_or_else(|| format!("Missing ffn norm: {}", layer.ffn_norm))?;
            self.metal_ctx.rmsnorm_enc(
                enc,
                &self.x_buf,
                ffn_norm_w,
                &self.x_norm_buf,
                dim,
                self.config.eps,
                self.config.zero_centered_rmsnorm,
            )?;

            // Fused W1+W3+SwiGLU: one dispatch instead of three.
            // h1_buf = silu(w1*x_norm) * (w3*x_norm) in one kernel call.
            {
                let w1 = self
                    .metal_weights
                    .get(&layer.feed_forward_w1)
                    .ok_or_else(|| format!("Missing {}", layer.feed_forward_w1))?;
                let w3 = self
                    .metal_weights
                    .get(&layer.feed_forward_w3)
                    .ok_or_else(|| format!("Missing {}", layer.feed_forward_w3))?;
                if w1.1 == w3.1 && matches!(w1.1, QuantType::Q4HQ | QuantType::Q4L) {
                    self.metal_ctx.w1w3_swiglu_multirow_enc(
                        enc, w1, w3, &self.x_norm_buf, &self.h1_buf, hidden_dim, dim,
                    )?;
                } else {
                    let jobs = [
                        lns_metal::GemvBatchJob { weights: w1, y: &self.h1_buf, out_dim: hidden_dim, in_dim: dim, qt: w1.1 },
                        lns_metal::GemvBatchJob { weights: w3, y: &self.h3_buf, out_dim: hidden_dim, in_dim: dim, qt: w3.1 },
                    ];
                    self.metal_ctx.gemv_batch_multirow_enc(enc, &self.x_norm_buf, &jobs)?;
                    self.metal_ctx.swiglu_enc(enc, &self.h1_buf, &self.h3_buf, hidden_dim)?;
                }
            }

            {
                let w2 = self
                    .metal_weights
                    .get(&layer.feed_forward_w2)
                    .ok_or_else(|| format!("Missing {}", layer.feed_forward_w2))?;
                self.metal_ctx.gemv_accumulate_multirow_enc(
                    enc,
                    w2,
                    &self.h1_buf,
                    &self.x_buf,
                    dim,
                    hidden_dim,
                )?;
            }
        }

        // 5. Logits projection
        if need_logits {
            // Final layer norm + logits projection (fused when possible)
            let final_norm_w = self
                .metal_f32_weights
                .get(&self.weights.norm_final)
                .ok_or_else(|| format!("Missing final norm: {}", self.weights.norm_final))?;
            let out_name = &self.output_weight_name;
            if let Some(gw) = self
                .metal_weights
                .get(out_name)
                .filter(|weights| weights.1 == QuantType::Q4L)
            {
                // Fused: RMSNorm + Q4L GEMV in one kernel (eliminates norm→logits barrier)
                self.metal_ctx.rmsnorm_gemv_multirow_enc(
                    enc,
                    &self.x_buf,
                    final_norm_w,
                    gw,
                    &self.logits_buf,
                    dim,
                    self.config.vocab_size,
                    self.config.eps,
                    self.config.zero_centered_rmsnorm,
                )?;
            } else {
                // Fallback: separate norm then f16 GEMV (or Q4HQ)
                // Note: Q4HQ lm_head is intentionally unfused here — with 62k+ threadgroups,
                // the fused path replicates norm computation per-TG and exceeds threadgroup
                // memory limits, hurting occupancy. Unfused is faster (x_normed fits in L2).
                self.metal_ctx.rmsnorm_enc(
                    enc,
                    &self.x_buf,
                    final_norm_w,
                    &self.x_norm_buf,
                    dim,
                    self.config.eps,
                    self.config.zero_centered_rmsnorm,
                )?;
                if let Some(gw) = self.metal_weights.get(out_name) {
                    let jobs = [lns_metal::GemvBatchJob {
                        weights: gw,
                        y: &self.logits_buf,
                        out_dim: self.config.vocab_size,
                        in_dim: dim,
                        qt: gw.1,
                    }];
                    self.metal_ctx
                        .gemv_batch_multirow_enc(enc, &self.x_norm_buf, &jobs)?;
                } else if let Some(gw) = self.metal_f16_weights.get(out_name) {
                    self.metal_ctx.f16_gemv_enc(
                        enc,
                        gw,
                        &self.x_norm_buf,
                        &self.logits_buf,
                        self.config.vocab_size,
                        dim,
                    )?;
                } else {
                    enc.end_encoding();
                    return Err(
                        "Output weight not on GPU — single-CB fast path unavailable".to_string()
                    );
                }
            }
            if !read_logits {
                // GPU argmax (greedy decode): two-pass within the same encoder, internal barrier.
                self.metal_ctx
                    .argmax_enc(enc, &self.logits_buf, self.config.vocab_size)?;
            }
        } else {
            // Prefill non-last token: skip norm+logits entirely
        }

        // 6. Close encoder, commit, wait
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // 7. Readback logits from GPU
        if need_logits && read_logits {
            let mut logits = vec![0.0f32; self.config.vocab_size];
            readback_from_buffer(&self.logits_buf, &mut logits, self.config.vocab_size);
            Ok(logits)
        } else {
            Ok(vec![])
        }
    }

    /// Stage-level profiling variant of forward_single_cb.
    /// Enabled only when LNS_PROFILE_STAGES=1. Splits the forward pass into
    /// separately-timed command buffers (all-layers + logits) and prints
    /// per-stage latency to stderr.
    fn forward_profiled(
        &mut self,
        token_id: u32,
        pos: usize,
        need_logits: bool,
        read_logits: bool,
    ) -> Result<Vec<f32>, String> {
        use std::time::Instant;

        let dim = self.config.dim;
        let head_dim = self.config.attention_head_dim;
        let attn_dim = self.config.n_heads * head_dim;
        let kv_dim = self.config.n_kv_heads * head_dim;
        let hidden_dim = self.config.hidden_dim;
        let gated = self.config.gated_query_attention;

        // Embedding upload
        let embedding_tensor = self
            .tensor_map
            .get(&self.weights.token_embedding)
            .ok_or_else(|| format!("Missing embedding: {}", self.weights.token_embedding))?;
        load_tensor_row(embedding_tensor, token_id as usize, dim, &mut self.x_cur)?;
        upload_to_buffer(&self.x_cur, &self.x_buf, dim);

        let rope_dim = {
            let rot = ((head_dim as f32) * self.config.partial_rotary_factor).round() as usize;
            if rot >= 2 && rot < head_dim { rot } else { head_dim }
        };

        let n_layers = self.weights.layer_weights.len();
        let window = self.config.sliding_window.unwrap_or(0);
        for l_idx in 0..n_layers {
            let kind = self
                .architecture
                .text_layers
                .get(l_idx)
                .cloned()
                .unwrap_or(TextLayerKind::FullAttention);
            if matches!(kind, TextLayerKind::FullAttention) {
                self.ensure_kv_page(l_idx, pos)?;
                self.evict_kv_pages(l_idx, pos, window);
                self.upload_block_table(l_idx);
            }
        }

        let kv_p_start = self.kv_p_start(pos);

        // ── All-layers stage ───────────────────────────────────────────────
        let t0 = Instant::now();
        {
            let cb = self.metal_ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();

            for l_idx in 0..n_layers {
                let layer = &self.weights.layer_weights[l_idx];
                let layer_kind = self
                    .architecture
                    .text_layers
                    .get(l_idx)
                    .cloned()
                    .unwrap_or(TextLayerKind::FullAttention);

                // Attention
                let attn_norm_w = self
                    .metal_f32_weights
                    .get(&layer.attention_norm)
                    .ok_or_else(|| format!("Missing attn norm: {}", layer.attention_norm))?;
                self.metal_ctx.rmsnorm_enc(
                    enc, &self.x_buf, attn_norm_w, &self.x_norm_buf,
                    dim, self.config.eps, self.config.zero_centered_rmsnorm,
                )?;
                MetalContext::barrier(enc, &[&self.x_norm_buf]);

                match layer_kind {
                    TextLayerKind::FullAttention => {
                        let (q_proj, k_proj, v_proj, o_proj, q_norm_name, k_norm_name) =
                            match &layer.attention {
                                HybridAttentionWeights::FullAttention {
                                    q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
                                } => (q_proj, k_proj, v_proj, o_proj, q_norm.as_ref(), k_norm.as_ref()),
                                _ => return Err(format!("Layer {l_idx}: expected full-attention")),
                            };
                        let qw = self.metal_weights.get(q_proj).ok_or_else(|| format!("Missing {q_proj}"))?;
                        let kw = self.metal_weights.get(k_proj).ok_or_else(|| format!("Missing {k_proj}"))?;
                        let vw = self.metal_weights.get(v_proj).ok_or_else(|| format!("Missing {v_proj}"))?;
                        if gated {
                            let jobs = [
                                lns_metal::GemvBatchJob { weights: qw, y: &self.h1_buf, out_dim: attn_dim * 2, in_dim: dim, qt: QuantType::Q4L },
                                lns_metal::GemvBatchJob { weights: kw, y: &self.k_buf, out_dim: kv_dim, in_dim: dim, qt: QuantType::Q4L },
                                lns_metal::GemvBatchJob { weights: vw, y: &self.v_buf, out_dim: kv_dim, in_dim: dim, qt: QuantType::Q4L },
                            ];
                            self.metal_ctx.gemv_batch_multirow_enc(enc, &self.x_norm_buf, &jobs)?;
                            MetalContext::barrier(enc, &[&self.h1_buf, &self.k_buf, &self.v_buf]);
                            self.metal_ctx.split_gated_query_enc(enc, &self.h1_buf, &self.q_buf, &self.h3_buf, attn_dim, head_dim)?;
                            MetalContext::barrier(enc, &[&self.q_buf, &self.h3_buf]);
                        } else {
                            let jobs = [
                                lns_metal::GemvBatchJob { weights: qw, y: &self.q_buf, out_dim: attn_dim, in_dim: dim, qt: QuantType::Q4L },
                                lns_metal::GemvBatchJob { weights: kw, y: &self.k_buf, out_dim: kv_dim, in_dim: dim, qt: QuantType::Q4L },
                                lns_metal::GemvBatchJob { weights: vw, y: &self.v_buf, out_dim: kv_dim, in_dim: dim, qt: QuantType::Q4L },
                            ];
                            self.metal_ctx.gemv_batch_multirow_enc(enc, &self.x_norm_buf, &jobs)?;
                            MetalContext::barrier(enc, &[&self.q_buf, &self.k_buf, &self.v_buf]);
                        }
                        if let Some(name) = q_norm_name {
                            let w = self.metal_f32_weights.get(name).ok_or_else(|| format!("Missing q_norm: {name}"))?;
                            self.metal_ctx.rmsnorm_rows_enc(enc, &self.q_buf, w, head_dim, self.config.n_heads, self.config.eps, self.config.zero_centered_rmsnorm)?;
                        }
                        if let Some(name) = k_norm_name {
                            let w = self.metal_f32_weights.get(name).ok_or_else(|| format!("Missing k_norm: {name}"))?;
                            self.metal_ctx.rmsnorm_rows_enc(enc, &self.k_buf, w, head_dim, self.config.n_kv_heads, self.config.eps, self.config.zero_centered_rmsnorm)?;
                        }
                        MetalContext::barrier(enc, &[&self.q_buf, &self.k_buf]);
                        self.metal_ctx.rope_with_freqs_enc(enc, &self.q_buf, &self.k_buf, &self.rope_inv_freqs_buf, pos, head_dim, rope_dim)?;
                        MetalContext::barrier(enc, &[&self.q_buf, &self.k_buf]);
                        {
                            let (k_data, v_data) = &self.paged_kv_data[l_idx];
                            let (ks_data, vs_data) = &self.paged_kv_scales[l_idx];
                            let bt_buf = &self.kv_block_table_bufs[l_idx];
                            self.metal_ctx.attention_paged_q8a_enc(enc, &self.q_buf, &self.k_buf, &self.v_buf, k_data, v_data, ks_data, vs_data, bt_buf, &self.attn_out_buf, pos, head_dim, self.config.n_heads, self.config.n_kv_heads, kv_p_start)?;
                        }
                        MetalContext::barrier(enc, &[&self.attn_out_buf]);
                        if gated {
                            self.metal_ctx.mul_inplace_enc(enc, &self.attn_out_buf, &self.h3_buf, attn_dim)?;
                            MetalContext::barrier(enc, &[&self.attn_out_buf]);
                        }
                        let ow = self.metal_weights.get(o_proj).ok_or_else(|| format!("Missing {o_proj}"))?;
                        self.metal_ctx.gemv_accumulate_multirow_enc(enc, ow, &self.attn_out_buf, &self.x_buf, dim, attn_dim)?;
                        MetalContext::barrier(enc, &[&self.x_buf]);
                    }
                    TextLayerKind::LinearAttention => {
                        let linear = self.config.linear_attention.as_ref().ok_or("Linear attention config missing")?;
                        let conv_dim = linear.conv_dim();
                        let value_dim = linear.value_dim();
                        let (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj, conv1d_weight, conv1d_bias, a_log, dt_bias, norm) =
                            match &layer.attention {
                                HybridAttentionWeights::LinearAttention { in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj, conv1d_weight, conv1d_bias, a_log, dt_bias, norm } =>
                                    (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj, conv1d_weight, conv1d_bias.as_ref(), a_log, dt_bias, norm),
                                _ => return Err(format!("Layer {l_idx}: expected linear-attention")),
                            };
                        let qkv_w = self.metal_weights.get(in_proj_qkv).ok_or_else(|| format!("Missing {in_proj_qkv}"))?;
                        let z_w = self.metal_weights.get(in_proj_z).ok_or_else(|| format!("Missing {in_proj_z}"))?;
                        let b_w = self.metal_weights.get(in_proj_b).ok_or_else(|| format!("Missing {in_proj_b}"))?;
                        let a_w = self.metal_weights.get(in_proj_a).ok_or_else(|| format!("Missing {in_proj_a}"))?;
                        let jobs = [
                            lns_metal::GemvBatchJob { weights: qkv_w, y: &self.h1_buf, out_dim: conv_dim, in_dim: dim, qt: QuantType::Q4L },
                            lns_metal::GemvBatchJob { weights: z_w, y: &self.h3_buf, out_dim: value_dim, in_dim: dim, qt: QuantType::Q4L },
                            lns_metal::GemvBatchJob { weights: b_w, y: &self.q_buf, out_dim: linear.num_value_heads, in_dim: dim, qt: QuantType::Q4L },
                            lns_metal::GemvBatchJob { weights: a_w, y: &self.k_buf, out_dim: linear.num_value_heads, in_dim: dim, qt: QuantType::Q4L },
                        ];
                        self.metal_ctx.gemv_batch_multirow_enc(enc, &self.x_norm_buf, &jobs)?;
                        MetalContext::barrier(enc, &[&self.h1_buf, &self.h3_buf, &self.q_buf, &self.k_buf]);
                        let conv_w_buf = self.metal_f32_weights.get(conv1d_weight).ok_or_else(|| format!("Missing {conv1d_weight}"))?;
                        let a_log_buf = self.metal_f32_weights.get(a_log).ok_or_else(|| format!("Missing {a_log}"))?;
                        let dt_bias_buf = self.metal_f32_weights.get(dt_bias).ok_or_else(|| format!("Missing {dt_bias}"))?;
                        let norm_buf = self.metal_f32_weights.get(norm).ok_or_else(|| format!("Missing {norm}"))?;
                        let bias_buf = conv1d_bias.and_then(|n| self.metal_f32_weights.get(n)).unwrap_or(&self.zero_buf);
                        let conv_state_buf = self.linear_conv_state_buffers[l_idx].as_ref().ok_or_else(|| format!("Missing conv state {l_idx}"))?;
                        let recurrent_state_buf = self.linear_recurrent_state_buffers[l_idx].as_ref().ok_or_else(|| format!("Missing recurrent state {l_idx}"))?;
                        self.metal_ctx.linear_attn_block_enc(enc, &self.h1_buf, conv_state_buf, conv_w_buf, bias_buf, &self.linear_conv_buf, &self.h3_buf, &self.q_buf, &self.k_buf, a_log_buf, dt_bias_buf, norm_buf, recurrent_state_buf, &self.attn_out_buf, linear.conv_kernel_size, conv_dim, linear.num_key_heads, linear.num_value_heads, linear.key_head_dim, linear.value_head_dim, self.config.eps)?;
                        MetalContext::barrier(enc, &[&self.attn_out_buf]);
                        let out_w = self.metal_weights.get(out_proj).ok_or_else(|| format!("Missing {out_proj}"))?;
                        self.metal_ctx.gemv_accumulate_multirow_enc(enc, out_w, &self.attn_out_buf, &self.x_buf, dim, value_dim)?;
                        MetalContext::barrier(enc, &[&self.x_buf]);
                    }
                    TextLayerKind::MlpOnly => {}
                    TextLayerKind::Unknown(kind) => { enc.end_encoding(); return Err(format!("Layer {l_idx}: unsupported kind '{kind}'")); }
                }

                // FFN
                let ffn_norm_w = self.metal_f32_weights.get(&layer.ffn_norm).ok_or_else(|| format!("Missing ffn norm: {}", layer.ffn_norm))?;
                self.metal_ctx.rmsnorm_enc(enc, &self.x_buf, ffn_norm_w, &self.x_norm_buf, dim, self.config.eps, self.config.zero_centered_rmsnorm)?;
                MetalContext::barrier(enc, &[&self.x_norm_buf]);
                let w1 = self.metal_weights.get(&layer.feed_forward_w1).ok_or_else(|| format!("Missing {}", layer.feed_forward_w1))?;
                let w3 = self.metal_weights.get(&layer.feed_forward_w3).ok_or_else(|| format!("Missing {}", layer.feed_forward_w3))?;
                let jobs = [
                    lns_metal::GemvBatchJob { weights: w1, y: &self.h1_buf, out_dim: hidden_dim, in_dim: dim, qt: QuantType::Q4L },
                    lns_metal::GemvBatchJob { weights: w3, y: &self.h3_buf, out_dim: hidden_dim, in_dim: dim, qt: QuantType::Q4L },
                ];
                self.metal_ctx.gemv_batch_multirow_enc(enc, &self.x_norm_buf, &jobs)?;
                MetalContext::barrier(enc, &[&self.h1_buf, &self.h3_buf]);
                self.metal_ctx.swiglu_enc(enc, &self.h1_buf, &self.h3_buf, hidden_dim)?;
                MetalContext::barrier(enc, &[&self.h1_buf]);
                let w2 = self.metal_weights.get(&layer.feed_forward_w2).ok_or_else(|| format!("Missing {}", layer.feed_forward_w2))?;
                self.metal_ctx.gemv_accumulate_multirow_enc(enc, w2, &self.h1_buf, &self.x_buf, dim, hidden_dim)?;
                MetalContext::barrier(enc, &[&self.x_buf]);
            }

            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let attn_ffn_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // ── Logits stage ────────────────────────────────────────────────────
        let t1 = Instant::now();
        if need_logits {
            let cb = self.metal_ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            let final_norm_w = self.metal_f32_weights.get(&self.weights.norm_final).ok_or_else(|| format!("Missing final norm: {}", self.weights.norm_final))?;
            let out_name = &self.output_weight_name;
            if let Some(gw) = self.metal_weights.get(out_name).filter(|w| w.1 == QuantType::Q4L) {
                self.metal_ctx.rmsnorm_gemv_multirow_enc(enc, &self.x_buf, final_norm_w, gw, &self.logits_buf, dim, self.config.vocab_size, self.config.eps, self.config.zero_centered_rmsnorm)?;
            } else if let Some(gw) = self.metal_weights.get(out_name).filter(|w| w.1 == QuantType::Q4HQ) {
                self.metal_ctx.q4hq_rmsnorm_gemv_multirow_enc(enc, &self.x_buf, final_norm_w, gw, &self.logits_buf, dim, self.config.vocab_size, self.config.eps, self.config.zero_centered_rmsnorm)?;
            } else {
                self.metal_ctx.rmsnorm_enc(enc, &self.x_buf, final_norm_w, &self.x_norm_buf, dim, self.config.eps, self.config.zero_centered_rmsnorm)?;
                MetalContext::barrier(enc, &[&self.x_norm_buf]);
                if let Some(gw) = self.metal_weights.get(out_name) {
                    let jobs = [lns_metal::GemvBatchJob { weights: gw, y: &self.logits_buf, out_dim: self.config.vocab_size, in_dim: dim, qt: gw.1 }];
                    self.metal_ctx.gemv_batch_multirow_enc(enc, &self.x_norm_buf, &jobs)?;
                } else if let Some(gw) = self.metal_f16_weights.get(out_name) {
                    self.metal_ctx.f16_gemv_enc(enc, gw, &self.x_norm_buf, &self.logits_buf, self.config.vocab_size, dim)?;
                }
            }
            MetalContext::barrier(enc, &[&self.logits_buf]);
            if !read_logits {
                self.metal_ctx.argmax_enc(enc, &self.logits_buf, self.config.vocab_size)?;
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let logits_ms = t1.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "[PROFILE pos={pos:4}] all_layers_attn+ffn={attn_ffn_ms:.1}ms  logits={logits_ms:.1}ms  total={:.1}ms",
            attn_ffn_ms + logits_ms
        );

        if need_logits && read_logits {
            let mut logits = vec![0.0f32; self.config.vocab_size];
            readback_from_buffer(&self.logits_buf, &mut logits, self.config.vocab_size);
            Ok(logits)
        } else {
            Ok(vec![])
        }
    }
}
