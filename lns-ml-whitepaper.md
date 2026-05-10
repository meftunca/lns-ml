# LNS-ML: Logarithmic Number System Quantization for Large Language Model Inference

**Version:** 1.0.1  
**Status:** Experimental — benchmarks and mathematical corrections in progress  
**Author:** lns-ml project  
**Hardware Reference:** Apple M2 Max 32GB Unified Memory

---

## Abstract

We present **LNS-ML**, a novel quantization framework for large language model (LLM) inference based on the Logarithmic Number System (LNS) with hierarchical block scaling. Unlike conventional linear quantization methods (GGML Q4_K_M, GPTQ, AWQ), LNS-ML exploits the empirically observed log-normal distribution of transformer weight matrices to achieve superior precision-per-bit ratios.

Current repository results show that the LNS memory layout and hierarchical scaling are promising, but older Q2_L-vs-Q8_0 and theoretical TPS claims are not accepted as validated benchmark results. All compression ratios below are now bounded by explicit bits-per-weight arithmetic, and runtime throughput is treated as a measured property rather than a bandwidth-only prediction.

---

## Table of Contents

1. [Motivation and Background](#1-motivation-and-background)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Architecture Specification](#3-architecture-specification)
4. [Quantization Types](#4-quantization-types)
5. [Encoding and Decoding Algorithms](#5-encoding-and-decoding-algorithms)
6. [Hardware Inference Pipeline](#6-hardware-inference-pipeline)
7. [Metal Kernel Implementation](#7-metal-kernel-implementation)
8. [Benchmark Results](#8-benchmark-results)
9. [Comparison with GGML](#9-comparison-with-ggml)
10. [Known Limitations](#10-known-limitations)
11. [Roadmap](#11-roadmap)
12. [Implementation Reference](#12-implementation-reference)

---

## 1. Motivation and Background

### 1.1 The Memory Bandwidth Problem

Modern LLM inference on consumer hardware is often strongly constrained by memory bandwidth, especially during autoregressive decode where weights are streamed repeatedly. This does not mean every quantized kernel is purely memory-bound. A software LNS kernel also spends integer and floating-point ALU work on code extraction, scale application, sign/zero handling, lookup-table decode, and f32 accumulation.

This observation motivates aggressive weight compression, but the throughput model must include both transferred bytes and decode arithmetic. Reducing bytes only improves tokens-per-second when the saved bandwidth is not replaced by a larger compute bottleneck.

### 1.2 Limitations of Linear Quantization

GGML, GPTQ, and AWQ all operate on linear quantization grids — they divide the representable range into equal intervals. This is statistically suboptimal for transformer weights, which empirically follow a **log-normal distribution**:

- The vast majority of weights (~80%) cluster near zero with small magnitudes
- A minority of weights (~5%) are large outliers
- The distribution is dense near zero and sparse at the extremes

A linear quantization grid wastes resolution on the sparse large-magnitude region and systematically **underflows** small-but-important near-zero weights. This is the root cause of perplexity degradation in aggressive linear quantization.

### 1.3 The LNS Opportunity

The Logarithmic Number System stores the logarithm of a value rather than the value itself. This provides **exponentially more resolution near zero**, exactly matching the shape of the transformer weight distribution. A 4-bit LNS representation can faithfully preserve weights that a 4-bit linear representation rounds to zero.

Furthermore, LNS converts multiplication to addition:

```
log(a × b) = log(a) + log(b)
```

On hardware without native LNS support, this property is not directly exploitable for matrix multiplication. The precision advantage remains applicable to weight storage, but the runtime path must be engineered around the target ISA. Current software kernels dequantize to ordinary floating-point values before FMA accumulation, so they cannot use the same linear INT4/INT8/FP8 tensor-core datapaths that conventional quantizers target.

---

## 2. Theoretical Foundation

### 2.1 Weight Distribution Analysis

Let $W$ be a weight matrix in a transformer layer. Empirically, the distribution of $|W_{ij}|$ is well-approximated by a log-normal distribution:

$$\log |W_{ij}| \sim \mathcal{N}(\mu, \sigma^2)$$

This motivates spending code points across logarithmic magnitude rather than uniformly in linear value space. It should not be read as a proof that a fixed LNS grid is globally optimal for model loss; the accepted criterion remains measured held-out NLL/PPL, logit KL, and runtime throughput.

Linear quantization, by contrast, concentrates most representable values in the high-magnitude region where probability mass is sparse, wasting bits on rarely-occurring values.

### 2.2 Dynamic Range vs. Resolution Trade-off

For a $k$-bit LNS representation with magnitude bits $M \in [1, 2^k - 1]$ and a reserved zero ($M = 0$):

**Representable values** within a block scaled by $S_{eff}$:

$$V = \{0\} \cup \{\pm S_{eff} \cdot 2^{m - \text{offset}} : m \in [1, 2^{k-1} - 1]\}$$

**Comparison with linear quantization:**

| Scheme       | Values near zero    | Values at max        | Distribution fit     |
| ------------ | ------------------- | -------------------- | -------------------- |
| Linear 4-bit | 1/8 of range        | 1/8 of range         | Uniform (poor)       |
| LNS 4-bit    | Exponentially dense | Exponentially sparse | Log-normal (optimal) |

### 2.3 Hierarchical Scale Theory

Single-scale LNS quantization has a critical weakness: a single outlier in a block forces `scale_global` to be large, causing all near-zero weights in the block to underflow to the zero bucket.

The hierarchical two-level scale system addresses this:

**Level 1 — Super-block scale** ($S_g$, f16):
Absorbs the maximum absolute value across the entire super-block. Outliers are contained at this level.

**Level 2 — Block-local scale** ($S_l$, 4-bit log):
Fine-tunes the dynamic range of each 32-element sub-block independently.

The effective scale for any weight is:

$$S_{eff} = S_g \cdot 2^{S_l - 7}$$

With $S_l \in [0, 15]$, the local scale multiplier ranges from $2^{-7}$ to $2^{8}$, providing 15 stops of local dynamic range adjustment. A block dominated by near-zero values receives a small $S_l$, preserving resolution for its specific range regardless of outliers in neighboring blocks.

---

## 3. Architecture Specification

### 3.1 Memory Layout

The fundamental unit of on-disk LNS-ML storage is the **super-block**. In the current repository implementation this is fixed at 256 elements, arranged as 8 sub-blocks of 32 values each.

```
On-disk Q4_L Super-Block Layout (256 elements, 8 sub-blocks)
┌─────────────────────────────────────────────────┐
│ scale_global (f16)          2 bytes             │
│ scale_local[4] (4-bit × 8)  4 bytes             │
├─────────────────────────────────────────────────┤
│ sub-block 0: qs[16]         16 bytes            │
│ sub-block 1: qs[16]         16 bytes            │
│ ...                                             │
│ sub-block 7: qs[16]         16 bytes            │
└─────────────────────────────────────────────────┘
Total: 6 + 128 = 134 bytes for 256 elements (4.19 bits/weight)
```

**Super-block size:** Fixed at 256 elements in the current repository format.

**Sub-block size:** Fixed at 32 elements for SIMD alignment.

**On-disk header size:** 6 bytes, tightly packed.

For Metal execution, Q4_L blocks are repacked at load time into a 144-byte GPU layout: 16 bytes of pre-expanded per-sub-block effective scales plus the original 128-byte weight payload.

### 3.2 Super-Block Header

```c
struct LnsSuperBlockHeader {
    uint16_t scale_global_bits;  // IEEE 754 f16 encoding
    uint8_t  scale_local[4];     // 8 × 4-bit nibbles, packed
};
```

**scale_local packing:** Two 4-bit values per byte. For sub-block index `i`:

- Byte index: `i >> 1`
- Nibble: `(i & 1) ? high nibble : low nibble`

### 3.3 File Format

Models are stored as a raw [rkyv](https://github.com/rkyv/rkyv) archive for zero-copy deserialization.

```
.lns file structure:
┌──────────────────────────────────────────────────────┐
│ rkyv-serialized LnsModel archive                    │
│  ├─ version                                          │
│  └─ tensors[]                                        │
│      ├─ name                                          │
│      ├─ quant_type                                    │
│      ├─ shape                                         │
│      └─ data (contiguous quantized bytes)             │
└──────────────────────────────────────────────────────┘
```

There is no separate magic header or standalone tensor index in the current repository format; the archived `LnsModel` itself is the file.

Zero-copy memory mapping means model load time is effectively instantaneous — the OS maps pages on demand rather than reading the entire file at startup.

---

## 4. Quantization Types

### 4.1 Type Definitions

| Type | Bits/weight | Magnitude bits | Sign bit | Zero reserved | Effective bits       |
| ---- | ----------- | -------------- | -------- | ------------- | -------------------- |
| Q2_L | 2           | 1              | 1        | M=0           | 1 magnitude level    |
| Q3_L | 3           | 2              | 1        | M=0           | 3 magnitude levels   |
| Q4_L | 4           | 3              | 1        | M=0           | 7 magnitude levels   |
| Q6_L | 6           | 5              | 1        | M=0           | 31 magnitude levels  |
| Q8_L | 8           | 7              | 1        | M=0           | 127 magnitude levels |

### 4.2 Representable Values per Type

For a block with effective scale $S_{eff}$:

**Q4_L** (implemented legacy grid, `MAG_OFFSET = 5`):
$$\{0, \pm S_{eff} \cdot 2^{-4}, \pm S_{eff} \cdot 2^{-3}, \pm S_{eff} \cdot 2^{-2}, \pm S_{eff} \cdot 2^{-1}, \pm S_{eff}, \pm 2S_{eff}, \pm 4S_{eff}\}$$

**Q2_L**:
$$\{0, +S_{eff}, -S_{eff}\}$$

The Q2_L dynamic range is intentionally narrow — only three representable values. The hierarchical scale system compensates by allowing each sub-block to independently calibrate its scale, effectively providing per-block ternary quantization with fine-grained amplitude control.

### 4.3 Memory Efficiency

Compression ratios must be computed from the full quantized payload size:

$$
r_{FP16} = \frac{16}{\text{bits-per-weight}_{quant}}
$$

For the current legacy disk formats, the super-block header is included in the
bits-per-weight value:

| Type | Bytes / 256 weights | Bits / weight | Max vs FP16 | Max vs FP32 |
| ---- | ------------------- | ------------- | ----------- | ----------- |
| FP32 | 1024                | 32.0000       | 0.50x       | 1.00x       |
| FP16 | 512                 | 16.0000       | 1.00x       | 2.00x       |
| Q8_L | 262                 | 8.1875        | 1.95x       | 3.91x       |
| Q4_L | 134                 | 4.1875        | 3.82x       | 7.64x       |
| Q2_L | 70                  | 2.1875        | 7.31x       | 14.63x      |

These are mathematical upper bounds for tensors stored entirely in that format.
Keeping embeddings, norm weights, routers, output heads, or metadata in FP16/F32
increases the denominator and therefore lowers the whole-archive compression
ratio. A reported archive-level ratio above the all-quantized mathematical bound
is a measurement or accounting error.

---

## 5. Encoding and Decoding Algorithms

### 5.1 Encoding (Quantization)

Given a tensor block of 256 FP32/FP16 values:

**Step 1 — Super-block scale extraction:**
$$S_g = \max_i |x_i|$$
Store as f16 in `scale_global_bits`.

**Step 2 — Sub-block local scale:**
For each 32-element sub-block $j$:
$$S_{l,j} = \text{round}\left(\log_2\left(\frac{\max_i |x_{ji}|}{S_g}\right) + 7\right)$$
Clamped to $[0, 15]$. Store as 4-bit nibble.

**Step 3 — Per-weight encoding:**
For each weight $x_i$ in sub-block $j$:

$$S_{eff} = S_g \cdot 2^{S_{l,j} - 7}$$

$$x_i' = x_i / S_{eff}$$

For decode parameters `mult` and `offset`, the continuous magnitude index is
the inverse of the decode exponent:

$$
L_{normalized} = \frac{\log_2(|x_i'|) + \text{offset}}{\text{mult}}
$$

The simple index-space encoder is therefore:

$$
M_i =
\begin{cases}
0, & x_i = 0 \text{ or } L_{normalized} < 0.5 \\
    \operatorname{clamp}(\text{round}(L_{normalized}), 1, 2^{\text{mag\_bits}} - 1), & \text{otherwise}
\end{cases}
$$

The division by `mult` is mandatory. Without it, Q8_L (`mult = 0.125`) would
incorrectly bury the first several non-zero quantization indices in the zero
bucket. Production encoders may improve on this index-space rule by checking
neighbouring decoded candidates under a linear-MSE or activation-weighted loss,
but the log-domain seed must still use the normalized inverse above.

**Step 4 — Packing:**
$$\text{nibble} = (S \ll \text{mag\_bits}) | M$$

where $S = (x_i < 0)$ ? 1 : 0.

Pack two nibbles per byte for Q4_L, four per byte for Q2_L.

### 5.2 Decoding (Dequantization)

Given a packed weight value `val`, effective scale $S_{eff}$, magnitude bits `mag_bits`, multiplier `mult`, and offset `offset`:

```
M        = val & ((1 << mag_bits) - 1)
sign_bit = (val >> mag_bits) & 1
magnitude = S_eff × exp2(M × mult - offset)
sign      = 1.0 - 2.0 × (sign_bit != 0)
result    = if M == 0 { 0.0 } else { magnitude × sign }
```

On GPU hot paths, this should not be implemented as a divergent branch. The
preferred representation is a decode lookup table whose zero index already
contains `0.0`, or an equivalent predicated/select operation selected by the
backend. The point is to avoid both control-flow divergence and unnecessary
per-weight mask work where a table entry can encode zero directly.

### 5.3 Type-specific Parameters

| Type | mag_bits | mult            | offset          |
| ---- | -------- | --------------- | --------------- |
| Q2_L | 1        | special ternary | special ternary |
| Q4_L | 3        | 1.0             | 5.0             |
| Q8_L | 7        | 0.125           | 15.0            |

Q2_L decodes `M=1` directly to `S_eff` and `M=0` to exact zero. Its current
encoder uses the linear-MSE midpoint `|x| < 0.5 * S_eff` for the zero decision,
not the generic log-index threshold.

---

## 6. Hardware Inference Pipeline

### 6.1 Overview

```
DRAM/Unified Memory
       │
    │  LNS-compressed weights
       ▼
┌─────────────────┐
│  Memory Load    │  ← uint4 vectorized reads (16-byte chunks)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Dequantization │  ← table/direct-codebook decode
│  (on-the-fly)   │     zero encoded as a decode-table value when possible
└────────┬────────┘
         │  f32 weights
         ▼
┌─────────────────┐
│  Matrix Multiply│  ← GEMV fused kernel
│  (GEMV/GEMM)    │     simd_sum reduction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Activations    │  ← RMSNorm, RoPE, SiLU/GELU
│  (f32)          │     kept in floating point
└─────────────────┘
```

**Key design decision:** Activations remain in FP32 throughout inference. Only weights are LNS-compressed. This avoids training-inference mismatch — the model was trained with floating-point activations and expects them.

### 6.2 Memory Bandwidth Analysis

For a 4B parameter model on M2 Max (400 GB/s bandwidth), the following table is
only a DRAM-traffic lower bound. It assumes every parameter is stored in the
listed format, no FP16 tensors are retained, and decode arithmetic is free.
Those assumptions are not true for the actual engine, so this table must not be
used as a TPS prediction.

| Format | Minimum weight bytes | DRAM-only upper bound |
| ------ | -------------------- | --------------------- |
| FP16   | ~8.00 GB             | ~50 tok/s             |
| Q4_K_M | ~2.50 GB             | ~160 tok/s            |
| Q4_L   | ~2.09 GB             | ~191 tok/s            |
| Q2_L   | ~1.09 GB             | ~366 tok/s            |

Measured throughput can be lower when the LNS decode path is compute-bound, when
command-buffer overhead dominates, or when FP16-retained tensors increase the
effective bytes transferred per token.

### 6.3 Dequantization Cost

The legacy mathematical form uses `exp2`, but the hot Metal Q4_L path does not
evaluate `exp2` per weight. It uses a small constant decode table for the eight
Q4_L magnitude levels, and Q4_HQ uses a fixed NF4-Z table. Even so, software LNS
decode is not free. Each packed weight still needs code extraction, table lookup
or equivalent constant selection, scale multiply, sign handling, and f32 FMA
accumulation.

This is the central hardware trade-off: LNS may reduce memory traffic and
preserve weight-distribution structure, but it gives up the most direct path to
linear INT4/INT8/FP8 matrix multiply accelerators. Any high-TPS claim must
therefore be backed by measured kernels on the target backend, not by a
bandwidth-only model.

---

## 7. Metal Kernel Implementation

### 7.1 Decode Kernel Strategy

Each GPU thread processes 4 weights (one `float4` store). For 256-element super-blocks, 64 threads cover one super-block. Vectorized byte reads minimize memory transaction overhead.

```metal
// Thread dispatch: one thread per 4 weights
kernel void q4l_decode_kernel_v2(
    device const LnsSuperBlockHeader *headers,
    device const uchar               *weights,
    device       float4              *out,
    constant     uint                &num_weights,
    uint gid [[ thread_position_in_grid ]]
)
```

### 7.2 GEMV Kernel Strategy

Fused dequantize + matrix-vector multiply kernel eliminates intermediate buffer allocation:

- One threadgroup per output row
- 256 threads per threadgroup (8 warps of 32)
- Each thread accumulates one dot product element
- `simd_sum()` for warp-level reduction
- Threadgroup shared memory for cross-warp reduction

### 7.3 RoPE Implementation

Standard split-head RoPE as used in Llama/Qwen:

```metal
// Correct split indexing: [0..d/2) and [d/2..d)
const uint half_head = head_dim / 2u;
const uint head_idx  = gid / half_head;
const uint pair_idx  = gid % half_head;
const uint base_idx  = head_idx * head_dim;

const float inv_freq = exp2(
    -(float)(pair_idx * 2u) / (float)head_dim * 13.2877f
);
```

**GQA support:** K/V heads use modular head mapping:

```metal
const uint k_head_idx = head_idx % kv_heads;
```

### 7.4 RMSNorm

Two-pass reduction using `simd_sum()` + threadgroup shared memory. Uses `rsqrt()` (hardware-accelerated on Apple Silicon) instead of `1/sqrt()`.

---

## 8. Benchmark Results

### 8.1 Primary Benchmark

**Model:** Qwen 3.5 4B  
**Hardware:** Apple M2 Max 32GB Unified Memory  
**Backend:** Metal (Apple Silicon GPU)  
**Date:** May 2026

| Metric                  | Value                           |
| ----------------------- | ------------------------------- |
| Format                  | Q4_L text-only bundle           |
| Runtime path            | Metal                           |
| Functional validation   | Single-turn chat smoke passes   |
| Context window          | 8192                            |
| **Throughput**          | **Pending revalidation**        |
| Perplexity (Wikitext-2) | Pending revalidation            |
| Hallucination rate      | Not yet systematically measured |

### 8.2 Size Comparison

For a 4B-parameter model, the all-quantized mathematical minima are:

| Format        | Minimum weight bytes | Max ratio vs FP16 |
| ------------- | -------------------- | ----------------- |
| FP16 baseline | ~8.00 GB             | 1.00x             |
| GGML Q4_K_M   | ~2.50 GB             | ~3.20x            |
| LNS Q4_L      | ~2.09 GB             | ~3.82x            |
| LNS Q2_L      | ~1.09 GB             | ~7.31x            |
| LNS Q4_HQ     | ~2.25 GB             | ~3.56x            |

Whole-archive sizes can only be larger than these minima when tensors are kept
in FP16/F32 or when metadata and bundle files are included. Therefore any
archive-level claim such as `40x` versus FP16 is mathematically impossible for
these formats and should be treated as an accounting bug unless the denominator
and numerator refer to different parameter sets.

### 8.3 Perplexity Targets (Theoretical)

| Type              | PPL increase vs FP16 | Fidelity  | Status              |
| ----------------- | -------------------- | --------- | ------------------- |
| Q8_L              | < 0.005              | 99.9%     | Planned             |
| Q8_0 (GGML ref)   | ~0.01                | 99.5%     | Reference           |
| Q6_L              | 0.01–0.02            | 99.7%     | Planned             |
| Q6_K (GGML ref)   | ~0.05                | 98.0%     | Reference           |
| **Q4_L**          | **0.03–0.08**        | **98.5%** | **In progress**     |
| Q4_K_M (GGML ref) | ~0.20–0.35           | 95.0%     | Reference           |
| Q3_L              | 0.15–0.30            | 95.0%     | Planned             |
| Q3_K_M (GGML ref) | ~0.60–1.20           | 80.0%     | Reference           |
| Q2_L              | TBD                  | TBD       | Requires validation |
| Q2_K (GGML ref)   | > 5.00               | 40.0%     | Reference           |

Formal Wikitext-2 validation for Q2_L is still pending in the current repository.

> ⚠️ These PPL values are planning targets and reference points, not repository-validated benchmark results.

---

## 9. Comparison with GGML

### 9.1 Architectural Differences

| Aspect                  | GGML                       | LNS-ML                                      |
| ----------------------- | -------------------------- | ------------------------------------------- |
| Quantization grid       | Linear (uniform intervals) | Logarithmic (exponential density near zero) |
| Scale hierarchy         | Single scale per block     | Two-level: super-block + sub-block          |
| Zero representation     | Nearest grid point         | Exact zero (reserved M=0)                   |
| Weight distribution fit | Uniform (suboptimal)       | Log-normal (optimal)                        |
| File format             | GGUF (parsed)              | rkyv (zero-copy)                            |
| Primary target          | Cross-platform CPU         | Metal-first, CUDA/ROCm planned              |

### 9.2 Why Q2-Class LNS Can Outperform Linear Low-Bit Formats

The core insight is theoretical: **a logarithmic grid can allocate its limited code points more efficiently than a linear grid when the underlying weight distribution is approximately log-normal.**

Concretely:

- GGML-style low-bit linear formats allocate uniformly spaced values in linear space, which can be poorly matched to heavy-tailed weight magnitudes.
- LNS-style Q2-class formats allocate their few effective values in a magnitude-aware way and rely on local scales to move the grid to each sub-block's range.

This is a motivation, not a validation result. The current legacy Q2_L format has only three effective values per sub-block, so it must be evaluated separately from the newer Q2_HQ/Q2HQM plan.

The hierarchical scale system further mitigates outlier sensitivity that would otherwise undermine this argument.

---

## 10. Known Limitations

### 10.1 Cache-Line Spanning

134-byte on-disk Q4_L super-blocks span 3 cache lines (64 bytes each) if read directly. The current Metal backend mitigates this by repacking each block into a 144-byte execution layout with a 16-byte scale header and 128-byte weight payload.

**Current status:** disk layout is compact; GPU layout is already partially normalized. A future SoA/AoSoA execution layout may reduce bandwidth waste further.

### 10.2 Fixed Super-block Size in Current Repo

The current repository fixes the super-block size at 256 elements. Supporting multiple super-block sizes would require converter, decoder, and GPU dispatch changes.

### 10.3 Post-Training Quantization Only

LNS-ML is currently applied as post-training quantization (PTQ). Models were trained assuming linear weight distributions. LNS-aware training (where the model's weight distribution is explicitly shaped toward log-normal during training) would likely yield further perplexity improvements but requires significant compute investment.

### 10.4 Quantization Conversion Cost

Converting FP16 tensors to LNS format requires computing `log2` for each parameter — approximately 5–10× slower than linear quantization conversion. This is a one-time offline cost and does not affect inference.

### 10.5 Q2_L Narrow Dynamic Range

Q2*L provides only three representable values per weight: $\{-S*{eff}, 0, +S\_{eff}\}$. Expressiveness is entirely dependent on the quality of the hierarchical scale system. Models with highly irregular weight distributions per sub-block may show higher perplexity degradation.

---

## 11. Roadmap

### Phase 1 — Foundation (Current)

- [x] Q4_L encoding/decoding
- [x] Q2_L encoding/decoding
- [x] rkyv zero-copy model format
- [x] safetensors converter
- [x] Apple Silicon Metal inference kernels
- [x] First token generation (Qwen 3.5 4B)
- [ ] Formal throughput revalidation after the Q4_HQ/Q2_HQ architecture changes
- [ ] Formal Wikitext-2 perplexity evaluation
- [ ] Q6_L, Q8_L implementation
- [ ] Stable multi-turn inference

### Phase 2 — Architecture Support

- [ ] Llama 3.x architecture
- [ ] Qwen 2.5 / 3.x architecture
- [ ] Mistral / Mixtral (sliding window attention)
- [ ] DeepSeek (MLA — multi-head latent attention)
- [ ] KV cache quantization

### Phase 3 — Acceleration

- [ ] CUDA PTX kernels
- [ ] ROCm HIP kernels
- [ ] Continuous batching
- [ ] Speculative decoding

### Phase 4 — Universal

- [ ] SPIR-V backend (Vulkan / OpenCL / WGSL)
- [ ] WebGPU inference
- [ ] OpenAI-compatible API server

### Phase 5 — Training

- [ ] LNS-aware LoRA fine-tuning
- [ ] LNS-native training (research)

---

## 12. Implementation Reference

### 12.1 Core Dequantization (Rust)

```rust
#[inline(always)]
pub fn lns_decode_q4l(val: u8, efektif_scale: f32) -> f32 {
    const GRID: [f32; 8] = [0.0, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0];
    let m = val & 0x07;
    let s = (val >> 3) & 0x01;
    let sign = 1.0 - 2.0 * s as f32;
    efektif_scale * GRID[m as usize] * sign
}

#[inline(always)]
pub fn effective_scale(scale_global: f16, scale_local: u8) -> f32 {
    f32::from(scale_global) * (scale_local as f32 - 7.0).exp2()
}
```

### 12.2 Encoding (Rust)

```rust
pub fn encode_block_q4l(
    weights: &[f32; 32],
    scale_global: f32,
    scale_local: u8,
) -> [u8; 16] {
    let s_eff = scale_global * (scale_local as f32 - 7.0).exp2();
    let mut out = [0u8; 16];

    for (i, &w) in weights.iter().enumerate() {
        let normalized = w / s_eff;
        let m = if normalized.abs() < 1e-10 {
            0u8
        } else {
            let mult = 1.0_f32;
            let offset = 5.0_f32;
            let l_normalized = (normalized.abs().log2() + offset) / mult;
            if l_normalized < 0.5 {
                0u8
            } else {
                (l_normalized.round() as u8).clamp(1, 7)
            }
        };
        let s = (w < 0.0) as u8;
        let nibble = (s << 3) | m;

        if i % 2 == 0 {
            out[i / 2] = nibble;
        } else {
            out[i / 2] |= nibble << 4;
        }
    }
    out
}
```

### 12.3 Metal Dispatch (Rust/objc2)

```rust
pub fn dispatch_q4l_gemv(
    encoder: &MTLComputeCommandEncoder,
    pipeline: &MTLComputePipelineState,
    headers: &MTLBuffer,
    weights: &MTLBuffer,
    x: &MTLBuffer,
    y: &MTLBuffer,
    nrows: u32,
    ncols: u32,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(headers), 0);
    encoder.set_buffer(1, Some(weights), 0);
    encoder.set_buffer(2, Some(x), 0);
    encoder.set_buffer(3, Some(y), 0);
    encoder.set_bytes(4, &ncols.to_le_bytes());

    // CRITICAL: threadgroup size must be 256 (8 warps × 32)
    encoder.dispatch_threadgroups(
        MTLSize { width: nrows as u64, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}
```

---

## References

1. Gustafson, J.L. & Yonemoto, I.T. (2017). _Beating Floating Point at its Own Game: Posit Arithmetic._ Supercomputing Frontiers and Innovations.
2. Dettmers, T. et al. (2022). _LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale._ NeurIPS 2022.
3. Frantar, E. et al. (2022). _GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers._ ICLR 2023.
4. Lin, J. et al. (2023). _AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration._
5. Ashkboos, S. et al. (2024). _QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs._
6. Kim, S. et al. (2023). _SqueezeLLM: Dense-and-Sparse Quantization._
7. Su, J. et al. (2021). _RoFormer: Enhanced Transformer with Rotary Position Embedding._

---

_This document reflects the state of lns-ml as of May 2026. Benchmark figures are preliminary and subject to revision upon formal evaluation._
