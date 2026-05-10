# LNS-ML Technical Specification v1.0

For the mathematical quantization note comparing the current LNS implementation against GGML-style linear quantization, see [LNS-vs-GGML-Mathematical-Analysis.md](/Users/mapletechnologies/Desktop/big_projects/lns-ml/LNS-vs-GGML-Mathematical-Analysis.md).

This specification defines the target feature parity between LNS-ML and `llama.cpp` (as of Q1 2025). The goal is to provide a high-performance, LNS-quantized alternative with full architectural support for modern LLMs.

## 1. Model Loading & Architecture

| Parameter           | Description                               | LNS-ML Status       |
| ------------------- | ----------------------------------------- | ------------------- |
| `n_ctx`             | Context window size (Target: up to 1M)    | [WIP] 8192 active   |
| `n_gpu_layers`      | Number of layers to offload to Metal      | [DONE] Full Offload |
| `n_batch`           | Batch size for prompt processing          | [TODO]              |
| `n_ubatch`          | Physical batch size                       | [TODO]              |
| `rope_freq_base`    | RoPE base frequency (e.g., 10k, 500k, 1M) | [DONE]              |
| `rope_freq_scale`   | RoPE linear scaling factor                | [DONE]              |
| `rope_scaling_type` | Support for YaRN, Linear, Dynamic         | [TODO]              |
| `gqa`               | Grouped Query Attention (Llama-3, Qwen)   | [DONE]              |
| `moe`               | Mixture of Experts (Mixtral, DeepSeek)    | [TODO]              |
| `swa`               | Sliding Window Attention (Mistral)        | [TODO]              |

## 2. Sampling Parameters

| Parameter           | Description                         | Default | Status |
| ------------------- | ----------------------------------- | ------- | ------ |
| `temp`              | Softmax temperature                 | 0.8     | [DONE] |
| `top_k`             | Top-K sampling                      | 40      | [DONE] |
| `top_p`             | Nucleus sampling                    | 0.9     | [DONE] |
| `min_p`             | Minimum probability threshold       | 0.05    | [DONE] |
| `repeat_penalty`    | Penalty for repeated tokens         | 1.1     | [DONE] |
| `repeat_last_n`     | Window for repetition penalty       | 64      | [DONE] |
| `presence_penalty`  | Presence penalty (alpha_presence)   | 0.0     | [DONE] |
| `frequency_penalty` | Frequency penalty (alpha_frequency) | 0.0     | [DONE] |

## 3. KV Cache Management

| Feature           | Description                                                     | Status |
| ----------------- | --------------------------------------------------------------- | ------ |
| `paged_attention` | Non-contiguous KV cache pages (16/32 tokens)                    | [TODO] |
| `kv_quant`        | KV cache quant — Q8L too coarse (8.3% step); Q8A linear planned | [WIP]  |
| `cache_type_k`    | Key cache: F16 (working); Q8A linear activation quant planned   | [WIP]  |
| `cache_type_v`    | Value cache: F16 (working); Q8A linear activation quant planned | [WIP]  |

## 4. Hardware Acceleration (Metal/macOS)

| Feature          | Description                              | Status |
| ---------------- | ---------------------------------------- | ------ |
| `flash_attn`     | Flash-Decoding kernel (online softmax)   | [DONE] |
| `fused_kernels`  | RMSNorm + QKV fusion                     | [DONE] |
| `async_dispatch` | Overlapping compute and memory transfers | [WIP]  |

## 5. Tokenization

| Feature         | Description                        | Status |
| --------------- | ---------------------------------- | ------ |
| `bpe`           | Byte-Pair Encoding                 | [DONE] |
| `tiktoken`      | Tiktoken-based (Llama-3)           | [DONE] |
| `byte_fallback` | Handling unknown chars via bytes   | [DONE] |
| `added_tokens`  | Handling special structural tokens | [DONE] |
