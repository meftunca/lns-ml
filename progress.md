# LNS-ML Project Progress & History Notes

This document tracks the evolution of the LNS-ML inference engine, including critical failures, technical debt resolution, and milestone achievements.

## ── 1. Phase: The Foundation (Initial Setup) ──

- **Goal**: Implementing 4-bit LNS (Logarithmic Number System) decoding on Apple Metal.
- **Progress**:
  - Basic `q4l_decode` kernels implemented.
  - `lns-core` quantization logic finalized.
- **Issue**: Initial inference was CPU-bound for Attention, causing high latency (ping-ponging data between GPU/CPU).

## ── 2. Phase: GPU Migration & The "Stability Crisis" ──

- **Goal**: Moving KV Cache and Multi-Head Attention entirely to the GPU to enable 2048+ token context.
- **What Went Wrong (The Breakdown)**:
  1.  **SwiGLU Activation Error**: We implemented `SiLU(x)` instead of `x * SiLU(x)`. This caused the model's signal to decay rapidly, leading to incoherent output.
  2.  **Attention Threading Bug**: The Metal kernel only wrote 32 out of 64 `head_dim` elements to the KV Cache. This resulted in "hallucinated" attention scores and rambling in Portuguese/Nonsense.
  3.  **Memory Leak Risks**: Constant buffer re-allocations in the chat loop threatened VRAM stability.
- **The Fixes**:
  - Corrected the SwiGLU gated linear unit math in `engine.rs`.
  - Refactored `fused_attention` kernel to properly loop over the entire head dimension using 32-thread warps.
  - Implemented `clear_kv_cache()` to reuse GPU buffers instead of re-allocating.

## ── 3. Phase: Stabilization & Intelligent Inference ──

- **Goal**: Making the model "smart" and instruction-compliant.
- **Progress**:
  - **Trie Tokenizer**: Replaced O(N) search with a fast Trie structure, enabling zero-latency encoding for 128k+ vocabularies (Llama-3).
  - **Chat Templates**: Added Llama-2, Llama-3, and ChatML support.
  - **Byte Fallback**: Resolved the `<0xHH>` hex-code issue in output, restoring UTF-8 integrity.
  - **RoPE Scaling**: Added infrastructure for massive context windows (up to 1M tokens) using linear frequency scaling.

## ── 4. Phase: The "llama.cpp" Parity Roadmap ──

- **Goal**: Achieving world-class feature parity.
- **Milestones**:
  - Created `SPEC.md` to define architectural targets.
  - Established `TODO.md` ledger for tracking PagedAttention, MoE, and SWA implementation.

---

### Status Summary

- **Correctness**: [STABLE] activations and attention verified.
- **Performance**: [HIGH] 100% GPU-resident inference pipeline.
- **Support**: [FLEXIBLE] Llama/Qwen/Mistral-ready infrastructure.

## ── 5. Phase: Qwen3.5 Runtime Hardening ──

- **Goal**: Reduce hybrid execution overhead and stop runaway memory duplication while debugging Qwen3.5 logits.
- **Progress**:
  - Removed the eager full-F32 embedding cache and switched token embedding lookup to row-wise decode.
  - Resolved tied output projection to the embedding matrix so `lm_head` is no longer materialized as an extra giant CPU copy during normal runtime.
  - Moved full-attention q/k row RMSNorm, partial RoPE, gated query split, and gate application onto Metal for the fast path.
  - Found the main Qwen3.5 correctness bug: full-attention `q_proj` outputs are laid out per head as `[q, gate]`, but the runtime had been splitting them globally as `[all_q][all_gate]`.
  - Fixed that gated-query split on both the CPU and Metal paths, which brought the first full-attention layers back in line with the Hugging Face reference instead of exploding at layer 3.
  - Switched user-facing Qwen ChatML prompting to the official `enable_thinking=false` variant so the CLI asks for a direct answer instead of printing visible chain-of-thought scaffolding.
  - Fixed tokenizer EOS detection for Qwen's `<|endoftext|>` token so greedy chat now stops cleanly after the answer.
  - Added `lns-convert --text-only` pruning for raw Qwen3.5 checkpoints, skipping `model.visual.*` and `mtp.*` tensors so text-runtime archives no longer require a hand-made stripped model copy.
  - Relaxed compatibility checks so raw Qwen3.5 configs with vision/MTP metadata are accepted for text inference as long as the text transformer mapping is complete.
- **Measured Outcome**:
  - Release `debug-prompt` smoke on `/tmp/qwen35-text-run/model.lns` runs in about 8.67s.
  - macOS `peak memory footprint` is about 4.63 GB on that smoke.
  - After chaining the Metal fast path into fewer command-buffer waits, warm release smokes now land around 4.44s to 5.94s with peak memory around 4.54 GB.
  - After folding vector RMSNorm into that same hot-path chain, warm release smokes now land around 3.61s to 5.65s with peak memory around 4.62 GB.
  - After the gated-query fix, the same `debug-prompt` run now produces meaningful top logits instead of digit/newline junk.
  - With visible thinking disabled, the top next-token logits for `Say only: merhaba` are now led by `"mer"`/`"Mer"`, and `lns-cli chat --temp 0.0` now returns a clean `merhaba`.
  - A real raw-model conversion using `cargo run --release -p lns-convert -- --input models/qwen3.5-4b --output /tmp/qwen3.5-4b-text-only.lns --quant Q4_L --quiet --text-only` completes successfully, but fresh Q4_L is no longer accepted as the Qwen3.5 quality path because it collapses top logits toward digits/newlines.
  - Fresh Q4_HQ conversion is the current Qwen3.5 quality path: `lns-cli chat --temp 0.0` returns a clean `hello`, built-in eval averages about 55 tok/s with calibration PPL 6.70, and a 512-token Wikitext-2 smoke reports PPL 10.6583.
- **Remaining Issue**:
  - Qwen3.5 Q4_HQ is now usable for basic direct-answer prompts, but Q4_L should be treated as a failed quality tier for this hybrid model unless a new calibration-aware encoder proves otherwise.

_Last Updated: 2026-05-07_
