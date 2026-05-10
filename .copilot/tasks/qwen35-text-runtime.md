## Summary

Enable text-only Qwen3.5 runtime execution in lns-ml so the real 4B model can be converted and smoke-tested.

## Context

- Existing work already covers config inspection, doctor diagnostics, shard-aware conversion, and hybrid tensor-name mapping.
- The remaining blocker for real text inference is execution support for Qwen3.5 hybrid layers.

## Existing System Analysis

- `lns-inference/src/model_config.rs` currently normalizes Llama, Mistral, and Qwen3 text configs.
- `lns-inference/src/transformer.rs` maps standard HF tensor names and a partial hybrid Qwen3.5 layout.
- `lns-inference/src/engine.rs` runs token-by-token autoregressive decoding and is the correct insertion point for a recurrent DeltaNet path.
- `lns-cli/src/main.rs` feeds prompts token-by-token, which means single-token recurrent execution is sufficient for an initial working path.

## Constraints

- Preserve existing Llama/Mistral/TinyLlama flows.
- Keep changes localized; avoid a full inference-architecture rewrite.
- Reuse existing Metal GEMV where possible.

## Assumptions Ledger

- [H] Token-by-token chat execution only needs recurrent linear attention, not chunked prefill kernels.
- [M] Text-only Qwen3.5 chat can ignore vision and MTP execution for standard autoregressive decoding.
- [M] Standard RoPE with partial rotary dimensions is sufficient for an initial text-only path.

## Options / Decision Matrix

- Localized runtime extension: chosen.
- Separate Qwen3.5-only engine: rejected due to duplication.
- External Python/Transformers fallback: rejected because it would not validate this runtime.

## Change Impact Map

- `lns-inference/src/model_config.rs`: runtime metadata.
- `lns-inference/src/transformer.rs`: exact tensor mapping for hybrid/full-attention Qwen3.5 weights.
- `lns-inference/src/engine.rs`: recurrent DeltaNet + gated full attention execution.
- `lns-convert/src/main.rs`: precision policy for small/stateful Qwen3.5 tensors.

## Risk Ledger

- R-01: Qwen3.5 RoPE details may still differ from the simplified text-only implementation.
- R-02: Large-memory runtime allocations for recurrent state may increase RAM use.
- R-03: Converter precision policy changes may affect archive size.

## Execution Plan

1. Extend normalized config with head-dimension and linear-attention metadata.
2. Expand Qwen3.5 tensor mapping to include gating, q/k norms, conv, and recurrent params.
3. Implement recurrent single-token DeltaNet execution.
4. Update converter precision policy for non-GEMV linear-attention tensors.
5. Build and run smoke validation.

## Verification Plan

- `cargo test -p lns-inference`
- `cargo test -p lns-cli --test smoke_cli`
- `cargo build --release -p lns-cli -p lns-inference`
- If conversion/runtime completes, run a one-shot Qwen3.5 text-only chat smoke.

## Decision Log

- Initial implementation targets text-only autoregressive decoding first.
- 2026-05-07: Removed the full decoded embedding cache from the runtime and resolved tied output projection to the embedding matrix to avoid multi-GB duplicated F32 copies.
- 2026-05-07: Full-attention GPU path now keeps q/k row RMSNorm, partial RoPE, gated query split, and post-attention gate application on Metal when the relevant weights are already on GPU.
- 2026-05-07: Release smoke on `/tmp/qwen35-text-run/model.lns` now runs in about 8.67s with `peak memory footprint` around 4.63 GB; logits remain numerically wrong, so the next work item is correctness, not more broad GPU plumbing.
- 2026-05-07: Activation flow is now more buffer-first: GPU residual projections are accumulated directly into `x_buf`, avoiding the extra `q_buf -> CPU -> x_buf` roundtrip on fast paths.
- 2026-05-07: After switching residual projections to direct GPU accumulation, release `debug-prompt` smoke improved to about 7.06s real with `peak memory footprint` still around 4.63 GB.
- 2026-05-07: Metal hot paths can now chain multiple kernels inside a single command buffer for the full-attention, linear-attention, and MLP Q4L fast paths instead of forcing a CPU-side wait after every individual dispatch.
- 2026-05-07: Post-build cold smoke can still be slower (~9.70s), but repeated warm release smokes now land around 4.44s to 5.94s with `peak memory footprint` around 4.54 GB and unchanged logits.
- 2026-05-07: Vector RMSNorm is now also encoded into the same command-buffer chain for guaranteed-GPU full-attention and MLP fast paths, removing two more per-layer waits from the hottest path.
- 2026-05-07: After chaining vector RMSNorm, repeated warm release smokes now measure about 3.61s to 5.65s real with logits unchanged and peak memory around 4.62 GB.
- 2026-05-07: Numerical reference comparison against Hugging Face isolated the first real divergence to the first `full_attention` layer rather than tokenizer or linear-attention code.
- 2026-05-07: Root cause was the Qwen3.5 full-attention `q_proj` layout: Hugging Face stores each head as `[q_head, gate_head]`, but the runtime had split the GEMV output as `[all_q][all_gate]`.
- 2026-05-07: Fixed gated-query splitting on both CPU and Metal, which pulled layer RMS values close to the Hugging Face reference and changed the top next-token logits from digit/newline garbage to meaningful text tokens.
- 2026-05-07: User-facing ChatML prompting for Qwen3.5 now uses the official `enable_thinking=false` branch, and tokenizer EOS handling now recognizes `<|endoftext|>`.
- 2026-05-07: Validation: `/Users/mapletechnologies/.cargo/target/release/lns-cli chat --model /tmp/qwen35-text-run/model.lns --prompt 'Say only: merhaba' --temp 0.0 --max-new-tokens 12` now returns `merhaba` cleanly.
- 2026-05-08: Added `lns-convert --text-only`, which prunes `model.visual.*` and `mtp.*` tensors so raw Qwen3.5 checkpoints can be converted into text-runtime archives without a separate stripped-model workspace.
- 2026-05-08: Compatibility checks now treat Qwen3.5 vision/MTP branches as ignorable for text inference; they stay visible as issues/details, but no longer make a complete text archive unsupported.
- 2026-05-08: End-to-end validation: a raw conversion to `/tmp/qwen3.5-4b-text-only.lns`, paired with copied `config.json` + `tokenizer.json`, yields `lns-cli doctor` support=`supported` and `lns-cli chat --prompt 'Say only: merhaba' --temp 0.0` => `merhaba`.
