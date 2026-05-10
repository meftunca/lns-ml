# LNS Math Spec Refresh

## Summary

Refresh the mathematical specification that compares LNS quantization against GGML-style linear quantization.

## Context

The original project framing already contained a strong mathematical argument for LNS, but the current implementation has evolved into a more precise two-level log quantizer (`scale_global` + per-sub-block `scale_local`). The written spec should match the code.

## Existing System Analysis

- `lns-core/src/quant/q4l.rs` implements Q4_L as a hierarchical log quantizer.
- A super-block stores one `scale_global`, eight packed 4-bit `scale_local` values, and 256 packed 4-bit signed magnitudes.
- Dequantization is branchless in the inner loop.
- The previous prose described a single-scale logarithmic mapping, which is directionally right but incomplete.

## Constraints

- Keep the math note aligned with the current Rust implementation.
- Do not overclaim hardware advantages that are not yet realized in kernels.
- Preserve the original motivation, but explicitly mark what changed.

## Assumptions Ledger

- [H] The user wants a durable repo document, not just a chat answer.
- [H] A dedicated markdown note is preferable to bloating the top-level `Spec.md`.
- [M] A short pointer from `Spec.md` is useful because the user referenced the spec directly.

## Options / Decision Matrix

- Update only `Spec.md`: rejected, too cramped for a full mathematical appendix.
- Create a dedicated math note: chosen, clearer and easier to maintain.
- Replace the original framing entirely: rejected, because the delta versus the old statement is important.

## Change Impact Map

- New math note in repo root for discoverability.
- Short pointer added to `Spec.md`.
- No runtime behavior changes.

## Risk Ledger

- Risk: overstating GGML internals. Mitigation: describe canonical blockwise linear quantization, not one exact GGML variant only.
- Risk: drift from implementation. Mitigation: anchor formulas to `q4l.rs` terminology and storage layout.

## Execution Plan

1. Read current Q4_L implementation.
2. Compare implementation with the old prose.
3. Write a new mathematical note with a clear delta section.
4. Add a short link from `Spec.md`.

## Verification Plan

- Confirm formulas match `q4l.rs` encode/decode behavior.
- Confirm note explicitly calls out what changed from the old spec text.
- Confirm `Spec.md` points to the new document.

## Decision Log

- 2026-05-07: Use a dedicated repo document for the mathematical comparison and keep `Spec.md` as the high-level feature spec.
- 2026-05-08: Begin v4 HQ integration with the Phase 0 format foundation: add HQ quant type ids, a self-describing payload header parser/writer, strict offset validation, and focused tests before implementing Q2_HQ/Q4_HQ encoders or GPU kernels.

## Integration Slice 2026-05-08

### Summary

Start implementation from the architecture-freeze foundation instead of jumping directly into kernels.

### Constraints

- This project is not actively deployed, so backward compatibility is not a primary constraint.
- Prefer clean v4/HQ archive architecture over compatibility-preserving shims when they conflict.
- Legacy `Q2_L`, `Q4_L`, and `Q8_L` may remain readable while convenient, but new writes can move to v4.
- Introduce HQ quant types as new ids while avoiding long-lived legacy bottlenecks.
- Reject malformed HQ payloads before any runtime kernel sees them.
- Keep crate root thin and place HQ payload logic in a quant submodule.

### Execution Plan

1. Add `Q4HQ`, `Q8HQ`, `Q4HQM`, `Q2HQ`, and `Q2HQM` enum ids.
2. Add an `hq` module with constants, header struct, serializer, parser, and bounds validation.
3. Add tests for roundtrip, reserved bits, invalid block sizes, overlapping ranges, and missing required metadata.
4. Update CLI labels so inspection commands display HQ quant ids correctly.
5. Run focused core tests and a workspace check where feasible.

### Verification Plan

- `cargo test -p lns-core hq`
- `cargo test -p lns-core format`
- `cargo check -p lns-cli`
- `cargo check -p lns-inference`
- `cargo check -p lns-convert`
- `cargo test -p lns-core`

### Verification Result

- 2026-05-08: `cargo test -p lns-core hq -- --nocapture` passed: 9 tests.
- 2026-05-08: `cargo test -p lns-core format -- --nocapture` passed: 6 tests.
- 2026-05-08: `cargo check -p lns-cli` passed.
- 2026-05-08: `cargo check -p lns-inference` passed.
- 2026-05-08: `cargo check -p lns-convert` passed.
- 2026-05-08: `cargo test -p lns-core` passed: 21 unit tests, 3 integration tests, and ignored doc-test unchanged.
- 2026-05-08: Focused `git diff --check` passed for the Phase 0 integration files.
- 2026-05-08: User explicitly removed backward-compatibility as a primary constraint; new archive writes can move to v4 by default.
- 2026-05-08: Added CPU `Q4HQSuperBlock` encode/decode, NF4-Z codebook constants, Q4HQ payload wrapping, and Q4HQ dense decode through `LnsTensor::to_f32`.
- 2026-05-08: `cargo test -p lns-core q4hq -- --nocapture` passed: 5 tests.
- 2026-05-08: `cargo test -p lns-core hq -- --nocapture` passed: 13 tests.
- 2026-05-08: `cargo test -p lns-core` passed: 25 unit tests, 3 integration tests, and ignored doc-test unchanged.
- 2026-05-08: `cargo check -p lns-cli`, `cargo check -p lns-inference`, and `cargo check -p lns-convert` passed after the Q4HQ slice.

### Completed Follow-Up

Converter support for `--quant Q4_HQ` is covered by the slice below.

## Integration Slice 2026-05-08: Q4_HQ Converter

### Summary

Expose the CPU Q4_HQ implementation through `lns-convert` so model archives can actually carry HQ payloads.

### Constraints

- New writes use v4 by default.
- `Q4_HQ` payloads must use the canonical HQ header and 144B block data.
- Keep the first converter slice CPU/runtime-decodable before adding native Metal kernels.

### Execution Plan

1. Add `Q4HQ` to `QuantFormat` parsing/display.
2. Emit `QuantType::Q4HQ` tensors with `q4hq_blocks_to_payload`.
3. Count `Q4_HQ` tensors in converter summaries.
4. Add an integration test that runs the converter binary and validates archive version, quant type, HQ header, and dense decode.

### Verification Result

- 2026-05-08: `cargo test -p lns-convert --test qwen3_convert converts_q4hq_payload_with_v4_archive -- --nocapture` passed.
- 2026-05-08: `cargo test -p lns-convert` passed: 7 unit tests and 2 integration tests.
- 2026-05-08: `cargo test -p lns-core` passed: 25 unit tests, 3 integration tests, ignored doc-test unchanged.
- 2026-05-08: `cargo check -p lns-cli` passed.
- 2026-05-08: `cargo check -p lns-inference` passed.

### Next Slice

Native Metal Q4_HQ GEMV is covered by the slice below.

## Integration Slice 2026-05-08: Native Q4_HQ Metal GEMV

### Summary

Route canonical Q4_HQ payloads into the Metal backend without dense F16 expansion.

### Constraints

- Reuse the existing 144B execution block shape: eight f16 direct scales plus 128B packed nibbles.
- Q4_HQ payload block data is already in execution layout, so Metal preparation must copy canonical block data directly after validating the HQ header.
- Dispatch must use the actual quant type attached to each Metal weight buffer, not a hardcoded Q4_L assumption.

### Execution Plan

1. Register Q4_HQ GEMV and accumulate Metal pipelines.
2. Add NF4-Z decode kernels for single-row and multirow GEMV.
3. Extend `prepare_tensor` to accept `QuantType::Q4HQ` and copy canonical block data from the HQ payload.
4. Store quant type on `MetalBuffer` and route batch/accumulate dispatch through the buffer's real type.
5. Load Q4_HQ tensors into `metal_weights` in `InferenceEngine::new`.
6. Add CPU/GPU Q4_HQ GEMV parity coverage.

### Verification Result

- 2026-05-08: `cargo check -p lns-metal` passed.
- 2026-05-08: `cargo check -p lns-inference` passed.
- 2026-05-08: `cargo test -p lns-metal --test consistency test_metal_gemv_q4hq_consistency -- --nocapture` passed with max observed row diff about `1.14e-5`.
- 2026-05-08: `cargo test -p lns-metal --test consistency -- --nocapture` passed: Q4_L and Q4_HQ parity ok; Q2_L/Q8_L tests remain skip-notices because those kernels are unavailable.
- 2026-05-08: `cargo test -p lns-inference` passed: 22 tests.
- 2026-05-08: `cargo test -p lns-convert` passed: 7 unit tests and 2 integration tests.
- 2026-05-08: `cargo test -p lns-core` passed: 25 unit tests, 3 integration tests, ignored doc-test unchanged.
- 2026-05-08: `cargo check -p lns-cli` passed.
- 2026-05-08: `cargo test -p lns-metal` still fails in the pre-existing `test_metal_rope_consistency` because the legacy `rope_optimized` pipeline is absent; this is unrelated to Q4_HQ GEMV and the Q4_HQ consistency path passes.

### Next Slice

The small real-model comparison is covered by the slice below.

## Integration Slice 2026-05-09: TinyLlama Q4_L vs Q4_HQ Smoke

### Summary

Compare a freshly converted TinyLlama Q4_L archive against a freshly converted TinyLlama Q4_HQ archive using the same release CLI eval path.

### Context

- Source checkpoint: `models/tinyllama1.1B/model.safetensors`.
- Temporary Q4_L bundle: `/tmp/lns-q4l-eval/model.lns`.
- Temporary Q4_HQ bundle: `/tmp/lns-q4hq-eval/model.lns`.
- Both archives use format version 4 and report runtime support as `supported`.

### Verification Result

- 2026-05-09: release Q4_L conversion completed in about `16.91s`; output archive `734M` by `du`, `769.48 MB` tensor payload by `inspect`.
- 2026-05-09: release Q4_HQ conversion completed in about `97.07s`; output archive `770M` by `du`, `807.34 MB` tensor payload by `inspect`.
- 2026-05-09: `lns-cli eval --gen-tokens 16` on Q4_L: average decode speed `171.2 tok/s`, calibration PPL `65.18` over 180 scored tokens.
- 2026-05-09: `lns-cli eval --gen-tokens 16` on Q4_HQ: average decode speed `183.9 tok/s`, calibration PPL `20.91` over 180 scored tokens.
- 2026-05-09: short Wikitext smoke (`--context 256 --stride 256 --max-tokens 512`) on Q4_L: 510 scored tokens, mean NLL `3.902459`, PPL `49.5241`, elapsed `4.74s`.
- 2026-05-09: matching short Wikitext smoke on Q4_HQ: 510 scored tokens, mean NLL `3.657245`, PPL `38.7544`, elapsed `4.20s`.

### Decision Log

- Q4_HQ is promising enough to keep as the near-term quality-tier default candidate: it improved both calibration PPL and short Wikitext PPL while also improving measured decode speed in this smoke.
- Q4_HQ conversion cost is now the visible bottleneck: TinyLlama conversion was roughly `5.7x` slower than Q4_L, so the next implementation slice should optimize the Q4_HQ encoder before large repeated model sweeps.

## Integration Slice 2026-05-09: Q4_HQ Encoder Throughput

### Summary

Reduce Q4_HQ offline conversion cost without changing the canonical payload contract or NF4-Z decode math.

### Existing System Analysis

The initial Q4_HQ encoder searched 33 log-spaced scale candidates for every 32-weight sub-block and ran two Lloyd refinement iterations on every candidate. TinyLlama conversion showed this as the first practical bottleneck: Q4_HQ took about `97.07s`, roughly `5.7x` the Q4_L conversion time.

### Change

- Keep the same 33-point coarse scale scan.
- Score all coarse candidates cheaply with the existing block-error objective.
- Retain only the best four distinct coarse candidates.
- Run Lloyd refinement only on those retained candidates plus the RMS/max seed.

### Verification Result

- 2026-05-09: `cargo test -p lns-core hq -- --nocapture` passed: 13 HQ-focused tests.
- 2026-05-09: `cargo test -p lns-convert --test qwen3_convert converts_q4hq_payload_with_v4_archive -- --nocapture` passed.
- 2026-05-09: `cargo test -p lns-metal --test consistency test_metal_gemv_q4hq_consistency -- --nocapture` passed with max observed row diff about `1.14e-5`.
- 2026-05-09: optimized release Q4_HQ TinyLlama conversion completed in about `50.91s`, down from `97.07s` before the change.
- 2026-05-09: optimized Q4_HQ archive remained `770M` by `du`.
- 2026-05-09: optimized `lns-cli eval --gen-tokens 16` on Q4_HQ: average decode speed `179.5 tok/s`, calibration PPL `20.88` over 180 scored tokens.
- 2026-05-09: optimized short Wikitext smoke (`--context 256 --stride 256 --max-tokens 512`) on Q4_HQ: 510 scored tokens, mean NLL `3.646078`, PPL `38.3241`, elapsed `4.34s`.

### Decision Log

- The optimized search keeps Q4_HQ quality in the same band as the original exhaustive-refine search while nearly halving conversion time on TinyLlama.
- Q4_HQ conversion is still about `3.0x` slower than Q4_L on this smoke, so further encoder work is useful before broad full-model sweeps.

### Next Slice

Choose between a second Q4_HQ encoder pass (SIMD/cache-level nearest-code optimisation) and starting Q2_HQ block implementation, depending on whether conversion speed or Q2 architecture coverage is more valuable next.

## Integration Slice 2026-05-09: Converter Memory Ceiling

### Problem

Fresh Qwen conversion was killed with exit `137` when the converter was run on a sharded 8.7G checkpoint. The old converter memory model was not acceptable for large models:

- read each safetensors shard fully into a `Vec<u8>`;
- copied every tensor into a `RawTensor` list before quantization;
- quantized tensors in parallel, multiplying live raw/f32/quant buffers;
- serialized the final rkyv archive into a second full `Vec<u8>` before writing.

### Change

- Switched safetensors input traversal to header + per-tensor range reads instead of whole-shard mmap/full-file reads.
- Added an input-side skip predicate so `--text-only` avoids reading skipped tensors at all.
- Converted tensors sequentially rather than through a Rayon `par_iter` over all raw tensors.
- Added `serialize_model_to_writer` in `lns-core` so rkyv output streams to the destination file instead of allocating a second full archive buffer.
- Avoided f32 expansion for tensors kept as F16: F16 raw bytes are moved directly into the output tensor data; BF16/F32 are converted directly to F16 output bytes.

### Verification Result

- `cargo test -p lns-convert` passed.
- `cargo test -p lns-core format -- --nocapture` passed.
- Qwen Q4_L text-only mixed-precision conversion now completes instead of being OOM-killed.
- Measured with `/usr/bin/time -l`:
  - before range reads: max RSS `8.15GB`, peak memory footprint `4.06GB`;
  - after range reads: max RSS `4.19GB`, peak memory footprint `5.13GB`.
- Output archive remains large: `/tmp/lns-qwen-q4l/model.lns` is `3.4G` on disk, with `3.69GB` tensor payload. The largest retained tensor is `model.language_model.embed_tokens.weight` at `1.27GB` F16.

### Decision Log

- The original 20GB-class behaviour was a converter architecture bug, not a necessary property of LNS quantization.
- A strict `<=2GB` converter RSS ceiling cannot be guaranteed by the current v4 single-rkyv archive when the output archive itself is >3GB and tensor data lives inside `Vec<u8>` fields in one `LnsModel`.
- Hitting a hard 2GB ceiling for Qwen-class archives requires a v5 streaming/slab archive layout: tensor metadata and tensor payloads must be written independently so conversion never owns the full output payload in memory.

## Spec Critique Response 2026-05-09

### Summary

Fold external spec criticism into the public whitepaper without weakening the v4 HQ direction.

### Corrections Applied

- Fixed legacy compression-ratio arithmetic in `lns-ml-whitepaper.md`: ratios are now derived from bits-per-weight including super-block headers. Legacy Q2_L is capped at `7.31x` vs FP16 before retained FP16/F32 tensors, so archive-level claims above that bound are explicitly treated as accounting errors.
- Replaced bandwidth-only TPS language with a hardware-aware caveat: software LNS decode may become compute-bound and cannot assume INT4/INT8/FP8 tensor-core style execution.
- Replaced hot-loop `exp2()` claims with the implemented table/direct-codebook decode model and clarified that measured kernels, not bandwidth math alone, decide TPS.
- Corrected the legacy encoder inverse formula: `L_normalized = (log2(|x'|) + offset) / mult`. The Q8_L catastrophic-threshold failure described by the critique is now explicitly prevented in the spec.
- Clarified zero handling: GPU hot paths should encode zero directly in the decode table or backend select/predication instead of paying an avoidable per-weight branch/mask when possible.
- Aligned whitepaper Q4_L parameters with the current code: `MAG_OFFSET = 5`, grid `{0, 1/16, 1/8, 1/4, 1/2, 1, 2, 4}`.

### Verification Notes

- `lns-core/src/quant/q8l.rs` already used the required inverse form by dividing by `0.125`; the failure was in old documentation, not the Q8_L encoder implementation.
- `lns-ml-math-spec.md` already used the safer v4 HQ hot-loop contract: no `exp`, no dynamic scale reconstruction, direct codebook lookup plus FMA.
