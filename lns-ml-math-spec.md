# LNS-ML: Mathematical Foundations & Architecture (v4 HQ)

> **Status:** Spec v4 HQ, May 2026. Supersedes v1-v3 for new HQ work.
> **Goal:** Minimise quantisation-induced output error at GGML-Q4-class and
> Q2-class memory budgets, while preserving a branch-light, bandwidth-efficient
> inference path on Apple Metal, CUDA, ROCm, and SPIR-V backends.
> **Non-goal:** This spec does not claim "zero hallucination". Quantisation can
> reduce or amplify hallucination risk, but hallucination is a model/runtime
> behaviour, not a scalar-quantizer property.

---

## 0. Executive Corrections

The v4 direction is correct in spirit: expensive quantizer choices belong in the
offline converter; the GEMV hot loop should not perform `exp2`, dynamic scale
hierarchy math, or sparse pointer chasing. However, several draft claims need
tightening before implementation:

| Draft claim                                                                    | v4 HQ correction                                                                                                                                                                                                                                                                    |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Per-channel AWQ can be fused into one block scale as `S_eff * alpha_j`.        | Exact AWQ requires either pre-scaling activations by `1 / alpha_j` outside GEMV, or restricting `alpha` to the block/group granularity. If weights are quantised after multiplying columns by `alpha_j`, the exact decoded original weight is divided by `alpha_j`, not multiplied. |
| NF4 removes the dead zone and guarantees zero underflow.                       | NF4 reduces central quantisation error and provides an exact-zero code, but every nearest-neighbour quantizer still has a finite zero bin. The zero bin must be measured and controlled, not declared gone.                                                                         |
| A `2B scale + 16B qs = 18B` sub-block should be padded/interleaved to 32B/64B. | Padding 18B sub-blocks destroys the bit budget. The correct GPU layout is structure-of-arrays: `scale[8]` followed by `qs[128]`, exactly the 144B layout already used by the Metal Q4 path.                                                                                         |
| Random outlier sidecars are bad, so any block can be replaced by Q8 inline.    | Variable-size inline blocks break O(1) row/block indexing. Promotion must be tensor-level, row/tile-level, or stored as contiguous promoted runs with separate branch-free kernels.                                                                                                 |
| "Zero-math inner loop" means no arithmetic.                                    | FMA is the work. The invariant is no transcendental decode math, no per-weight branches, and no scale hierarchy reconstruction in the hot loop.                                                                                                                                     |
| Lower scalar weight MSE automatically means fewer hallucinations.              | Weight MSE is only a proxy. The converter must optimise held-out NLL, final-logit KL, top-k rank stability, and layer output error. Hallucination reduction can only be claimed empirically as reduced quantisation-induced behaviour drift.                                        |
| Q2 can be treated as a smaller Q4.                                             | Q2 is a separate deployment tier. It needs learned four-value codebooks, stronger calibration gates, and an explicit Q4/Q8 rescue budget; otherwise PPL drift can dominate the memory win.                                                                                          |
| The best archive is the one with the lowest PPL.                               | The deployment archive is chosen from a Pareto frontier: lowest NLL/KL under a bit budget and the highest measured TPS under the runtime precision floor. Quality wins until the agreed drift budget is met; after that, TPS and memory bandwidth decide.                           |

---

## 1. Notation

| Symbol        | Meaning                                                                  |
| ------------- | ------------------------------------------------------------------------ |
| `w_ij`        | true weight at output row `i`, input channel `j`                         |
| `x_j`         | activation input channel `j`                                             |
| `a_j`         | calibration salience, default `E[x_j^2]`                                 |
| `alpha_j`     | AWQ column scale, positive; default `1`                                  |
| `z_ij`        | AWQ-transformed weight, `alpha_j * w_ij`                                 |
| `d_b`         | direct f16 scale for quantisation group `b`                              |
| `C[k]`        | signed 4-bit HQ codebook value for nibble `k in 0..15`                   |
| `C2[k]`       | signed 2-bit HQ codebook value for code `k in 0..3`                      |
| `q_t`         | packed quantisation code for element `t` inside a block                  |
| `lambda_j`    | weighted objective coefficient, `a_j / alpha_j^2`                        |
| `H_G`         | damped activation covariance for input group `G`, `X_G^T X_G / n + mu I` |
| `p_fp`, `p_q` | full-precision and quantised next-token distributions                    |
| `D_KL`        | `sum_v p_fp(v) log(p_fp(v) / p_q(v))`                                    |

All runtime decode formulas below are the contract. Encoder algorithms may be
improved as long as they decode to the same values.

---

## 2. Dequantization Contracts

### 2.1 Legacy Q4_L

Legacy `Q4L` is a reference and convenience decode path, not a hard
compatibility constraint for new architecture work:

```text
S_eff = S_g * 2^(S_l - 7)
hat_w = sign * S_eff * 2^(m - 5) * 1[m != 0]
```

Current Metal loading already expands each 134B disk block into a 144B GPU
block with eight f16 `S_eff` values plus 128B packed nibbles. HQ work may reuse
this 144B execution shape, but it should not preserve legacy semantics when a
cleaner v4 path is better for quality or throughput.

### 2.2 Legacy Q2_L

Legacy `Q2L` remains useful as a compact baseline, but Q2_HQ does not need to
preserve its duplicate-zero/sign-magnitude encoding:

```text
S_eff = S_g * 2^(S_l - 7)
hat_w = sign * S_eff * 1[m != 0]
```

The stored code is `sign | magnitude`, but `m = 0` decodes to exact zero for
both sign values. Therefore legacy Q2_L has only three effective values per
sub-block: `{-S_eff, 0, +S_eff}`. This is excellent for speed and archive size,
but any HQ successor must treat the duplicate-zero behaviour as an empirical
choice, not a mathematical requirement.

### 2.3 Q4_HQ Base Format

`Q4_HQ` replaces the sign/magnitude logarithmic nibble with a direct codebook
index:

```text
hat_z_t = d_b * C[q_t]
```

If AWQ is disabled, `hat_w_t = hat_z_t`. If per-channel AWQ is enabled:

```text
z_ij     = alpha_j * w_ij
hat_z_ij = d_b * C[q_ij]
hat_w_ij = hat_z_ij / alpha_j
```

At GEMV time this is evaluated by either of the equivalent forms:

```text
y_i = sum_j (hat_z_ij / alpha_j) * x_j
y_i = sum_j  hat_z_ij * (x_j / alpha_j)
```

The second form is preferred: pre-scale the activation vector once outside the
GEMV inner loop. For RMSNorm-fed projections this pre-scale is fused with the
RMSNorm gain pass.

### 2.4 Q2_HQ Extreme-Compression Format

`Q2_HQ` is a direct two-bit codebook format:

```text
q_t in 0..3
hat_z_t = d_b * C2[q_t]
```

If AWQ is enabled, the same exactness rule applies as Q4_HQ:

```text
z_ij      = alpha_j * w_ij
hat_w_ij = (d_b * C2[q_ij]) / alpha_j
```

Q2_HQ is not expected to preserve all tensors at Q2 quality. It is an
extreme-compression tier that must be paired with calibration-selected Q4_HQ,
Q8_HQ, or F16 rescue for sensitive tensors and tile runs.

### 2.5 Q8_HQ Promotion Format

`Q8_HQ` is a symmetric linear int8 quantizer with one direct f16 scale per
32-weight sub-block:

```text
q_t in [-127, 127]
hat_z_t = d_b * q_t / 127
```

This is intentionally not legacy `Q8_L`. Q8 promotion exists to protect
outliers and sensitive tensors; a linear int8 grid usually has lower absolute
error for already-scaled local blocks than a logarithmic 8-bit grid.

---

## 3. HQ Codebook

### 3.1 Default Codebook: NF4-Z

The default `Q4_HQ` codebook is a zero-exact NF4-style table, normalised to
`max(abs(C)) = 1`:

```cpp
// Q4_HQ NF4-Z, nibble is a direct index, not sign|magnitude.
constant float Q4_HQ_NF4_Z[16] = {
    -1.0000000, -0.6961928, -0.5250731, -0.3949175,
    -0.2844414, -0.1847734, -0.0910500,  0.0000000,
     0.0795803,  0.1609302,  0.2461123,  0.3379152,
     0.4407098,  0.5626170,  0.7229568,  1.0000000,
};
```

Properties:

- One exact zero code. No duplicate zero code is allowed; duplicate zero wastes
  one of the 16 representable values.
- The table is slightly asymmetric because exact zero consumes a codepoint in a
  signed distribution. This is expected and preferable to forcing symmetry.
- Values are finite constants in shader code. Runtime does not evaluate CDFs,
  exponentials, or inverse-normal functions.

### 3.2 Codebook Registry

v4 readers may support a small registry selected by `codebook_id`:

| ID  | Name             | Use case                                       |
| --- | ---------------- | ---------------------------------------------- |
| 0   | `NF4_Z`          | default for approximately normal local weights |
| 1   | `LLOYD_WEIGHTED` | optional globally trained Lloyd-Max table      |
| 2   | `E2M1_FP4`       | sub-Gaussian / RMSNorm-adjacent tensors        |
| 3   | `Q4L_LOG_DELTA5` | compatibility experiments only                 |

Per-tensor learned 16-value codebooks are not default for Q4_HQ because they add
metadata, shader indirection, and validation complexity. They may be revisited
only after fixed-codebook PPL is measured.

### 3.3 Q2_HQ Codebooks

Q2_HQ is the exception: with only four codepoints, codebook placement matters
too much to rely on a single universal table. The default Q2_HQ encoder should
test at least two profiles per tensor and keep the one with better held-out
NLL/KL under the global bit budget.

**Dense four-code profile:** learn or select four monotonic values
`C2[0..3]`, normalised to `max(abs(C2)) = 1`. Exact zero is optional. This is
the preferred default for dense projection matrices because all four codes carry
signal.

**Ternary exact-zero profile:** effective values are `{-1, 0, +1}` with one
duplicate zero code. This is allowed only when the validation contract prefers a
wider zero bin, for example on sparse or highly clipped tensors. Duplicate zero
is therefore a measured compression tactic, not a default ideology.

For Q2_HQ custom codebooks, the `custom_codebook_off` payload stores `f32[4]`
followed by zero padding to the same alignment used by Q4_HQ `f32[16]` tables.
Readers must reject NaN, infinity, non-monotonic codebooks, and all-zero
codebooks. The hot loop still performs only a two-bit lookup, scale multiply,
and FMA.

---

## 4. Encoder Objective

### 4.1 Output-Weighted Error

For a linear layer `y = W x`, the most relevant local error is output error:

```text
E ||(W - hat_W) x||_2^2
```

The diagonal approximation used by the converter is:

```text
sum_j a_j * ||W[:, j] - hat_W[:, j]||_2^2
```

where `a_j = E[x_j^2]` from calibration activations. If calibration data is not
available, set `a_j = 1`.

### 4.2 Second-Order Calibration Objective

The diagonal objective in Section 4.1 is the default because it is cheap and
stable. When calibration activations are available, v4 HQ should also support a
GPTQ/OBS-style second-order encoder for high-impact tensors.

For an input group `G` of 128-256 contiguous columns, collect calibration
activations `X_G` and form a damped covariance:

```text
H_G = X_G^T X_G / n + mu I
mu  = damp * mean(diag(X_G^T X_G / n))
damp in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
```

For each output row, the local output-error objective is:

```text
e_i = W_i,G - hat_W_i,G
loss_i,G = e_i^T H_G e_i
```

The diagonal method is the special case where only `diag(H_G)` is kept. The
second-order path therefore improves code and scale selection without changing
the Q4_HQ or Q8_HQ runtime format.

Reference error-feedback encoder:

1. Compute a Cholesky factorisation or inverse `H_inv = H_G^-1` of the damped
   covariance in f32/f64.
2. Process columns in a deterministic activation-descending or block-local
   order.
3. Quantise the current column/block with the Q4_HQ block search.
4. Propagate the residual to the remaining unquantised columns using the
   inverse-Hessian update. Here `u_j` is the mutable transformed weight column
   currently being quantised and `hat_u_j` is its decoded quantised value, not
   the packed 4-bit code:

```text
err_j = (u_j - hat_u_j) / H_inv[j, j]
u_k   = u_k - err_j * H_inv[j, k]       for each unquantised k after j
```

5. Re-run the final block assignment against rounded f16 scales and measure the
   true layer output error on held-out calibration windows.

This path is converter-only. If it overfits or regresses held-out NLL, fall back
to the diagonal weighted encoder for that tensor.

### 4.3 AWQ Transform

AWQ scales are chosen per input channel:

```text
alpha_j(gamma) = clamp((sqrt(a_j) / geom_mean_k sqrt(a_k))^gamma, alpha_min, alpha_max)
```

Recommended search:

```text
gamma in {0.00, 0.25, 0.50, 0.75, 1.00}
alpha_min = 1/16
alpha_max = 16
```

Pick the `gamma` that minimises calibration output MSE after quantization. This
turns AWQ from a fixed formula into an offline model-selection problem.

Important exactness rule:

```text
quantize z_ij = alpha_j * w_ij
runtime activation is x_j / alpha_j
```

Multiplying the final decoded weight by `alpha_j` without dividing the
activation changes the model and is invalid.

### 4.4 Block Scale, Clipping, and Code Search

For each 32-weight sub-block `b`, define transformed weights `z_t = alpha_j w_t`
and objective weights:

```text
lambda_t = a_j / alpha_j^2
```

The block objective is:

```text
min_{d > 0, q_t in 0..15}  sum_t lambda_t * (z_t - d * C[q_t])^2
```

For a fixed `d`, code assignment is nearest-neighbour:

```text
q_t(d) = argmin_k (z_t - d * C[k])^2
```

For fixed codes, the optimal positive scale is:

```text
d* = max(eps, sum_t lambda_t * C[q_t] * z_t / sum_t lambda_t * C[q_t]^2)
```

Reference encoder:

1. Seed `d` from both RMS and percentile scales:
   `rms(z) / rms(C)` and `percentile_99(abs(z))`.
2. Add clipped candidates from `percentile_p(abs(z)) / max(abs(C))` for
   `p in {99.9, 99.5, 99.0, 98.0, 97.0}`. Clipping is allowed only when it
   lowers the weighted objective or held-out layer output error.
3. Search 32-64 log-spaced `d` candidates around the seeds.
4. Assign codes by nearest neighbour for each candidate.
5. Refine `d` with the closed form above for 2 Lloyd iterations.
6. Store `f16(d)` and re-run final code assignment against the rounded scale.

This is offline work. It is intentionally more expensive than the current Q4_L
encoder because it buys lower PPL without touching inference latency.

The converter must report per-tensor saturation rate:

```text
saturation_fraction = count(abs(C[q_t]) == max(abs(C))) / n
saturation_mse      = sum_{saturated} lambda_t * (z_t - d_b C[q_t])^2
```

High saturation with high output error is a promotion signal. High saturation
with lower held-out NLL is acceptable because intentional clipping can improve
the bulk of a heavy-tailed block.

### 4.5 Zero Bin Accounting

Let `z0 = 0` be the exact-zero code and `z_near` be the closest non-zero
codebook value after scaling. The zero bin boundary is the arithmetic midpoint:

```text
|z| < d_b * |z_near| / 2       when zero and z_near are adjacent
```

For asymmetric NF4-Z, compute the positive and negative zero thresholds
separately. The converter must report:

```text
zero_fraction = count(q_t == zero_code) / n
zero_mse      = sum_{q_t == zero_code} lambda_t * z_t^2
```

The target is not zero `zero_fraction`; the target is that zero assignments are
chosen because they minimise the weighted objective and do not dominate layer
output error.

### 4.6 NLL/KL-Gated Model Selection

The final converter decision is not made by weight error alone. For each model
candidate under the same bit budget, evaluate a held-out calibration stream with
the full forward pass and score:

```text
NLL_q      = -mean_t log p_q(token_t | context_t)
KL_logits  = mean_t D_KL(p_fp(. | context_t) || p_q(. | context_t))
rank_drift = mean_t rank_distance(top_k_fp, top_k_q)
```

Candidate ranking is lexicographic:

1. reject candidates that exceed the bits/weight budget or CPU/GPU parity
   tolerance;
2. minimise `NLL_q` on held-out text;
3. break ties with `KL_logits` and top-k rank stability;
4. break remaining ties with throughput and archive size.

This directly targets PPL and quantisation-induced behaviour drift. It still
does not prove hallucination reduction, but it prevents accepting a low-MSE
quantizer that changes next-token distributions in sensitive regions.

---

## 5. Layout and File Format

### 5.1 Outer Container

The current repository uses an rkyv-archived `LnsModel` outer container:

```rust
struct LnsTensor {
    name: String,
    shape: Vec<u64>,
    quant_type: u8,
    data: Vec<u8>,
}
```

v4 keeps this outer model shape. New HQ formats are introduced as new
`quant_type` values, not as reinterpretations of legacy `Q4L`:

| Code | Type    | Notes                                     |
| ---- | ------- | ----------------------------------------- |
| 2    | `Q4L`   | legacy log-grid baseline                  |
| 5    | `Q8L`   | legacy logarithmic Q8 baseline            |
| 6    | `Q4HQ`  | v4 direct-scale NF4-Z / codebook registry |
| 7    | `Q8HQ`  | v4 direct-scale linear int8               |
| 8    | `Q4HQM` | v4 mixed Q4HQ + promoted contiguous runs  |
| 9    | `Q2HQ`  | v4 direct-scale two-bit codebook          |
| 10   | `Q2HQM` | v4 mixed Q2HQ + Q4HQ/Q8HQ rescue runs     |

New archives are v4 by default. Reading v2/v3 tensors is an implementation
convenience while useful for local experiments; it is not an architecture freeze
criterion and must not force compatibility shims into the HQ hot path.

HQ tensor `data` is self-describing because the outer `LnsTensor` has no
separate flags field. Every `Q2HQ`, `Q4HQ`, `Q8HQ`, `Q2HQM`, and `Q4HQM`
payload begins with a little-endian header:

```text
struct HqTensorPayloadHeader {
  magic:              [u8; 4],  // "LHQ4"
  header_bytes:       u16,      // sizeof header, allows extension
  block_bytes:        u16,      // 80/96 for Q2HQ, 144 for Q4HQ, 272 for Q8HQ
  flags:              u32,      // bit 0 AWQ inv_alpha present
                   // bit 1 segmented promotion table present
                   // bit 2 custom codebook present
                   // bits 3-31 reserved, must be zero
  codebook_id:        u8,       // quant-type-specific registry id; ignored by Q8HQ
  reserved0:          [u8; 7],  // must be zero
  inv_alpha_count:    u32,      // 0 or input dimension
  segment_count:      u32,      // 0 for non-segmented tensors
  custom_codebook_off:u64,      // 0 unless flags.bit2; f32[4] for Q2HQ, f32[16] for Q4HQ
  inv_alpha_off:      u64,      // 0 unless flags.bit0; f16[inv_alpha_count]
  segment_table_off:  u64,      // 0 unless flags.bit1; Segment[segment_count]
  block_data_off:     u64,      // first HQ block, 16B aligned
  block_data_bytes:   u64,
}
```

All offsets are relative to the first byte of `LnsTensor.data`. Readers must
reject payloads whose offsets are unaligned, overlapping, out of bounds, or whose
reserved fields are non-zero. For Phase 1, `flags = 0`, `codebook_id = 0`, and
`block_data_off` points directly to the raw Q4_HQ blocks.

Block data is row-major. For a 2-D tensor shaped `[rows, cols]`, each row stores
`ceil(cols / 256)` blocks and pads the final block if `cols` is not a multiple
of 256. Padding values must either decode to exact zero or be ignored by a
tail-aware kernel. For branch-light backends, tensors with partial final blocks
should use an exact-zero codebook profile. Padding is not included in quality
metrics or bit-budget numerators. Non-matrix tensors should use F16/F32 unless a
dedicated kernel consumes their HQ layout directly.

### 5.2 Q4_HQ Block Layout

The canonical Q4_HQ super-block stores 256 weights as eight 32-weight
sub-blocks:

```text
Offset  Size  Field
0       16B   scale[8]      eight f16 direct scales d_b
16      128B  qs[128]       256 packed 4-bit codebook indices
```

Total: 144B per 256 weights = 4.50 bits/weight.

This is the same GPU stride as the current Metal `block_q4_l` layout and is the
preferred hot-path representation. The 18B sub-block shape is only a conceptual
unit; it must not be individually padded to cache-line boundaries.

### 5.3 Q2_HQ Block Layouts

Q2_HQ keeps 256 weights per super-block and packs four codes per byte. Two
scale granularities are valid:

```text
Q2HQ-32
Offset  Size  Field
0       16B   scale[8]      eight f16 direct scales d_b, one per 32 weights
16      64B   qs[64]        256 packed 2-bit codebook indices
Total: 80B per 256 weights = 2.50 bits/weight

Q2HQ-16
Offset  Size  Field
0       32B   scale[16]     sixteen f16 direct scales d_b, one per 16 weights
32      64B   qs[64]        256 packed 2-bit codebook indices
Total: 96B per 256 weights = 3.00 bits/weight
```

`Q2HQ-32` is the default speed/memory profile. `Q2HQ-16` is a quality profile
for tensors where the extra scale bits reduce held-out NLL enough to justify the
cost. The `block_bytes` field distinguishes the two layouts. Runtimes must not
guess scale granularity from tensor names.

### 5.4 Q8_HQ Block Layout

```text
Offset  Size  Field
0       16B   scale[8]      eight f16 direct scales d_b
16      256B  qs[256]       signed int8 values
```

Total: 272B per 256 weights = 8.50 bits/weight.

The int8 code `-128` is reserved and must not be emitted by the encoder. Readers
may treat it as invalid data or clamp it to `-127` in diagnostic tooling, but
production inference should reject malformed payloads.

### 5.5 AWQ Metadata

Per-channel AWQ stores `inv_alpha_j = 1 / alpha_j` as f16 with length equal to
the tensor input dimension. Its amortised cost is:

```text
bits_per_weight_awq = 16 / rows
```

For 2048-8192 output rows this is roughly 0.002-0.008 bits/weight. Small
tensors may disable AWQ if the metadata is not worth the gain.

The runtime may materialise `x_awq[j] = x[j] * inv_alpha_j` in one vector pass.
For RMSNorm-fed projections, combine it with RMSNorm:

```text
x_awq[j] = x[j] * inv_rms * norm_weight[j] * inv_alpha_j
```

### 5.6 Mixed-Promotion Payloads

Fine-grained promotion must preserve sequential memory access. A flat array of
variable-size Q4/Q8 blocks is forbidden because it requires prefix sums or
pointer chasing in GEMV.

Allowed promotion layouts:

1. **Tensor-level promotion:** the whole tensor is `Q8HQ`. This is simplest and
   should be implemented first for `o_proj`, `down_proj`, `lm_head`, and any
   calibration-identified sensitive tensors.
2. **Segmented tile promotion:** store branch-free contiguous runs:

```text
struct Segment {
    start_block: u32,
    block_count: u32,
    dtype: u8,          // Q4HQ or Q8HQ
    data_offset: u64,
}
```

Each segment is dispatched with the matching kernel. There is no per-weight or
per-block branch inside the kernel. Adjacent segments with the same dtype must
be merged by the converter.

For Q2HQM, the same segment table is used, but valid segment dtypes are `Q2HQ`,
`Q4HQ`, and `Q8HQ`. Adjacent Q2 segments with different scale granularities must
not be merged unless their `block_bytes` and codebook metadata match exactly.

### 5.7 Precompiled LNS Archive Optimisations

The `.lns` archive should become a precompiled inference artefact, not merely a
compressed weight dump. Precompilation has two distinct goals:

1. **Speed:** remove load-time repacking, shape inference, and backend-specific
   pointer preparation from the critical startup path.
2. **Quality:** preserve the exact offline calibration decisions that produced
   the best held-out NLL/KL under the selected bit budget.

The canonical HQ payload remains the source of truth. Optional precompiled
chunks may be added only when they are deterministic caches derived from the
canonical payload and can be ignored safely by older runtimes.

Recommended archive additions:

- **Tensor manifest:** per tensor byte offset, shape, dtype, block size,
  row-block count, row stride in blocks, flags, and backend compatibility hash.
- **Backend-ready execution blob:** optional Metal/CUDA/ROCm/SPIR-V chunk whose
  tensor blocks are already ordered, padded, and aligned for the target kernels.
  The current Metal Q4 path's 144B block is a good canonical execution stride;
  future HQ archives should avoid doing this repack at model load when the blob
  was produced for the same backend ABI.
- **Calibration report:** code entropy, zero fraction, saturation fraction,
  layer-output MSE, final-logit KL, held-out NLL/PPL, selected AWQ `gamma`, and
  promotion decisions per tensor. This metadata is not used in the hot loop, but
  it makes quality reproducible and auditable.
- **Fused vector cache:** optional f16 vectors such as
  `norm_weight[j] * inv_alpha[j]` for RMSNorm-fed projections. This is useful
  only when it removes a runtime vector multiply and its metadata cost is below
  the measured speed benefit.
- **Tokenizer/runtime config snapshot:** tokenizer hash, chat-template hash,
  RoPE parameters, sliding-window settings, and special token ids. These do not
  change quantisation quality directly, but mismatches can dominate PPL and
  generation behaviour more than the quantizer itself.

Hard rules:

- Do not store `scale * C[k]` tables per sub-block by default. For Q4_HQ that
  would add `8 * 16` f16 values per 256 weights, increasing the block by 256B
  and destroying the 4.5-4.8 bpw target.
- Do not make precompiled backend blobs mandatory. A runtime must be able to
  ignore them and execute from the canonical HQ payload.
- Do not let archive caches silently change numerics. Every precompiled chunk
  must be covered by a hash over the canonical tensor payload, codebook id,
  AWQ metadata, segment table, backend ABI version, and kernel layout version.
- Padding added for alignment must decode to zero and must be excluded from
  PPL, NLL, KL, and bits/weight metrics.

---

## 6. GEMV Hot Loop

### 6.1 Q2_HQ Decode

`Q2_HQ` decodes four weights per byte:

```cpp
inline float4 q2hq_decode_byte(uchar raw_byte, float scale, constant float *C2) {
  uint q0 =  raw_byte        & 0x03u;
  uint q1 = (raw_byte >> 2u) & 0x03u;
  uint q2 = (raw_byte >> 4u) & 0x03u;
  uint q3 = (raw_byte >> 6u) & 0x03u;
  return float4(scale * C2[q0], scale * C2[q1], scale * C2[q2], scale * C2[q3]);
}
```

For fixed registry codebooks, `C2` should be a shader constant. For per-tensor
learned codebooks, pass the four f32 constants once per dispatch. Do not store
per-block expanded decode tables in the archive.

### 6.2 Q4_HQ Decode

`Q4_HQ` nibbles are direct codebook indices:

```cpp
inline float2 q4hq_decode_byte(uchar raw_byte, float scale) {
    uint lo = raw_byte & 0x0Fu;
    uint hi = raw_byte >> 4u;
    return float2(scale * Q4_HQ_NF4_Z[lo], scale * Q4_HQ_NF4_Z[hi]);
}
```

The inner loop invariant:

- no `exp`, `exp2`, `pow`, or logarithm;
- no reconstruction of `S_g * 2^(S_l - offset)`;
- no sign/magnitude branch;
- one direct nibble lookup and FMA accumulation per weight.

Optional micro-optimisation: precompute `scale * C[k]` for a sub-block in
registers or threadgroup memory when that reduces repeated multiplications on a
specific backend. This is backend policy, not a file-format requirement.

### 6.3 Q8_HQ Decode

```cpp
decoded = scale * ((float)q_i) * (1.0f / 127.0f);
acc = fma(decoded, x_i, acc);
```

For promoted tensors or segments, Q8_HQ should have a native GPU GEMV kernel.
Decoding Q8 to a dense f16 matrix at load time is acceptable as a temporary
bring-up path but is not the v4 target because it spends too much memory.

### 6.4 Runtime Precision Floors

Weight quantisation is only one source of PPL drift. The runtime must keep the
following precision floors unless a measured replacement improves held-out
NLL/KL:

- GEMV/GEMM accumulation uses f32 accumulators, even when inputs and decoded
  weights are f16-compatible.
- RMSNorm variance, residual additions, RoPE angle generation, attention score
  accumulation, online softmax max/sum, and final logits are computed in f32 or
  a numerically equivalent mixed-precision path.
- KV cache compression must be validated separately from weight quantisation.
  Q8 per-head or per-block KV is acceptable when long-context NLL and attention
  output error match the f16 cache within tolerance. Q4 KV is not a default v4
  target because it can dominate long-context behaviour drift.
- Sampling transforms operate on f32 logits after all penalties, masks, and
  temperature scaling. Quantisation should not change token filtering order by
  accident.

These floors are not dogma; they are guardrails. A lower-precision kernel can be
accepted only if it beats the floor on the same validation contract.

### 6.5 Throughput Contract

The performance target is maximum realised tokens/sec for a fixed quality budget,
not outscoring GGML on every benchmark. GGML-relative throughput is useful as a
diagnostic, but the v4 pass/fail gate is the model's own Pareto frontier:

```text
maximise   TPS_prefill, TPS_decode
subject to DeltaNLL <= budget_nll
           KL_logits <= budget_kl
           bpw       <= budget_bpw
           memory    <= budget_memory
           parity    <= tolerance
```

Every accepted runtime path must preserve these invariants:

- no per-token heap allocation in the decode loop;
- no load-time expansion of Q2/Q4/Q8 HQ weights into dense f16/f32 except as a
  temporary bring-up fallback that is reported separately;
- O(1) tensor row and block indexing;
- no per-weight or random per-block branch in GEMV kernels;
- no sparse sidecar pointer chasing in the hot loop;
- command-buffer and encoder count are measured for prefill and decode;
- host-to-device transfers during decode are limited to the current token,
  scalar parameters, and explicitly measured small vectors;
- prefill and decode are benchmarked separately because they stress different
  kernels and memory paths.

The converter may produce several candidate archives for the same model: Q2HQM,
Q4HQ, Q4HQ+AWQ, Q4HQ+Q8 rescue, and mixed variants. The runtime chooses no
default until those candidates are plotted as `(bpw, DeltaNLL, KL, TPS_prefill,
TPS_decode, peak_memory)`. A candidate that is slower and lower quality than
another candidate at the same or higher bpw is rejected.

---

## 7. Promotion Policy

Promotion is a budgeted optimisation, not a threshold guessed in isolation.

### 7.1 Block Error Score

For each Q4_HQ block, compute the relative weighted error:

```text
R_b = sum_t lambda_t * (z_t - d_b C[q_t])^2
      / (sum_t lambda_t * z_t^2 + eps)
```

Also compute the Q8_HQ error `R_b^8`. The block benefit is:

```text
G_b = R_b - R_b^8
```

For Q2_HQ blocks, use the same score with `C2[q_t]` and compare against the
best available rescue candidate: Q2HQ-16, Q4_HQ, or Q8_HQ.

### 7.2 Budget Constraint

If Q4_HQ costs 4.50 bpw and Q8_HQ costs 8.50 bpw, replacing a fraction `p` of
weights with Q8_HQ costs:

```text
bpw = 4.50 + 4.00 * p + metadata
```

To stay near 4.70 bpw, the replacement budget is roughly `p <= 0.05` before
metadata. The converter therefore sorts candidates by `G_b` and promotes only
the best blocks, rows, or tensors until the explicit bit budget is exhausted.

### 7.3 Tensor Sensitivity

A tensor may be promoted wholesale when calibration shows Q4 error has high
loss impact:

```text
rho_T = DeltaLoss_T(Q4_HQ) / max(DeltaLoss_T(Q8_HQ), eps)
promote if rho_T > rho_threshold and bit budget allows
```

Default prior for transformer LLMs:

- likely sensitive: `o_proj`, `down_proj`, `lm_head` / tied output projection;
- usually less sensitive: `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`;
- keep high precision unless proven safe: embeddings, norm weights, router
  weights, tiny control tensors.

Name-based rules are only a bootstrap. Calibration loss wins when available.

### 7.4 Loss-Aware Promotion Queue

The converter should maintain a promotion queue ranked by measured loss impact,
not by raw block error alone. For each candidate tensor, row group, or contiguous
tile run, estimate:

```text
benefit = NLL(Q4HQ) - NLL(candidate)
cost    = bpw(candidate) - bpw(Q4HQ)
score   = benefit / max(cost, eps)
```

When full NLL evaluation for every small segment is too expensive, use the
following fallback order:

1. layer output error on held-out activations;
2. final-logit KL after replacing one tensor at a time;
3. weighted block benefit `G_b`;
4. name-based priors.

Promotion stops when the global bpw budget is exhausted or when the next
candidate fails to improve held-out NLL/KL beyond measurement noise. This avoids
spending Q8_HQ budget on visually large weight errors that do not affect the
language-model distribution.

### 7.5 Q2 Rescue Policy

Q2_HQ requires a stricter rescue policy than Q4_HQ. A pure-Q2 model is valid
only if it passes held-out NLL/KL and deterministic generation drift checks.
Otherwise, the converter must build a Q2HQM plan under an explicit target such
as 2.7, 3.0, or 3.5 bpw.

Default Q2 prior:

- keep embeddings, norm weights, router weights, and tiny control tensors in
  F16/F32 unless a Q2/Q4 candidate is explicitly validated;
- try Q4_HQ for `lm_head`, `o_proj`, and `down_proj` before spending Q8_HQ;
- reserve Q8_HQ for tensors or tile runs where Q4_HQ still fails NLL/KL or
  top-k stability gates;
- prefer `Q2HQ-16` over Q4_HQ when the extra scales recover most of the loss at
  lower bpw.

Cost model from a `Q2HQ-32` base:

```text
promote p4 fraction from Q2HQ-32 to Q4HQ: bpw = 2.50 + 2.00 * p4
promote p8 fraction from Q2HQ-32 to Q8HQ: bpw = 2.50 + 6.00 * p8
switch p16 fraction from Q2HQ-32 to Q2HQ-16: bpw = 2.50 + 0.50 * p16
```

At Q2-class budgets, the promotion queue must rank candidates by NLL/KL benefit
per added bit. Raw block error is too weak a proxy for this tier.

---

## 8. Bit Budget

| Stack                        |      Bits/weight | Notes                                                        |
| ---------------------------- | ---------------: | ------------------------------------------------------------ |
| Legacy Q2_L disk             |             2.19 | current compact ternary storage, three effective values      |
| Q2_HQ-32 base                |             2.50 | direct f16 scales every 32 weights, learned 4-value codebook |
| Q2_HQ-32 + AWQ               | `2.50 + 16/rows` | usually still near 2.50 for large matrices                   |
| Q2_HQ-16 quality             |             3.00 | direct f16 scales every 16 weights                           |
| Q2_HQ-32 + 10% Q4_HQ rescue  |            ~2.70 | before segment metadata                                      |
| Q2_HQ-32 + 20% Q4_HQ rescue  |            ~2.90 | practical extreme-compression target                         |
| Legacy Q4_L disk             |             4.19 | current compact v2/v3 storage                                |
| Q4_HQ base                   |             4.50 | direct f16 scales, NF4-Z codebook                            |
| Q4_HQ + AWQ                  | `4.50 + 16/rows` | usually < 4.51 for large matrices                            |
| Q4_HQ + 5% Q8_HQ replacement |            ~4.70 | before tiny segment metadata                                 |
| Full Q8_HQ tensor            |             8.50 | use only where PPL justifies it                              |

The v4 target is not the smallest possible archive. There are two explicit
quality tiers:

- Q2-class: lowest acceptable PPL in the 2.5-3.5 bpw regime, with Q4/Q8 rescue
  driven by NLL/KL benefit per added bit.
- Q4-class: lowest PPL in the 4.5-4.8 bpw regime with a hot loop that remains
  bandwidth-shaped.

---

## 9. Validation Contract

Every encoder/runtime change must report the following on the same model and
dataset windows:

1. PPL on Wikitext-2 or a stronger held-out corpus, with context/stride stated.
2. Held-out NLL delta versus the same model in F16/BF16 or the highest-precision
   available reference path.
3. Final-logit KL: `D_KL(p_fp || p_q)` on calibration and held-out windows.
4. Layer output MSE: `||Y_fp - Y_quant||_F^2 / ||Y_fp||_F^2` on calibration data.
5. Top-k logit rank stability for the final projection, including top-1 flip
   rate and top-5 set overlap.
6. Code entropy, zero-code fraction, and saturation fraction per tensor.
7. Average bits/weight including scales, AWQ vectors, segment tables, and
   promoted blocks.
8. Tokens/sec for prefill and decode separately.
9. CPU/GPU decode parity for Q2_HQ, Q4_HQ, and Q8_HQ blocks.
10. For Q2-class archives, report the rescue mix: Q2HQ-32, Q2HQ-16, Q4HQ,
    Q8HQ, F16/F32 fractions and their individual NLL/KL contribution.
11. Peak resident memory, archive size, model-load time, first-token latency,
    per-token decode latency distribution, and command-buffer count.
12. Pareto frontier table for all candidate archives under the same tokenizer,
    runtime config, prompt windows, and backend.

For hallucination-sensitive claims, also run a fixed qualitative regression set
with deterministic decoding (`temperature = 0`, fixed prompt list, fixed max
tokens). The required output is not a subjective score; it is a diff report:

- answer changed / unchanged;
- exact-match or judge-model score when a gold answer exists;
- refusal/tool-call format changed / unchanged when applicable;
- first divergent token index versus the reference model.

Claims about hallucination must be phrased as empirical model-output results,
not as mathematical guarantees from the quantizer.

---

## 10. Architecture Freeze Contract

The architecture is not considered frozen when the equations look good. It is
frozen only when the validation contract proves that the file format, converter,
runtime precision floor, and backend kernels form a stable Pareto surface.

Freeze criteria:

1. `Q2_HQ`, `Q2HQM`, `Q4_HQ`, and `Q8_HQ` decode parity are tested on CPU and
   at least one GPU backend. Legacy `Q2_L`/`Q4_L` parity is useful for baseline
   comparison, but not required for forward architecture compatibility.
2. Q2-class and Q4-class candidates are benchmarked on the same held-out PPL/NLL
   windows and the same throughput prompts.
3. The selected default archive is not dominated in `(DeltaNLL, KL, TPS, bpw,
peak_memory)` by another implemented candidate.
4. The runtime has no mandatory dense expansion path for HQ weights.
5. Precompiled backend chunks are optional caches with hash validation and a
   canonical-payload fallback.
6. Numeric precision floors are either followed or replaced by kernels that pass
   the same NLL/KL gate.
7. Format version, quant type ids, payload headers, segment tables, codebook
   ids, and alignment rules have roundtrip tests and malformed-input rejection
   tests.
8. A regression dashboard or report records every accepted model artefact with
   PPL, NLL, KL, TPS, memory, archive size, and generation-diff results.

Until these criteria pass, v4 HQ is an implementation plan, not a frozen
architecture. After they pass, changes that alter tensor payload layout, decode
math, scale granularity, or precision floors require a new spec revision and a
fresh Pareto comparison.

---

## 11. Implementation Plan

### Phase 0: Measurement and Archive Contract

- Add a reproducible evaluation harness for PPL, held-out NLL, final-logit KL,
  top-k stability, and deterministic generation diffs.
- Add an HQ tensor payload parser/writer with manifest hashes and strict offset
  validation before adding more quantizers.
- Add optional precompiled backend chunk support behind a feature flag. The
  runtime must verify hashes and fall back to the canonical payload.
- Record tokenizer/config hashes in the archive so PPL regressions caused by
  preprocessing drift are not misattributed to quantisation.

### Phase 1: Q4_HQ Base

- Add `QuantType::Q4HQ` and make newly written archives v4 by default.
- Implement `Q4HQSuperBlock { scale: [u16; 8], qs: [u8; 128] }`.
- Add CPU encode/decode using NF4-Z and weighted scale search.
- Add Metal Q4_HQ GEMV by reusing the current 144B block stride with a 16-entry
  signed LUT.
- Validate CPU/GPU parity and PPL against current Q4_L.

### Phase 1.25: Q2_HQ Base

- Add `QuantType::Q2HQ` and `QuantType::Q2HQM`.
- Implement `Q2HQ-32` and `Q2HQ-16` block layouts with direct f16 scales and
  packed two-bit codebook indices.
- Add per-tensor learned four-value codebook selection and optional ternary-zero
  profile selection under held-out NLL/KL gates.
- Add native Metal Q2_HQ GEMV that decodes four weights per byte and keeps f32
  accumulation.
- Validate Q2_L, Q2_HQ-32, Q2_HQ-16, and Q2HQM rescue plans separately.

### Phase 1.5: Calibration-Driven Encoder Quality

- Add calibration activation capture and held-out text windows.
- Add diagonal activation-weighted scale search, percentile clipping candidates,
  zero/saturation telemetry, and NLL/KL scoring.
- Add second-order GPTQ/OBS-style encoding for sensitive tensors behind a
  converter flag.
- Accept the second-order path only when held-out NLL/KL improves over the
  diagonal encoder.

### Phase 2: Native Q8_HQ Promotion

- Add `Q8HQSuperBlock { scale: [u16; 8], qs: [i8; 256] }`.
- Implement native Metal GEMV for Q8_HQ.
- Replace current mixed-precision load-time Q8->F16 mirror with native Q8_HQ
  where possible.
- Start with tensor-level promotion for sensitive tensors, then rank promotion
  candidates by loss benefit per added bit.

### Phase 3: AWQ Calibration

- Add calibration activation collection per input channel.
- Store f16 `inv_alpha` metadata for Q4_HQ tensors.
- Fuse `inv_alpha` into RMSNorm-fed GEMV inputs; add a vector pre-scale path for
  non-RMSNorm projections.
- Select `gamma` by calibration output MSE.

### Phase 4: Segmented Tile Promotion

- Add `Q4HQM` only after tensor-level promotion is measured.
- Generate contiguous Q4/Q8 segments under an explicit bpw budget.
- Dispatch one branch-free kernel per segment or per merged segment group.
- Reject layouts that require random sparse scatter-adds in the decode hot path.

---

## 12. Current Repository Delta

As of 2026-05-08:

- `Q4_L` already uses `MAG_OFFSET = 5`, linear-MSE magnitude assignment, and a
  mean-of-sub-block-max global scale in `lns-core/src/quant/q4l.rs`.
- Newly written archives use format version 4 by default. Backward
  compatibility with v2/v3 is a convenience path, not an architecture
  constraint.
- `Q4_HQ` now has a CPU block contract in `lns-core`: NF4-Z constants,
  `Q4HQSuperBlock { scale_bits: [u16; 8], qs: [u8; 128] }`, 144B canonical
  block size, payload wrapping, payload validation, and dense decode.
- `lns-convert --quant Q4_HQ` writes v4 archives with canonical HQ payloads.
- Metal can now consume Q4_HQ payloads natively for GEMV: the backend validates
  the HQ header, copies canonical 144B block data directly into a Metal buffer,
  and dispatches NF4-Z Q4_HQ kernels without dense F16 expansion.
- A TinyLlama smoke comparison on 2026-05-09 found Q4_HQ promising but encoder
  heavy: fresh Q4_L conversion took about 16.91s and produced a 734M archive,
  while fresh Q4_HQ conversion took about 97.07s and produced a 770M archive.
  On `lns-cli eval --gen-tokens 16`, Q4_HQ improved average decode speed
  `171.2 -> 183.9 tok/s` and calibration PPL `65.18 -> 20.91`; on a 512-token
  Wikitext smoke it improved PPL `49.5241 -> 38.7544`.
- The first Q4_HQ encoder throughput pass keeps the canonical payload and
  NF4-Z decode contract unchanged but refines only the best coarse scale
  candidates. TinyLlama Q4_HQ conversion improved to about 50.91s, with
  calibration PPL `20.88` and 512-token Wikitext smoke PPL `38.3241`.
- `Q2_L` currently uses a compact 70B/256-weight legacy ternary block:
  `scale_global`, packed 4-bit local scales, and 2-bit sign/magnitude weights.
  It is a valid baseline, but Q2_HQ should be introduced as a new quant type
  rather than carrying over this encoding.
- Metal repacks legacy 134B Q4_L disk blocks into 144B GPU blocks with
  pre-expanded `efektif_scale[8]`; Q4_HQ already stores that execution layout
  canonically.
- `--mixed-precision` currently promotes selected tensors to legacy `Q8_L`, but
  non-Q4 two-dimensional tensors are mirrored to dense f16 for GPU GEMV. That is
  a useful bring-up path, not the final v4 memory model.
- The next high-leverage step is either a second Q4_HQ encoder throughput pass
  or the first `Q2_HQ` block implementation. After that, continue with
  `Q2HQM`, native `Q8_HQ` promotion, and measured AWQ as shared rescue
  mechanisms.

---

_Document version 4.0 HQ — May 2026._
