# LNS vs. GGML: Mathematical Quantization Analysis

This note consolidates the mathematical framing of the project's LNS quantization architecture against GGML-style linear quantization and updates the earlier specification so it matches the current implementation.

## Status

The original framing was directionally correct, but the current `Q4_L` implementation is more specific than the older text implied.

The main change is this:

- The old text described LNS as a single-scale logarithmic quantizer.
- The current implementation is a two-level hierarchical log quantizer:
  - one `scale_global` per 256-weight super-block
  - one packed 4-bit `scale_local` per 32-weight sub-block
  - one 4-bit signed log-coded weight per element

So the thesis has not changed, but the actual mathematics are now sharper and stronger than the original summary.

## 1. Weight Distributions and Why Quantization Is Hard

LLM weight tensors are typically concentrated around zero, with a high density of small values and a relatively small number of larger magnitudes. In practice this is often described as Gaussian-like near the center, with heavy tails and per-channel or per-block scale variation.

Quantization is therefore a mapping from a continuous variable $W \in \mathbb{R}$ to a discrete codebook of finite size.

The critical design question is:

$$
\text{How should finite code points be distributed over magnitude space?}
$$

GGML-style linear quantization and LNS answer this question differently.

## 2. Canonical GGML-Style Linear Quantization

For a standard blockwise linear quantizer, weights are mapped onto a uniformly spaced grid.

### Quantization map

In the simplest symmetric form:

$$
q = \operatorname{round}\!\left(\frac{W}{\Delta}\right)
$$

where:

- $W$ is the original weight
- $q$ is the quantized integer code
- $\Delta$ is the block scale or step size

Depending on the exact GGML format, there may also be an offset or blockwise affine correction, but the core property is unchanged: the codebook is evenly spaced in linear space.

### Dequantization map

$$
W' = q \cdot \Delta
$$

### Characteristic

Adjacent representable values differ by a constant amount:

$$
W'_{k+1} - W'_k = \Delta
$$

This means the maximum absolute error is uniformly bounded by:

$$
|W - W'| \le \frac{\Delta}{2}
$$

for the ideal symmetric rounding case.

## 3. Current LNS Q4_L Mathematics

The current `Q4_L` format in [lns-core/src/quant/q4l.rs](/Users/mapletechnologies/Desktop/big_projects/lns-ml/lns-core/src/quant/q4l.rs) is not a flat log quantizer. It is hierarchical.

### Storage layout

Each 256-weight super-block stores:

- `scale_global` as `f16`
- `scale_local[8]` packed into 4 bytes, one 4-bit local scale per 32-weight sub-block
- 256 packed 4-bit weight codes

So the effective quantization scale is not just one global factor. It is:

$$
\operatorname{efektif\_scale}_b = \operatorname{scale\_global} \cdot 2^{(s_b - 7)}
$$

where:

- $b$ is the sub-block index
- $s_b \in \{0,1,\dots,15\}$ is the 4-bit local scale code for that sub-block

### Global scale

For each 256-weight super-block:

$$
\operatorname{scale\_global} \approx \max_i |W_i|
$$

with the implementation storing it as `f16` after rounding.

### Local scale

For each 32-weight sub-block:

$$
s_b = \operatorname{clamp}_{[0,15]}\left(\operatorname{round}\left(\log_2\left(\frac{\max_{i \in b}|W_i|}{\operatorname{scale\_global}}\right) + 7\right)\right)
$$

This is the first major refinement over the old spec text. The quantizer does not force the whole 256-weight block to share one log grid. It first adapts the grid per 32-weight region.

### Weight code

Each weight stores:

- 1 sign bit
- 3 magnitude bits

The magnitude code is:

$$
M \in \{0,1,2,3,4,5,6,7\}
$$

with:

- $M = 0$ meaning exact zero
- $M \in \{1,\dots,7\}$ meaning a nonzero logarithmic level

The encoder computes:

$$
M =
\begin{cases}
0, & |W| \approx 0 \\
\min\left(7,\max\left(1,\operatorname{round}\left(\log_2\left(\frac{|W|}{\operatorname{efektif\_scale}_b}\right)+7\right)\right)\right), & \text{otherwise}
\end{cases}
$$

more precisely with the implementation collapsing non-positive rounded magnitudes to zero.

### Dequantization map

The current decoder reconstructs:

$$
W' = \operatorname{sign}(W) \cdot \mathbf{1}_{M \ne 0} \cdot \operatorname{efektif\_scale}_b \cdot 2^{(M-7)}
$$

This is the exact structural difference from the older one-line spec. The old form:

$$
W' = \operatorname{scale} \cdot 2^{M-7} \cdot \operatorname{sign}(W)
$$

was conceptually right, but incomplete. In the current implementation, `scale` is really:

$$
\operatorname{scale} = \operatorname{scale\_global} \cdot 2^{(s_b - 7)}
$$

and therefore the real decoded value is:

$$
W' = \operatorname{sign}(W) \cdot \mathbf{1}_{M \ne 0} \cdot \operatorname{scale\_global} \cdot 2^{(s_b - 7)} \cdot 2^{(M - 7)}
$$

or equivalently:

$$
W' = \operatorname{sign}(W) \cdot \mathbf{1}_{M \ne 0} \cdot \operatorname{scale\_global} \cdot 2^{(s_b + M - 14)}
$$

## 4. What Actually Changed Relative to the Original Spec Text

The original statement was mostly correct at the intuition level, but several points are now clearer.

### Change 1: LNS is hierarchical, not single-scale

Old framing:

- one `scale`
- one log mapping for the full region

Current implementation:

- one global outlier absorber per 256 weights
- one local log-scale adaptation per 32 weights
- one log-coded magnitude per weight

This matters because it further reduces the damage caused by mixed-magnitude blocks.

### Change 2: Zero is represented explicitly

The old prose implicitly treated LNS as a pure logarithmic grid. The implementation is more practical:

$$
M = 0 \Rightarrow W' = 0
$$

This gives exact sparsity handling and avoids meaningless log reconstruction near machine-zero.

### Change 3: The phrase “steps get denser near zero” needs refinement

That sentence is intuitively useful, but mathematically it can mislead if read literally.

Inside one sub-block, the nonzero levels are geometric:

$$
\operatorname{efektif\_scale}_b \cdot 2^{-6},
\operatorname{efektif\_scale}_b \cdot 2^{-5},
\dots,
\operatorname{efektif\_scale}_b \cdot 2^0
$$

So the spacing is not continuously densifying toward zero. Instead:

- zero is exact
- nonzero levels are exponentially spaced
- local scale adapts the level set to each 32-weight region

The more precise statement is:

> LNS allocates quantization resolution proportionally in magnitude space, not uniformly in linear space.

### Change 4: The old “single outlier breaks the block” claim is still true for linear quantization, but Q4_L now counters it in two stages

The original contrast said GGML suffers from outlier-driven weight squashing, while LNS does not.

The current implementation supports that claim more concretely than before because it has two protections:

1. `scale_global` absorbs super-block-wide outliers.
2. `scale_local` restores local resolution inside each 32-weight sub-block.

This means Q4_L does not merely avoid uniform linear squashing by using logarithms; it also avoids over-tying all 256 weights to one local resolution regime.

## 5. Error Analysis

## 5.1 GGML-Style Linear Quantization

For linear quantization, the error is naturally described in absolute terms.

If the step size is $\Delta$, then ideally:

$$
|W - W'| \le \frac{\Delta}{2}
$$

This is acceptable for larger values, but for small values the relative error becomes unstable:

$$
\frac{|W - W'|}{|W|}
$$

As $W \to 0$, this can blow up. In practical terms, a weight of size $0.003$ can be rounded to zero if the block scale was set by a much larger neighbor.

That is the mathematical core of “weight squashing.”

## 5.2 LNS Q4_L

For a logarithmic codebook, the more relevant error measure is multiplicative or relative distortion.

If a nonzero value is mapped to the nearest power-of-two level relative to the local effective scale, then the reconstruction ratio is approximately bounded by a multiplicative factor around the midpoint between adjacent log levels.

Since adjacent nonzero levels differ by a factor of $2$, nearest-log rounding implies a worst-case multiplicative distortion of approximately:

$$
2^{1/2}
$$

between codebook levels before considering the local-scale adaptation.

That sounds coarse in isolation, but Q4_L compensates in two ways:

- the local sub-block scale moves the active grid to the neighborhood of the data
- exact zero is available as a dedicated code

So the practical outcome is not “small absolute error everywhere.” It is:

- better preservation of relative structure across magnitudes
- less catastrophic collapse of near-zero-but-meaningful weights
- much stronger resilience to blockwise dynamic-range mismatch

This is the mathematically important distinction: LNS is not minimizing the same notion of error as linear quantization.

## 6. Why LNS Better Matches the Weight Distribution

If the weight distribution concentrates strongly near zero while spanning multiple orders of magnitude, then a uniform linear codebook spends too many code points describing large linear intervals that contain relatively little mass.

LNS instead distributes representable values across orders of magnitude.

In information-theoretic language, the codebook is closer to the geometry of the source distribution when magnitudes vary multiplicatively rather than additively.

This does not mean LNS is universally optimal, but it does mean that for heavy-tailed, zero-centered neural weight distributions, it is mathematically better aligned with the data than a uniformly spaced linear grid.

## 7. Hardware and Kernel Consequences

The original spec correctly identified the hardware tradeoff, and that still holds.

### GGML-style linear dequantization

Typical inner form:

$$
W' = q \cdot \Delta
$$

This maps naturally to cheap fused multiply-add style arithmetic.

### LNS dequantization

Current inner form:

$$
W' = \operatorname{sign}(W) \cdot \mathbf{1}_{M \ne 0} \cdot \operatorname{efektif\_scale}_b \cdot 2^{(M-7)}
$$

This introduces additional structure:

- sign extraction
- zero masking
- power-of-two magnitude reconstruction
- local-scale reconstruction

In the current implementation, that is still more compute-shaped than a purely linear dequantizer, even though the inner loop is branchless.

So the updated statement is:

- GGML is still closer to a memory-bandwidth-limited dequantization regime.
- LNS still pays extra arithmetic structure cost on conventional floating-point hardware.
- Q4_L reduces some of that cost by using branchless reconstruction and discrete exponent levels, but it does not yet eliminate the underlying exp-like reconstruction cost.

That is why the architectural upside of LNS remains tied to future fused kernels, table-based decode, or more specialized hardware paths.

## 8. Information-Theoretic Perspective

The original Shannon-entropy framing is still valid, but it should be stated more carefully.

Given a fixed number of states, a quantizer is better if it assigns its resolution where the source distribution actually has probability mass and task-relevant sensitivity.

Linear quantization distributes states uniformly in value space.
LNS distributes states approximately uniformly in log-magnitude space, with additional per-sub-block adaptation.

That means Q4_L spends more of its representational budget distinguishing multiplicative differences among small and medium magnitudes instead of reserving equal linear spacing for regions that may contain little useful density.

The practical claim is therefore not that LNS magically maximizes Shannon entropy in every formal sense. The safer and more precise statement is:

> For weight distributions with strong central concentration and multi-order dynamic range, Q4_L uses its finite codebook more in line with the empirical geometry of the data than a uniformly spaced linear quantizer.

## 9. Final Mathematical Judgment

The old headline conclusion remains directionally correct, but it is worth rewriting in code-accurate form.

### Old simplified conclusion

- GGML is a hardware-friendly linear quantizer.
- LNS is a mathematically distribution-aligned logarithmic quantizer.

### Updated conclusion

GGML-style quantization is a blockwise linear approximation scheme that optimizes implementation simplicity and cheap arithmetic, but it is vulnerable to dynamic-range mismatch and outlier-driven collapse of small weights.

The current `Q4_L` design is a hierarchical logarithmic quantizer that combines:

- super-block global scaling
- sub-block local log scaling
- per-weight signed log-coded magnitudes
- exact zero representation

Mathematically, that makes it much better suited to preserving relative structure in wide-dynamic-range, zero-centered neural weight distributions.

The remaining cost is hardware-side: conventional ALUs handle linear dequantization more naturally than hierarchical log reconstruction. That is the tradeoff.

## 10. One-Sentence Distillation

GGML compresses weights on a uniform linear grid.
`Q4_L` compresses them on a hierarchical logarithmic grid whose scale adapts to the local magnitude structure of the tensor.

That is the core mathematical difference.
