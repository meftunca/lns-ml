#include <metal_stdlib>
using namespace metal;

/// Q4_L super-block layout — must match the Rust `Q4LSuperBlock` #[repr(C)].
///
/// Total: 2 + 4 + 128 = 134 bytes per 256-weight block.
///
/// Weight nibble encoding:
///   bit 3      → sign  S  (0 = positive, 1 = negative)
///   bits 2..0  → magnitude M ∈ [0, 7]
///     M = 0            → exact zero (sparsity short-circuit)
///     M ∈ [1, 7]       → ±efektif_scale × 2^(M − 7)
///
/// Effective scale per sub-block (32 weights):
///   efektif_scale = scale_global × 2^(scale_local[i] − 7)
struct Q4LSuperBlock {
    ushort scale_global_bits;   // f16 global scale — stored as raw IEEE 754 half bits
    uchar4 scale_local;         // 8 × 4-bit local scales; 2 per byte, lo nibble first
    uchar  weights[128];        // 256 × 4-bit nibbles:  [sign(1)|mag(3)], 2 per byte
};

/// Return the 4-bit local scale for sub-block `sub_idx` (0..8).
static inline uchar get_scale_local(uchar4 sl, uint sub_idx) {
    uint byte_idx = sub_idx / 2u;
    uint shift    = (sub_idx % 2u) * 4u;
    return (sl[byte_idx] >> shift) & 0xFu;
}

/// Return the 4-bit nibble [sign(1)|mag(3)] for weight at `pos` (0..256) within a block.
static inline uchar get_nibble(device const uchar *w, uint pos) {
    uint byte_idx = pos / 2u;
    uint shift    = (pos % 2u) * 4u;
    return (w[byte_idx] >> shift) & 0xFu;
}

/// Q4_L decode kernel — one thread per output weight element.
///
/// Buffer slots:
///   0 — device const Q4LSuperBlock *blocks    (input super-blocks)
///   1 — device       float         *out       (output f32 weights, num_weights elements)
///   2 — constant     uint          &num_weights
kernel void q4l_decode_kernel(
    device const Q4LSuperBlock *blocks      [[ buffer(0) ]],
    device       float         *out         [[ buffer(1) ]],
    constant     uint          &num_weights [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= num_weights) { return; }

    // ── Locate super-block and position within it ─────────────────────────
    const uint block_idx    = gid / 256u;
    const uint pos_in_block = gid % 256u;
    const uint sub_idx      = pos_in_block / 32u;   // sub-block index 0..8

    device const Q4LSuperBlock &sb = blocks[block_idx];

    // ── Decode scale_global: f16 bit-pattern → f32 ────────────────────────
    const float scale_global = (float)as_type<half>(sb.scale_global_bits);

    // ── Effective scale for this sub-block ────────────────────────────────
    // efektif_scale = scale_global × 2^(sl − 7)
    const uchar sl            = get_scale_local(sb.scale_local, sub_idx);
    const float efektif_scale = scale_global * powr(2.0f, (float)sl - 7.0f);

    // ── Decode weight nibble ──────────────────────────────────────────────
    const uchar nibble   = get_nibble(sb.weights, pos_in_block);
    const uchar sign_bit = (nibble >> 3u) & 1u;
    const uchar m        = nibble & 0x7u;

    // ── Branchless dequantization (M = 0 → 0.0 via is_nonzero mask) ──────
    const float is_nonzero = (m != 0u) ? 1.0f : 0.0f;
    const float magnitude  = efektif_scale * powr(2.0f, (float)m - 7.0f);
    const float sign       = (sign_bit != 0u) ? -1.0f : 1.0f;

    out[gid] = magnitude * sign * is_nonzero;
}
