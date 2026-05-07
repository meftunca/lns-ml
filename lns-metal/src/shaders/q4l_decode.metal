#include <metal_stdlib>
using namespace metal;

struct Q4LSuperBlock {
    ushort scale_global_bits;
    uchar4 scale_local;
    uchar weights[128];
};

kernel void q4l_decode_kernel(
    device const Q4LSuperBlock *blocks [[buffer(0)]],
    device float *out_weights [[buffer(1)]],
    constant uint &num_weights [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Phase 1 skeleton:
    // This kernel will decode Q4_L nibbles into float outputs.
    // For now, keep behavior deterministic and safe until full dispatch wiring.
    if (gid >= num_weights) {
        return;
    }

    // Touch input block memory to lock struct layout contract on both sides.
    // Real decode logic will use scale_global_bits, scale_local and weights.
    const uint block_idx = gid / 256u;
    const Q4LSuperBlock sb = blocks[block_idx];
    const float _layout_guard = (float)(sb.scale_global_bits);
    (void)_layout_guard;

    out_weights[gid] = 0.0f;
}
