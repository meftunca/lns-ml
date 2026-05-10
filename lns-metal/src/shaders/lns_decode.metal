#include <metal_stdlib>
using namespace metal;

// ── Yapısal Hizalama ─────────────────────────────────────────────────────────

struct LnsSuperBlockHeader {
    half scale_global;
    uchar scale_local[4];   
    ushort padding;         
};

struct alignas(16) block_q4_l {
    half  efektif_scale[8]; // 16 Byte
    uchar qs[128];          // 128 Byte
};

// ── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

// LNS exponent lookup (Q4_L, spec v3): exp2(m - δ) with δ = 5 for m in [1,7].
// m=0 is exact zero. Must match the encoder constants `MAG_OFFSET = 5` and
// `GRID = {0, 1/16, 1/8, 1/4, 1/2, 1, 2, 4}` in lns-core/src/quant/q4l.rs.
//
// Spec v1 used δ=7 → grid {1/64..1}. Empirically that left 95% of weights
// crowded into 3-4 levels and clipped the upper tail. v3 (δ=5) covers
// [1/16, 4], aligning the dense bulk to four grid points and giving the
// outliers four octaves of headroom before clipping into m=7.
constant float LNS_EXP_LUT[8] = {
    0.0f,      // m=0 → exact zero
    0.0625f,   // m=1 → 2^-4
    0.125f,    // m=2 → 2^-3
    0.25f,     // m=3 → 2^-2
    0.5f,      // m=4 → 2^-1
    1.0f,      // m=5 → 2^0
    2.0f,      // m=6 → 2^1
    4.0f,      // m=7 → 2^2
};

inline float2 lns_decode_byte(uchar raw_byte, float scale) {
    const uint m0   = raw_byte & 7u;
    const float sgn0 = 1.0f - 2.0f * (float)((raw_byte >> 3u) & 1u);
    const float val0 = sgn0 * scale * LNS_EXP_LUT[m0];

    const uint m1   = (raw_byte >> 4u) & 7u;
    const float sgn1 = 1.0f - 2.0f * (float)(raw_byte >> 7u);
    const float val1 = sgn1 * scale * LNS_EXP_LUT[m1];

    return float2(val0, val1);
}

constant float Q4HQ_NF4_Z_CODEBOOK[16] = {
    -1.0000000f, -0.6961928f, -0.5250731f, -0.3949175f,
    -0.2844414f, -0.1847734f, -0.0910500f,  0.0000000f,
     0.0795803f,  0.1609302f,  0.2461123f,  0.3379152f,
     0.4407098f,  0.5626170f,  0.7229568f,  1.0000000f,
};

inline float2 q4hq_decode_byte(uchar raw_byte, float scale) {
    const uint lo = raw_byte & 0x0fu;
    const uint hi = raw_byte >> 4u;
    return float2(
        scale * Q4HQ_NF4_Z_CODEBOOK[lo],
        scale * Q4HQ_NF4_Z_CODEBOOK[hi]
    );
}

inline float silu_f32(float x) {
    return x / (1.0f + exp(-x));
}

inline float softplus_f32(float x) {
    return x > 20.0f ? x : log(1.0f + exp(x));
}

kernel void rmsnorm_kernel(
    device const float *x             [[buffer(0)]],
    device const float *w             [[buffer(1)]],
    device float       *out           [[buffer(2)]],
    constant uint      &dim           [[buffer(3)]],
    constant float     &eps           [[buffer(4)]],
    constant uint      &zero_centered [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    // 256 threads = 8 warps of 32.  simd_sum reduces within each warp (no barrier).
    // One threadgroup barrier merges 8 warp sums.
    threadgroup float warp_sums[8];
    const uint warp_id = tid >> 5u;
    const uint lane_id = tid & 31u;

    float acc = 0.0f;
    for (uint idx = tid; idx < dim; idx += 256u) {
        const float v = x[idx];
        acc += v * v;
    }
    // Reduce within warp (no barrier needed)
    acc = simd_sum(acc);
    if (lane_id == 0u) warp_sums[warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 sums the 8 warp accumulators
    float total = 0.0f;
    if (tid == 0u) {
        for (uint w = 0u; w < 8u; ++w) total += warp_sums[w];
        warp_sums[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = rsqrt(warp_sums[0] / (float)dim + eps);
    for (uint idx = tid; idx < dim; idx += 256u) {
        const float scale = zero_centered != 0u ? (1.0f + w[idx]) : w[idx];
        out[idx] = x[idx] * inv_rms * scale;
    }
}

kernel void rmsnorm_rows_kernel(
    device float       *x             [[buffer(0)]],
    device const float *w             [[buffer(1)]],
    constant uint      &row_dim       [[buffer(2)]],
    constant uint      &rows          [[buffer(3)]],
    constant float     &eps           [[buffer(4)]],
    constant uint      &zero_centered [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row_idx [[threadgroup_position_in_grid]]
) {
    if (row_idx >= rows) return;

    threadgroup float warp_sums[8];
    const uint warp_id = tid >> 5u;
    const uint lane_id = tid & 31u;
    const uint row_base = row_idx * row_dim;

    float acc = 0.0f;
    for (uint idx = tid; idx < row_dim; idx += 256u) {
        const float v = x[row_base + idx];
        acc += v * v;
    }
    acc = simd_sum(acc);
    if (lane_id == 0u) warp_sums[warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    if (tid == 0u) {
        for (uint w = 0u; w < 8u; ++w) total += warp_sums[w];
        warp_sums[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = rsqrt(warp_sums[0] / (float)row_dim + eps);
    for (uint idx = tid; idx < row_dim; idx += 256u) {
        const float scale = zero_centered != 0u ? (1.0f + w[idx]) : w[idx];
        x[row_base + idx] = x[row_base + idx] * inv_rms * scale;
    }
}

// ── RoPE with precomputed per-dimension inverse frequencies ────────────────
// Supports YaRN, LLaMA3-extended RoPE, and any scaling mode where inv_freqs[i]
// differs from the plain base-theta formula.  Precomputed on the Rust side at
// model load time; angle_i = pos * inv_freqs[i].
kernel void rope_with_freqs(
    device float       *q          [[buffer(0)]],
    device float       *k          [[buffer(1)]],
    constant uint      &pos        [[buffer(2)]],
    constant uint      &head_dim   [[buffer(3)]],
    constant uint      &rotary_dim [[buffer(4)]],
    device const float *inv_freqs  [[buffer(5)]],  // length = rotary_dim / 2
    constant uint      &q_heads    [[buffer(6)]],
    constant uint      &k_heads    [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint half_rotary_dim = rotary_dim >> 1u;
    if (half_rotary_dim == 0u) return;

    const uint i = gid % half_rotary_dim;
    const uint h = gid / half_rotary_dim;

    const float theta = (float)pos * inv_freqs[i];
    float cos_t;
    const float sin_t = sincos(theta, cos_t);

    if (h < q_heads) {
        const uint base = h * head_dim + i;
        const float2 val = float2(q[base], q[base + half_rotary_dim]);
        q[base]                   = val.x * cos_t - val.y * sin_t;
        q[base + half_rotary_dim] = val.x * sin_t + val.y * cos_t;
    }
    if (h < k_heads) {
        const uint base = h * head_dim + i;
        const float2 val = float2(k[base], k[base + half_rotary_dim]);
        k[base]                   = val.x * cos_t - val.y * sin_t;
        k[base + half_rotary_dim] = val.x * sin_t + val.y * cos_t;
    }
}

kernel void split_gated_query_kernel(
    device const float *input   [[buffer(0)]],
    device float       *q_out   [[buffer(1)]],
    device float       *gate_out[[buffer(2)]],
    constant uint      &dim     [[buffer(3)]],
    constant uint      &head_dim[[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dim || head_dim == 0u) {
        return;
    }
    const uint head_idx = gid / head_dim;
    const uint lane = gid % head_dim;
    const uint base = head_idx * (head_dim * 2u) + lane;
    q_out[gid] = input[base];
    gate_out[gid] = 1.0f / (1.0f + exp(-input[base + head_dim]));
}

kernel void mul_inplace_kernel(
    device float       *x    [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    constant uint      &dim  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dim) {
        return;
    }
    x[gid] *= gate[gid];
}

kernel void add_inplace_kernel(
    device float       *x   [[buffer(0)]],
    device const float *y   [[buffer(1)]],
    constant uint      &dim [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dim) {
        return;
    }
    x[gid] += y[gid];
}

// ── Flash Attention Decode ──────────────────────────────────────────────────
//
// Online-softmax single-query attention (Flash-Decoding style).
// Supports any sequence length — no fixed scores buffer.
// KV cache stored in float16 to halve memory usage.
// threadgroup size: 128 threads; one threadgroup per Q head.
//
// NOTE on KV cache quantization: Q8L (logarithmic) is a poor fit for KV cache
// activations because the ~8.3% step size near the max causes attention score
// errors > 8%, breaking greedy decoding.  F16 (~0.05% step) is used here.
// A dedicated activation-quantization format (e.g., linear int8 or "Q8A") is
// planned as a future improvement once the forward-pass quality is stable.
//
// Online softmax recurrence (Milakov & Gimelshein, 2018):
//   For each position p:
//     m_new = max(m, s_p)
//     alpha  = exp(m - m_new)
//     beta   = exp(s_p - m_new)
//     acc[i] = alpha * acc[i] + beta * V[p][i]
//     l_new  = alpha * l + beta
//   output[i] = acc[i] / l

#define FA_THREADS 128u
#define FA_MAX_HEAD_DIM 256u

kernel void flash_attention_decode(
    device const float *q          [[buffer(0)]],  // [n_heads * head_dim] f32
    device const float *k_in       [[buffer(1)]],  // [kv_dim] f32 — current-token K
    device const float *v_in       [[buffer(2)]],  // [kv_dim] f32 — current-token V
    device half        *k_cache    [[buffer(3)]],  // [cache_len * kv_dim] f16
    device half        *v_cache    [[buffer(4)]],  // [cache_len * kv_dim] f16
    device float       *out        [[buffer(5)]],  // [n_heads * head_dim] f32
    constant uint      &pos        [[buffer(6)]],
    constant uint      &head_dim   [[buffer(7)]],
    constant uint      &n_heads    [[buffer(8)]],
    constant uint      &n_kv_heads [[buffer(9)]],
    constant uint      &window_size[[buffer(10)]], // 0 = full attention; >0 = SWA ring buffer
    uint tid [[thread_position_in_threadgroup]],
    uint hid [[threadgroup_position_in_grid]]   // Q head index
) {
    const uint rep    = n_heads / n_kv_heads;
    const uint kv_h   = hid / rep;
    const uint kv_dim = n_kv_heads * head_dim;
    const float scale = rsqrt((float)head_dim);

    // ── 1. Write current K, V to cache (f32 → f16) ──────────────────────
    // For SWA: ring-buffer write at (pos % window_size).
    // For full attention: linear write at pos.
    const uint write_slot = (window_size > 0u) ? (pos % window_size) : pos;
    for (uint i = tid; i < head_dim; i += FA_THREADS) {
        k_cache[write_slot * kv_dim + kv_h * head_dim + i] = (half)k_in[kv_h * head_dim + i];
        v_cache[write_slot * kv_dim + kv_h * head_dim + i] = (half)v_in[kv_h * head_dim + i];
    }
    threadgroup_barrier(mem_flags::mem_device);   // ensure write visible to loop below

    // ── 2. Online-softmax attention scan ────────────────────────────────
    // For SWA: only attend to the last window_size tokens.
    const uint p_start = (window_size > 0u && pos >= window_size) ? (pos - window_size + 1u) : 0u;

    threadgroup float acc[FA_MAX_HEAD_DIM];      // running output accumulator
    threadgroup float stats[2];                  // [0]=m (running max), [1]=l (running sum)

    // Initialise
    for (uint i = tid; i < head_dim; i += FA_THREADS) acc[i] = 0.0f;
    if (tid == 0) { stats[0] = -1e38f; stats[1] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float *qh = q + hid * head_dim;

    for (uint p = p_start; p <= pos; ++p) {
        const uint cache_slot = (window_size > 0u) ? (p % window_size) : p;
        device const half *kh = k_cache + cache_slot * kv_dim + kv_h * head_dim;
        device const half *vh = v_cache + cache_slot * kv_dim + kv_h * head_dim;

        // Reduce dot Q·K[p] across 4 warps: simd_sum within warp (free),
        // store to TG array, 1 barrier, then all threads sum all 4 entries.
        threadgroup float warp_sums_fa[4];
        float dot = 0.0f;
        for (uint i = tid; i < head_dim; i += FA_THREADS) {
            dot += qh[i] * (float)kh[i];
        }
        dot = simd_sum(dot);
        if ((tid & 31u) == 0u) warp_sums_fa[tid >> 5u] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // All threads read all 4 warp sums — same value, no extra barrier needed
        float score = (warp_sums_fa[0] + warp_sums_fa[1] +
                       warp_sums_fa[2] + warp_sums_fa[3]) * scale;

        // Online softmax update
        float m     = stats[0];
        float l     = stats[1];
        float m_new = max(m, score);
        float alpha = exp(m - m_new);
        float beta  = exp(score - m_new);

        // acc = alpha * acc + beta * V[p];   output in head_dim chunks
        for (uint i = tid; i < head_dim; i += FA_THREADS) {
            acc[i] = alpha * acc[i] + beta * (float)vh[i];
        }
        if (tid == 0) {
            stats[0] = m_new;
            stats[1] = alpha * l + beta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── 3. Normalise and write output ───────────────────────────────────
    float l_inv = 1.0f / stats[1];
    for (uint i = tid; i < head_dim; i += FA_THREADS) {
        out[hid * head_dim + i] = acc[i] * l_inv;
    }
}

// ── Flash Attention Decode — Q8A KV Cache ─────────────────────────────────
//
// Same online-softmax decode as flash_attention_decode, but the KV cache is
// stored as signed int8 with a per-head-per-position f32 scale factor.
//
// Format:
//   k_cache  : [cache_len * kv_dim]   i8   — quantized K values
//   v_cache  : [cache_len * kv_dim]   i8   — quantized V values
//   k_scales : [cache_len * n_kv_heads] f32 — scale per (slot, kv_head)
//   v_scales : [cache_len * n_kv_heads] f32 — scale per (slot, kv_head)
//
// Quantization: x_i8 = round(clamp(x / scale, -127, 127))
//               scale = max(|x_j|) / 127   (0 → 1 to avoid div-by-zero)
// Dequantization: x_f32 = x_i8 * scale
//
// Memory vs F16: for head_dim=256, n_kv_heads=4 (Qwen3.5-4B):
//   F16:  cache_len * kv_dim * 2  bytes
//   Q8A:  cache_len * kv_dim * 1  bytes + cache_len * n_kv_heads * 4  bytes
//   Savings: ~49%  (scale overhead < 1%)
//
kernel void flash_attention_q8a(
    device const float  *q           [[buffer(0)]],   // [n_heads * head_dim] f32
    device const float  *k_in        [[buffer(1)]],   // [kv_dim] f32 — current-token K
    device const float  *v_in        [[buffer(2)]],   // [kv_dim] f32 — current-token V
    device char         *k_cache     [[buffer(3)]],   // [cache_len * kv_dim] i8
    device char         *v_cache     [[buffer(4)]],   // [cache_len * kv_dim] i8
    device float        *k_scales    [[buffer(5)]],   // [cache_len * n_kv_heads] f32
    device float        *v_scales    [[buffer(6)]],   // [cache_len * n_kv_heads] f32
    device float        *out         [[buffer(7)]],   // [n_heads * head_dim] f32
    constant uint       &pos         [[buffer(8)]],
    constant uint       &head_dim    [[buffer(9)]],
    constant uint       &n_heads     [[buffer(10)]],
    constant uint       &n_kv_heads  [[buffer(11)]],
    constant uint       &window_size [[buffer(12)]],  // 0 = full attention; >0 = SWA
    uint tid [[thread_position_in_threadgroup]],
    uint hid [[threadgroup_position_in_grid]]
) {
    const uint rep    = n_heads / n_kv_heads;
    const uint kv_h   = hid / rep;
    const uint kv_dim = n_kv_heads * head_dim;
    const float scale = rsqrt((float)head_dim);

    const uint write_slot = (window_size > 0u) ? (pos % window_size) : pos;

    // ── 1. Quantize and write K ──────────────────────────────────────────
    threadgroup float kvq_warp_sums[4];

    float local_max = 0.0f;
    for (uint i = tid; i < head_dim; i += FA_THREADS) {
        local_max = max(local_max, abs(k_in[kv_h * head_dim + i]));
    }
    local_max = simd_max(local_max);
    if ((tid & 31u) == 0u) kvq_warp_sums[tid >> 5u] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // All threads compute the same k_max by reading all 4 entries
    float k_max = max(max(kvq_warp_sums[0], kvq_warp_sums[1]),
                      max(kvq_warp_sums[2], kvq_warp_sums[3]));
    float k_scale = k_max == 0.0f ? 1.0f : k_max / 127.0f;
    if (tid == 0) k_scales[write_slot * n_kv_heads + kv_h] = k_scale;
    // Vectorized char4 write — head_dim is always a multiple of 4
    const uint kv_base_k = write_slot * kv_dim + kv_h * head_dim;
    for (uint i4 = tid * 4u; i4 + 3u < head_dim; i4 += FA_THREADS * 4u) {
        const uint src = kv_h * head_dim + i4;
        char4 out4;
        out4.x = (char)clamp(round(k_in[src    ] / k_scale), -127.0f, 127.0f);
        out4.y = (char)clamp(round(k_in[src + 1] / k_scale), -127.0f, 127.0f);
        out4.z = (char)clamp(round(k_in[src + 2] / k_scale), -127.0f, 127.0f);
        out4.w = (char)clamp(round(k_in[src + 3] / k_scale), -127.0f, 127.0f);
        *(device char4*)(k_cache + kv_base_k + i4) = out4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 2. Quantize and write V ──────────────────────────────────────────
    float local_max_v = 0.0f;
    for (uint i = tid; i < head_dim; i += FA_THREADS) {
        local_max_v = max(local_max_v, abs(v_in[kv_h * head_dim + i]));
    }
    local_max_v = simd_max(local_max_v);
    if ((tid & 31u) == 0u) kvq_warp_sums[tid >> 5u] = local_max_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v_max = max(max(kvq_warp_sums[0], kvq_warp_sums[1]),
                      max(kvq_warp_sums[2], kvq_warp_sums[3]));
    float v_scale = v_max == 0.0f ? 1.0f : v_max / 127.0f;
    if (tid == 0) v_scales[write_slot * n_kv_heads + kv_h] = v_scale;
    const uint kv_base_v = write_slot * kv_dim + kv_h * head_dim;
    for (uint i4 = tid * 4u; i4 + 3u < head_dim; i4 += FA_THREADS * 4u) {
        const uint src = kv_h * head_dim + i4;
        char4 out4;
        out4.x = (char)clamp(round(v_in[src    ] / v_scale), -127.0f, 127.0f);
        out4.y = (char)clamp(round(v_in[src + 1] / v_scale), -127.0f, 127.0f);
        out4.z = (char)clamp(round(v_in[src + 2] / v_scale), -127.0f, 127.0f);
        out4.w = (char)clamp(round(v_in[src + 3] / v_scale), -127.0f, 127.0f);
        *(device char4*)(v_cache + kv_base_v + i4) = out4;
    }
    threadgroup_barrier(mem_flags::mem_device);   // all cache writes visible to loop below

    // ── 3. Online-softmax attention scan ────────────────────────────────
    const uint p_start = (window_size > 0u && pos >= window_size) ? (pos - window_size + 1u) : 0u;

    threadgroup float acc[FA_MAX_HEAD_DIM];
    threadgroup float stats[2];   // [0]=m, [1]=l

    for (uint i = tid; i < head_dim; i += FA_THREADS) acc[i] = 0.0f;
    if (tid == 0) { stats[0] = -1e38f; stats[1] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float *qh = q + hid * head_dim;

    for (uint p = p_start; p <= pos; ++p) {
        const uint cs = (window_size > 0u) ? (p % window_size) : p;
        device const char *kh = k_cache + cs * kv_dim + kv_h * head_dim;
        float ks = k_scales[cs * n_kv_heads + kv_h];

        // Reduce dot Q·K[p] across 4 warps
        threadgroup float warp_sums_fa[4];
        float dot = 0.0f;
        for (uint i = tid; i < head_dim; i += FA_THREADS) {
            dot += qh[i] * ((float)kh[i] * ks);
        }
        dot = simd_sum(dot);
        if ((tid & 31u) == 0u) warp_sums_fa[tid >> 5u] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score = (warp_sums_fa[0] + warp_sums_fa[1] +
                       warp_sums_fa[2] + warp_sums_fa[3]) * scale;

        float m     = stats[0];
        float l     = stats[1];
        float m_new = max(m, score);
        float alpha = exp(m - m_new);
        float beta  = exp(score - m_new);

        device const char *vh = v_cache + cs * kv_dim + kv_h * head_dim;
        const float beta_vs = beta * v_scales[cs * n_kv_heads + kv_h];
        for (uint i = tid; i < head_dim; i += FA_THREADS) {
            acc[i] = alpha * acc[i] + beta_vs * (float)vh[i];
        }
        if (tid == 0) {
            stats[0] = m_new;
            stats[1] = alpha * l + beta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── 4. Normalise and write output ────────────────────────────────────
    float l_inv = 1.0f / stats[1];
    for (uint i = tid; i < head_dim; i += FA_THREADS) {
        out[hid * head_dim + i] = acc[i] * l_inv;
    }
}

// ── Paged Q8A Flash Attention ─────────────────────────────────────────────
// Same int8+scale KV quantization as flash_attention_q8a, but KV cache is
// addressed through a block table: logical_page → physical_page.
// This decouples the virtual position space from physical memory layout,
// enabling just-in-time page allocation and SWA page eviction on the CPU.
//
// Buffer layout:
//   0: q (float*)              [n_heads * head_dim]
//   1: k_in (float*)           [kv_dim]   — current token K
//   2: v_in (float*)           [kv_dim]   — current token V
//   3: k_cache (char*)         [num_phys_pages * PAGE_SZ * kv_dim]  i8
//   4: v_cache (char*)         idem
//   5: k_scales (float*)       [num_phys_pages * PAGE_SZ * n_kv_heads]
//   6: v_scales (float*)       idem
//   7: block_table (uint*)     [MAX_LOGICAL_PAGES]  logical→physical
//   8: out (float*)            [n_heads * head_dim]
//   9..13: pos, head_dim, n_heads, n_kv_heads, p_start (uint constants)
//
#define PAGED_PAGE_SZ 16u

kernel void flash_attention_paged_q8a(
    device const float  *q           [[buffer(0)]],
    device const float  *k_in        [[buffer(1)]],
    device const float  *v_in        [[buffer(2)]],
    device char         *k_cache     [[buffer(3)]],
    device char         *v_cache     [[buffer(4)]],
    device float        *k_scales    [[buffer(5)]],
    device float        *v_scales    [[buffer(6)]],
    device const uint   *block_table [[buffer(7)]],
    device float        *out         [[buffer(8)]],
    constant uint       &pos         [[buffer(9)]],
    constant uint       &head_dim    [[buffer(10)]],
    constant uint       &n_heads     [[buffer(11)]],
    constant uint       &n_kv_heads  [[buffer(12)]],
    constant uint       &p_start     [[buffer(13)]],
    uint tid [[thread_position_in_threadgroup]],
    uint hid [[threadgroup_position_in_grid]]
) {
    const uint rep    = n_heads / n_kv_heads;
    const uint kv_h   = hid / rep;
    const uint kv_dim = n_kv_heads * head_dim;
    const float qk_scale = rsqrt((float)head_dim);

    // Physical slot for the current write position
    const uint write_lp   = pos / PAGED_PAGE_SZ;
    const uint write_off  = pos % PAGED_PAGE_SZ;
    const uint write_phys = block_table[write_lp];
    const uint write_slot = write_phys * PAGED_PAGE_SZ + write_off;

    // ── 1. Quantize and write K ──────────────────────────────────────────
    threadgroup float kvq_warp_sums2[4];

    float local_max = 0.0f;
    for (uint i = tid; i < head_dim; i += FA_THREADS)
        local_max = max(local_max, abs(k_in[kv_h * head_dim + i]));
    local_max = simd_max(local_max);
    if ((tid & 31u) == 0u) kvq_warp_sums2[tid >> 5u] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_max = max(max(kvq_warp_sums2[0], kvq_warp_sums2[1]),
                      max(kvq_warp_sums2[2], kvq_warp_sums2[3]));
    float k_scale = k_max == 0.0f ? 1.0f : k_max / 127.0f;
    if (tid == 0) k_scales[write_slot * n_kv_heads + kv_h] = k_scale;
    const uint kv_base_k2 = write_slot * kv_dim + kv_h * head_dim;
    for (uint i4 = tid * 4u; i4 + 3u < head_dim; i4 += FA_THREADS * 4u) {
        const uint src = kv_h * head_dim + i4;
        char4 out4;
        out4.x = (char)clamp(round(k_in[src    ] / k_scale), -127.0f, 127.0f);
        out4.y = (char)clamp(round(k_in[src + 1] / k_scale), -127.0f, 127.0f);
        out4.z = (char)clamp(round(k_in[src + 2] / k_scale), -127.0f, 127.0f);
        out4.w = (char)clamp(round(k_in[src + 3] / k_scale), -127.0f, 127.0f);
        *(device char4*)(k_cache + kv_base_k2 + i4) = out4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 2. Quantize and write V ──────────────────────────────────────────
    float local_max_v = 0.0f;
    for (uint i = tid; i < head_dim; i += FA_THREADS)
        local_max_v = max(local_max_v, abs(v_in[kv_h * head_dim + i]));
    local_max_v = simd_max(local_max_v);
    if ((tid & 31u) == 0u) kvq_warp_sums2[tid >> 5u] = local_max_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v_max = max(max(kvq_warp_sums2[0], kvq_warp_sums2[1]),
                      max(kvq_warp_sums2[2], kvq_warp_sums2[3]));
    float v_scale = v_max == 0.0f ? 1.0f : v_max / 127.0f;
    if (tid == 0) v_scales[write_slot * n_kv_heads + kv_h] = v_scale;
    const uint kv_base_v2 = write_slot * kv_dim + kv_h * head_dim;
    for (uint i4 = tid * 4u; i4 + 3u < head_dim; i4 += FA_THREADS * 4u) {
        const uint src = kv_h * head_dim + i4;
        char4 out4;
        out4.x = (char)clamp(round(v_in[src    ] / v_scale), -127.0f, 127.0f);
        out4.y = (char)clamp(round(v_in[src + 1] / v_scale), -127.0f, 127.0f);
        out4.z = (char)clamp(round(v_in[src + 2] / v_scale), -127.0f, 127.0f);
        out4.w = (char)clamp(round(v_in[src + 3] / v_scale), -127.0f, 127.0f);
        *(device char4*)(v_cache + kv_base_v2 + i4) = out4;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ── 3. Online-softmax attention scan (paged) ─────────────────────────
    threadgroup float acc[FA_MAX_HEAD_DIM];
    threadgroup float stats[2];   // [0]=m, [1]=l

    for (uint i = tid; i < head_dim; i += FA_THREADS) acc[i] = 0.0f;
    if (tid == 0) { stats[0] = -1e38f; stats[1] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float *qh = q + hid * head_dim;

    for (uint p = p_start; p <= pos; ++p) {
        const uint lp   = p / PAGED_PAGE_SZ;
        const uint loff = p % PAGED_PAGE_SZ;
        const uint phys = block_table[lp];
        const uint slot = phys * PAGED_PAGE_SZ + loff;

        device const char *kh = k_cache + slot * kv_dim + kv_h * head_dim;
        float ks = k_scales[slot * n_kv_heads + kv_h];

        // Reduce dot Q·K[p] across 4 warps
        threadgroup float warp_sums_fa2[4];
        float dot = 0.0f;
        for (uint i = tid; i < head_dim; i += FA_THREADS)
            dot += qh[i] * ((float)kh[i] * ks);
        dot = simd_sum(dot);
        if ((tid & 31u) == 0u) warp_sums_fa2[tid >> 5u] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score = (warp_sums_fa2[0] + warp_sums_fa2[1] +
                       warp_sums_fa2[2] + warp_sums_fa2[3]) * qk_scale;

        float m     = stats[0];
        float l     = stats[1];
        float m_new = max(m, score);
        float alpha = exp(m - m_new);
        float beta  = exp(score - m_new);

        device const char *vh = v_cache + slot * kv_dim + kv_h * head_dim;
        const float beta_vs = beta * v_scales[slot * n_kv_heads + kv_h];
        for (uint i = tid; i < head_dim; i += FA_THREADS)
            acc[i] = alpha * acc[i] + beta_vs * (float)vh[i];
        if (tid == 0) {
            stats[0] = m_new;
            stats[1] = alpha * l + beta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── 4. Normalise and write output ────────────────────────────────────
    float l_inv = 1.0f / stats[1];
    for (uint i = tid; i < head_dim; i += FA_THREADS)
        out[hid * head_dim + i] = acc[i] * l_inv;
}

// ── GEMV ───────────────────────────────────────────────────────────────────

kernel void q4l_gemv_optimized(
    device const block_q4_l *blocks  [[ buffer(0) ]],
    device const float      *x       [[ buffer(1) ]],
    device       float      *y       [[ buffer(2) ]],
    constant     uint       &ncols   [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    const uint row_idx = bid;
    const uint blocks_per_row = ncols >> 8u;
    const uint row_block_start = row_idx * blocks_per_row;
    
    const uint warp_id = tid >> 5u;
    const uint lane_id = tid & 31u;
    
    float acc = 0.0f;

    for (uint b = warp_id; b < blocks_per_row; b += 8u) {
        device const block_q4_l &block = blocks[row_block_start + b];
        const float scale = (float)block.efektif_scale[lane_id >> 2u];
        uchar4 w_bytes = *(device const uchar4*)(block.qs + (lane_id << 2u));

        const uint col_base = (b << 8u) + (lane_id << 3u);
        device const float *x_ptr = x + col_base;
        float4 x_vals   = *(device const float4*)(x_ptr);
        float4 x_vals_2 = *(device const float4*)(x_ptr + 4);

        float2 w01 = lns_decode_byte(w_bytes[0], scale);
        float2 w23 = lns_decode_byte(w_bytes[1], scale);
        float2 w45 = lns_decode_byte(w_bytes[2], scale);
        float2 w67 = lns_decode_byte(w_bytes[3], scale);

        acc += w01[0] * x_vals[0] + w01[1] * x_vals[1];
        acc += w23[0] * x_vals[2] + w23[1] * x_vals[3];
        acc += w45[0] * x_vals_2[0] + w45[1] * x_vals_2[1];
        acc += w67[0] * x_vals_2[2] + w67[1] * x_vals_2[3];
    }

    acc = simd_sum(acc);
    threadgroup float warp_sums[8]; 
    if (lane_id == 0u) warp_sums[warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0u) {
        float final_acc = 0.0f;
        for (uint i = 0; i < 8u; ++i) final_acc += warp_sums[i];
        y[row_idx] = final_acc;
    }
}

kernel void q4l_gemv_accumulate_optimized(
    device const block_q4_l *blocks  [[ buffer(0) ]],
    device const float      *x       [[ buffer(1) ]],
    device       float      *y       [[ buffer(2) ]],
    constant     uint       &ncols   [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    const uint row_idx = bid;
    const uint blocks_per_row = ncols >> 8u;
    const uint row_block_start = row_idx * blocks_per_row;

    const uint warp_id = tid >> 5u;
    const uint lane_id = tid & 31u;

    float acc = 0.0f;

    for (uint b = warp_id; b < blocks_per_row; b += 8u) {
        device const block_q4_l &block = blocks[row_block_start + b];
        const float scale = (float)block.efektif_scale[lane_id >> 2u];
        uchar4 w_bytes = *(device const uchar4*)(block.qs + (lane_id << 2u));

        const uint col_base = (b << 8u) + (lane_id << 3u);
        device const float *x_ptr = x + col_base;
        float4 x_vals   = *(device const float4*)(x_ptr);
        float4 x_vals_2 = *(device const float4*)(x_ptr + 4);

        float2 w01 = lns_decode_byte(w_bytes[0], scale);
        float2 w23 = lns_decode_byte(w_bytes[1], scale);
        float2 w45 = lns_decode_byte(w_bytes[2], scale);
        float2 w67 = lns_decode_byte(w_bytes[3], scale);

        acc += w01[0] * x_vals[0] + w01[1] * x_vals[1];
        acc += w23[0] * x_vals[2] + w23[1] * x_vals[3];
        acc += w45[0] * x_vals_2[0] + w45[1] * x_vals_2[1];
        acc += w67[0] * x_vals_2[2] + w67[1] * x_vals_2[3];
    }

    acc = simd_sum(acc);
    threadgroup float warp_sums[8];
    if (lane_id == 0u) warp_sums[warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float final_acc = 0.0f;
        for (uint i = 0; i < 8u; ++i) final_acc += warp_sums[i];
        y[row_idx] += final_acc;
    }
}

kernel void q4hq_gemv_optimized(
    device const block_q4_l *blocks  [[ buffer(0) ]],
    device const float      *x       [[ buffer(1) ]],
    device       float      *y       [[ buffer(2) ]],
    constant     uint       &ncols   [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    const uint row_idx = bid;
    const uint blocks_per_row = ncols >> 8u;
    const uint row_block_start = row_idx * blocks_per_row;

    const uint warp_id = tid >> 5u;
    const uint lane_id = tid & 31u;

    float acc = 0.0f;

    for (uint b = warp_id; b < blocks_per_row; b += 8u) {
        device const block_q4_l &block = blocks[row_block_start + b];
        const float scale = (float)block.efektif_scale[lane_id >> 2u];
        uchar4 w_bytes = *(device const uchar4*)(block.qs + (lane_id << 2u));

        const uint col_base = (b << 8u) + (lane_id << 3u);
        device const float *x_ptr = x + col_base;
        float4 x_vals   = *(device const float4*)(x_ptr);
        float4 x_vals_2 = *(device const float4*)(x_ptr + 4);

        float2 w01 = q4hq_decode_byte(w_bytes[0], scale);
        float2 w23 = q4hq_decode_byte(w_bytes[1], scale);
        float2 w45 = q4hq_decode_byte(w_bytes[2], scale);
        float2 w67 = q4hq_decode_byte(w_bytes[3], scale);

        acc += w01[0] * x_vals[0] + w01[1] * x_vals[1];
        acc += w23[0] * x_vals[2] + w23[1] * x_vals[3];
        acc += w45[0] * x_vals_2[0] + w45[1] * x_vals_2[1];
        acc += w67[0] * x_vals_2[2] + w67[1] * x_vals_2[3];
    }

    acc = simd_sum(acc);
    threadgroup float warp_sums[8];
    if (lane_id == 0u) warp_sums[warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float final_acc = 0.0f;
        for (uint i = 0; i < 8u; ++i) final_acc += warp_sums[i];
        y[row_idx] = final_acc;
    }
}

kernel void q4hq_gemv_accumulate_optimized(
    device const block_q4_l *blocks  [[ buffer(0) ]],
    device const float      *x       [[ buffer(1) ]],
    device       float      *y       [[ buffer(2) ]],
    constant     uint       &ncols   [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    const uint row_idx = bid;
    const uint blocks_per_row = ncols >> 8u;
    const uint row_block_start = row_idx * blocks_per_row;

    const uint warp_id = tid >> 5u;
    const uint lane_id = tid & 31u;

    float acc = 0.0f;

    for (uint b = warp_id; b < blocks_per_row; b += 8u) {
        device const block_q4_l &block = blocks[row_block_start + b];
        const float scale = (float)block.efektif_scale[lane_id >> 2u];
        uchar4 w_bytes = *(device const uchar4*)(block.qs + (lane_id << 2u));

        const uint col_base = (b << 8u) + (lane_id << 3u);
        device const float *x_ptr = x + col_base;
        float4 x_vals   = *(device const float4*)(x_ptr);
        float4 x_vals_2 = *(device const float4*)(x_ptr + 4);

        float2 w01 = q4hq_decode_byte(w_bytes[0], scale);
        float2 w23 = q4hq_decode_byte(w_bytes[1], scale);
        float2 w45 = q4hq_decode_byte(w_bytes[2], scale);
        float2 w67 = q4hq_decode_byte(w_bytes[3], scale);

        acc += w01[0] * x_vals[0] + w01[1] * x_vals[1];
        acc += w23[0] * x_vals[2] + w23[1] * x_vals[3];
        acc += w45[0] * x_vals_2[0] + w45[1] * x_vals_2[1];
        acc += w67[0] * x_vals_2[2] + w67[1] * x_vals_2[3];
    }

    acc = simd_sum(acc);
    threadgroup float warp_sums[8];
    if (lane_id == 0u) warp_sums[warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float final_acc = 0.0f;
        for (uint i = 0; i < 8u; ++i) final_acc += warp_sums[i];
        y[row_idx] += final_acc;
    }
}

kernel void f16_gemv_kernel(
    device const half  *weights [[buffer(0)]],
    device const float *x       [[buffer(1)]],
    device       float *y       [[buffer(2)]],
    constant     uint  &ncols   [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]]
) {
    const uint row_idx = bid;
    const uint row_start = row_idx * ncols;
    const uint warp_id = tid >> 5u;
    const uint lane_id = tid & 31u;

    float acc = 0.0f;
    for (uint col = tid; col < ncols; col += 256u) {
        acc += (float)weights[row_start + col] * x[col];
    }
    // simd_sum within warp, then 1 barrier for 8 warp sums
    threadgroup float f16_warp_sums[8];
    acc = simd_sum(acc);
    if (lane_id == 0u) f16_warp_sums[warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float total = 0.0f;
        for (uint w = 0u; w < 8u; ++w) total += f16_warp_sums[w];
        y[row_idx] = total;
    }
}

// ── Multi-row F16 GEMV: 4 output rows per threadgroup, x cached in TG memory.
//
// Each threadgroup of 256 threads handles 4 contiguous output rows.  64 threads
// per row (= 2 SIMD warps) sweep the input dimension in cooperative tiles so
// `x[]` is read from device memory **once** per threadgroup instead of once
// per output row.  For Qwen-3.5-4B's 248 320×2560 lm_head this collapses
// 248k threadgroups down to 62k and removes ~95 % of the redundant `x` traffic.
kernel void f16_gemv_multirow(
    device const half  *weights [[buffer(0)]],
    device const float *x       [[buffer(1)]],
    device       float *y       [[buffer(2)]],
    constant     uint  &ncols   [[buffer(3)]],
    constant     uint  &nrows   [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]]
) {
    constexpr uint MR_ROWS    = 4u;
    constexpr uint MR_THREADS = 256u;
    constexpr uint MR_PER_ROW = MR_THREADS / MR_ROWS; // 64
    constexpr uint TILE       = 256u;                 // x tile size

    const uint row_block = bid * MR_ROWS;
    const uint local_row = tid / MR_PER_ROW;          // 0..3
    const uint lane      = tid - local_row * MR_PER_ROW; // 0..63
    const uint row_idx   = row_block + local_row;
    const bool row_valid = row_idx < nrows;

    threadgroup float x_tile[TILE];
    float acc = 0.0f;

    for (uint base = 0u; base < ncols; base += TILE) {
        const uint chunk = min(TILE, ncols - base);
        if (tid < chunk) {
            x_tile[tid] = x[base + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_valid) {
            const uint row_start = row_idx * ncols + base;
            for (uint c = lane; c < chunk; c += MR_PER_ROW) {
                acc += (float)weights[row_start + c] * x_tile[c];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reduce 64 lanes per row (= 2 warps).  simd_sum collapses each warp;
    // 2 partial sums merge through a tiny TG buffer.
    threadgroup float row_partials[MR_ROWS * 2];
    acc = simd_sum(acc);
    if ((lane & 31u) == 0u && row_valid) {
        row_partials[local_row * 2u + (lane >> 5u)] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0u && row_valid) {
        y[row_idx] = row_partials[local_row * 2u] + row_partials[local_row * 2u + 1u];
    }
}

kernel void swiglu_kernel(
    device float       *h1    [[buffer(0)]],
    device const float *h3    [[buffer(1)]],
    constant uint      &dim   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dim) {
        return;
    }
    const float v1 = h1[gid];
    h1[gid] = silu_f32(v1) * h3[gid];
}

kernel void linear_attn_conv1d(
    device const float *input      [[buffer(0)]],
    device float       *state      [[buffer(1)]],
    device const float *weights    [[buffer(2)]],
    device const float *bias       [[buffer(3)]],
    device float       *output     [[buffer(4)]],
    constant uint      &kernel_sz  [[buffer(5)]],
    constant uint      &conv_dim   [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= conv_dim) {
        return;
    }

    const uint state_base = gid * kernel_sz;
    for (uint idx = 0; idx + 1u < kernel_sz; ++idx) {
        state[state_base + idx] = state[state_base + idx + 1u];
    }
    state[state_base + kernel_sz - 1u] = input[gid];

    float acc = bias[gid];
    const uint weight_base = gid * kernel_sz;
    for (uint idx = 0; idx < kernel_sz; ++idx) {
        acc += state[state_base + idx] * weights[weight_base + idx];
    }
    output[gid] = silu_f32(acc);
}

kernel void linear_attn_recurrent(
    device const float *conv_out        [[buffer(0)]],
    device const float *z_in            [[buffer(1)]],
    device const float *beta_logits     [[buffer(2)]],
    device const float *a_in            [[buffer(3)]],
    device const float *a_log           [[buffer(4)]],
    device const float *dt_bias         [[buffer(5)]],
    device const float *norm_w          [[buffer(6)]],
    device float       *recurrent_state [[buffer(7)]],
    device float       *output          [[buffer(8)]],
    constant uint      &num_key_heads   [[buffer(9)]],
    constant uint      &num_value_heads [[buffer(10)]],
    constant uint      &key_head_dim    [[buffer(11)]],
    constant uint      &value_head_dim  [[buffer(12)]],
    constant float     &eps             [[buffer(13)]],
    uint tid [[thread_position_in_threadgroup]],
    uint v_head [[threadgroup_position_in_grid]]
) {
    if (v_head >= num_value_heads || key_head_dim > 128u || value_head_dim > 128u) {
        return;
    }

    const bool value_lane_active = tid < value_head_dim;

    const uint rep = num_value_heads / num_key_heads;
    const uint k_head = v_head / rep;
    const uint key_dim = num_key_heads * key_head_dim;
    const uint value_offset = key_dim * 2u + v_head * value_head_dim;
    const uint q_offset = k_head * key_head_dim;
    const uint k_offset = key_dim + k_head * key_head_dim;

    threadgroup float q_sq[128];
    threadgroup float k_sq[128];
    threadgroup float q_normed[128];
    threadgroup float k_normed[128];
    threadgroup float core_sq[128];

    if (tid < key_head_dim) {
        const float qv = conv_out[q_offset + tid];
        const float kv = conv_out[k_offset + tid];
        q_sq[tid] = qv * qv;
        k_sq[tid] = kv * kv;
    } else {
        q_sq[tid] = 0.0f;
        k_sq[tid] = 0.0f;
    }
    // simd_sum within warp (no barrier), then 1 barrier to merge warp sums
    threadgroup float la_warp[4];
    float qss = (tid < key_head_dim) ? q_sq[tid] : 0.0f;
    float kss = (tid < key_head_dim) ? k_sq[tid] : 0.0f;
    qss = simd_sum(qss);
    kss = simd_sum(kss);
    if ((tid & 31u) == 0u) { la_warp[tid >> 5u] = qss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qss_total = 0.0f;
    if (tid < 4u) qss_total = la_warp[tid];
    qss_total = simd_sum(qss_total);
    if ((tid & 31u) == 0u) { la_warp[tid >> 5u] = kss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float kss_total = 0.0f;
    if (tid < 4u) kss_total = la_warp[tid];
    kss_total = simd_sum(kss_total);
    if (tid < key_head_dim) { q_sq[tid] = qss_total; k_sq[tid] = kss_total; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float q_inv = rsqrt(q_sq[0] + eps) * rsqrt((float)key_head_dim);
    const float k_inv = rsqrt(k_sq[0] + eps);

    if (tid < key_head_dim) {
        q_normed[tid] = conv_out[q_offset + tid] * q_inv;
        k_normed[tid] = conv_out[k_offset + tid] * k_inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float beta = 1.0f / (1.0f + exp(-beta_logits[v_head]));
    const float decay = exp(-exp(a_log[v_head]) * softplus_f32(a_in[v_head] + dt_bias[v_head]));
    const float value = value_lane_active ? conv_out[value_offset + tid] : 0.0f;
    const uint state_base = v_head * key_head_dim * value_head_dim + tid;

    float kv_mem = 0.0f;
    if (value_lane_active) {
        for (uint k_idx = 0; k_idx < key_head_dim; ++k_idx) {
            const uint state_idx = state_base + k_idx * value_head_dim;
            const float decayed = recurrent_state[state_idx] * decay;
            recurrent_state[state_idx] = decayed;
            kv_mem += decayed * k_normed[k_idx];
        }
    }

    const float delta = (value - kv_mem) * beta;
    float core = 0.0f;
    if (value_lane_active) {
        for (uint k_idx = 0; k_idx < key_head_dim; ++k_idx) {
            const uint state_idx = state_base + k_idx * value_head_dim;
            const float updated = recurrent_state[state_idx] + k_normed[k_idx] * delta;
            recurrent_state[state_idx] = updated;
            core += updated * q_normed[k_idx];
        }
    }

    core_sq[tid] = core * core;
    // simd_sum within warp then 1 barrier
    threadgroup float la_warp2[4];
    float css = simd_sum(core_sq[tid]);
    if ((tid & 31u) == 0u) la_warp2[tid >> 5u] = css;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float css_total = 0.0f;
    if (tid < 4u) css_total = la_warp2[tid];
    css_total = simd_sum(css_total);
    if (tid == 0u) core_sq[0] = css_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = rsqrt(core_sq[0] / (float)value_head_dim + eps);
    if (value_lane_active) {
        const float gate = silu_f32(z_in[v_head * value_head_dim + tid]);
        output[v_head * value_head_dim + tid] = core * inv_rms * norm_w[tid] * gate;
    }
}

// ── Multi-row Q4L GEMV (4 output rows per threadgroup) ───────────────────
//
// Layout: 256 threads = 8 warps × 32 lanes.
// Rows R+0..R+3 are each computed by 2 warps (64 threads), so 4×64=256.
// Each warp handles alternating blocks; within a block, each lane processes
// 8 weight elements (4 bytes = uchar4) × 1 input vector chunk.
//
// Compared to q4l_gemv_optimized (1 row/TG):
//   - 4× the arithmetic per threadgroup launch → ~3-4× throughput
//   - Same weight bytes read per launch → same bandwidth, better utilization
//   - register pressure slightly higher but within M-series limits
//
// dispatch: threadgroups = ceil(n_rows / 4),  threads_per_TG = 256

#define MR_ROWS  4u       // output rows per threadgroup
#define MR_WARPS 2u       // warps per row  (2 × 32 = 64 threads/row)

kernel void q4l_gemv_multirow(
    device const block_q4_l *blocks [[ buffer(0) ]],
    device const float      *x      [[ buffer(1) ]],
    device       float      *y      [[ buffer(2) ]],
    constant     uint       &ncols  [[ buffer(3) ]],
    constant     uint       &nrows  [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    // Which row inside this threadgroup (0-3) and which warp within that row.
    const uint local_row  = tid >> 6u;       // 0..3  (tid / 64)
    const uint row_tid    = tid & 63u;       // 0..63 (tid % 64)
    const uint warp_id    = row_tid >> 5u;   // 0..1  (row_tid / 32)
    const uint lane_id    = row_tid & 31u;   // 0..31

    const uint row_idx = bid * MR_ROWS + local_row;
    if (row_idx >= nrows) return;

    const uint blocks_per_row = ncols >> 8u;
    const uint row_block_start = row_idx * blocks_per_row;

    float acc = 0.0f;
    for (uint b = warp_id; b < blocks_per_row; b += MR_WARPS) {
        device const block_q4_l &block = blocks[row_block_start + b];
        const float scale = (float)block.efektif_scale[lane_id >> 2u];
        uchar4 w_bytes = *(device const uchar4*)(block.qs + (lane_id << 2u));

        const uint col_base = (b << 8u) + (lane_id << 3u);
        device const float *xp = x + col_base;
        float4 x0 = *(device const float4*)(xp);
        float4 x1 = *(device const float4*)(xp + 4);

        float2 w01 = lns_decode_byte(w_bytes[0], scale);
        float2 w23 = lns_decode_byte(w_bytes[1], scale);
        float2 w45 = lns_decode_byte(w_bytes[2], scale);
        float2 w67 = lns_decode_byte(w_bytes[3], scale);

        acc += w01[0]*x0[0] + w01[1]*x0[1];
        acc += w23[0]*x0[2] + w23[1]*x0[3];
        acc += w45[0]*x1[0] + w45[1]*x1[1];
        acc += w67[0]*x1[2] + w67[1]*x1[3];
    }

    // Reduce within SIMD (32 threads → 1)
    acc = simd_sum(acc);

    // Reduce 2 warp sums per row into a single value.
    // We store all 8 warp sums (4 rows × 2 warps) in one shared array.
    threadgroup float warp_sums[8]; // [row0_w0, row0_w1, row1_w0, row1_w1, ...]
    if (lane_id == 0u) warp_sums[local_row * MR_WARPS + warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < MR_ROWS) {
        y[bid * MR_ROWS + tid] = warp_sums[tid * MR_WARPS] + warp_sums[tid * MR_WARPS + 1u];
    }
}

kernel void q4l_gemv_accumulate_multirow(
    device const block_q4_l *blocks [[ buffer(0) ]],
    device const float      *x      [[ buffer(1) ]],
    device       float      *y      [[ buffer(2) ]],
    constant     uint       &ncols  [[ buffer(3) ]],
    constant     uint       &nrows  [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    const uint local_row  = tid >> 6u;
    const uint row_tid    = tid & 63u;
    const uint warp_id    = row_tid >> 5u;
    const uint lane_id    = row_tid & 31u;

    const uint row_idx = bid * MR_ROWS + local_row;
    if (row_idx >= nrows) return;

    const uint blocks_per_row = ncols >> 8u;
    const uint row_block_start = row_idx * blocks_per_row;

    float acc = 0.0f;
    for (uint b = warp_id; b < blocks_per_row; b += MR_WARPS) {
        device const block_q4_l &block = blocks[row_block_start + b];
        const float scale = (float)block.efektif_scale[lane_id >> 2u];
        uchar4 w_bytes = *(device const uchar4*)(block.qs + (lane_id << 2u));

        const uint col_base = (b << 8u) + (lane_id << 3u);
        device const float *xp = x + col_base;
        float4 x0 = *(device const float4*)(xp);
        float4 x1 = *(device const float4*)(xp + 4);

        float2 w01 = lns_decode_byte(w_bytes[0], scale);
        float2 w23 = lns_decode_byte(w_bytes[1], scale);
        float2 w45 = lns_decode_byte(w_bytes[2], scale);
        float2 w67 = lns_decode_byte(w_bytes[3], scale);

        acc += w01[0]*x0[0] + w01[1]*x0[1];
        acc += w23[0]*x0[2] + w23[1]*x0[3];
        acc += w45[0]*x1[0] + w45[1]*x1[1];
        acc += w67[0]*x1[2] + w67[1]*x1[3];
    }

    acc = simd_sum(acc);

    threadgroup float warp_sums[8];
    if (lane_id == 0u) warp_sums[local_row * MR_WARPS + warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < MR_ROWS) {
        y[bid * MR_ROWS + tid] += warp_sums[tid * MR_WARPS] + warp_sums[tid * MR_WARPS + 1u];
    }
}

// ── Fused W1+W3+SwiGLU multirow GEMV for Q4HQ FFN ────────────────────────
//
// Replaces three separate dispatches (w1_gemv, w3_gemv, swiglu) with one.
// Each threadgroup computes 4 rows of:
//   h_out[row] = silu(w1[row, :] · x) * (w3[row, :] · x)
// where silu(x) = x * sigmoid(x).
//
// Both w1 and w3 blocks are read in the same loop; their dot products are
// accumulated in separate registers, then the SwiGLU gate is applied at
// final-reduction time — no extra device memory round-trip for h1 or h3.
//
// dispatch: threadgroups = ceil(ffn_dim / MR_ROWS), threads = 256

kernel void q4l_w1w3_swiglu_multirow(
    device const block_q4_l *w1_blocks [[ buffer(0) ]],
    device const block_q4_l *w3_blocks [[ buffer(1) ]],
    device const float      *x         [[ buffer(2) ]],
    device       float      *h_out     [[ buffer(3) ]],
    constant     uint       &ncols     [[ buffer(4) ]],
    constant     uint       &nrows     [[ buffer(5) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    const uint local_row = tid >> 6u;
    const uint row_tid   = tid & 63u;
    const uint warp_id   = row_tid >> 5u;
    const uint lane_id   = row_tid & 31u;

    const uint row_idx = bid * MR_ROWS + local_row;
    if (row_idx >= nrows) return;

    const uint blocks_per_row  = ncols >> 8u;
    const uint row_block_start = row_idx * blocks_per_row;

    float acc_w1 = 0.0f;
    float acc_w3 = 0.0f;

    for (uint b = warp_id; b < blocks_per_row; b += MR_WARPS) {
        const uint block_idx = row_block_start + b;
        const uint col_base  = (b << 8u) + (lane_id << 3u);

        device const block_q4_l &blk1 = w1_blocks[block_idx];
        const float scale1 = (float)blk1.efektif_scale[lane_id >> 2u];
        uchar4 wb1 = *(device const uchar4*)(blk1.qs + (lane_id << 2u));

        device const block_q4_l &blk3 = w3_blocks[block_idx];
        const float scale3 = (float)blk3.efektif_scale[lane_id >> 2u];
        uchar4 wb3 = *(device const uchar4*)(blk3.qs + (lane_id << 2u));

        device const float *xp = x + col_base;
        float4 x0 = *(device const float4*)(xp);
        float4 x1 = *(device const float4*)(xp + 4);

        float2 w1_01 = lns_decode_byte(wb1[0], scale1); float2 w1_23 = lns_decode_byte(wb1[1], scale1);
        float2 w1_45 = lns_decode_byte(wb1[2], scale1); float2 w1_67 = lns_decode_byte(wb1[3], scale1);
        float2 w3_01 = lns_decode_byte(wb3[0], scale3); float2 w3_23 = lns_decode_byte(wb3[1], scale3);
        float2 w3_45 = lns_decode_byte(wb3[2], scale3); float2 w3_67 = lns_decode_byte(wb3[3], scale3);

        acc_w1 += w1_01[0]*x0[0] + w1_01[1]*x0[1] + w1_23[0]*x0[2] + w1_23[1]*x0[3];
        acc_w1 += w1_45[0]*x1[0] + w1_45[1]*x1[1] + w1_67[0]*x1[2] + w1_67[1]*x1[3];

        acc_w3 += w3_01[0]*x0[0] + w3_01[1]*x0[1] + w3_23[0]*x0[2] + w3_23[1]*x0[3];
        acc_w3 += w3_45[0]*x1[0] + w3_45[1]*x1[1] + w3_67[0]*x1[2] + w3_67[1]*x1[3];
    }

    acc_w1 = simd_sum(acc_w1);
    acc_w3 = simd_sum(acc_w3);

    threadgroup float ws1[8];
    threadgroup float ws3[8];
    if (lane_id == 0u) {
        ws1[local_row * MR_WARPS + warp_id] = acc_w1;
        ws3[local_row * MR_WARPS + warp_id] = acc_w3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < MR_ROWS) {
        const uint out_row = bid * MR_ROWS + tid;
        if (out_row < nrows) {
            float v1 = ws1[tid * MR_WARPS] + ws1[tid * MR_WARPS + 1u];
            float v3 = ws3[tid * MR_WARPS] + ws3[tid * MR_WARPS + 1u];
            h_out[out_row] = v1 * (1.0f / (1.0f + exp(-v1))) * v3;
        }
    }
}

kernel void q4hq_w1w3_swiglu_multirow(
    device const block_q4_l *w1_blocks [[ buffer(0) ]],   // [out_rows × ncols/256] Q4HQ
    device const block_q4_l *w3_blocks [[ buffer(1) ]],   // [out_rows × ncols/256] Q4HQ
    device const float      *x         [[ buffer(2) ]],   // [ncols] input
    device       float      *h_out     [[ buffer(3) ]],   // [out_rows] output
    constant     uint       &ncols     [[ buffer(4) ]],
    constant     uint       &nrows     [[ buffer(5) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    const uint local_row = tid >> 6u;
    const uint row_tid   = tid & 63u;
    const uint warp_id   = row_tid >> 5u;
    const uint lane_id   = row_tid & 31u;

    const uint row_idx = bid * MR_ROWS + local_row;
    if (row_idx >= nrows) return;

    const uint blocks_per_row  = ncols >> 8u;
    const uint row_block_start = row_idx * blocks_per_row;

    float acc_w1 = 0.0f;
    float acc_w3 = 0.0f;

    for (uint b = warp_id; b < blocks_per_row; b += MR_WARPS) {
        const uint block_idx = row_block_start + b;
        const uint col_base  = (b << 8u) + (lane_id << 3u);

        device const block_q4_l &blk1 = w1_blocks[block_idx];
        const float scale1 = (float)blk1.efektif_scale[lane_id >> 2u];
        uchar4 wb1 = *(device const uchar4*)(blk1.qs + (lane_id << 2u));

        device const block_q4_l &blk3 = w3_blocks[block_idx];
        const float scale3 = (float)blk3.efektif_scale[lane_id >> 2u];
        uchar4 wb3 = *(device const uchar4*)(blk3.qs + (lane_id << 2u));

        device const float *xp = x + col_base;
        float4 x0 = *(device const float4*)(xp);
        float4 x1 = *(device const float4*)(xp + 4);

        float2 w1_01 = q4hq_decode_byte(wb1[0], scale1); float2 w1_23 = q4hq_decode_byte(wb1[1], scale1);
        float2 w1_45 = q4hq_decode_byte(wb1[2], scale1); float2 w1_67 = q4hq_decode_byte(wb1[3], scale1);
        float2 w3_01 = q4hq_decode_byte(wb3[0], scale3); float2 w3_23 = q4hq_decode_byte(wb3[1], scale3);
        float2 w3_45 = q4hq_decode_byte(wb3[2], scale3); float2 w3_67 = q4hq_decode_byte(wb3[3], scale3);

        acc_w1 += w1_01[0]*x0[0] + w1_01[1]*x0[1] + w1_23[0]*x0[2] + w1_23[1]*x0[3];
        acc_w1 += w1_45[0]*x1[0] + w1_45[1]*x1[1] + w1_67[0]*x1[2] + w1_67[1]*x1[3];

        acc_w3 += w3_01[0]*x0[0] + w3_01[1]*x0[1] + w3_23[0]*x0[2] + w3_23[1]*x0[3];
        acc_w3 += w3_45[0]*x1[0] + w3_45[1]*x1[1] + w3_67[0]*x1[2] + w3_67[1]*x1[3];
    }

    acc_w1 = simd_sum(acc_w1);
    acc_w3 = simd_sum(acc_w3);

    threadgroup float ws1[8];  // warp sums for w1
    threadgroup float ws3[8];  // warp sums for w3
    if (lane_id == 0u) {
        ws1[local_row * MR_WARPS + warp_id] = acc_w1;
        ws3[local_row * MR_WARPS + warp_id] = acc_w3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < MR_ROWS) {
        const uint out_row = bid * MR_ROWS + tid;
        if (out_row < nrows) {
            float v1 = ws1[tid * MR_WARPS] + ws1[tid * MR_WARPS + 1u];
            float v3 = ws3[tid * MR_WARPS] + ws3[tid * MR_WARPS + 1u];
            // SwiGLU: silu(v1) * v3, silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
            h_out[out_row] = v1 * (1.0f / (1.0f + exp(-v1))) * v3;
        }
    }
}

kernel void q4hq_gemv_multirow(
    device const block_q4_l *blocks [[ buffer(0) ]],
    device const float      *x      [[ buffer(1) ]],
    device       float      *y      [[ buffer(2) ]],
    constant     uint       &ncols  [[ buffer(3) ]],
    constant     uint       &nrows  [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    const uint local_row  = tid >> 6u;
    const uint row_tid    = tid & 63u;
    const uint warp_id    = row_tid >> 5u;
    const uint lane_id    = row_tid & 31u;

    const uint row_idx = bid * MR_ROWS + local_row;
    if (row_idx >= nrows) return;

    const uint blocks_per_row = ncols >> 8u;
    const uint row_block_start = row_idx * blocks_per_row;

    float acc = 0.0f;
    for (uint b = warp_id; b < blocks_per_row; b += MR_WARPS) {
        device const block_q4_l &block = blocks[row_block_start + b];
        const float scale = (float)block.efektif_scale[lane_id >> 2u];
        uchar4 w_bytes = *(device const uchar4*)(block.qs + (lane_id << 2u));

        const uint col_base = (b << 8u) + (lane_id << 3u);
        device const float *xp = x + col_base;
        float4 x0 = *(device const float4*)(xp);
        float4 x1 = *(device const float4*)(xp + 4);

        float2 w01 = q4hq_decode_byte(w_bytes[0], scale);
        float2 w23 = q4hq_decode_byte(w_bytes[1], scale);
        float2 w45 = q4hq_decode_byte(w_bytes[2], scale);
        float2 w67 = q4hq_decode_byte(w_bytes[3], scale);

        acc += w01[0]*x0[0] + w01[1]*x0[1];
        acc += w23[0]*x0[2] + w23[1]*x0[3];
        acc += w45[0]*x1[0] + w45[1]*x1[1];
        acc += w67[0]*x1[2] + w67[1]*x1[3];
    }

    acc = simd_sum(acc);

    threadgroup float warp_sums[8];
    if (lane_id == 0u) warp_sums[local_row * MR_WARPS + warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < MR_ROWS) {
        y[bid * MR_ROWS + tid] = warp_sums[tid * MR_WARPS] + warp_sums[tid * MR_WARPS + 1u];
    }
}

kernel void q4hq_gemv_accumulate_multirow(
    device const block_q4_l *blocks [[ buffer(0) ]],
    device const float      *x      [[ buffer(1) ]],
    device       float      *y      [[ buffer(2) ]],
    constant     uint       &ncols  [[ buffer(3) ]],
    constant     uint       &nrows  [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    const uint local_row  = tid >> 6u;
    const uint row_tid    = tid & 63u;
    const uint warp_id    = row_tid >> 5u;
    const uint lane_id    = row_tid & 31u;

    const uint row_idx = bid * MR_ROWS + local_row;
    if (row_idx >= nrows) return;

    const uint blocks_per_row = ncols >> 8u;
    const uint row_block_start = row_idx * blocks_per_row;

    float acc = 0.0f;
    for (uint b = warp_id; b < blocks_per_row; b += MR_WARPS) {
        device const block_q4_l &block = blocks[row_block_start + b];
        const float scale = (float)block.efektif_scale[lane_id >> 2u];
        uchar4 w_bytes = *(device const uchar4*)(block.qs + (lane_id << 2u));

        const uint col_base = (b << 8u) + (lane_id << 3u);
        device const float *xp = x + col_base;
        float4 x0 = *(device const float4*)(xp);
        float4 x1 = *(device const float4*)(xp + 4);

        float2 w01 = q4hq_decode_byte(w_bytes[0], scale);
        float2 w23 = q4hq_decode_byte(w_bytes[1], scale);
        float2 w45 = q4hq_decode_byte(w_bytes[2], scale);
        float2 w67 = q4hq_decode_byte(w_bytes[3], scale);

        acc += w01[0]*x0[0] + w01[1]*x0[1];
        acc += w23[0]*x0[2] + w23[1]*x0[3];
        acc += w45[0]*x1[0] + w45[1]*x1[1];
        acc += w67[0]*x1[2] + w67[1]*x1[3];
    }

    acc = simd_sum(acc);

    threadgroup float warp_sums[8];
    if (lane_id == 0u) warp_sums[local_row * MR_WARPS + warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < MR_ROWS) {
        const uint out_row = bid * MR_ROWS + tid;
        if (out_row < nrows) {
            y[out_row] += warp_sums[tid * MR_WARPS] + warp_sums[tid * MR_WARPS + 1u];
        }
    }
}

// ── Fused RMSNorm + Q4L GEMV (eliminates one inter-encoder barrier) ───────
//
// Combined kernel: computes RMSNorm(x, w_norm) in threadgroup shared memory,
// then immediately uses the normalized result as input to Q4L GEMV — without
// writing the norm output to device memory.
//
// This removes the RMSNorm→GEMV barrier that appears before every attention
// and FFN projection.  For Qwen3.5-4B (32 layers × 2 norms = 64 barriers):
//   Each eliminated barrier saves ~20µs → 64 × 20µs = ~1.3ms/token.
//
// Constraints:
//   - dim must be a multiple of 256.
//   - Works for multi-row (4 rows/TG) variant only (same as q4l_gemv_multirow).
//
// dispatch: threadgroups = ceil(out_rows / 4), threads = 256

#define FUSED_DIM_MAX 4096u   // threadgroup shared mem for x_normed

kernel void q4l_rmsnorm_gemv_multirow(
    device const float      *x         [[ buffer(0) ]],   // [dim] input
    device const float      *norm_w    [[ buffer(1) ]],   // [dim] RMSNorm weight
    device const block_q4_l *blocks    [[ buffer(2) ]],   // GEMV weight matrix
    device       float      *y         [[ buffer(3) ]],   // [out_rows] output
    constant     uint       &dim       [[ buffer(4) ]],   // = ncols for GEMV
    constant     uint       &out_rows  [[ buffer(5) ]],
    constant     float      &eps       [[ buffer(6) ]],
    constant     uint       &zero_cent [[ buffer(7) ]],   // zero_centered_rmsnorm
    uint tid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    // ── Phase 1: RMSNorm — all 256 threads collaborate ──────────────────
    // simd_sum within warps, then 1 TG barrier to merge 8 warp sums.
    threadgroup float rn_warp_sums[8];
    const uint rn_warp_id = tid >> 5u;
    const uint rn_lane_id = tid & 31u;

    float acc_sq = 0.0f;
    for (uint idx = tid; idx < dim; idx += 256u) {
        float v = x[idx];
        acc_sq += v * v;
    }
    acc_sq = simd_sum(acc_sq);
    if (rn_lane_id == 0u) rn_warp_sums[rn_warp_id] = acc_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rn_total = 0.0f;
    if (tid == 0u) {
        for (uint w = 0u; w < 8u; ++w) rn_total += rn_warp_sums[w];
        rn_warp_sums[0] = rn_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = rsqrt(rn_warp_sums[0] / (float)dim + eps);

    threadgroup float x_normed[FUSED_DIM_MAX];
    for (uint idx = tid; idx < dim; idx += 256u) {
        float scale_n = zero_cent != 0u ? (1.0f + norm_w[idx]) : norm_w[idx];
        x_normed[idx] = x[idx] * inv_rms * scale_n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: Multi-row Q4L GEMV with vectorized TG reads ─────────────
    const uint local_row = tid >> 6u;
    const uint row_tid   = tid & 63u;
    const uint warp_id   = row_tid >> 5u;
    const uint lane_id   = row_tid & 31u;

    const uint row_idx = bid * MR_ROWS + local_row;

    float acc = 0.0f;
    if (row_idx < out_rows) {
        const uint blocks_per_row  = dim >> 8u;
        const uint row_block_start = row_idx * blocks_per_row;

        for (uint b = warp_id; b < blocks_per_row; b += MR_WARPS) {
            device const block_q4_l &block = blocks[row_block_start + b];
            const float scale = (float)block.efektif_scale[lane_id >> 2u];
            uchar4 w_bytes = *(device const uchar4*)(block.qs + (lane_id << 2u));

            const uint col_base = (b << 8u) + (lane_id << 3u);
            float4 x0 = *(threadgroup const float4*)(&x_normed[col_base]);
            float4 x1 = *(threadgroup const float4*)(&x_normed[col_base + 4u]);

            float2 w01 = lns_decode_byte(w_bytes[0], scale);
            float2 w23 = lns_decode_byte(w_bytes[1], scale);
            float2 w45 = lns_decode_byte(w_bytes[2], scale);
            float2 w67 = lns_decode_byte(w_bytes[3], scale);

            acc += w01[0]*x0[0] + w01[1]*x0[1];
            acc += w23[0]*x0[2] + w23[1]*x0[3];
            acc += w45[0]*x1[0] + w45[1]*x1[1];
            acc += w67[0]*x1[2] + w67[1]*x1[3];
        }
    }

    acc = simd_sum(acc);
    threadgroup float warp_sums[8];
    if (lane_id == 0u) warp_sums[local_row * MR_WARPS + warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < MR_ROWS) {
        const uint out_row = bid * MR_ROWS + tid;
        if (out_row < out_rows) {
            y[out_row] = warp_sums[tid * MR_WARPS] + warp_sums[tid * MR_WARPS + 1u];
        }
    }
}

// ── Fused RMSNorm + Q4L GEMV Accumulate ─────────────────────────────────
// Same as above but accumulates into y instead of overwriting (for o_proj, w2).

kernel void q4l_rmsnorm_gemv_accumulate_multirow(
    device const float      *x         [[ buffer(0) ]],
    device const float      *norm_w    [[ buffer(1) ]],
    device const block_q4_l *blocks    [[ buffer(2) ]],
    device       float      *y         [[ buffer(3) ]],
    constant     uint       &dim       [[ buffer(4) ]],
    constant     uint       &out_rows  [[ buffer(5) ]],
    constant     float      &eps       [[ buffer(6) ]],
    constant     uint       &zero_cent [[ buffer(7) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    // ── Phase 1: RMSNorm with simd_sum (no barrier tree) ─────────────────
    threadgroup float rn_warp_sums[8];
    const uint rn_warp_id = tid >> 5u;
    const uint rn_lane_id = tid & 31u;

    float acc_sq = 0.0f;
    for (uint idx = tid; idx < dim; idx += 256u) {
        float v = x[idx];
        acc_sq += v * v;
    }
    acc_sq = simd_sum(acc_sq);
    if (rn_lane_id == 0u) rn_warp_sums[rn_warp_id] = acc_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rn_total = 0.0f;
    if (tid == 0u) {
        for (uint w = 0u; w < 8u; ++w) rn_total += rn_warp_sums[w];
        rn_warp_sums[0] = rn_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = rsqrt(rn_warp_sums[0] / (float)dim + eps);

    threadgroup float x_normed[FUSED_DIM_MAX];
    for (uint idx = tid; idx < dim; idx += 256u) {
        float scale_n = zero_cent != 0u ? (1.0f + norm_w[idx]) : norm_w[idx];
        x_normed[idx] = x[idx] * inv_rms * scale_n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: Multi-row Q4L GEMV with vectorized TG reads ─────────────
    const uint local_row = tid >> 6u;
    const uint row_tid   = tid & 63u;
    const uint warp_id   = row_tid >> 5u;
    const uint lane_id   = row_tid & 31u;

    const uint row_idx = bid * MR_ROWS + local_row;

    float acc = 0.0f;
    if (row_idx < out_rows) {
        const uint blocks_per_row  = dim >> 8u;
        const uint row_block_start = row_idx * blocks_per_row;

        for (uint b = warp_id; b < blocks_per_row; b += MR_WARPS) {
            device const block_q4_l &block = blocks[row_block_start + b];
            const float scale = (float)block.efektif_scale[lane_id >> 2u];
            uchar4 w_bytes = *(device const uchar4*)(block.qs + (lane_id << 2u));

            const uint col_base = (b << 8u) + (lane_id << 3u);
            float4 x0 = *(threadgroup const float4*)(&x_normed[col_base]);
            float4 x1 = *(threadgroup const float4*)(&x_normed[col_base + 4u]);

            float2 w01 = lns_decode_byte(w_bytes[0], scale);
            float2 w23 = lns_decode_byte(w_bytes[1], scale);
            float2 w45 = lns_decode_byte(w_bytes[2], scale);
            float2 w67 = lns_decode_byte(w_bytes[3], scale);

            acc += w01[0]*x0[0] + w01[1]*x0[1];
            acc += w23[0]*x0[2] + w23[1]*x0[3];
            acc += w45[0]*x1[0] + w45[1]*x1[1];
            acc += w67[0]*x1[2] + w67[1]*x1[3];
        }
    }

    acc = simd_sum(acc);
    threadgroup float warp_sums[8];
    if (lane_id == 0u) warp_sums[local_row * MR_WARPS + warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < MR_ROWS) {
        const uint out_row = bid * MR_ROWS + tid;
        if (out_row < out_rows) {
            y[out_row] += warp_sums[tid * MR_WARPS] + warp_sums[tid * MR_WARPS + 1u];
        }
    }
}

// ── Fused RMSNorm + Q4HQ GEMV ─────────────────────────────────────────────
// Same as q4l_rmsnorm_gemv_multirow but uses q4hq_decode_byte.
// Used for the lm_head projection (vocab_size × dim) which stores weights in Q4HQ.
// Fusing eliminates the norm→lm_head inter-encoder barrier and avoids writing
// x_normed to device memory and re-reading it for the large vocab GEMV.
//
// dispatch: threadgroups = ceil(vocab_size / 4), threads_per_threadgroup = 256

kernel void q4hq_rmsnorm_gemv_multirow(
    device const float      *x         [[ buffer(0) ]],   // [dim] input
    device const float      *norm_w    [[ buffer(1) ]],   // [dim] RMSNorm weight
    device const block_q4_l *blocks    [[ buffer(2) ]],   // Q4HQ weight matrix [out_rows × dim]
    device       float      *y         [[ buffer(3) ]],   // [out_rows] output
    constant     uint       &dim       [[ buffer(4) ]],   // = ncols for GEMV
    constant     uint       &out_rows  [[ buffer(5) ]],
    constant     float      &eps       [[ buffer(6) ]],
    constant     uint       &zero_cent [[ buffer(7) ]],   // zero_centered_rmsnorm flag
    uint tid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    // ── Phase 1: RMSNorm ──────────────────────────────────────────────────
    threadgroup float rn_warp_sums[8];
    const uint rn_warp_id = tid >> 5u;
    const uint rn_lane_id = tid & 31u;

    float acc_sq = 0.0f;
    for (uint idx = tid; idx < dim; idx += 256u) {
        float v = x[idx];
        acc_sq += v * v;
    }
    acc_sq = simd_sum(acc_sq);
    if (rn_lane_id == 0u) rn_warp_sums[rn_warp_id] = acc_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rn_total = 0.0f;
    if (tid == 0u) {
        for (uint w = 0u; w < 8u; ++w) rn_total += rn_warp_sums[w];
        rn_warp_sums[0] = rn_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = rsqrt(rn_warp_sums[0] / (float)dim + eps);

    threadgroup float x_normed[FUSED_DIM_MAX];
    for (uint idx = tid; idx < dim; idx += 256u) {
        float scale_n = zero_cent != 0u ? (1.0f + norm_w[idx]) : norm_w[idx];
        x_normed[idx] = x[idx] * inv_rms * scale_n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: Multi-row Q4HQ GEMV ────────────────────────────────────
    const uint local_row = tid >> 6u;
    const uint row_tid   = tid & 63u;
    const uint warp_id   = row_tid >> 5u;
    const uint lane_id   = row_tid & 31u;

    const uint row_idx = bid * MR_ROWS + local_row;

    float acc = 0.0f;
    if (row_idx < out_rows) {
        const uint blocks_per_row  = dim >> 8u;
        const uint row_block_start = row_idx * blocks_per_row;

        for (uint b = warp_id; b < blocks_per_row; b += MR_WARPS) {
            device const block_q4_l &block = blocks[row_block_start + b];
            const float scale = (float)block.efektif_scale[lane_id >> 2u];
            uchar4 w_bytes = *(device const uchar4*)(block.qs + (lane_id << 2u));

            const uint col_base = (b << 8u) + (lane_id << 3u);
            float4 x0 = *(threadgroup const float4*)(&x_normed[col_base]);
            float4 x1 = *(threadgroup const float4*)(&x_normed[col_base + 4u]);

            float2 w01 = q4hq_decode_byte(w_bytes[0], scale);
            float2 w23 = q4hq_decode_byte(w_bytes[1], scale);
            float2 w45 = q4hq_decode_byte(w_bytes[2], scale);
            float2 w67 = q4hq_decode_byte(w_bytes[3], scale);

            acc += w01[0]*x0[0] + w01[1]*x0[1];
            acc += w23[0]*x0[2] + w23[1]*x0[3];
            acc += w45[0]*x1[0] + w45[1]*x1[1];
            acc += w67[0]*x1[2] + w67[1]*x1[3];
        }
    }

    acc = simd_sum(acc);
    threadgroup float warp_sums[8];
    if (lane_id == 0u) warp_sums[local_row * MR_WARPS + warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < MR_ROWS) {
        const uint out_row = bid * MR_ROWS + tid;
        if (out_row < out_rows) {
            y[out_row] = warp_sums[tid * MR_WARPS] + warp_sums[tid * MR_WARPS + 1u];
        }
    }
}

// ── GPU Argmax (greedy decode — eliminates 248K×4B readback) ─────────────
//
// Two-pass parallel argmax over logits[vocab_size]:
//   Pass 1: each threadgroup reduces CHUNK elements → 1 (val, idx) pair.
//   Pass 2: a single threadgroup reduces all partial results.
//
// For vocab=248320, chunk_size=256: pass1 = 970 TGs, pass2 = 1 TG.
// Total GPU→CPU readback: 4 bytes (winning token ID) vs 992 KB.
//
// Buffer layout:
//   0: logits   float[vocab_size]
//   1: partials float2[ceil(vocab_size/ARGMAX_CHUNK)]  — (max_val, max_idx as float)
//   2: result   uint[1]

#define ARGMAX_CHUNK 256u

kernel void argmax_pass1(
    device const float *logits   [[ buffer(0) ]],
    device       float *partials [[ buffer(1) ]],   // interleaved (val, idx) pairs
    constant     uint  &n        [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint bid [[ threadgroup_position_in_grid ]]
) {
    const uint base = bid * ARGMAX_CHUNK;
    const uint idx  = base + tid;

    float my_val = (idx < n) ? logits[idx] : -1e38f;
    uint  my_idx = (idx < n) ? idx : 0u;

    // ── Intra-warp reduce (simd_shuffle_down, zero barriers) ────────────
    for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
        float ov = simd_shuffle_down(my_val, offset);
        uint  oi = simd_shuffle_down(my_idx, offset);
        if (ov > my_val) { my_val = ov; my_idx = oi; }
    }

    // ── Inter-warp: one barrier to merge 8 warp bests ───────────────────
    threadgroup float sh_val[8];
    threadgroup uint  sh_idx[8];
    const uint warp_id = tid >> 5u;
    const uint lane_id = tid & 31u;
    if (lane_id == 0u) { sh_val[warp_id] = my_val; sh_idx[warp_id] = my_idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Warp 0 reduces the 8 warp bests — only lanes 0..7 do work
    if (warp_id == 0u) {
        float v = (lane_id < 8u) ? sh_val[lane_id] : -1e38f;
        uint  i = (lane_id < 8u) ? sh_idx[lane_id] : 0u;
        for (ushort offset = 4u; offset > 0u; offset >>= 1u) {
            float ov = simd_shuffle_down(v, offset);
            uint  oi = simd_shuffle_down(i, offset);
            if (ov > v) { v = ov; i = oi; }
        }
        if (lane_id == 0u) {
            partials[bid * 2u]      = v;
            partials[bid * 2u + 1u] = (float)i;
        }
    }
}

kernel void argmax_pass2(
    device const float *partials [[ buffer(0) ]],   // (val, idx) pairs
    device       uint  *result   [[ buffer(1) ]],
    constant     uint  &n_parts  [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]]
) {
    // Each thread finds its local best over its slice of partial results
    float my_val = -1e38f;
    uint  my_idx = 0u;
    for (uint i = tid; i < n_parts; i += ARGMAX_CHUNK) {
        float v = partials[i * 2u];
        if (v > my_val) { my_val = v; my_idx = (uint)partials[i * 2u + 1u]; }
    }

    // ── Intra-warp reduce (simd_shuffle_down, zero barriers) ────────────
    for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
        float ov = simd_shuffle_down(my_val, offset);
        uint  oi = simd_shuffle_down(my_idx, offset);
        if (ov > my_val) { my_val = ov; my_idx = oi; }
    }

    // ── Inter-warp: one barrier to merge 8 warp bests ───────────────────
    threadgroup float sh_val[8];
    threadgroup uint  sh_idx[8];
    const uint warp_id = tid >> 5u;
    const uint lane_id = tid & 31u;
    if (lane_id == 0u) { sh_val[warp_id] = my_val; sh_idx[warp_id] = my_idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Warp 0 reduces the 8 warp bests
    if (warp_id == 0u) {
        float v = (lane_id < 8u) ? sh_val[lane_id] : -1e38f;
        uint  i = (lane_id < 8u) ? sh_idx[lane_id] : 0u;
        for (ushort offset = 4u; offset > 0u; offset >>= 1u) {
            float ov = simd_shuffle_down(v, offset);
            uint  oi = simd_shuffle_down(i, offset);
            if (ov > v) { v = ov; i = oi; }
        }
        if (lane_id == 0u) { result[0] = i; }
    }
}
