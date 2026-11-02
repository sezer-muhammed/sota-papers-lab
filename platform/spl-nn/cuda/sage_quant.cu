// SageAttention prologue: optional QK LayerNorm, K smoothing, per-warp INT8
// Q, per-block INT8 K, per-channel FP8 V (transposed and key-permuted for the
// FP8 mma), and KV merging. Owner: nn-sage.
//
// Three launches per attention call:
//   sg_prep      per 128-query tile: LayerNorm and INT8-quantise Q (one scale
//                per 16 rows, softmax scale * log2(e) folded in), and write
//                partial K/V channel statistics of the same tokens
//   sg_finalize  per (batch, head): K mean, V mean and the V scale
//   sg_quant_kv  per 64 (merged) keys: pool members, LayerNorm, subtract
//                the K mean, INT8 K with one scale per block, FP8 V
//
// Layouts written (b = batch, h = head):
//   q8      i8  [b][h][tq_pad][D]
//   q_scale f32 [b][h][tq_pad / 16]
//   k8      i8  [b][h][tk_pad][D]
//   k_scale f32 [b][h][tk_pad / 64]
//   v8      u8  [b][h][tk_pad / 64][D][64]   E4M3, keys permuted by sg_vpos
//   stats   f32 [b][h][4][D]                 K mean, V mean, V scale, 1 / V scale
//   kbias   f32 [tk_pad]                     log2(merged count); -inf on padding
//
// KV merging (GPA-VGGT-style, arXiv 2610.37000): within each frame the patch
// grid is pooled in s x s cells (mean of K and of V; partial cells at the
// right and bottom edges pool what they cover) and each pooled key gets a
// +log(count) logit bias so it carries the attention mass of its members.
// The `special` camera and register tokens at the start of each frame are
// kept as they are, and so is every token of the first `dense_frames` frames
// (VGGT-Ω's reference frame). Queries are never merged.

#define SG_CH (SG_D / 8)  // 16-byte bf16 chunks per head row

// LayerNorm over one head row whose SG_D channels are spread over SG_CH
// consecutive lanes, 8 per lane. Every lane of the warp must call it.
__device__ __forceinline__ void sg_layernorm(float* x, const float* w, const float* bias, int ch, float eps) {
    float s = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) s += x[e];
#pragma unroll
    for (int o = 1; o < SG_CH; o <<= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s * (1.0f / SG_D);
    float s2 = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const float d = x[e] - mean;
        s2 += d * d;
    }
#pragma unroll
    for (int o = 1; o < SG_CH; o <<= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
    const float rstd = rsqrtf(s2 * (1.0f / SG_D) + eps);
#pragma unroll
    for (int e = 0; e < 8; e++) x[e] = (x[e] - mean) * rstd * w[ch * 8 + e] + bias[ch * 8 + e];
}

__device__ __forceinline__ uint4 sg_load8(const sg_u16* p, bool ok) {
    return ok ? __ldg(reinterpret_cast<const uint4*>(p)) : make_uint4(0, 0, 0, 0);
}

// grid (q_tiles, heads, batch), 256 threads.
extern "C" __global__ void __launch_bounds__(256) sg_prep(
    const sg_u16* __restrict__ q, const sg_u16* __restrict__ k, const sg_u16* __restrict__ v,
    const float* __restrict__ ln,  // [4][D]: q weight, q bias, k weight, k bias
    sg_i8* __restrict__ q8, float* __restrict__ q_scale, float* __restrict__ part, const SgGeom g) {
    constexpr int TPI = 256 / SG_CH;         // tokens per iteration
    constexpr int NI = SG_QTILE / TPI;       // iterations
    const int tile = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, ch = tid % SG_CH;
    const long long bh = (long long)b * g.heads + h;

    __shared__ unsigned gmax[SG_QTILE / SG_QGROUP];
    __shared__ float red[8][4][SG_D];
    if (tid < SG_QTILE / SG_QGROUP) gmax[tid] = 0u;

    float ks[8], vs[8], vmn[8], vmx[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
        ks[e] = 0.f;
        vs[e] = 0.f;
        vmn[e] = 3.0e38f;
        vmx[e] = -3.0e38f;
    }
    float qv[NI][8];
    __syncthreads();

#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int tl = tid / SG_CH + i * TPI;
        const int tok = tile * SG_QTILE + tl;
        const bool ok = tok < g.tokens;
        float fq[8], fk[8], fv[8];
        sg_unpack8(sg_load8(q + b * g.q_sb + (long long)tok * g.q_st + h * g.q_sh + ch * 8, ok), fq);
        sg_unpack8(sg_load8(k + b * g.k_sb + (long long)tok * g.k_st + h * g.k_sh + ch * 8, ok), fk);
        sg_unpack8(sg_load8(v + b * g.v_sb + (long long)tok * g.v_st + h * g.v_sh + ch * 8, ok), fv);
        if (g.flags & SG_LN_Q) sg_layernorm(fq, ln, ln + SG_D, ch, g.ln_eps);
        if (g.flags & SG_LN_K) sg_layernorm(fk, ln + 2 * SG_D, ln + 3 * SG_D, ch, g.ln_eps);
        float amax = 0.f;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            if (!ok) fq[e] = 0.f;
            qv[i][e] = fq[e];
            amax = fmaxf(amax, fabsf(fq[e]));
            if (ok) {
                ks[e] += fk[e];
                vs[e] += fv[e];
                vmn[e] = fminf(vmn[e], fv[e]);
                vmx[e] = fmaxf(vmx[e], fv[e]);
            }
        }
        // A warp's tokens lie in one 16-row group (32 / SG_CH <= 16 tokens, aligned).
        amax = sg_warp_max(amax);
        if (lane == 0) atomicMax(&gmax[tl / SG_QGROUP], __float_as_uint(amax));
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int tl = tid / SG_CH + i * TPI;
        const float amax = __uint_as_float(gmax[tl / SG_QGROUP]);
        const float inv = amax > 0.f ? 127.0f / amax : 0.f;
        const float* f = qv[i];
        uint2 w;
        w.x = sg_s8x4(f[0] * inv, f[1] * inv, f[2] * inv, f[3] * inv);
        w.y = sg_s8x4(f[4] * inv, f[5] * inv, f[6] * inv, f[7] * inv);
        *reinterpret_cast<uint2*>(q8 + (bh * g.tq_pad + tile * SG_QTILE + tl) * SG_D + ch * 8) = w;
    }
    if (tid < SG_QTILE / SG_QGROUP)
        q_scale[bh * (g.tq_pad / SG_QGROUP) + tile * (SG_QTILE / SG_QGROUP) + tid] =
            __uint_as_float(gmax[tid]) * (1.0f / 127.0f) * g.q_mul;

    // Channel statistics: lanes with the same chunk, then the warps.
#pragma unroll
    for (int o = SG_CH; o < 32; o <<= 1) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            ks[e] += __shfl_xor_sync(0xffffffffu, ks[e], o);
            vs[e] += __shfl_xor_sync(0xffffffffu, vs[e], o);
            vmn[e] = fminf(vmn[e], __shfl_xor_sync(0xffffffffu, vmn[e], o));
            vmx[e] = fmaxf(vmx[e], __shfl_xor_sync(0xffffffffu, vmx[e], o));
        }
    }
    if (lane < SG_CH) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            red[warp][0][ch * 8 + e] = ks[e];
            red[warp][1][ch * 8 + e] = vs[e];
            red[warp][2][ch * 8 + e] = vmn[e];
            red[warp][3][ch * 8 + e] = vmx[e];
        }
    }
    __syncthreads();
    for (int c = tid; c < 4 * SG_D; c += 256) {
        const int stat = c / SG_D, d = c % SG_D;
        float r = red[0][stat][d];
#pragma unroll
        for (int w = 1; w < 8; w++) {
            const float x = red[w][stat][d];
            r = stat < 2 ? r + x : (stat == 2 ? fminf(r, x) : fmaxf(r, x));
        }
        part[((bh * g.q_tiles + tile) * 4 + stat) * SG_D + d] = r;
    }
}

// grid (heads, batch), 256 threads.
extern "C" __global__ void __launch_bounds__(256) sg_finalize(const float* __restrict__ part, float* __restrict__ stats,
                                                             const SgGeom g) {
    const int h = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
    const long long bh = (long long)b * g.heads + h;
    __shared__ float st[4][SG_D];
    for (int c = tid; c < 4 * SG_D; c += 256) {
        const int stat = c / SG_D, d = c % SG_D;
        const float* p = part + bh * g.q_tiles * 4 * SG_D + stat * SG_D + d;
        float r0 = stat < 2 ? 0.f : (stat == 2 ? 3.0e38f : -3.0e38f);
        float r1 = r0;
        int t = 0;
        for (; t + 1 < g.q_tiles; t += 2) {
            const float x0 = p[(long long)t * 4 * SG_D], x1 = p[(long long)(t + 1) * 4 * SG_D];
            if (stat < 2) {
                r0 += x0;
                r1 += x1;
            } else if (stat == 2) {
                r0 = fminf(r0, x0);
                r1 = fminf(r1, x1);
            } else {
                r0 = fmaxf(r0, x0);
                r1 = fmaxf(r1, x1);
            }
        }
        if (t < g.q_tiles) {
            const float x0 = p[(long long)t * 4 * SG_D];
            r0 = stat < 2 ? r0 + x0 : (stat == 2 ? fminf(r0, x0) : fmaxf(r0, x0));
        }
        st[stat][c % SG_D] = stat < 2 ? r0 + r1 : (stat == 2 ? fminf(r0, r1) : fmaxf(r0, r1));
    }
    __syncthreads();
    for (int d = tid; d < SG_D; d += 256) {
        const float n = (float)g.tokens;
        const float mk = (g.flags & SG_SMOOTH_K) ? st[0][d] / n : 0.f;
        const float mv = (g.flags & SG_SMOOTH_V) ? st[1][d] / n : 0.f;
        const float amax = fmaxf(st[3][d] - mv, mv - st[2][d]);
        const float scale = amax > 0.f ? amax / g.v_range : 1.0f;
        float* o = stats + bh * 4 * SG_D;
        o[d] = mk;
        o[SG_D + d] = mv;
        o[2 * SG_D + d] = scale;
        o[3 * SG_D + d] = 1.0f / scale;
    }
}

// Members of merged key j (within one batch item): count, and the token of
// member m (0 <= m < count). Dense when g.merge == 1.
__device__ __forceinline__ int sg_members(const SgGeom& g, int j, int& base, int& nx) {
    if (g.merge == 1) {
        base = j;
        nx = 1;
        return 1;
    }
    const int dense = g.dense_frames * g.frame_tokens;
    if (j < dense) {
        base = j;
        nx = 1;
        return 1;
    }
    const int f = g.dense_frames + (j - dense) / g.merged_frame, loc = (j - dense) % g.merged_frame;
    if (loc < g.special) {
        base = f * g.frame_tokens + loc;
        nx = 1;
        return 1;
    }
    const int cell = loc - g.special;
    const int gy = cell / g.merged_w, gx = cell % g.merged_w;
    const int y0 = gy * g.merge, x0 = gx * g.merge;
    const int ny = min(g.merge, g.grid_h - y0);
    nx = min(g.merge, g.grid_w - x0);
    base = f * g.frame_tokens + g.special + y0 * g.grid_w + x0;
    return ny * nx;
}

// grid (tk_pad / 64, heads, batch), 256 threads.
extern "C" __global__ void __launch_bounds__(256) sg_quant_kv(
    const sg_u16* __restrict__ k, const sg_u16* __restrict__ v, const float* __restrict__ ln,
    const float* __restrict__ stats, sg_i8* __restrict__ k8, float* __restrict__ k_scale, sg_u8* __restrict__ v8,
    float* __restrict__ kbias, const SgGeom g) {
    constexpr int TPI = 256 / SG_CH;
    constexpr int NI = SG_KBLOCK / TPI;
    const int blk = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, ch = tid % SG_CH;
    const long long bh = (long long)b * g.heads + h;
    const int nblk = g.tk_pad / SG_KBLOCK;
    const int cells = g.merge * g.merge;  // warp-uniform member loop bound

    __shared__ unsigned kmax;
    __shared__ __align__(16) sg_u8 vt[SG_D * SG_KBLOCK];
    if (tid == 0) kmax = 0u;
    __syncthreads();

    const float* st = stats + bh * 4 * SG_D;
    float kv[NI][8];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int kl = tid / SG_CH + i * TPI;
        const int j = blk * SG_KBLOCK + kl;
        const bool ok = j < g.tk;
        int base = 0, nx = 1;
        const int cnt = ok ? sg_members(g, j, base, nx) : 0;
        float ksum[8], vsum[8];
#pragma unroll
        for (int e = 0; e < 8; e++) ksum[e] = vsum[e] = 0.f;
        for (int m = 0; m < cells; m++) {
            const bool mok = m < cnt;
            const int tok = base + (mok ? (m / nx) * g.grid_w + m % nx : 0);
            float fk[8], fv[8];
            sg_unpack8(sg_load8(k + b * g.k_sb + (long long)tok * g.k_st + h * g.k_sh + ch * 8, mok), fk);
            sg_unpack8(sg_load8(v + b * g.v_sb + (long long)tok * g.v_st + h * g.v_sh + ch * 8, mok), fv);
            if (g.flags & SG_LN_K) sg_layernorm(fk, ln + 2 * SG_D, ln + 3 * SG_D, ch, g.ln_eps);
            if (mok) {
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    ksum[e] += fk[e];
                    vsum[e] += fv[e];
                }
            }
        }
        const float rc = cnt > 0 ? 1.0f / (float)cnt : 0.f;
        float vq[8];
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int d = ch * 8 + e;
            kv[i][e] = ok ? ksum[e] * rc - st[d] : 0.f;
            amax = fmaxf(amax, fabsf(kv[i][e]));
            vq[e] = ok ? (vsum[e] * rc - st[SG_D + d]) * st[3 * SG_D + d] : 0.f;
        }
        // Transposed V tile, keys permuted for the FP8 P fragment.
        const int col = (kl & ~15) + sg_vpos(kl & 15);
#pragma unroll
        for (int e = 0; e < 8; e += 2) {
            const sg_u32 two = sg_e4m3x2(vq[e], vq[e + 1]);
            vt[(ch * 8 + e) * SG_KBLOCK + col] = (sg_u8)(two & 0xff);
            vt[(ch * 8 + e + 1) * SG_KBLOCK + col] = (sg_u8)(two >> 8);
        }
    }
    amax = sg_warp_max(amax);
    if (lane == 0) atomicMax(&kmax, __float_as_uint(amax));
    __syncthreads();

    const float km = __uint_as_float(kmax);
    const float inv = km > 0.f ? 127.0f / km : 0.f;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int kl = tid / SG_CH + i * TPI;
        const float* f = kv[i];
        uint2 w;
        w.x = sg_s8x4(f[0] * inv, f[1] * inv, f[2] * inv, f[3] * inv);
        w.y = sg_s8x4(f[4] * inv, f[5] * inv, f[6] * inv, f[7] * inv);
        *reinterpret_cast<uint2*>(k8 + (bh * g.tk_pad + blk * SG_KBLOCK + kl) * SG_D + ch * 8) = w;
    }
    if (tid == 0) k_scale[bh * nblk + blk] = km * (1.0f / 127.0f);
    sg_u8* vo = v8 + (bh * nblk + blk) * (long long)(SG_D * SG_KBLOCK);
    for (int c = tid; c < SG_D * SG_KBLOCK / 16; c += 256)
        reinterpret_cast<uint4*>(vo)[c] = reinterpret_cast<const uint4*>(vt)[c];

    if (g.merge > 1 && b == 0 && h == 0 && tid < SG_KBLOCK) {
        const int j = blk * SG_KBLOCK + tid;
        int base, nx;
        kbias[j] = j < g.tk ? log2f((float)sg_members(g, j, base, nx)) : __int_as_float(0xff800000);
    }
}
