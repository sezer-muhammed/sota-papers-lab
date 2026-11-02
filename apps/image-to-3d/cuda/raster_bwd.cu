// Chained per-Gaussian raster backward. Owner: raster (E3).
// Spec: spec-raster Sections 5.1-5.2. Base: the per-Gaussian backward of Taming-3DGS as
// refined by Faster-GS (Apache-2.0): a warp owns a bucket of 32 consecutive tile-list
// Gaussians (lane l owns Gaussian 32b + l) and the tile's pixels flow through the warp
// diagonally, lane l handling list entry s - l at step s, the mutable pixel state passed one
// lane up with __shfl_up_sync each step. Written from the papers and the spec; no code copied.
//
// Our changes:
// - Chains: a warp processes the CHAIN_K consecutive buckets of a chain, so the forward
//   checkpoints only every CHAIN_LEN Gaussians. Between buckets the pixel state (A, T) goes
//   through the chain's own checkpoint slot (lane 31 stores it, the next bucket's staging
//   loads it back from L2), so shared memory holds only a 64-entry staging ring per warp
//   (3.3 KB instead of 4 KB of per-pixel state plus the ring): 24 warps per SM instead of 12.
// - Active-pixel lists: only pixels whose last contributor is at or past the bucket start.
// - Per-pixel constants (dL/dC, gac, pixel centre, n_contrib, subtile bit, incoming state)
//   staged in the ring by list index, the next 32 entries prefetched into registers while the
//   current 32 are processed.
// - Warp-uniform votes skip steps where no lane's Gaussian reaches its pixel; inside, the
//   per-lane decisions are selects, not divergent branches.
// - The forward's subtile mask applied per (Gaussian, pixel): both kernels blend exactly the
//   same pairs.
//
// Reductions: each lane sums its Gaussian's gradient over every pixel of the tile in
// registers (the per-Gaussian design's reduction), so a Gaussian costs exactly one set of
// atomics per tile it is listed in; lanes whose sums are all zero skip theirs.
//
// Gradient conventions (gsplat v_means2d / v_conics / v_colors / v_opacities): w.r.t. the
// pixel mean (mx, my), the scalar-B conic (A, B, C) of sigma = 0.5 (A dx^2 + C dy^2) + B dx dy,
// the effective opacity o and the clamped colour (zeroed for clamped channels).

struct BwdWarp {
    float4 cst[STAGE];  // (dL/dC.rgb, gac), ring by list index % STAGE
    float4 pxi[STAGE];  // (centre x, centre y, n_contrib bits, px | subtile one-hot << 8)
    float4 sin[STAGE];  // state (A.rgb, T) entering the bucket's first Gaussian
#if EXTRA_CH > 0
    float cst_x[EXTRA_CH][STAGE];
    float sin_x[EXTRA_CH][STAGE];
#endif
    uint8_t list[TILE_PX];  // active pixels of the current bucket, increasing px
};

// One staged ring entry in registers.
struct BwdStage {
    float4 cst, pxi, sin;
#if EXTRA_CH > 0
    float cx[EXTRA_CH], sx[EXTRA_CH];
#endif
};

// Read-only inputs of one frame (kernel arguments plus constants derived from them).
struct BwdFrame {
    const float* out_rgb;
    const float* out_T;
    const uint32_t* out_nc;
    const float* out_extra;
    const float* d_rgb;
    const float* d_alpha;
    const float* d_extra;
    float* ckpt;
    float* ck_extra;
    uint32_t width, P;
    float bg0, bg1, bg2;
};

// The staged values of list entry i of chain c's current bucket (first: the chain's first
// bucket, whose incoming state comes from the forward's checkpoint; later buckets read the
// state lane 31 stored in the same slot).
SPL_INLINE BwdStage bwd_load(const BwdFrame& F, const BwdWarp& S, uint32_t i, uint32_t c, bool first,
                             bool chain_start, uint32_t tx0, uint32_t ty0) {
    BwdStage o;
    const uint32_t px = S.list[i];
    const uint32_t gx = tx0 + px % TILE_W, gy = ty0 + px / TILE_W;
    const uint32_t pix = gy * F.width + gx;
    const float gr = F.d_rgb[pix], gg = F.d_rgb[F.P + pix], gb = F.d_rgb[2 * F.P + pix];
    const float da = F.d_alpha != nullptr ? F.d_alpha[pix] : 0.0f;
    const float Tf = F.out_T[pix];
    o.cst = make_float4(gr, gg, gb, Tf * (da - (gr * F.bg0 + gg * F.bg1 + gb * F.bg2)));
    o.pxi = make_float4((float)gx + 0.5f, (float)gy + 0.5f, __uint_as_float(F.out_nc[pix]),
                        __uint_as_float(px | (1u << (8 + rs_subtile(px)))));
    const size_t e = (size_t)c * TILE_PX + px;
    float4 st = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (!first || !chain_start) {
#if CKPT_F16
        const uint2 h = ((const uint2*)F.ckpt)[e];
        st = make_float4(f16_to_f32((uint16_t)(h.x & 0xffffu)), f16_to_f32((uint16_t)(h.x >> 16)),
                         f16_to_f32((uint16_t)(h.y & 0xffffu)), f16_to_f32((uint16_t)(h.y >> 16)));
#else
        st = ((const float4*)F.ckpt)[e];
#endif
    }
    if (first) {
        // Forward checkpoint (C, T) -> remaining colour A = Cf - C, Cf = out_rgb - T_f bg.
        st.x = fmaf(-Tf, F.bg0, F.out_rgb[pix]) - st.x;
        st.y = fmaf(-Tf, F.bg1, F.out_rgb[F.P + pix]) - st.y;
        st.z = fmaf(-Tf, F.bg2, F.out_rgb[2 * F.P + pix]) - st.z;
    }
    o.sin = st;
#if EXTRA_CH > 0
#pragma unroll
    for (int k = 0; k < EXTRA_CH; ++k) {
        o.cx[k] = F.d_extra[k * F.P + pix];
        const float s = (!first || !chain_start) ? F.ck_extra[((size_t)c * EXTRA_CH + k) * TILE_PX + px] : 0.0f;
        o.sx[k] = first ? F.out_extra[k * F.P + pix] - s : s;
    }
#endif
    return o;
}

SPL_INLINE void bwd_store(BwdWarp& S, uint32_t i, const BwdStage& o) {
    const uint32_t slot = i % STAGE;
    S.cst[slot] = o.cst;
    S.pxi[slot] = o.pxi;
    S.sin[slot] = o.sin;
#if EXTRA_CH > 0
#pragma unroll
    for (int k = 0; k < EXTRA_CH; ++k) {
        S.cst_x[k][slot] = o.cx[k];
        S.sin_x[k][slot] = o.sx[k];
    }
#endif
}

// Persistent warps (BWD_WARPS per block) take chains from *chain_ctr. Per pixel constants
// come from the forward outputs and the loss: d_rgb[3][P], d_alpha[P] (nullable),
// d_extra[E][P]. Accumulates the 9 values (mx, my, A, B, C, o, r, g, b), plus abs (2, when
// FLAG_ABS_GRAD and g2d_abs) and extra (E), into g2d[9][v_cap], g2d_abs[2][v_cap],
// g2d_extra[E][v_cap]. Uses each processed chain's checkpoint slot as scratch.
extern "C" __global__ void __launch_bounds__(BWD_THREADS)
    raster_bwd(const IterParams* __restrict__ params, const FrameCam* __restrict__ cam,
               const SplatRec* __restrict__ rec, const uint32_t* __restrict__ inst_v,
               const uint32_t* __restrict__ tile_start, const uint32_t* __restrict__ chain_off,
               const uint32_t* __restrict__ tile_max_nc, float* __restrict__ ckpt, uint32_t c_cap,
               const float* __restrict__ out_rgb, const float* __restrict__ out_T,
               const uint32_t* __restrict__ out_nc, const float* __restrict__ out_extra,
               const float* __restrict__ d_rgb, const float* __restrict__ d_alpha,
               const float* __restrict__ d_extra, uint32_t v_cap, float* __restrict__ g2d,
               float* __restrict__ g2d_abs, float* __restrict__ g2d_extra, uint32_t* __restrict__ chain_ctr,
               const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    __shared__ BwdWarp s_warp[BWD_WARPS];
    const uint32_t lane = threadIdx.x & 31;
    BwdWarp& S = s_warp[threadIdx.x >> 5];
    // Ring slots outside the current entries are read (and their results discarded) by lanes
    // in the pipeline ramps: keep them finite.
    for (uint32_t i = lane; i < STAGE; i += 32) {
        S.cst[i] = S.pxi[i] = S.sin[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#if EXTRA_CH > 0
#pragma unroll
        for (int k = 0; k < EXTRA_CH; ++k) S.cst_x[k][i] = S.sin_x[k][i] = 0.0f;
#endif
    }
    __syncwarp();

    const uint32_t n_chains = status[ST_N_CHAINS];
    const uint32_t width = cam->width, height = cam->height, grid_w = cam->grid_w, n_tiles = cam->n_tiles;
    BwdFrame F;
    F.out_rgb = out_rgb;
    F.out_T = out_T;
    F.out_nc = out_nc;
    F.out_extra = out_extra;
    F.d_rgb = d_rgb;
    F.d_alpha = d_alpha;
    F.d_extra = d_extra;
    F.ckpt = ckpt;
    F.ck_extra = ckpt + (size_t)c_cap * TILE_PX * CKPT_MAIN_WORDS;
    F.width = width;
    F.P = width * height;
    F.bg0 = params->frame.bg[0];
    F.bg1 = params->frame.bg[1];
    F.bg2 = params->frame.bg[2];
    const bool abs_on = (params->frame.flags & FLAG_ABS_GRAD) != 0 && g2d_abs != nullptr;
    const float ln2 = 0.6931471805599453f;

    for (;;) {
        uint32_t c = 0;
        if (lane == 0) c = atomicAdd(chain_ctr, 1u);
        c = __shfl_sync(FULL_MASK, c, 0);
        if (c >= n_chains) break;

        // Tile of chain c: the last t with chain_off[t] <= c (chain_off is non-decreasing).
        uint32_t lo = 0, hi = n_tiles;
        while (hi - lo > 1) {
            const uint32_t mid = (lo + hi) >> 1;
            if (chain_off[mid] <= c) lo = mid;
            else hi = mid;
        }
        const uint32_t t = lo;
        const uint32_t b0 = (c - chain_off[t]) * CHAIN_K;
        const uint32_t max_nc = tile_max_nc[t];
        if (b0 * BUCKET >= max_nc) continue;
        const uint32_t s0 = tile_start[t];
        const uint32_t n_t = tile_start[t + 1] - s0;
        const uint32_t tx0 = (t % grid_w) * TILE_W, ty0 = (t / grid_w) * TILE_H;

        // Active pixels of the first bucket: inside the image with a contributor at or past 32 b0.
        uint32_t n_act = 0;
        for (uint32_t base = 0; base < TILE_PX; base += 32) {
            const uint32_t px = base + lane;
            const uint32_t gx = tx0 + px % TILE_W, gy = ty0 + px / TILE_W;
            const bool act = gx < width && gy < height && out_nc[gy * width + gx] > b0 * BUCKET;
            const uint32_t ball = __ballot_sync(FULL_MASK, act);
            if (act) S.list[n_act + __popc(ball & lanemask_lt())] = (uint8_t)px;
            n_act += __popc(ball);
        }
        __syncwarp();

        const uint32_t n_buckets = (n_t + BUCKET - 1) / BUCKET;
        const uint32_t b_end = min(min(b0 + CHAIN_K, n_buckets), (max_nc + BUCKET - 1) / BUCKET);
        for (uint32_t b = b0; b < b_end; ++b) {
            const uint32_t g0 = b * BUCKET;
            if (b != b0) {
                // Keep the pixels whose last contributor is at or past this bucket (in place).
                uint32_t m = 0;
                for (uint32_t base = 0; base < n_act; base += 32) {
                    const uint32_t i = base + lane;
                    uint32_t px = 0;
                    bool act = false;
                    if (i < n_act) {
                        px = S.list[i];
                        act = out_nc[(ty0 + px / TILE_W) * width + tx0 + px % TILE_W] > g0;
                    }
                    const uint32_t ball = __ballot_sync(FULL_MASK, act);
                    __syncwarp();  // every lane has read its entry before any is overwritten
                    if (act) S.list[m + __popc(ball & lanemask_lt())] = (uint8_t)px;
                    m += __popc(ball);
                }
                n_act = m;
                __syncwarp();
            }
            if (n_act == 0) break;
            const bool first = b == b0, last = b + 1 == b_end;

            // Stage entries [0, 32) now, prefetch [32, 64) into registers.
            if (lane < n_act) bwd_store(S, lane, bwd_load(F, S, lane, c, first, b0 == 0, tx0, ty0));
            BwdStage nxt;
            if (32 + lane < n_act) nxt = bwd_load(F, S, 32 + lane, c, first, b0 == 0, tx0, ty0);

            // This lane's Gaussian.
            const uint32_t g = g0 + lane;
            const bool has_g = g < n_t && g < max_nc;
            uint32_t v = 0;
            float4 ra = make_float4(0.0f, 0.0f, 0.0f, 0.0f), rb = ra, rc = ra;
            uint32_t gmask = 0;
            if (has_g) {
                v = inst_v[s0 + g];
                const float4* r4 = (const float4*)(rec + v);
                ra = r4[0];
                rb = r4[1];
                rc = r4[2];
                gmask = rs_subtile_mask(rs_ellipse(ra.x, ra.y, ra.z, ra.w, rb.x, rc.y), tx0, ty0);
            }
            // Subtile bits aligned with the staged one-hot (bits 8..15 of pxi.w).
            const uint32_t gmask8 = gmask << 8;
            // Mean gradient from the pre-scaled conic: A dx + B dy = -ln2 (2 ca dx + cb dy).
            const float ca2 = 2.0f * ra.z, cbv = ra.w, cc2 = 2.0f * rb.x;
            const float col0 = rb.z, col1 = rb.w, col2 = rc.x;
#if EXTRA_CH > 0
            float xcol[EXTRA_CH];
#pragma unroll
            for (int k = 0; k < EXTRA_CH; ++k) xcol[k] = k == 0 ? rc.z : 0.0f;
            float g_x[EXTRA_CH], sX[EXTRA_CH];
#pragma unroll
            for (int k = 0; k < EXTRA_CH; ++k) g_x[k] = sX[k] = 0.0f;
#endif
            float g_mx = 0.0f, g_my = 0.0f, g_a = 0.0f, g_b = 0.0f, g_c = 0.0f, g_o = 0.0f;
            float g_r = 0.0f, g_g = 0.0f, g_bl = 0.0f, a_mx = 0.0f, a_my = 0.0f;
            float sA0 = 0.0f, sA1 = 0.0f, sA2 = 0.0f, sT = 0.0f;
            __syncwarp();

            const uint32_t steps = n_act + 31;
            for (uint32_t s = 0; s < steps; ++s) {
                if ((s & 31u) == 0 && s != 0 && s < n_act) {
                    __syncwarp();  // the slots being refilled were last read 2 steps ago
                    if (s + lane < n_act) bwd_store(S, s + lane, nxt);
                    if (s + 32 + lane < n_act) nxt = bwd_load(F, S, s + 32 + lane, c, first, b0 == 0, tx0, ty0);
                    __syncwarp();
                }
                // Incoming state: from the lane below, or for lane 0 from the ring.
                sA0 = __shfl_up_sync(FULL_MASK, sA0, 1);
                sA1 = __shfl_up_sync(FULL_MASK, sA1, 1);
                sA2 = __shfl_up_sync(FULL_MASK, sA2, 1);
                sT = __shfl_up_sync(FULL_MASK, sT, 1);
#if EXTRA_CH > 0
#pragma unroll
                for (int k = 0; k < EXTRA_CH; ++k) sX[k] = __shfl_up_sync(FULL_MASK, sX[k], 1);
#endif
                if (lane == 0 && s < n_act) {
                    const float4 st = S.sin[s % STAGE];
                    sA0 = st.x;
                    sA1 = st.y;
                    sA2 = st.z;
                    sT = st.w;
#if EXTRA_CH > 0
#pragma unroll
                    for (int k = 0; k < EXTRA_CH; ++k) sX[k] = S.sin_x[k][s % STAGE];
#endif
                }
                const int32_t i = (int32_t)s - (int32_t)lane;
                const bool in = i >= 0 && (uint32_t)i < n_act;
                const uint32_t slot = (uint32_t)i % STAGE;
                const float4 pi = S.pxi[slot];
                const uint32_t word = __float_as_uint(pi.w);
                const bool pre = in && g < __float_as_uint(pi.z) && (gmask8 & word) != 0u;
                if (__any_sync(FULL_MASK, pre)) {
                    const float dx = ra.x - pi.x, dy = ra.y - pi.y;
                    const float e = rs_power(ra.z, ra.w, rb.x, dx, dy);
                    float G, a_raw, alpha;
                    const bool valid = rs_alpha(rb.y, e, G, a_raw, alpha) && pre;
                    if (__any_sync(FULL_MASK, valid)) {
                        const float4 k4 = S.cst[slot];  // (gc.rgb, gac)
                        const float w = valid ? alpha * sT : 0.0f;
                        const float one_m = 1.0f - alpha;
                        g_r = fmaf(w, k4.x, g_r);
                        g_g = fmaf(w, k4.y, g_g);
                        g_bl = fmaf(w, k4.z, g_bl);
                        sA0 = fmaf(-col0, w, sA0);
                        sA1 = fmaf(-col1, w, sA1);
                        sA2 = fmaf(-col2, w, sA2);
                        float gcol = k4.x * col0 + k4.y * col1 + k4.z * col2;
                        float gA = k4.x * sA0 + k4.y * sA1 + k4.z * sA2;
#if EXTRA_CH > 0
#pragma unroll
                        for (int k = 0; k < EXTRA_CH; ++k) {
                            const float gx = S.cst_x[k][slot];
                            g_x[k] = fmaf(w, gx, g_x[k]);
                            sX[k] = fmaf(-xcol[k], w, sX[k]);
                            gcol = fmaf(gx, xcol[k], gcol);
                            gA = fmaf(gx, sX[k], gA);
                        }
#endif
#if BLEND == BLEND_FASTERGS
                        const float ra_ = 1.0f / fmaxf(one_m, ONE_MINUS_ALPHA_MIN);
                        const bool through = valid;
#else
                        const float ra_ = 1.0f / one_m;  // alpha <= ALPHA_MAX < 1
                        const bool through = valid && a_raw <= ALPHA_MAX;
#endif
                        const float d_alpha_i = fmaf(gcol, sT, (k4.w - gA) * ra_);
                        g_o += through ? G * d_alpha_i : 0.0f;
                        // q = o G dL/dalpha = -dL/dsigma.
                        const float q = through ? a_raw * d_alpha_i : 0.0f;
                        const float qx = q * dx, qy = q * dy;
                        g_a = fmaf(qx, dx, g_a);
                        g_b = fmaf(qx, dy, g_b);
                        g_c = fmaf(qy, dy, g_c);
                        const float tmx = fmaf(ca2, qx, cbv * qy);
                        const float tmy = fmaf(cbv, qx, cc2 * qy);
                        g_mx += tmx;
                        g_my += tmy;
                        a_mx += fabsf(tmx);
                        a_my += fabsf(tmy);
                        sT = valid ? sT * one_m : sT;
                    }
                }
                if (lane == 31 && in && !last) {
                    // State after this bucket, for the next bucket of the chain.
                    const size_t e = (size_t)c * TILE_PX + (word & 0xffu);
#if CKPT_F16
                    const uint32_t w0 = (uint32_t)f32_to_f16(sA0) | ((uint32_t)f32_to_f16(sA1) << 16);
                    const uint32_t w1 = (uint32_t)f32_to_f16(sA2) | ((uint32_t)f32_to_f16(sT) << 16);
                    ((uint2*)ckpt)[e] = make_uint2(w0, w1);
#else
                    ((float4*)ckpt)[e] = make_float4(sA0, sA1, sA2, sT);
#endif
#if EXTRA_CH > 0
#pragma unroll
                    for (int k = 0; k < EXTRA_CH; ++k)
                        F.ck_extra[((size_t)c * EXTRA_CH + k) * TILE_PX + (word & 0xffu)] = sX[k];
#endif
                }
            }
            __syncwarp();  // lane 31's state stores are visible to the next bucket's staging

            // One atomic set per (Gaussian, tile); lanes without a contribution skip theirs.
            bool any = g_mx != 0.0f || g_my != 0.0f || g_a != 0.0f || g_b != 0.0f || g_c != 0.0f || g_o != 0.0f ||
                       g_r != 0.0f || g_g != 0.0f || g_bl != 0.0f;
#if EXTRA_CH > 0
#pragma unroll
            for (int k = 0; k < EXTRA_CH; ++k) any = any || g_x[k] != 0.0f;
#endif
            if (has_g && any) {
                const uint32_t fl = __float_as_uint(rc.w);
                // dL/dsigma = -q: the q sums are negated; A and C carry the 1/2; the mean
                // terms were summed with the pre-scaled conic (factor -ln2, see above).
                atomicAdd(&g2d[0 * (size_t)v_cap + v], ln2 * g_mx);
                atomicAdd(&g2d[1 * (size_t)v_cap + v], ln2 * g_my);
                atomicAdd(&g2d[2 * (size_t)v_cap + v], -0.5f * g_a);
                atomicAdd(&g2d[3 * (size_t)v_cap + v], -g_b);
                atomicAdd(&g2d[4 * (size_t)v_cap + v], -0.5f * g_c);
                atomicAdd(&g2d[5 * (size_t)v_cap + v], g_o);
                if (!(fl & 1u)) atomicAdd(&g2d[6 * (size_t)v_cap + v], g_r);
                if (!(fl & 2u)) atomicAdd(&g2d[7 * (size_t)v_cap + v], g_g);
                if (!(fl & 4u)) atomicAdd(&g2d[8 * (size_t)v_cap + v], g_bl);
                if (abs_on) {
                    atomicAdd(&g2d_abs[v], ln2 * a_mx);
                    atomicAdd(&g2d_abs[(size_t)v_cap + v], ln2 * a_my);
                }
#if EXTRA_CH > 0
#pragma unroll
                for (int k = 0; k < EXTRA_CH; ++k) atomicAdd(&g2d_extra[(size_t)k * v_cap + v], g_x[k]);
#endif
            }
        }
    }
}
