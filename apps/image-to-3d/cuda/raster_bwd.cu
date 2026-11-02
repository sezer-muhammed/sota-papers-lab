// Chained per-Gaussian raster backward. Owner: raster (E3).
// Spec: spec-raster Sections 5.1-5.2. Base: the per-Gaussian backward of Taming-3DGS as
// refined by Faster-GS (Apache-2.0): a warp owns a bucket of 32 consecutive tile-list
// Gaussians (lane l owns Gaussian 32b + l) and the tile's pixels flow through the warp
// diagonally, lane l handling list entry s - l at step s, the mutable pixel state passed one
// lane up with __shfl_up_sync each step. Written from the papers and the spec; no code copied.
//
// Our changes: chains of CHAIN_K buckets per warp with the pixel state (A, T) kept in shared
// memory between buckets, so the forward checkpoints only every CHAIN_LEN Gaussians; an
// active-pixel list (pixels whose last contributor is past the bucket start) instead of the
// whole tile; per-pixel constants staged in a 64-entry ring instead of shuffled; the
// forward's subtile mask applied per (Gaussian, pixel); SoA atomics that skip all-zero lanes.
//
// Reductions: each lane sums its Gaussian's gradient over every pixel of the tile in
// registers (the per-Gaussian design's reduction), so a Gaussian costs exactly one set of
// atomics per tile it is listed in.
//
// Gradient conventions (gsplat v_means2d / v_conics / v_colors / v_opacities): w.r.t. the
// pixel mean (mx, my), the scalar-B conic (A, B, C) of sigma = 0.5 (A dx^2 + C dy^2) + B dx dy,
// the effective opacity o and the clamped colour (zeroed for clamped channels).

struct BwdWarp {
    float4 state[TILE_PX];  // (A.rgb, T) per tile pixel
#if EXTRA_CH > 0
    float state_x[EXTRA_CH][TILE_PX];
    float cst_x[EXTRA_CH][STAGE];
#endif
    float4 cst[STAGE];  // (dL/dC.rgb, gac) of list entries, ring by list index % STAGE
    uint32_t cst_nc[STAGE];
    uint8_t list[TILE_PX];  // active pixels of the current bucket, increasing px
};

// Persistent warps (BWD_WARPS per block) take chains from *chain_ctr. Per pixel constants
// come from the forward outputs and the loss: d_rgb[3][P], d_alpha[P] (nullable),
// d_extra[E][P]. Accumulates the 9 values (mx, my, A, B, C, o, r, g, b), plus abs (2, when
// FLAG_ABS_GRAD and g2d_abs) and extra (E), into g2d[9][v_cap], g2d_abs[2][v_cap],
// g2d_extra[E][v_cap].
extern "C" __global__ void __launch_bounds__(BWD_THREADS)
    raster_bwd(const IterParams* __restrict__ params, const FrameCam* __restrict__ cam,
               const SplatRec* __restrict__ rec, const uint32_t* __restrict__ inst_v,
               const uint32_t* __restrict__ tile_start, const uint32_t* __restrict__ chain_off,
               const uint32_t* __restrict__ tile_max_nc, const float* __restrict__ ckpt, uint32_t c_cap,
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

    const uint32_t n_chains = status[ST_N_CHAINS];
    const uint32_t width = cam->width, height = cam->height, grid_w = cam->grid_w, n_tiles = cam->n_tiles;
    const uint32_t P = width * height;
    const float bg0 = params->frame.bg[0], bg1 = params->frame.bg[1], bg2 = params->frame.bg[2];
    const bool abs_on = (params->frame.flags & FLAG_ABS_GRAD) != 0 && g2d_abs != nullptr;
    const float4* ck_main = (const float4*)ckpt;
    const float* ck_extra = ckpt + (size_t)c_cap * TILE_PX * CKPT_MAIN_WORDS;

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

        // Initial state from the chain's checkpoint: A = Cf - C_ckpt, T = T_ckpt (chain 0 of a
        // tile starts from C = 0, T = 1 and has no stored checkpoint).
        for (uint32_t i = lane; i < n_act; i += 32) {
            const uint32_t px = S.list[i];
            const uint32_t pix = (ty0 + px / TILE_W) * width + tx0 + px % TILE_W;
            const float Tf = out_T[pix];
            float4 ck = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
            if (b0 != 0) {
                const size_t e = (size_t)c * TILE_PX + px;
#if CKPT_F16
                const uint2 h = ((const uint2*)ckpt)[e];
                ck = make_float4(f16_to_f32((uint16_t)(h.x & 0xffffu)), f16_to_f32((uint16_t)(h.x >> 16)),
                                 f16_to_f32((uint16_t)(h.y & 0xffffu)), f16_to_f32((uint16_t)(h.y >> 16)));
#else
                ck = ck_main[e];
#endif
            }
            S.state[px] = make_float4(fmaf(-Tf, bg0, out_rgb[pix]) - ck.x, fmaf(-Tf, bg1, out_rgb[P + pix]) - ck.y,
                                      fmaf(-Tf, bg2, out_rgb[2 * P + pix]) - ck.z, ck.w);
#if EXTRA_CH > 0
#pragma unroll
            for (int k = 0; k < EXTRA_CH; ++k) {
                const float c0x = b0 != 0 ? ck_extra[((size_t)c * EXTRA_CH + k) * TILE_PX + px] : 0.0f;
                S.state_x[k][px] = out_extra[k * P + pix] - c0x;
            }
#endif
        }
        __syncwarp();

        const uint32_t n_buckets = (n_t + BUCKET - 1) / BUCKET;
        const uint32_t b_end = min(b0 + CHAIN_K, n_buckets);
        for (uint32_t b = b0; b < b_end; ++b) {
            const uint32_t g0 = b * BUCKET;
            if (g0 >= max_nc) break;
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
            }
            if (n_act == 0) break;

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
                gmask = rs_subtile_mask(ra.x, ra.y, ra.z, ra.w, rb.x, rc.y, (float)tx0, (float)ty0);
            }
            // Unscaled conic for the gradients: A = -2 ln2 ca, B = -ln2 cb, C = -2 ln2 cc.
            const float ln2 = 0.6931471805599453f;
            const float cA = -2.0f * ln2 * ra.z, cB = -ln2 * ra.w, cC = -2.0f * ln2 * rb.x;
            const float col0 = rb.z, col1 = rb.w, col2 = rc.x;
            float xcol[EXTRA_CH > 0 ? EXTRA_CH : 1];
#pragma unroll
            for (int k = 0; k < EXTRA_CH; ++k) xcol[k] = k == 0 ? rc.z : 0.0f;

            float g_mx = 0.0f, g_my = 0.0f, g_a = 0.0f, g_b = 0.0f, g_c = 0.0f, g_o = 0.0f;
            float g_r = 0.0f, g_g = 0.0f, g_bl = 0.0f, a_mx = 0.0f, a_my = 0.0f;
            float g_x[EXTRA_CH > 0 ? EXTRA_CH : 1];
#pragma unroll
            for (int k = 0; k < EXTRA_CH; ++k) g_x[k] = 0.0f;

            // State carried through the lanes: (A.rgb, T) (+ A.extra).
            float sA0 = 0.0f, sA1 = 0.0f, sA2 = 0.0f, sT = 0.0f;
            float sX[EXTRA_CH > 0 ? EXTRA_CH : 1];
#pragma unroll
            for (int k = 0; k < EXTRA_CH; ++k) sX[k] = 0.0f;

            const uint32_t steps = n_act + 31;
            for (uint32_t s = 0; s < steps; ++s) {
                if ((s & 31u) == 0 && s < n_act) {
                    // Stage the constants of list entries [s, s + 32).
                    __syncwarp();
                    const uint32_t i = s + lane;
                    if (i < n_act) {
                        const uint32_t px = S.list[i];
                        const uint32_t pix = (ty0 + px / TILE_W) * width + tx0 + px % TILE_W;
                        const float gr = d_rgb[pix], gg = d_rgb[P + pix], gb = d_rgb[2 * P + pix];
                        const float da = d_alpha != nullptr ? d_alpha[pix] : 0.0f;
                        const float gac = out_T[pix] * (da - (gr * bg0 + gg * bg1 + gb * bg2));
                        S.cst[i % STAGE] = make_float4(gr, gg, gb, gac);
                        S.cst_nc[i % STAGE] = out_nc[pix];
#if EXTRA_CH > 0
#pragma unroll
                        for (int k = 0; k < EXTRA_CH; ++k) S.cst_x[k][i % STAGE] = d_extra[k * P + pix];
#endif
                    }
                    __syncwarp();
                }
                // Incoming state: from the lane below, or for lane 0 from shared memory.
                sA0 = __shfl_up_sync(FULL_MASK, sA0, 1);
                sA1 = __shfl_up_sync(FULL_MASK, sA1, 1);
                sA2 = __shfl_up_sync(FULL_MASK, sA2, 1);
                sT = __shfl_up_sync(FULL_MASK, sT, 1);
#pragma unroll
                for (int k = 0; k < EXTRA_CH; ++k) sX[k] = __shfl_up_sync(FULL_MASK, sX[k], 1);
                if (lane == 0 && s < n_act) {
                    const uint32_t px = S.list[s];
                    const float4 st = S.state[px];
                    sA0 = st.x;
                    sA1 = st.y;
                    sA2 = st.z;
                    sT = st.w;
#if EXTRA_CH > 0
#pragma unroll
                    for (int k = 0; k < EXTRA_CH; ++k) sX[k] = S.state_x[k][px];
#endif
                }
                const int32_t i = (int32_t)s - (int32_t)lane;
                if (i >= 0 && (uint32_t)i < n_act) {
                    const uint32_t px = S.list[i];
                    const uint32_t slot = (uint32_t)i % STAGE;
                    if (g < S.cst_nc[slot] && ((gmask >> rs_subtile(px)) & 1u)) {
                        const float pcx = (float)(tx0 + px % TILE_W) + 0.5f;
                        const float pcy = (float)(ty0 + px / TILE_W) + 0.5f;
                        const float dx = ra.x - pcx, dy = ra.y - pcy;
                        const float e = rs_power(ra.z, ra.w, rb.x, dx, dy);
                        float G, a_raw, alpha;
                        if (rs_alpha(rb.y, e, G, a_raw, alpha)) {
                            const float4 k4 = S.cst[slot];  // (gc.rgb, gac)
                            const float w = alpha * sT;
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
                            const float ra_ = 1.0f / fmaxf(one_m, ONE_MINUS_ALPHA_MIN);
                            const float d_alpha_i = fmaf(gcol, sT, (k4.w - gA) * ra_);
#if BLEND == BLEND_FASTERGS
                            const bool through = true;
#else
                            const bool through = a_raw <= ALPHA_MAX;
#endif
                            if (through) {
                                g_o = fmaf(G, d_alpha_i, g_o);
                                // q = o G dL/dalpha = -dL/dsigma.
                                const float q = a_raw * d_alpha_i;
                                const float qx = q * dx, qy = q * dy;
                                g_a = fmaf(qx, dx, g_a);
                                g_b = fmaf(qx, dy, g_b);
                                g_c = fmaf(qy, dy, g_c);
                                const float tmx = fmaf(cA, qx, cB * qy);
                                const float tmy = fmaf(cB, qx, cC * qy);
                                g_mx += tmx;
                                g_my += tmy;
                                a_mx += fabsf(tmx);
                                a_my += fabsf(tmy);
                            }
                            sT = sT * one_m;
                        }
                    }
                    if (lane == 31) {
                        S.state[px] = make_float4(sA0, sA1, sA2, sT);
#if EXTRA_CH > 0
#pragma unroll
                        for (int k = 0; k < EXTRA_CH; ++k) S.state_x[k][px] = sX[k];
#endif
                    }
                }
            }
            __syncwarp();

            // One atomic set per (Gaussian, tile); lanes without a contribution skip theirs.
            bool any = g_mx != 0.0f || g_my != 0.0f || g_a != 0.0f || g_b != 0.0f || g_c != 0.0f || g_o != 0.0f ||
                       g_r != 0.0f || g_g != 0.0f || g_bl != 0.0f;
#pragma unroll
            for (int k = 0; k < EXTRA_CH; ++k) any = any || g_x[k] != 0.0f;
            if (has_g && any) {
                const uint32_t fl = __float_as_uint(rc.w);
                // sigma-gradients: dL/dsigma = -q, so negate the q sums; A and C carry the 1/2.
                atomicAdd(&g2d[0 * (size_t)v_cap + v], -g_mx);
                atomicAdd(&g2d[1 * (size_t)v_cap + v], -g_my);
                atomicAdd(&g2d[2 * (size_t)v_cap + v], -0.5f * g_a);
                atomicAdd(&g2d[3 * (size_t)v_cap + v], -g_b);
                atomicAdd(&g2d[4 * (size_t)v_cap + v], -0.5f * g_c);
                atomicAdd(&g2d[5 * (size_t)v_cap + v], g_o);
                if (!(fl & 1u)) atomicAdd(&g2d[6 * (size_t)v_cap + v], g_r);
                if (!(fl & 2u)) atomicAdd(&g2d[7 * (size_t)v_cap + v], g_g);
                if (!(fl & 4u)) atomicAdd(&g2d[8 * (size_t)v_cap + v], g_bl);
                if (abs_on) {
                    atomicAdd(&g2d_abs[v], a_mx);
                    atomicAdd(&g2d_abs[(size_t)v_cap + v], a_my);
                }
#pragma unroll
                for (int k = 0; k < EXTRA_CH; ++k) atomicAdd(&g2d_extra[(size_t)k * v_cap + v], g_x[k]);
            }
        }
    }
}
