// Tile rasterizer forward: train, infer and count modes. Owner: raster (E3).
// Spec: spec-raster Section 4 (gsplat blend semantics by default, BLEND_FASTERGS as a flag).
// The warp-per-subtile layout follows Faster-GS's warp-level culling (Apache-2.0); the
// ellipse-vs-subtile test and the rest are our own code.
//
// One block per tile (TILE_W x TILE_H, TILE_PX threads); warp w owns the 8x4 subtile
// (w % 2, w / 2), lane l the pixel (l % 8, l / 8) inside it. Warps walk the tile list on
// their own, 32 records per chunk: lane l holds record c0 + l (the tile's warps read the
// same records, so most come from L1), tests it against the warp's subtile
// (rs_subtile_hit), and the warp ballots, parks the chunk in its own shared-memory slot and
// blends only the hits, two per iteration with selects instead of branches. The next chunk's
// records are already in flight (its inst_v two chunks ahead). No block-wide barrier at all
// (measured with block-wide batches: half the warp cycles waited at the batch barrier for
// the busiest subtile), so a warp whose 32 pixels are done simply stops. Chained checkpoints
// (C, T) every CHAIN_LEN Gaussians of the tile list in train mode (spec-raster 5.2).

struct FwdRec {
    float4 a, b, c;
};

SPL_INLINE FwdRec fwd_load(const SplatRec* __restrict__ rec, uint32_t v) {
    const float4* r4 = (const float4*)(rec + v);
    FwdRec r;
    r.a = r4[0];
    r.b = r4[1];
    r.c = r4[2];
    return r;
}

// One blend step of a pixel (spec-raster 4.2), branch-free. `ok_in`: the Gaussian exists
// and passed the subtile test.
SPL_INLINE bool fwd_blend(const float4& ra, const float4& rb, const float4& rc, bool ok_in, float pcx, float pcy,
                          float* C, float& T, uint32_t& nc, bool& done, uint32_t idx) {
    const float dx = ra.x - pcx, dy = ra.y - pcy;
    const float e = rs_power(ra.z, ra.w, rb.x, dx, dy);
    float G, a_raw, alpha;
    const bool ok = rs_alpha(rb.y, e, G, a_raw, alpha) && ok_in && !done;
    const float Tn = T * (1.0f - alpha);
#if BLEND == BLEND_FASTERGS
    const bool blend = ok;
    const bool stop = ok && Tn < T_MIN;  // blended, then stop
#else
    const bool stop = ok && Tn <= T_MIN;  // stop before blending
    const bool blend = ok && !stop;
#endif
    const float w = blend ? alpha * T : 0.0f;
    C[0] = fmaf(rb.z, w, C[0]);
    C[1] = fmaf(rb.w, w, C[1]);
    C[2] = fmaf(rc.x, w, C[2]);
#if EXTRA_CH > 0
    C[3] = fmaf(rc.z, w, C[3]);
#endif
    T = blend ? Tn : T;
    nc = blend ? idx : nc;
    done = done || stop;
    return blend;
}

// mode: MODE_TRAIN (ckpt, out_nc, tile_max_nc), MODE_INFER (colour and T only),
// MODE_COUNT (count[v] += masked pixels the Gaussian blends into; warp-aggregated).
// Outputs are planar CHW: out_rgb[3][P] (background composited), out_T[P], out_nc[P],
// out_extra[EXTRA_CH][P]. ckpt layout: raster_common.cuh (c_cap chains). Train mode:
// tile_max_nc must be zero on entry (atomicMax; raster_fwd's stage function clears it).
extern "C" __global__ void __launch_bounds__(TILE_PX)
    raster_fwd(const IterParams* __restrict__ params, const FrameCam* __restrict__ cam,
               const SplatRec* __restrict__ rec, const uint32_t* __restrict__ inst_v,
               const uint32_t* __restrict__ tile_start, const uint32_t* __restrict__ chain_off, uint32_t mode,
               const uint8_t* __restrict__ mask, float* __restrict__ out_rgb, float* __restrict__ out_T,
               uint32_t* __restrict__ out_nc, float* __restrict__ out_extra, float* __restrict__ ckpt,
               uint32_t c_cap, uint32_t* __restrict__ tile_max_nc, uint32_t* __restrict__ count,
               const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    const uint32_t t = blockIdx.x;
    if (t >= cam->n_tiles) return;

    __shared__ float4 s_rec[FWD_WARPS][32 * 3];
    __shared__ uint32_t s_v[FWD_WARPS][32];

    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const uint32_t width = cam->width, height = cam->height, grid_w = cam->grid_w;
    const uint32_t tx0 = (t % grid_w) * TILE_W, ty0 = (t / grid_w) * TILE_H;
    const uint32_t sx0 = tx0 + (warp % SUB_COLS) * SUB_W, sy0 = ty0 + (warp / SUB_COLS) * SUB_H;
    const uint32_t lx = sx0 - tx0 + (lane % SUB_W), ly = sy0 - ty0 + (lane / SUB_W);
    const uint32_t px = ly * TILE_W + lx;
    const uint32_t gx = tx0 + lx, gy = ty0 + ly;
    const bool inside = gx < width && gy < height;
    const uint32_t pix = gy * width + gx;
    const float pcx = (float)gx + 0.5f, pcy = (float)gy + 0.5f;

    const uint32_t s0 = tile_start[t];
    const uint32_t n = tile_start[t + 1] - s0;
    const bool train = mode == MODE_TRAIN;
    const bool counting = mode == MODE_COUNT;
    const bool m_px = counting && inside && mask[pix] != 0;

    float C[N_CH];
#pragma unroll
    for (int k = 0; k < N_CH; ++k) C[k] = 0.0f;
    float T = 1.0f;
    uint32_t nc = 0;
    bool done = !inside;

    float4* ck_main = (float4*)ckpt;
    float* ck_extra = ckpt + (size_t)c_cap * TILE_PX * CKPT_MAIN_WORDS;
    const uint32_t chain0 = train ? chain_off[t] : 0u;
    float4* srec = s_rec[warp];
    uint32_t* sv = s_v[warp];

    // Software pipeline: records of chunk c0 in `cur`, inst_v of chunk c0 + 32 in `v_next`.
    uint32_t v_cur = lane < n ? inst_v[s0 + lane] : 0u;
    uint32_t v_next = 32 + lane < n ? inst_v[s0 + 32 + lane] : 0u;
    FwdRec cur;
    if (lane < n) cur = fwd_load(rec, v_cur);

    for (uint32_t c0 = 0; c0 < n; c0 += 32) {
        if (__all_sync(FULL_MASK, done)) break;
        if (train && c0 != 0 && c0 % CHAIN_LEN == 0 && !done) {
            const size_t e = (size_t)(chain0 + c0 / CHAIN_LEN) * TILE_PX + px;
#if CKPT_F16
            const uint32_t w0 = (uint32_t)f32_to_f16(C[0]) | ((uint32_t)f32_to_f16(C[1]) << 16);
            const uint32_t w1 = (uint32_t)f32_to_f16(C[2]) | ((uint32_t)f32_to_f16(T) << 16);
            ((uint2*)ckpt)[e] = make_uint2(w0, w1);
#else
            ck_main[e] = make_float4(C[0], C[1], C[2], T);
#endif
#pragma unroll
            for (int k = 0; k < EXTRA_CH; ++k)
                ck_extra[((size_t)(chain0 + c0 / CHAIN_LEN) * EXTRA_CH + k) * TILE_PX + px] = C[3 + k];
        }
        const uint32_t j = c0 + lane;
        const bool hit = j < n && rs_subtile_hit(rs_ellipse(cur.a.x, cur.a.y, cur.a.z, cur.a.w, cur.b.x, cur.c.y),
                                                 sx0, sy0);
        uint32_t hits = __ballot_sync(FULL_MASK, hit);
        if (hits) {
            __syncwarp();  // the previous chunk's readers are done with the slot
            if (hit) {
                srec[lane * 3 + 0] = cur.a;
                srec[lane * 3 + 1] = cur.b;
                srec[lane * 3 + 2] = cur.c;
                sv[lane] = v_cur;
            }
            __syncwarp();
        }
        // Next chunk in flight while this one blends.
        v_cur = v_next;
        if (j + 32 < n) cur = fwd_load(rec, v_cur);
        v_next = j + 64 < n ? inst_v[s0 + j + 64] : 0u;

        while (hits) {
            const uint32_t j1 = __ffs(hits) - 1;
            hits &= hits - 1;
            const bool two = hits != 0;
            const uint32_t j2 = two ? __ffs(hits) - 1 : j1;
            hits &= hits - 1;
            const float4 a1 = srec[j1 * 3 + 0], b1 = srec[j1 * 3 + 1], e1 = srec[j1 * 3 + 2];
            const float4 a2 = srec[j2 * 3 + 0], b2 = srec[j2 * 3 + 1], e2 = srec[j2 * 3 + 2];
            const bool bl1 = fwd_blend(a1, b1, e1, true, pcx, pcy, C, T, nc, done, c0 + j1 + 1);
            const bool bl2 = fwd_blend(a2, b2, e2, two, pcx, pcy, C, T, nc, done, c0 + j2 + 1);
            if (counting) {
                const uint32_t m1 = __ballot_sync(FULL_MASK, bl1 && m_px);
                const uint32_t m2 = __ballot_sync(FULL_MASK, bl2 && m_px);
                if (lane == 0 && m1) atomicAdd(&count[sv[j1]], (uint32_t)__popc(m1));
                if (lane == 0 && m2) atomicAdd(&count[sv[j2]], (uint32_t)__popc(m2));
            }
            if (__all_sync(FULL_MASK, done)) break;
        }
    }

    if (inside) {
        const uint32_t P = width * height;
        const float* bg = params->frame.bg;
        out_rgb[pix] = fmaf(T, bg[0], C[0]);
        out_rgb[P + pix] = fmaf(T, bg[1], C[1]);
        out_rgb[2 * P + pix] = fmaf(T, bg[2], C[2]);
        out_T[pix] = T;
        if (train) out_nc[pix] = nc;
#pragma unroll
        for (int k = 0; k < EXTRA_CH; ++k) out_extra[k * P + pix] = C[3 + k];
    }
    if (train) {
        const uint32_t m = __reduce_max_sync(FULL_MASK, nc);
        if (lane == 0 && m) atomicMax(&tile_max_nc[t], m);
    }
}
