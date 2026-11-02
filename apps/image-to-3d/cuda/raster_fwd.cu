// Tile rasterizer forward: train, infer and count modes. Owner: raster (E3).
// Spec: spec-raster Section 4 (gsplat blend semantics by default, BLEND_FASTERGS as a flag).
// Warp-level subtile culling as in Faster-GS (Apache-2.0); the ellipse-vs-subtile test and
// everything else here is our own code.
//
// One block per tile (TILE_W x TILE_H, TILE_PX threads); warp w owns the 8x4 subtile
// (w % 2, w / 2), lane l the pixel (l % 8, l / 8) inside it. Batches of BATCH records go
// to shared memory together with each record's 8-bit subtile mask (rs_subtile_mask,
// computed once per record by the thread that loads it); per 32-record chunk a warp ballots
// the records whose mask has its bit and blends only those. Chained checkpoints (C, T) are
// written every CHAIN_LEN Gaussians of the tile list in train mode (spec-raster 5.2).

// mode: MODE_TRAIN (ckpt, out_nc, tile_max_nc), MODE_INFER (colour and T only),
// MODE_COUNT (count[v] += masked pixels the Gaussian blends into; warp-aggregated).
// Outputs are planar CHW: out_rgb[3][P] (background composited), out_T[P], out_nc[P],
// out_extra[EXTRA_CH][P]. ckpt layout: raster_common.cuh (c_cap chains).
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

    __shared__ float4 s_rec[BATCH * 3];
    __shared__ uint32_t s_v[BATCH];
    __shared__ uint8_t s_mask[BATCH];
    __shared__ uint32_t s_nc[FWD_WARPS];

    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const uint32_t width = cam->width, height = cam->height, grid_w = cam->grid_w;
    const uint32_t tx0 = (t % grid_w) * TILE_W, ty0 = (t / grid_w) * TILE_H;
    const uint32_t lx = (warp % SUB_COLS) * SUB_W + (lane % SUB_W);
    const uint32_t ly = (warp / SUB_COLS) * SUB_H + (lane / SUB_W);
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

    for (uint32_t base = 0; base < n; base += BATCH) {
        if (__syncthreads_and(done)) break;
        // BATCH records per batch, TILE_PX loaders (16x12 tiles: 192 threads, two passes).
        for (uint32_t i = tid; i < BATCH && base + i < n; i += TILE_PX) {
            const uint32_t v = inst_v[s0 + base + i];
            const float4* r4 = (const float4*)(rec + v);
            const float4 a = r4[0], b = r4[1], c = r4[2];
            s_rec[i * 3 + 0] = a;
            s_rec[i * 3 + 1] = b;
            s_rec[i * 3 + 2] = c;
            s_v[i] = v;
            s_mask[i] = (uint8_t)rs_subtile_mask(a.x, a.y, a.z, a.w, b.x, c.y, (float)tx0, (float)ty0);
        }
        __syncthreads();
        if (__all_sync(FULL_MASK, done)) continue;
        const uint32_t nb = min((uint32_t)BATCH, n - base);
        for (uint32_t c0 = 0; c0 < nb; c0 += 32) {
            const uint32_t gi = base + c0;  // tile-list index of the chunk's first Gaussian
            if (train && gi != 0 && gi % CHAIN_LEN == 0 && !done) {
                const size_t e = (size_t)(chain0 + gi / CHAIN_LEN) * TILE_PX + px;
#if CKPT_F16
                const uint32_t w0 = (uint32_t)f32_to_f16(C[0]) | ((uint32_t)f32_to_f16(C[1]) << 16);
                const uint32_t w1 = (uint32_t)f32_to_f16(C[2]) | ((uint32_t)f32_to_f16(T) << 16);
                ((uint2*)ckpt)[e] = make_uint2(w0, w1);
#else
                ck_main[e] = make_float4(C[0], C[1], C[2], T);
#endif
#pragma unroll
                for (int k = 0; k < EXTRA_CH; ++k)
                    ck_extra[((size_t)(chain0 + gi / CHAIN_LEN) * EXTRA_CH + k) * TILE_PX + px] = C[3 + k];
            }
            const uint32_t k = c0 + lane;
            uint32_t hits = __ballot_sync(FULL_MASK, k < nb && ((s_mask[k] >> warp) & 1u));
            while (hits) {
                const uint32_t jj = __ffs(hits) - 1;
                hits &= hits - 1;
                const float4 ra = s_rec[(c0 + jj) * 3 + 0];
                const float4 rb = s_rec[(c0 + jj) * 3 + 1];
                bool blended = false;
                if (!done) {
                    const float dx = ra.x - pcx, dy = ra.y - pcy;
                    const float e = rs_power(ra.z, ra.w, rb.x, dx, dy);
                    float G, a_raw, alpha;
                    if (rs_alpha(rb.y, e, G, a_raw, alpha)) {
                        const float Tn = T * (1.0f - alpha);
#if BLEND == BLEND_FASTERGS
                        const bool stop = false;
#else
                        const bool stop = Tn <= T_MIN;
#endif
                        if (stop) {
                            done = true;
                        } else {
                            const float4 rc = s_rec[(c0 + jj) * 3 + 2];
                            const float w = alpha * T;
                            C[0] = fmaf(rb.z, w, C[0]);
                            C[1] = fmaf(rb.w, w, C[1]);
                            C[2] = fmaf(rc.x, w, C[2]);
#if EXTRA_CH > 0
                            C[3] = fmaf(rc.z, w, C[3]);
#endif
                            T = Tn;
                            nc = gi + jj + 1;
                            blended = true;
#if BLEND == BLEND_FASTERGS
                            if (T < T_MIN) done = true;
#endif
                        }
                    }
                }
                if (counting) {
                    const uint32_t bm = __ballot_sync(FULL_MASK, blended && m_px);
                    if (lane == 0 && bm) atomicAdd(&count[s_v[c0 + jj]], (uint32_t)__popc(bm));
                }
                if (__all_sync(FULL_MASK, done)) break;
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
        if (lane == 0) s_nc[warp] = m;
        __syncthreads();
        if (tid == 0) {
            uint32_t mt = 0;
#pragma unroll
            for (int w = 0; w < FWD_WARPS; ++w) mt = max(mt, s_nc[w]);
            tile_max_nc[t] = mt;
        }
    }
}
