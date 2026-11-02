// Training loss 0.8 L1 + 0.2 (1 - SSIM), forward and backward in one pass over the
// render, plus target resampling and the VCD error map. Owner: loss (E4).
// Spec: spec-train Section 3. Design after fused-ssim (MIT, a7c48d6): separable 11-tap
// Gaussian sums in shared memory, the SSIM partials w.r.t. the moment maps, and the
// backward as a second separable convolution of those partials. Our own code.
//
// Images are planar f32 [3][H][W] with H, W = IterParams.frame.render_h/w (the graph
// replays at a fixed size; sizes are never kernel arguments, the grid covers the largest
// render and surplus blocks exit). blockIdx.z is the colour channel. Sums go to a 2-float
// device accumulator, already divided by N = 3 H W per iteration: sums[0] += mean |x - y|,
// sums[1] += mean S; read every ~100 iterations.

#define SSIM_C1 1.0e-4f
#define SSIM_C2 9.0e-4f
#ifndef LAMBDA_L1
#define LAMBDA_L1 0.8f
#endif
#ifndef LAMBDA_DSSIM
#define LAMBDA_DSSIM 0.2f
#endif

__constant__ float c_gauss[11] = {0.0010283801f, 0.0075987581f, 0.0360007721f, 0.1093606895f,
                                  0.2130055377f, 0.2660117249f, 0.2130055377f, 0.1093606895f,
                                  0.0360007721f, 0.0075987581f, 0.0010283801f};

// S and its scaled partials a, b, d (spec-train 3.1) from the five moments of one pixel.
// k = -lambda_dssim / N. A clamped S has zero partials.
SPL_INLINE float ssim_pixel(float mx, float my, float mxx, float myy, float mxy, float k, float& pa, float& pb,
                            float& pd) {
    float sx = fmaxf(0.f, mxx - mx * mx);
    float sy = fmaxf(0.f, myy - my * my);
    float sxy = mxy - mx * my;
    float A = mx * mx + my * my + SSIM_C1;
    float B = sx + sy + SSIM_C2;
    float Cn = 2.f * mx * my + SSIM_C1;
    float D = 2.f * sxy + SSIM_C2;
    float rAB = 1.f / (A * B);
    float S = Cn * D * rAB;
    if (S > 1.f || S < -1.f) {
        pa = pb = pd = 0.f;
        return fminf(1.f, fmaxf(-1.f, S));
    }
    float rA = 1.f / A, rB = 1.f / B;
    float dm1 = 2.f * my * (D - Cn) * rAB + 2.f * mx * S * (rB - rA);
    float dm2 = -S * rB;
    float dm12 = 2.f * Cn * rAB;
    pa = k * dm1;
    pb = k * dm2;
    pd = k * dm12;
    return S;
}

// 4-byte global -> shared async copy; writes 0 when !valid (src-size 0, nothing read).
SPL_INLINE void cp_async4_zfill(float* smem, const float* gmem, bool valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(smem_u32(smem)), "l"(gmem), "r"(valid ? 4 : 0)
                 : "memory");
}

SPL_INLINE float sgn(float x) { return x > 0.f ? 1.f : (x < 0.f ? -1.f : 0.f); }

// Block sum of two values; thread 0 of the block adds them (times scale) into the f64
// accumulator (f32 atomics over ~10^4 blocks x 100 iterations lose ~1e-6 of S).
template <int THREADS>
SPL_INLINE void block_sums(float a, float b, double scale, double* sums) {
    __shared__ float red[2][THREADS / 32];
    a = warp_sum(a);
    b = warp_sum(b);
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if ((tid & 31) == 0) {
        red[0][tid >> 5] = a;
        red[1][tid >> 5] = b;
    }
    __syncthreads();
    if (tid < 32) {
        a = tid < THREADS / 32 ? red[0][tid] : 0.f;
        b = tid < THREADS / 32 ? red[1][tid] : 0.f;
        a = warp_sum(a);
        b = warp_sum(b);
        if (tid == 0) {
            atomicAdd(&sums[0], (double)a * scale);
            atomicAdd(&sums[1], (double)b * scale);
        }
    }
}

// ---------------------------------------------------------------- resample_target

// PyTorch interpolate(mode="bilinear", antialias=True) weights along one axis: output i of
// `out` from `in` samples; returns the first source index and fills up to `max_taps`
// normalized weights.
SPL_INLINE int aa_weights(int i, int in, int out, float* w, int max_taps, int& taps) {
    float scale = (float)in / (float)out;
    float support = scale >= 1.f ? scale : 1.f;
    float inv = scale >= 1.f ? 1.f / scale : 1.f;
    float center = scale * (i + 0.5f);
    int xmin = max((int)(center - support + 0.5f), 0);
    int xmax = min((int)(center + support + 0.5f), in);
    taps = min(xmax - xmin, max_taps);
    float total = 0.f;
    for (int j = 0; j < taps; ++j) {
        float x = fabsf((j + xmin - center + 0.5f) * inv);
        float v = x < 1.f ? 1.f - x : 0.f;
        w[j] = v;
        total += v;
    }
    float r = total > 0.f ? 1.f / total : 0.f;
    for (int j = 0; j < taps; ++j) w[j] *= r;
    return xmin;
}

#define RESAMPLE_MAX_TAPS 34  // support 2 s + 1 at s <= 16

// Cached u8 RGB (pool + IterParams.target_offset, target_w x target_h) -> f32 planar at
// the render size: copy at scale 1, separable antialiased tent filter otherwise (6.5).
// One thread per output pixel, all three channels.
extern "C" __global__ void resample_target(const IterParams* __restrict__ params, const uint8_t* __restrict__ pool,
                                           float* __restrict__ target, const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    const IterParams& P = *params;
    const int W = P.frame.render_w, H = P.frame.render_h;
    const int tw = P.target_w, th = P.target_h;
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const uint8_t* src = pool + P.target_offset;
    const size_t hw = (size_t)W * H, o = (size_t)y * W + x;
    const float k = 1.f / 255.f;
    if (tw == W && th == H) {
        const uint8_t* s = src + 3 * o;
        target[o] = s[0] * k;
        target[hw + o] = s[1] * k;
        target[2 * hw + o] = s[2] * k;
        return;
    }
    float wx[RESAMPLE_MAX_TAPS], wy[RESAMPLE_MAX_TAPS];
    int nx, ny;
    int x0 = aa_weights(x, tw, W, wx, RESAMPLE_MAX_TAPS, nx);
    int y0 = aa_weights(y, th, H, wy, RESAMPLE_MAX_TAPS, ny);
    float r = 0.f, g = 0.f, b = 0.f;
    for (int j = 0; j < ny; ++j) {
        const uint8_t* row = src + 3 * ((size_t)(y0 + j) * tw + x0);
        float hr = 0.f, hg = 0.f, hb = 0.f;
        for (int i = 0; i < nx; ++i) {
            hr += wx[i] * row[3 * i];
            hg += wx[i] * row[3 * i + 1];
            hb += wx[i] * row[3 * i + 2];
        }
        r += wy[j] * hr;
        g += wy[j] * hg;
        b += wy[j] * hb;
    }
    target[o] = r * k;
    target[hw + o] = g * k;
    target[2 * hw + o] = b * k;
}

// ---------------------------------------------------------------- v1: two kernels, 16 x 16

#define V1_T 16
#define V1_S (V1_T + 10)

// v1 pass 1: five separable 11x11 Gaussian sums per channel, S and the three scaled
// partials a, b, d (planes [3 partials][3 channels][H][W]); block-reduced sums into `sums`.
extern "C" __global__ void ssim_fwd(const IterParams* __restrict__ params, const float* __restrict__ x, const float* __restrict__ y,
             float* __restrict__ partials, double* __restrict__ sums, const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    const int W = params->frame.render_w, H = params->frame.render_h;
    const int bx = blockIdx.x * V1_T, by = blockIdx.y * V1_T, ch = blockIdx.z;
    if (bx >= W || by >= H) return;
    const size_t hw = (size_t)W * H;
    const float* X = x + ch * hw;
    const float* Y = y + ch * hw;
    __shared__ float sx[V1_S][V1_S], sy[V1_S][V1_S];
    __shared__ float hs[5][V1_S][V1_T];
    const int tid = threadIdx.y * V1_T + threadIdx.x;
    for (int i = tid; i < V1_S * V1_S; i += 256) {
        int r = i / V1_S, c = i % V1_S;
        int gx = bx - 5 + c, gy = by - 5 + r;
        bool in = gx >= 0 && gx < W && gy >= 0 && gy < H;
        sx[r][c] = in ? X[(size_t)gy * W + gx] : 0.f;
        sy[r][c] = in ? Y[(size_t)gy * W + gx] : 0.f;
    }
    __syncthreads();
    for (int i = tid; i < V1_S * V1_T; i += 256) {
        int r = i / V1_T, c = i % V1_T;
        float a = 0.f, b = 0.f, aa = 0.f, bb = 0.f, ab = 0.f;
#pragma unroll
        for (int d = 0; d < 11; ++d) {
            float w = c_gauss[d], u = sx[r][c + d], v = sy[r][c + d];
            a += w * u;
            b += w * v;
            aa += w * u * u;
            bb += w * v * v;
            ab += w * u * v;
        }
        hs[0][r][c] = a, hs[1][r][c] = b, hs[2][r][c] = aa, hs[3][r][c] = bb, hs[4][r][c] = ab;
    }
    __syncthreads();
    const int c = threadIdx.x, r = threadIdx.y, gx = bx + c, gy = by + r;
    float s_sum = 0.f, l1_sum = 0.f;
    if (gx < W && gy < H) {
        float m[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int d = 0; d < 11; ++d) {
            float w = c_gauss[d];
#pragma unroll
            for (int q = 0; q < 5; ++q) m[q] += w * hs[q][r + d][c];
        }
        const float k = -LAMBDA_DSSIM / (3.f * (float)hw);
        float pa, pb, pd;
        s_sum = ssim_pixel(m[0], m[1], m[2], m[3], m[4], k, pa, pb, pd);
        l1_sum = fabsf(sx[r + 5][c + 5] - sy[r + 5][c + 5]);
        size_t o = (size_t)gy * W + gx;
        partials[(0 * 3 + ch) * hw + o] = pa;
        partials[(1 * 3 + ch) * hw + o] = pb;
        partials[(2 * 3 + ch) * hw + o] = pd;
    }
    block_sums<256>(l1_sum, s_sum, 1.0 / (3.0 * (double)hw), sums);
}

// v1 pass 2: dL/dx = w*a + 2x (w*b) + y (w*d) + (l1/N) sign(x - y).
extern "C" __global__ void ssim_bwd(const IterParams* __restrict__ params, const float* __restrict__ x, const float* __restrict__ y,
             const float* __restrict__ partials, float* __restrict__ d_x, const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    const int W = params->frame.render_w, H = params->frame.render_h;
    const int bx = blockIdx.x * V1_T, by = blockIdx.y * V1_T, ch = blockIdx.z;
    if (bx >= W || by >= H) return;
    const size_t hw = (size_t)W * H;
    __shared__ float sp[3][V1_S][V1_S];
    __shared__ float hb[3][V1_S][V1_T];
    const int tid = threadIdx.y * V1_T + threadIdx.x;
    for (int i = tid; i < V1_S * V1_S; i += 256) {
        int r = i / V1_S, c = i % V1_S;
        int gx = bx - 5 + c, gy = by - 5 + r;
        bool in = gx >= 0 && gx < W && gy >= 0 && gy < H;
        size_t o = (size_t)gy * W + gx;
#pragma unroll
        for (int q = 0; q < 3; ++q) sp[q][r][c] = in ? partials[(q * 3 + ch) * hw + o] : 0.f;
    }
    __syncthreads();
    for (int i = tid; i < V1_S * V1_T; i += 256) {
        int r = i / V1_T, c = i % V1_T;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f;
#pragma unroll
        for (int d = 0; d < 11; ++d) {
            float w = c_gauss[d];
            s0 += w * sp[0][r][c + d];
            s1 += w * sp[1][r][c + d];
            s2 += w * sp[2][r][c + d];
        }
        hb[0][r][c] = s0, hb[1][r][c] = s1, hb[2][r][c] = s2;
    }
    __syncthreads();
    const int c = threadIdx.x, r = threadIdx.y, gx = bx + c, gy = by + r;
    if (gx >= W || gy >= H) return;
    float a = 0.f, b = 0.f, d = 0.f;
#pragma unroll
    for (int k = 0; k < 11; ++k) {
        float w = c_gauss[k];
        a += w * hb[0][r + k][c];
        b += w * hb[1][r + k][c];
        d += w * hb[2][r + k][c];
    }
    size_t o = ch * hw + (size_t)gy * W + gx;
    float xv = x[o], yv = y[o];
    d_x[o] = a + 2.f * xv * b + yv * d + (LAMBDA_L1 / (3.f * (float)hw)) * sgn(xv - yv);
}

// ---------------------------------------------------------------- v2: one kernel, 32 x 16

#define F_TW 32
#define F_TH 16
#define F_RW (F_TW + 10)  // forward region (outputs + backward halo)
#define F_RH (F_TH + 10)
#define F_IW (F_TW + 20)  // input tile (region + forward halo)
#define F_IH (F_TH + 20)
#define F_THREADS (F_TW * F_TH)
#define F_HB_SEGS 11      // ceil(F_TW / 3) column triples in phase 4
#define F_HB_W (3 * F_HB_SEGS)

// Shared memory, reused across phases (46.2 KB static; one padding row of hs, and A's
// slack after pa, keep the register-blocked loops free of index clamps):
//   A: input x, y [2][F_IH][F_IW]          -> partials a, b, d [3][F_RH][F_RW]
//   B: horizontal sums [5][F_IH][F_RW]     -> horizontal partial sums [3][F_RH][F_HB_W]
struct FusedSmem {
    union {
        float in[2][F_IH][F_IW];
        float pa[3][F_RH][F_RW];
    } a;
    union {
        float hs[5][F_IH + 1][F_RW];
        float hb[3][F_RH][F_HB_W];
    } b;
};

template <bool EVAL>
SPL_INLINE void ssim_fused_body(const IterParams* __restrict__ params, const float* __restrict__ x,
                                const float* __restrict__ y, float* __restrict__ d_x, float* __restrict__ err,
                                float* __restrict__ minmax, double* __restrict__ sums) {
    const int W = params->frame.render_w, H = params->frame.render_h;
    const int bx = blockIdx.x * F_TW, by = blockIdx.y * F_TH, ch = blockIdx.z;
    if (bx >= W || by >= H) return;
    const size_t hw = (size_t)W * H;
    const float* X = x + ch * hw;
    const float* Y = y + ch * hw;
    __shared__ FusedSmem sm;
    const int tid = threadIdx.y * F_TW + threadIdx.x;
    const int ox = bx - 10, oy = by - 10;  // input tile origin

    // 1. Input tile with zero padding, as 4-byte cp.async copies (src-size 0 zero-fills
    //    outside the image): every load of the tile is in flight at once instead of one
    //    load-store round trip per loop iteration.
    for (int i = tid; i < F_IH * F_IW; i += F_THREADS) {
        int r = i / F_IW, c = i - r * F_IW;
        int gx = ox + c, gy = oy + r;
        bool in = gx >= 0 && gx < W && gy >= 0 && gy < H;
        size_t o = in ? (size_t)gy * W + gx : 0;
        cp_async4_zfill(&sm.a.in[0][r][c], X + o, in);
        cp_async4_zfill(&sm.a.in[1][r][c], Y + o, in);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    // 2. Horizontal sums over the region's columns, 3 columns per thread (13 input
    //    values in registers instead of 33 shared loads); the L1 sum of the output tile.
    if (tid < F_IH * (F_RW / 3)) {
        const int r = tid / (F_RW / 3), c0 = (tid - r * (F_RW / 3)) * 3;
        float u[13], v[13];
#pragma unroll
        for (int j = 0; j < 13; ++j) {
            u[j] = sm.a.in[0][r][c0 + j];
            v[j] = sm.a.in[1][r][c0 + j];
        }
#pragma unroll
        for (int o = 0; o < 3; ++o) {
            float a = 0.f, b = 0.f, aa = 0.f, bb = 0.f, ab = 0.f;
#pragma unroll
            for (int d = 0; d < 11; ++d) {
                // fma(w u, u) keeps u^2 unrounded: the variance m2 - mu^2 cancels in flat
                // regions, and a rounded u^2 costs 1e-5 of the gradient there.
                float w = c_gauss[d], uu = u[o + d], vv = v[o + d];
                float wu = w * uu, wv = w * vv;
                a += wu;
                b += wv;
                aa = fmaf(wu, uu, aa);
                bb = fmaf(wv, vv, bb);
                ab = fmaf(wu, vv, ab);
            }
            sm.b.hs[0][r][c0 + o] = a, sm.b.hs[1][r][c0 + o] = b, sm.b.hs[2][r][c0 + o] = aa;
            sm.b.hs[3][r][c0 + o] = bb, sm.b.hs[4][r][c0 + o] = ab;
        }
    }
    float l1 = 0.f;
    {
        int gx = bx + threadIdx.x, gy = by + threadIdx.y;
        if (gx < W && gy < H)
            l1 = fabsf(sm.a.in[0][threadIdx.y + 10][threadIdx.x + 10] - sm.a.in[1][threadIdx.y + 10][threadIdx.x + 10]);
    }
    __syncthreads();

    // 3. Vertical sums, 3 rows per thread: S over the region, partials (zero outside the
    //    image) into A.
    const float k = -LAMBDA_DSSIM / (3.f * (float)hw);
    float s_sum = 0.f;
    if (tid < F_RW * ((F_RH + 2) / 3)) {
        const int c = tid % F_RW, r0 = (tid / F_RW) * 3;
        float m[3][5];
#pragma unroll
        for (int o = 0; o < 3; ++o)
#pragma unroll
            for (int q = 0; q < 5; ++q) m[o][q] = 0.f;
#pragma unroll
        for (int j = 0; j < 13; ++j) {
            const int rr = r0 + j;  // row F_IH is padding, read only for discarded outputs
            float h[5];
#pragma unroll
            for (int q = 0; q < 5; ++q) h[q] = sm.b.hs[q][rr][c];
#pragma unroll
            for (int o = 0; o < 3; ++o) {
                const int d = j - o;
                if (d >= 0 && d < 11) {
#pragma unroll
                    for (int q = 0; q < 5; ++q) m[o][q] = fmaf(c_gauss[d], h[q], m[o][q]);
                }
            }
        }
#pragma unroll
        for (int o = 0; o < 3; ++o) {
            const int r = r0 + o;
            if (r >= F_RH) break;
            int gx = ox + 5 + c, gy = oy + 5 + r;
            bool inside = gx >= 0 && gx < W && gy >= 0 && gy < H;
            bool center = r >= 5 && r < 5 + F_TH && c >= 5 && c < 5 + F_TW;
            float pa = 0.f, pb = 0.f, pd = 0.f;
            if (inside && (!EVAL || center)) {
                float S = ssim_pixel(m[o][0], m[o][1], m[o][2], m[o][3], m[o][4], k, pa, pb, pd);
                if (center) s_sum += S;
            }
            if (!EVAL) {  // A's input tile is dead after phase 2's barrier
                sm.a.pa[0][r][c] = pa;
                sm.a.pa[1][r][c] = pb;
                sm.a.pa[2][r][c] = pd;
            }
        }
    }
    if (EVAL) {
        // Per-pixel mean abs error over the channels (channel-0 blocks), its min / max.
        if (ch == 0) {
            int gx = bx + threadIdx.x, gy = by + threadIdx.y;
            float e = 0.f, emin = 3.0e38f;
            if (gx < W && gy < H) {
                size_t o = (size_t)gy * W + gx;
                e = (fabsf(x[o] - y[o]) + fabsf(x[hw + o] - y[hw + o]) + fabsf(x[2 * hw + o] - y[2 * hw + o])) *
                    (1.f / 3.f);
                err[o] = e;
                emin = e;
            }
            emin = warp_min(emin);
            float emax = warp_max(e);
            if ((tid & 31) == 0) {
                // Non-negative floats order like their bit patterns.
                atomicMin(reinterpret_cast<unsigned int*>(&minmax[0]), __float_as_uint(emin));
                atomicMax(reinterpret_cast<unsigned int*>(&minmax[1]), __float_as_uint(emax));
            }
        }
        block_sums<F_THREADS>(l1, s_sum, 1.0 / (3.0 * (double)hw), sums);
        return;
    }
    __syncthreads();

    // 4. Horizontal sums of the partials, 3 output columns per thread (33 columns, the
    //    last one unused, so the stride-3 accesses stay free of bank conflicts).
    if (tid < F_RH * F_HB_SEGS) {
        const int r = tid / F_HB_SEGS, c0 = (tid - r * F_HB_SEGS) * 3;
#pragma unroll
        for (int q = 0; q < 3; ++q) {
            float u[13];
#pragma unroll
            for (int j = 0; j < 13; ++j) u[j] = (&sm.a.pa[q][r][0])[c0 + j];  // index F_RW: next row, discarded
#pragma unroll
            for (int o = 0; o < 3; ++o) {
                float acc = 0.f;
#pragma unroll
                for (int d = 0; d < 11; ++d) acc = fmaf(c_gauss[d], u[o + d], acc);
                sm.b.hb[q][r][c0 + o] = acc;
            }
        }
    }
    __syncthreads();

    // 5. Vertical sums and the gradient, 2 output rows per thread.
    if (tid < F_TW * (F_TH / 2)) {
        const int c = tid % F_TW, r0 = (tid / F_TW) * 2;
        float acc[2][3] = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};
#pragma unroll
        for (int j = 0; j < 12; ++j) {
            float h0 = sm.b.hb[0][r0 + j][c], h1 = sm.b.hb[1][r0 + j][c], h2 = sm.b.hb[2][r0 + j][c];
#pragma unroll
            for (int o = 0; o < 2; ++o) {
                const int d = j - o;
                if (d >= 0 && d < 11) {
                    acc[o][0] = fmaf(c_gauss[d], h0, acc[o][0]);
                    acc[o][1] = fmaf(c_gauss[d], h1, acc[o][1]);
                    acc[o][2] = fmaf(c_gauss[d], h2, acc[o][2]);
                }
            }
        }
        const int gx = bx + c;
#pragma unroll
        for (int o = 0; o < 2; ++o) {
            const int gy = by + r0 + o;
            if (gx < W && gy < H) {
                size_t idx = (size_t)gy * W + gx;
                float xv = X[idx], yv = Y[idx];
                d_x[ch * hw + idx] = acc[o][0] + 2.f * xv * acc[o][1] + yv * acc[o][2] +
                                     (LAMBDA_L1 / (3.f * (float)hw)) * sgn(xv - yv);
            }
        }
    }
    block_sums<F_THREADS>(l1, s_sum, 1.0 / (3.0 * (double)hw), sums);
}

// v2: one kernel, 32x16 output tile, recomputes the forward over the halo in smem.
extern "C" __global__ void ssim_fused(const IterParams* __restrict__ params, const float* __restrict__ x, const float* __restrict__ y,
               float* __restrict__ d_x, double* __restrict__ sums, const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    ssim_fused_body<false>(params, x, y, d_x, nullptr, nullptr, sums);
}

// Eval mode for VCD (5.4): e(p) = mean_c |x - y|, min/max into minmax[2] (initialized by
// the host to [+big, 0]), and the view loss sums for E = 0.8 L1 + 0.2 (1 - SSIM).
extern "C" __global__ void error_map(const IterParams* __restrict__ params, const float* __restrict__ x, const float* __restrict__ y,
              float* __restrict__ err, float* __restrict__ minmax, double* __restrict__ sums) {
    ssim_fused_body<true>(params, x, y, nullptr, err, minmax, sums);
}
