// Lightroom-style Detail panel on display-referred sRGB (see src/detail.rs).
//
// Planes are gamma-encoded Y' (BT.709 luma), Cb = B' - Y' and Cr = R' - Y'
// for one horizontal strip of the image (with halo rows). Noise sigmas are
// looked up per pixel from a 16-bin noise profile indexed by Y'.

#define NBINS 16

// Mirrors `struct Detail` in src/detail.rs (all 4-byte fields, same order).
struct Detail {
    int w, rows;
    int lw, lrows, f;
    int row0, out_rows;
    // Luminance noise reduction: window radius, 1 / (2 sigma_s^2), range
    // sigma as a multiple of the luma noise, blend.
    int l_rad;
    float l_inv2ss, l_k, l_mix;
    // Colour noise reduction on the 1/f grid: window radius, 1 / (2 sigma_s^2),
    // chroma range sigma floor and multiple of the chroma noise, luma range
    // sigma floor and multiple of the luma noise, blend, and the threshold
    // (multiple of the chroma noise) below which a pixel's own chroma detail
    // counts as noise.
    int c_rad;
    float c_inv2ss, c_floor, c_k, cy_floor, cy_k, c_mix, cu_k;
    // Sharpening: Gaussian radius in taps and sigma, gain, texture threshold,
    // noise threshold (multiple of the luma noise), share of the overshoot
    // kept past the local range, masking threshold (0 = off), halo window.
    int g_rad;
    float g_sigma, s_amount, s_tex, s_noise, s_halo, s_mask;
    int s_win;
    float luma_sigma[NBINS];
    float chroma_sigma[NBINS];
};

__device__ __forceinline__ float profile(const float* t, float y) {
    float f = fminf(fmaxf(y * (float)NBINS - 0.5f, 0.0f), (float)(NBINS - 1));
    int i = min((int)f, NBINS - 2);
    float u = f - (float)i;
    return t[i] + u * (t[i + 1] - t[i]);
}

__device__ __forceinline__ float srgb01(unsigned short v) { return (float)v * (1.0f / 65535.0f); }

extern "C" __global__ void ycc_kernel(const unsigned short* __restrict__ rgb, int n, float* __restrict__ Y,
                                      float* __restrict__ Cb, float* __restrict__ Cr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float r = srgb01(rgb[3 * (size_t)i]), g = srgb01(rgb[3 * (size_t)i + 1]), b = srgb01(rgb[3 * (size_t)i + 2]);
    float y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    Y[i] = y;
    Cb[i] = b - y;
    Cr[i] = r - y;
}

// Bilateral filter of Y'. The range term compares each neighbour with the
// 3x3 mean rather than the noisy centre, so isolated noise spikes are
// smoothed too; its sigma follows the noise at that brightness.
extern "C" __global__ void luma_nr_kernel(const Detail p, const float* __restrict__ Y, float* __restrict__ Yd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.rows) return;
    float ref = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
        const float* row = Y + (size_t)min(max(y + dy, 0), p.rows - 1) * p.w;
        for (int dx = -1; dx <= 1; dx++) ref += row[min(max(x + dx, 0), p.w - 1)];
    }
    ref *= (1.0f / 9.0f);
    float sig = p.l_k * profile(p.luma_sigma, ref);
    float inv2sr = 0.5f / (sig * sig);
    float sw = 0.0f, sv = 0.0f;
    for (int dy = -p.l_rad; dy <= p.l_rad; dy++) {
        const float* row = Y + (size_t)min(max(y + dy, 0), p.rows - 1) * p.w;
        for (int dx = -p.l_rad; dx <= p.l_rad; dx++) {
            float v = row[min(max(x + dx, 0), p.w - 1)];
            float d = v - ref;
            float wt = __expf(-(float)(dx * dx + dy * dy) * p.l_inv2ss - d * d * inv2sr);
            sw += wt;
            sv += wt * v;
        }
    }
    size_t i = (size_t)y * p.w + x;
    Yd[i] = Y[i] + p.l_mix * (sv / sw - Y[i]);
}

// f x f box average of Y', Cb and Cr onto the coarse chroma grid.
extern "C" __global__ void chroma_down_kernel(const Detail p, const float* __restrict__ Y, const float* __restrict__ Cb,
                                              const float* __restrict__ Cr, float* __restrict__ Yl,
                                              float* __restrict__ Cbl, float* __restrict__ Crl) {
    int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= p.lw || ly >= p.lrows) return;
    float sy = 0.0f, sb = 0.0f, sr = 0.0f;
    int n = 0;
    for (int y = ly * p.f; y < min(ly * p.f + p.f, p.rows); y++) {
        for (int x = lx * p.f; x < min(lx * p.f + p.f, p.w); x++) {
            size_t i = (size_t)y * p.w + x;
            sy += Y[i];
            sb += Cb[i];
            sr += Cr[i];
            n++;
        }
    }
    float inv = 1.0f / (float)max(n, 1);
    size_t o = (size_t)ly * p.lw + lx;
    Yl[o] = sy * inv;
    Cbl[o] = sb * inv;
    Crl[o] = sr * inv;
}

// Cross-bilateral filter of the coarse chroma: neighbours count when both
// their luma and their chroma are close, so colour does not bleed across
// edges in either.
extern "C" __global__ void chroma_nr_kernel(const Detail p, const float* __restrict__ Yl, const float* __restrict__ Cbl,
                                            const float* __restrict__ Crl, float* __restrict__ Cbo,
                                            float* __restrict__ Cro) {
    int lx = blockIdx.x * blockDim.x + threadIdx.x, ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= p.lw || ly >= p.lrows) return;
    size_t o = (size_t)ly * p.lw + lx;
    float y0 = Yl[o], b0 = Cbl[o], r0 = Crl[o];
    float sc = fmaxf(p.c_floor, p.c_k * profile(p.chroma_sigma, y0));
    float sy = p.cy_floor + p.cy_k * profile(p.luma_sigma, y0);
    float kc = 0.5f / (sc * sc), ky = 0.5f / (sy * sy);
    float sw = 0.0f, sb = 0.0f, sr = 0.0f;
    for (int dy = -p.c_rad; dy <= p.c_rad; dy++) {
        int yy = min(max(ly + dy, 0), p.lrows - 1);
        for (int dx = -p.c_rad; dx <= p.c_rad; dx++) {
            size_t j = (size_t)yy * p.lw + min(max(lx + dx, 0), p.lw - 1);
            float dl = Yl[j] - y0, db = Cbl[j] - b0, dr = Crl[j] - r0;
            float wt = __expf(-(float)(dx * dx + dy * dy) * p.c_inv2ss - dl * dl * ky - (db * db + dr * dr) * kc);
            sw += wt;
            sb += wt * Cbl[j];
            sr += wt * Crl[j];
        }
    }
    Cbo[o] = sb / sw;
    Cro[o] = sr / sw;
}

// Separable Gaussian of the (denoised) luma: the unsharp mask's blur.
extern "C" __global__ void blur_h_kernel(const Detail p, const float* __restrict__ src, float* __restrict__ dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.rows) return;
    float k = -0.5f / (p.g_sigma * p.g_sigma), s = 0.0f, ws = 0.0f;
    const float* row = src + (size_t)y * p.w;
    for (int d = -p.g_rad; d <= p.g_rad; d++) {
        float wt = __expf(k * (float)(d * d));
        s += wt * row[min(max(x + d, 0), p.w - 1)];
        ws += wt;
    }
    dst[(size_t)y * p.w + x] = s / ws;
}

extern "C" __global__ void blur_v_kernel(const Detail p, const float* __restrict__ src, float* __restrict__ dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.rows) return;
    float k = -0.5f / (p.g_sigma * p.g_sigma), s = 0.0f, ws = 0.0f;
    for (int d = -p.g_rad; d <= p.g_rad; d++) {
        float wt = __expf(k * (float)(d * d));
        s += wt * src[(size_t)min(max(y + d, 0), p.rows - 1) * p.w + x];
        ws += wt;
    }
    dst[(size_t)y * p.w + x] = s / ws;
}

__device__ __forceinline__ unsigned short to16(float v) {
    return (unsigned short)(fminf(fmaxf(v, 0.0f), 1.0f) * 65535.0f + 0.5f);
}

// Output rows: chroma as the denoised coarse grid upsampled (bilinear) plus
// the pixel's own chroma detail (its chroma minus the unfiltered coarse grid
// upsampled the same way), soft-thresholded against the chroma noise: real
// colour edges and detail stand well above the noise and come back exactly,
// grain does not. Then the unsharp mask on the denoised luma (soft
// texture/noise threshold, edge mask, halo clamp) and back to 16-bit R'G'B'.
__device__ __forceinline__ float bilerp(const float* __restrict__ t, int lw, int x0, int x1, int y0, int y1, float tx,
                                        float ty) {
    float top = t[(size_t)y0 * lw + x0] + tx * (t[(size_t)y0 * lw + x1] - t[(size_t)y0 * lw + x0]);
    float bot = t[(size_t)y1 * lw + x0] + tx * (t[(size_t)y1 * lw + x1] - t[(size_t)y1 * lw + x0]);
    return top + ty * (bot - top);
}

extern "C" __global__ void finish_kernel(const Detail p, const float* __restrict__ Y, const float* __restrict__ Yd,
                                         const float* __restrict__ G, const float* __restrict__ Cb,
                                         const float* __restrict__ Cr, const float* __restrict__ Cb0,
                                         const float* __restrict__ Cr0, const float* __restrict__ Cbf,
                                         const float* __restrict__ Crf, unsigned short* __restrict__ out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || oy >= p.out_rows) return;
    int y = oy + p.row0;
    size_t i = (size_t)y * p.w + x;
    float yv = Yd[i], cb = Cb[i], cr = Cr[i];

    if (p.c_mix > 0.0f) {
        float fx = fminf(fmaxf(((float)x + 0.5f) / (float)p.f - 0.5f, 0.0f), (float)(p.lw - 1));
        float fy = fminf(fmaxf(((float)y + 0.5f) / (float)p.f - 0.5f, 0.0f), (float)(p.lrows - 1));
        int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
        int x1 = min(x0 + 1, p.lw - 1), y1 = min(y0 + 1, p.lrows - 1);
        float tx = fx - (float)x0, ty = fy - (float)y0;
        float db = cb - bilerp(Cb0, p.lw, x0, x1, y0, y1, tx, ty);
        float dr = cr - bilerp(Cr0, p.lw, x0, x1, y0, y1, tx, ty);
        float t = fmaxf(0.5f * p.c_floor, p.cu_k * profile(p.chroma_sigma, Y[i]));
        float m2 = db * db + dr * dr;
        float keep = m2 / (m2 + t * t);
        float nb = bilerp(Cbf, p.lw, x0, x1, y0, y1, tx, ty) + keep * db;
        float nr = bilerp(Crf, p.lw, x0, x1, y0, y1, tx, ty) + keep * dr;
        cb += p.c_mix * (nb - cb);
        cr += p.c_mix * (nr - cr);
        // Chroma averaged from brighter neighbours may not fit this pixel's
        // luma: scale it toward grey (same luma, same hue) instead of
        // letting a channel clip.
        float dg = -(0.2126f * cr + 0.0722f * cb) * (1.0f / 0.7152f);
        float k = 1.0f;
        float diffs[3] = {cr, dg, cb};
        for (int c = 0; c < 3; c++) {
            float d = diffs[c];
            if (d < 0.0f && yv + d < 0.0f) k = fminf(k, yv / -d);
            if (d > 0.0f && yv + d > 1.0f) k = fminf(k, (1.0f - yv) / d);
        }
        k = fmaxf(k, 0.0f);
        cb *= k;
        cr *= k;
    }

    if (p.s_amount > 0.0f) {
        float d = yv - G[i];
        float t = p.s_tex + p.s_noise * profile(p.luma_sigma, yv);
        float gain = p.s_amount * d * d / (d * d + t * t);
        if (p.s_mask > 0.0f) {
            const float* row = G + (size_t)y * p.w;
            float gx = row[min(x + 1, p.w - 1)] - row[max(x - 1, 0)];
            float gy = G[(size_t)min(y + 1, p.rows - 1) * p.w + x] - G[(size_t)max(y - 1, 0) * p.w + x];
            // Edge contrast across about one blur width.
            float e = sqrtf(gx * gx + gy * gy) * fmaxf(p.g_sigma, 1.0f);
            float m = fminf(fmaxf((e - 0.5f * p.s_mask) / p.s_mask, 0.0f), 1.0f);
            gain *= m * m * (3.0f - 2.0f * m);
        }
        float ys = yv + gain * d;
        float lo = yv, hi = yv;
        for (int dy = -p.s_win; dy <= p.s_win; dy++) {
            const float* row = Yd + (size_t)min(max(y + dy, 0), p.rows - 1) * p.w;
            for (int dx = -p.s_win; dx <= p.s_win; dx++) {
                float v = row[min(max(x + dx, 0), p.w - 1)];
                lo = fminf(lo, v);
                hi = fmaxf(hi, v);
            }
        }
        float yc = fminf(fmaxf(ys, lo), hi);
        // A luma change moves R', G' and B' alike; keep it from clipping a
        // channel, which would shift saturated colours' hue.
        float r0 = yv + cr, b0 = yv + cb, g0 = yv - (0.2126f * cr + 0.0722f * cb) * (1.0f / 0.7152f);
        float cmin = fminf(r0, fminf(g0, b0)), cmax = fmaxf(r0, fmaxf(g0, b0));
        float delta = yc + p.s_halo * (ys - yc) - yv;
        yv += fminf(fmaxf(delta, fminf(0.0f, -cmin)), fmaxf(0.0f, 1.0f - cmax));
    }

    float r = yv + cr, b = yv + cb;
    float g = (yv - 0.2126f * r - 0.0722f * b) * (1.0f / 0.7152f);
    size_t o = 3 * ((size_t)oy * p.w + x);
    out[o] = to16(r);
    out[o + 1] = to16(g);
    out[o + 2] = to16(b);
}
