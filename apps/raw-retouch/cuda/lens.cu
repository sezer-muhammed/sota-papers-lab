// Lens correction from the camera's embedded profile, on linear RGB
// (interleaved, row-major). See src/lens.rs for the model.
//
// Every output pixel at offset d from the image centre reads channel c of
// the source at centre + (d / zoom) * S_c(r), where r = |d / zoom| over the
// half-diagonal and S_c is the channel's radial scale (distortion times
// lateral chromatic aberration). The read is Catmull-Rom bicubic, clamped to
// the range of the four nearest source pixels so edges get no ringing, and
// is multiplied by the vignetting gain at the source radius.

// The lookup table holds LUT_N rows of (S_r, S_g, S_b, gain) sampled
// uniformly on r in [0, rmax], linearly interpolated, clamped at the ends.
__device__ __forceinline__ float lut_at(const float* __restrict__ lut, int n, float rmax, float r, int ch) {
    float f = fminf(fmaxf(r * ((float)(n - 1) / rmax), 0.0f), (float)(n - 1));
    int i = min((int)f, n - 2);
    float t = f - (float)i;
    float a = lut[4 * i + ch], b = lut[4 * (i + 1) + ch];
    return a + t * (b - a);
}

__device__ __forceinline__ void catmull_rom(float t, float w[4]) {
    float t2 = t * t, t3 = t2 * t;
    w[0] = -0.5f * t3 + t2 - 0.5f * t;
    w[1] = 1.5f * t3 - 2.5f * t2 + 1.0f;
    w[2] = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
    w[3] = 0.5f * t3 - 0.5f * t2;
}

// Channel c at (x, y) in pixel-centre coordinates, edges clamped.
__device__ float sample(const float* __restrict__ src, int w, int h, float x, float y, int c) {
    x = fminf(fmaxf(x, 0.0f), (float)(w - 1));
    y = fminf(fmaxf(y, 0.0f), (float)(h - 1));
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float wx[4], wy[4];
    catmull_rom(x - (float)x0, wx);
    catmull_rom(y - (float)y0, wy);
    int xs[4];
    for (int i = 0; i < 4; i++) xs[i] = 3 * min(max(x0 - 1 + i, 0), w - 1) + c;
    float acc = 0.0f, lo = 3.4e38f, hi = -3.4e38f;
    for (int j = 0; j < 4; j++) {
        const float* row = src + 3 * (size_t)min(max(y0 - 1 + j, 0), h - 1) * (size_t)w;
        float v0 = row[xs[0]], v1 = row[xs[1]], v2 = row[xs[2]], v3 = row[xs[3]];
        acc += wy[j] * (wx[0] * v0 + wx[1] * v1 + wx[2] * v2 + wx[3] * v3);
        if (j == 1 || j == 2) {
            lo = fminf(lo, fminf(v1, v2));
            hi = fmaxf(hi, fmaxf(v1, v2));
        }
    }
    return fminf(fmaxf(acc, lo), hi);
}

extern "C" __global__ void lens_kernel(const float* __restrict__ src, float* __restrict__ dst, int w, int h,
                                       const float* __restrict__ lut, int n, float rmax, float inv_half_diag,
                                       float inv_zoom) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float cx = 0.5f * (float)(w - 1), cy = 0.5f * (float)(h - 1);
    float dx = ((float)x - cx) * inv_zoom, dy = ((float)y - cy) * inv_zoom;
    float r = sqrtf(dx * dx + dy * dy) * inv_half_diag;
    float* out = dst + 3 * ((size_t)y * (size_t)w + (size_t)x);
    for (int c = 0; c < 3; c++) {
        float s = lut_at(lut, n, rmax, r, c);
        float v = sample(src, w, h, cx + dx * s, cy + dy * s, c);
        out[c] = v * lut_at(lut, n, rmax, r * s, 3);
    }
}

// Lateral chromatic aberration on the RAW samples, before demosaicing (a
// colour-difference demosaic would otherwise drag red and blue edges partly
// onto green's, leaving a residue no later correction can model). Green
// sites are copied; a red or blue site reads its own colour's sub-lattice
// (every other row and column, starting at (ox, oy)) at the radially scaled
// position, Catmull-Rom clamped to the four nearest samples. LUT rows hold
// (S_r, 1, S_b, 1) for CA alone. `cfa` packs the 2x2 pattern as c00 | c01 << 2
// | c10 << 4 | c11 << 6 (0 = R, 1 = G, 2 = B).
__device__ float sample_lattice(const unsigned short* __restrict__ src, int w, int h, int ox, int oy, float x,
                                float y) {
    int lw = (w - ox + 1) / 2, lh = (h - oy + 1) / 2;
    float u = fminf(fmaxf((x - (float)ox) * 0.5f, 0.0f), (float)(lw - 1));
    float v = fminf(fmaxf((y - (float)oy) * 0.5f, 0.0f), (float)(lh - 1));
    int u0 = (int)floorf(u), v0 = (int)floorf(v);
    float wu[4], wv[4];
    catmull_rom(u - (float)u0, wu);
    catmull_rom(v - (float)v0, wv);
    int xs[4];
    for (int i = 0; i < 4; i++) xs[i] = ox + 2 * min(max(u0 - 1 + i, 0), lw - 1);
    float acc = 0.0f, lo = 3.4e38f, hi = -3.4e38f;
    for (int j = 0; j < 4; j++) {
        const unsigned short* row = src + (size_t)(oy + 2 * min(max(v0 - 1 + j, 0), lh - 1)) * (size_t)w;
        float a0 = row[xs[0]], a1 = row[xs[1]], a2 = row[xs[2]], a3 = row[xs[3]];
        acc += wv[j] * (wu[0] * a0 + wu[1] * a1 + wu[2] * a2 + wu[3] * a3);
        if (j == 1 || j == 2) {
            lo = fminf(lo, fminf(a1, a2));
            hi = fmaxf(hi, fmaxf(a1, a2));
        }
    }
    return fminf(fmaxf(acc, lo), hi);
}

extern "C" __global__ void ca_mosaic_kernel(const unsigned short* __restrict__ src, unsigned short* __restrict__ dst,
                                            int w, int h, int cfa, const float* __restrict__ lut, int n, float rmax,
                                            float inv_half_diag) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    size_t i = (size_t)y * (size_t)w + (size_t)x;
    int c = (cfa >> (2 * ((y & 1) * 2 + (x & 1)))) & 3;
    if (c != 0 && c != 2) {
        dst[i] = src[i];
        return;
    }
    // The site of this colour in the 2x2 pattern.
    int ox = 0, oy = 0;
    for (int k = 0; k < 4; k++) {
        if (((cfa >> (2 * k)) & 3) == c) {
            ox = k & 1;
            oy = k >> 1;
            break;
        }
    }
    float cx = 0.5f * (float)(w - 1), cy = 0.5f * (float)(h - 1);
    float dx = (float)x - cx, dy = (float)y - cy;
    float s = lut_at(lut, n, rmax, sqrtf(dx * dx + dy * dy) * inv_half_diag, c);
    float v = sample_lattice(src, w, h, ox, oy, cx + dx * s, cy + dy * s);
    dst[i] = (unsigned short)fminf(fmaxf(v + 0.5f, 0.0f), 65535.0f);
}

// The same for three-channel RAWs (Sony's lossless M/S sizes): red and blue
// resampled on the interleaved samples, green copied.
extern "C" __global__ void ca_rgb_kernel(const unsigned short* __restrict__ src, unsigned short* __restrict__ dst,
                                         int w, int h, const float* __restrict__ lut, int n, float rmax,
                                         float inv_half_diag) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    size_t i = 3 * ((size_t)y * (size_t)w + (size_t)x);
    float cx = 0.5f * (float)(w - 1), cy = 0.5f * (float)(h - 1);
    float dx = (float)x - cx, dy = (float)y - cy;
    float r = sqrtf(dx * dx + dy * dy) * inv_half_diag;
    dst[i + 1] = src[i + 1];
    for (int c = 0; c < 3; c += 2) {
        float s = lut_at(lut, n, rmax, r, c);
        float sx = fminf(fmaxf(cx + dx * s, 0.0f), (float)(w - 1));
        float sy = fminf(fmaxf(cy + dy * s, 0.0f), (float)(h - 1));
        int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
        float wx[4], wy[4];
        catmull_rom(sx - (float)x0, wx);
        catmull_rom(sy - (float)y0, wy);
        float acc = 0.0f, lo = 3.4e38f, hi = -3.4e38f;
        for (int j = 0; j < 4; j++) {
            const unsigned short* row = src + 3 * (size_t)min(max(y0 - 1 + j, 0), h - 1) * (size_t)w;
            for (int k = 0; k < 4; k++) {
                float a = row[3 * min(max(x0 - 1 + k, 0), w - 1) + c];
                acc += wy[j] * wx[k] * a;
                if ((j == 1 || j == 2) && (k == 1 || k == 2)) {
                    lo = fminf(lo, a);
                    hi = fmaxf(hi, a);
                }
            }
        }
        dst[i + c] = (unsigned short)fminf(fmaxf(fminf(fmaxf(acc, lo), hi) + 0.5f, 0.0f), 65535.0f);
    }
}
