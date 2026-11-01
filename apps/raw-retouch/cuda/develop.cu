// Develop kernels: normalize, white balance, highlight clip, superpixel,
// downscale, full-resolution demosaic. Owner: develop.
//
// Every path sees each raw sample the same way (sample_value): black
// subtracted per CFA position, scaled so the sensor white is 1, as-shot white
// balance applied, then clamped at the smallest multiplier and divided by it.
// The host folds all of that into one scale per position. A pixel that
// clipped in any channel therefore ends at exactly 1.0 there, and a fully
// clipped pixel is exactly (1, 1, 1): clipped highlights are neutral.
// Nothing is clamped at zero until the end, so averages of shadow noise stay
// unbiased.

struct Mosaic {
    int width;
    int height;
    int cfa[4];      // colour at (0,0), (0,1), (1,0), (1,1): 0 = R, 1 = G, 2 = B
    float black[4];  // per CFA position, or per channel for linear data
    float scale[4];  // wb / ((white - black) * min(wb))
    float white;     // sensor white in raw units
};

__device__ __forceinline__ float sample_value(unsigned short v, const Mosaic& m, int slot) {
    return fminf(((float)v - m.black[slot]) * m.scale[slot], 1.0f);
}

__device__ __forceinline__ int cfa_slot(int x, int y) {
    return ((y & 1) << 1) | (x & 1);
}

// One atomic per warp: the clipped-pixel counter is touched by every thread.
__device__ __forceinline__ void count_clipped(bool hit, unsigned int* clipped) {
    unsigned int mask = __ballot_sync(0xffffffffu, hit);
    unsigned int lane = (threadIdx.y * blockDim.x + threadIdx.x) & 31u;
    if (lane == 0 && mask) atomicAdd(clipped, __popc(mask));
}

// Half-size RGB from each 2x2 Bayer block: R, mean of the two greens, B.
// A block counts as clipped when any of its samples reached the sensor white.
extern "C" __global__ void superpixel(const unsigned short* raw, Mosaic m, float* out,
                                      int out_w, int out_h, unsigned int* clipped) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    bool hit = false;
    if (x < out_w && y < out_h) {
        float rgb[3] = {0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int slot = 0; slot < 4; slot++) {
            unsigned short v = raw[(size_t)(2 * y + (slot >> 1)) * m.width + 2 * x + (slot & 1)];
            hit |= (float)v >= m.white;
            rgb[m.cfa[slot]] += sample_value(v, m, slot);
        }
        size_t o = 3 * ((size_t)y * out_w + x);
        out[o] = rgb[0];
        out[o + 1] = rgb[1] * 0.5f;
        out[o + 2] = rgb[2];
    }
    count_clipped(hit, clipped);
}

// Linear (already demosaiced) RAWs: the same per-sample transform, per channel.
extern "C" __global__ void linear_prep(const unsigned short* raw, Mosaic m, float* out,
                                       unsigned int* clipped) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    bool hit = false;
    if (x < m.width && y < m.height) {
        size_t i = 3 * ((size_t)y * m.width + x);
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            unsigned short v = raw[i + c];
            hit |= (float)v >= m.white;
            out[i + c] = sample_value(v, m, c);
        }
    }
    count_clipped(hit, clipped);
}

// Exact area-average resize of interleaved RGB: every output pixel is the
// coverage-weighted mean of the input box it covers, so nothing aliases.
// The weight sum is accumulated alongside, so a box of 1.0s stays exactly 1.0.
extern "C" __global__ void area_downscale(const float* in, int in_w, int in_h, float* out,
                                          int out_w, int out_h) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= out_w || oy >= out_h) return;
    float sx = (float)in_w / (float)out_w;
    float sy = (float)in_h / (float)out_h;
    float x0 = ox * sx, x1 = fminf((ox + 1) * sx, (float)in_w);
    float y0 = oy * sy, y1 = fminf((oy + 1) * sy, (float)in_h);
    int ix0 = (int)floorf(x0), ix1 = min((int)ceilf(x1), in_w);
    int iy0 = (int)floorf(y0), iy1 = min((int)ceilf(y1), in_h);
    float r = 0.0f, g = 0.0f, b = 0.0f, wsum = 0.0f;
    for (int iy = iy0; iy < iy1; iy++) {
        float wy = fminf((float)(iy + 1), y1) - fmaxf((float)iy, y0);
        if (wy <= 0.0f) continue;
        const float* row = in + 3 * (size_t)iy * in_w;
        for (int ix = ix0; ix < ix1; ix++) {
            float wx = fminf((float)(ix + 1), x1) - fmaxf((float)ix, x0);
            if (wx <= 0.0f) continue;
            float w = wx * wy;
            r += w * row[3 * ix];
            g += w * row[3 * ix + 1];
            b += w * row[3 * ix + 2];
            wsum += w;
        }
    }
    size_t o = 3 * ((size_t)oy * out_w + ox);
    out[o] = fminf(fmaxf(r / wsum, 0.0f), 1.0f);
    out[o + 1] = fminf(fmaxf(g / wsum, 0.0f), 1.0f);
    out[o + 2] = fminf(fmaxf(b / wsum, 0.0f), 1.0f);
}

// ---------------------------------------------------------------------------
// Full-resolution demosaic: RCD (ratio corrected demosaicing, Luis Sanz
// Rodriguez, as shipped in RawTherapee and darktable), re-implemented as one
// kernel per step. Reads past the border mirror the image without repeating
// the edge sample, which keeps the CFA phase, so the border needs no
// separate code path. One change from the reference: the low-pass ratio is
// 2a / (a + b) with both terms floored, instead of adding epsilon to the
// denominator, so flat and fully clipped areas come out exactly unchanged.

#define RCD_EPS 1e-5f
#define RCD_EPS_SQ 1e-10f

__device__ __forceinline__ int mirror(int i, int n) {
    i = i < 0 ? -i : i;
    return i >= n ? 2 * (n - 1) - i : i;
}

__device__ __forceinline__ float at(const float* p, int x, int y, int w, int h) {
    return p[(size_t)mirror(y, h) * w + mirror(x, w)];
}

// Channel c of the interleaved output at a mirrored position.
__device__ __forceinline__ float at3(const float* p, int x, int y, int c, int w, int h) {
    return p[3 * ((size_t)mirror(y, h) * w + mirror(x, w)) + c];
}

__device__ __forceinline__ float sq(float v) { return v * v; }

// The ratio by which a neighbour's colour is corrected toward this pixel:
// 2a / (a + b) on the low-passed signal, exactly 1 when the two agree (so
// flat and clipped areas reproduce exactly) and bounded in (0, 2).
__device__ __forceinline__ float lp_ratio(float a, float b) {
    a = fmaxf(a, RCD_EPS);
    b = fmaxf(b, RCD_EPS);
    return (a + a) / (a + b);
}

// Picks the more decisive of the pixel's own direction and its diagonal
// neighbours' average (both in 0..1, 0.5 = no preference).
__device__ __forceinline__ float refine(float central, float neighbourhood) {
    return fabsf(0.5f - central) < fabsf(0.5f - neighbourhood) ? neighbourhood : central;
}

// a * b + (1 - a) * c, written so equal b and c give back exactly c.
__device__ __forceinline__ float intp(float a, float b, float c) { return a * (b - c) + c; }

extern "C" __global__ void mosaic_prep(const unsigned short* raw, Mosaic m, float* cfa) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= m.width || y >= m.height) return;
    size_t i = (size_t)y * m.width + x;
    cfa[i] = sample_value(raw[i], m, cfa_slot(x, y));
}

// Step 1: vertical/horizontal discrimination from squared colour-difference
// high-pass filters summed over three lines (vh: 0 = vertical, 1 = horizontal).
// Step 2: low-pass filter at red and blue sites.
extern "C" __global__ void rcd_directions(const float* cfa, Mosaic m, float* vh, float* lpf) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = m.width, h = m.height;
    if (x >= w || y >= h) return;
    float v_stat = 0.0f, h_stat = 0.0f;
    #pragma unroll
    for (int d = -1; d <= 1; d++) {
        int yy = y + d, xx = x + d;
        v_stat += sq((at(cfa, x, yy - 3, w, h) - at(cfa, x, yy - 1, w, h) - at(cfa, x, yy + 1, w, h) +
                      at(cfa, x, yy + 3, w, h)) -
                     3.0f * (at(cfa, x, yy - 2, w, h) + at(cfa, x, yy + 2, w, h)) +
                     6.0f * at(cfa, x, yy, w, h));
        h_stat += sq((at(cfa, xx - 3, y, w, h) - at(cfa, xx - 1, y, w, h) - at(cfa, xx + 1, y, w, h) +
                      at(cfa, xx + 3, y, w, h)) -
                     3.0f * (at(cfa, xx - 2, y, w, h) + at(cfa, xx + 2, y, w, h)) +
                     6.0f * at(cfa, xx, y, w, h));
    }
    v_stat = fmaxf(RCD_EPS_SQ, v_stat);
    h_stat = fmaxf(RCD_EPS_SQ, h_stat);
    size_t i = (size_t)y * w + x;
    vh[i] = v_stat / (v_stat + h_stat);
    if (m.cfa[cfa_slot(x, y)] != 1) {
        lpf[i] = at(cfa, x, y, w, h) +
                 0.5f * (at(cfa, x, y - 1, w, h) + at(cfa, x, y + 1, w, h) + at(cfa, x - 1, y, w, h) +
                         at(cfa, x + 1, y, w, h)) +
                 0.25f * (at(cfa, x - 1, y - 1, w, h) + at(cfa, x + 1, y - 1, w, h) +
                          at(cfa, x - 1, y + 1, w, h) + at(cfa, x + 1, y + 1, w, h));
    }
}

// Step 3: green at red and blue sites from ratio-corrected cardinal
// neighbours; every site also copies its own sample into the output.
extern "C" __global__ void rcd_green(const float* cfa, const float* vh, const float* lpf, Mosaic m,
                                     float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = m.width, h = m.height;
    if (x >= w || y >= h) return;
    size_t i = (size_t)y * w + x;
    int color = m.cfa[cfa_slot(x, y)];
    float c0 = cfa[i];
    out[3 * i + color] = c0;
    if (color == 1) return;

    float n1 = at(cfa, x, y - 1, w, h), s1 = at(cfa, x, y + 1, w, h);
    float w1 = at(cfa, x - 1, y, w, h), e1 = at(cfa, x + 1, y, w, h);
    float n2 = at(cfa, x, y - 2, w, h), s2 = at(cfa, x, y + 2, w, h);
    float w2 = at(cfa, x - 2, y, w, h), e2 = at(cfa, x + 2, y, w, h);
    float n_grad = RCD_EPS + fabsf(n1 - s1) + fabsf(c0 - n2) + fabsf(n1 - at(cfa, x, y - 3, w, h)) +
                   fabsf(n2 - at(cfa, x, y - 4, w, h));
    float s_grad = RCD_EPS + fabsf(n1 - s1) + fabsf(c0 - s2) + fabsf(s1 - at(cfa, x, y + 3, w, h)) +
                   fabsf(s2 - at(cfa, x, y + 4, w, h));
    float w_grad = RCD_EPS + fabsf(w1 - e1) + fabsf(c0 - w2) + fabsf(w1 - at(cfa, x - 3, y, w, h)) +
                   fabsf(w2 - at(cfa, x - 4, y, w, h));
    float e_grad = RCD_EPS + fabsf(w1 - e1) + fabsf(c0 - e2) + fabsf(e1 - at(cfa, x + 3, y, w, h)) +
                   fabsf(e2 - at(cfa, x + 4, y, w, h));

    float lc = lpf[i];
    float n_est = n1 * lp_ratio(lc, at(lpf, x, y - 2, w, h));
    float s_est = s1 * lp_ratio(lc, at(lpf, x, y + 2, w, h));
    float w_est = w1 * lp_ratio(lc, at(lpf, x - 2, y, w, h));
    float e_est = e1 * lp_ratio(lc, at(lpf, x + 2, y, w, h));

    float v_est = (s_grad * n_est + n_grad * s_est) / (n_grad + s_grad);
    float h_est = (w_grad * e_est + e_grad * w_est) / (e_grad + w_grad);

    float disc = refine(vh[i], 0.25f * (at(vh, x - 1, y - 1, w, h) + at(vh, x + 1, y - 1, w, h) +
                                        at(vh, x - 1, y + 1, w, h) + at(vh, x + 1, y + 1, w, h)));
    out[3 * i + 1] = intp(disc, h_est, v_est);
}

__device__ __forceinline__ float diagonal_hpf(const float* cfa, int x, int y, int sx, int w, int h) {
    // sx = +1: the NW-SE diagonal (P); sx = -1: the NE-SW diagonal (Q).
    return sq((at(cfa, x - 3 * sx, y - 3, w, h) - at(cfa, x - sx, y - 1, w, h) -
               at(cfa, x + sx, y + 1, w, h) + at(cfa, x + 3 * sx, y + 3, w, h)) -
              3.0f * (at(cfa, x - 2 * sx, y - 2, w, h) + at(cfa, x + 2 * sx, y + 2, w, h)) +
              6.0f * at(cfa, x, y, w, h));
}

// Step 4.1: diagonal discrimination at red and blue sites (pq: 0 = P, 1 = Q).
extern "C" __global__ void rcd_diagonals(const float* cfa, Mosaic m, float* pq) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = m.width, h = m.height;
    if (x >= w || y >= h || m.cfa[cfa_slot(x, y)] == 1) return;
    float p_stat = fmaxf(RCD_EPS_SQ, diagonal_hpf(cfa, x - 1, y - 1, 1, w, h) +
                                         diagonal_hpf(cfa, x, y, 1, w, h) +
                                         diagonal_hpf(cfa, x + 1, y + 1, 1, w, h));
    float q_stat = fmaxf(RCD_EPS_SQ, diagonal_hpf(cfa, x + 1, y - 1, -1, w, h) +
                                         diagonal_hpf(cfa, x, y, -1, w, h) +
                                         diagonal_hpf(cfa, x - 1, y + 1, -1, w, h));
    pq[(size_t)y * w + x] = p_stat / (p_stat + q_stat);
}

// Step 4.2: red at blue sites and blue at red sites, from colour differences
// along the diagonals (whose sites hold the missing colour as raw samples).
extern "C" __global__ void rcd_rb_at_rb(const float* cfa, const float* pq, Mosaic m, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = m.width, h = m.height;
    if (x >= w || y >= h) return;
    int color = m.cfa[cfa_slot(x, y)];
    if (color == 1) return;
    int c = 2 - color;
    size_t i = (size_t)y * w + x;

    float disc = refine(pq[i], 0.25f * (at(pq, x - 1, y - 1, w, h) + at(pq, x + 1, y - 1, w, h) +
                                        at(pq, x - 1, y + 1, w, h) + at(pq, x + 1, y + 1, w, h)));
    float g0 = out[3 * i + 1];
    float nw = at(cfa, x - 1, y - 1, w, h), ne = at(cfa, x + 1, y - 1, w, h);
    float sw = at(cfa, x - 1, y + 1, w, h), se = at(cfa, x + 1, y + 1, w, h);

    float nw_grad = RCD_EPS + fabsf(nw - se) + fabsf(nw - at(cfa, x - 3, y - 3, w, h)) +
                    fabsf(g0 - at3(out, x - 2, y - 2, 1, w, h));
    float ne_grad = RCD_EPS + fabsf(ne - sw) + fabsf(ne - at(cfa, x + 3, y - 3, w, h)) +
                    fabsf(g0 - at3(out, x + 2, y - 2, 1, w, h));
    float sw_grad = RCD_EPS + fabsf(ne - sw) + fabsf(sw - at(cfa, x - 3, y + 3, w, h)) +
                    fabsf(g0 - at3(out, x - 2, y + 2, 1, w, h));
    float se_grad = RCD_EPS + fabsf(nw - se) + fabsf(se - at(cfa, x + 3, y + 3, w, h)) +
                    fabsf(g0 - at3(out, x + 2, y + 2, 1, w, h));

    float nw_est = nw - at3(out, x - 1, y - 1, 1, w, h);
    float ne_est = ne - at3(out, x + 1, y - 1, 1, w, h);
    float sw_est = sw - at3(out, x - 1, y + 1, 1, w, h);
    float se_est = se - at3(out, x + 1, y + 1, 1, w, h);

    float p_est = (nw_grad * se_est + se_grad * nw_est) / (nw_grad + se_grad);
    float q_est = (ne_grad * sw_est + sw_grad * ne_est) / (ne_grad + sw_grad);
    out[3 * i + c] = g0 + intp(disc, q_est, p_est);
}

// Step 4.3: red and blue at green sites from cardinal colour differences.
extern "C" __global__ void rcd_rb_at_g(const float* vh, Mosaic m, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = m.width, h = m.height;
    if (x >= w || y >= h || m.cfa[cfa_slot(x, y)] != 1) return;
    size_t i = (size_t)y * w + x;

    float disc = refine(vh[i], 0.25f * (at(vh, x - 1, y - 1, w, h) + at(vh, x + 1, y - 1, w, h) +
                                        at(vh, x - 1, y + 1, w, h) + at(vh, x + 1, y + 1, w, h)));
    float g0 = out[3 * i + 1];
    float n1 = RCD_EPS + fabsf(g0 - at3(out, x, y - 2, 1, w, h));
    float s1 = RCD_EPS + fabsf(g0 - at3(out, x, y + 2, 1, w, h));
    float w1 = RCD_EPS + fabsf(g0 - at3(out, x - 2, y, 1, w, h));
    float e1 = RCD_EPS + fabsf(g0 - at3(out, x + 2, y, 1, w, h));
    float gn = at3(out, x, y - 1, 1, w, h), gs = at3(out, x, y + 1, 1, w, h);
    float gw = at3(out, x - 1, y, 1, w, h), ge = at3(out, x + 1, y, 1, w, h);

    #pragma unroll
    for (int c = 0; c <= 2; c += 2) {
        float cn = at3(out, x, y - 1, c, w, h), cs = at3(out, x, y + 1, c, w, h);
        float cw = at3(out, x - 1, y, c, w, h), ce = at3(out, x + 1, y, c, w, h);
        float sn_abs = fabsf(cn - cs);
        float ew_abs = fabsf(cw - ce);
        float n_grad = n1 + sn_abs + fabsf(cn - at3(out, x, y - 3, c, w, h));
        float s_grad = s1 + sn_abs + fabsf(cs - at3(out, x, y + 3, c, w, h));
        float w_grad = w1 + ew_abs + fabsf(cw - at3(out, x - 3, y, c, w, h));
        float e_grad = e1 + ew_abs + fabsf(ce - at3(out, x + 3, y, c, w, h));
        float n_est = cn - gn, s_est = cs - gs, w_est = cw - gw, e_est = ce - ge;
        float v_est = (n_grad * s_est + s_grad * n_est) / (n_grad + s_grad);
        float h_est = (e_grad * w_est + w_grad * e_est) / (e_grad + w_grad);
        out[3 * i + c] = g0 + intp(disc, h_est, v_est);
    }
}

extern "C" __global__ void clamp_unit(float* data, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = fminf(fmaxf(data[i], 0.0f), 1.0f);
}
