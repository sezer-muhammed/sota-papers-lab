// Render kernels: white balance delta, Rec.2020, exposure, local tone,
// global tone curve, vibrance/saturation, sRGB output, statistics. Owner: render.
//
// A parameter set ("candidate") arrives as CAND_STRIDE floats packed by
// src/render/curves.rs: camera -> linear Rec.2020 matrices with the white
// balance delta folded in, exposure gain, base-layer shift, vibrance and
// saturation, then three lookup tables: the global tone curve (base curve and
// Contrast), Highlights/Shadows by base layer, and Whites/Blacks by pixel
// luminance. The kernels hold no slider logic of their own, so the CPU tests
// of the slider shapes describe exactly what renders.
// Every #define below mirrors a constant in src/render/layout.rs; a unit test
// keeps the two in step.

#define CAND_STRIDE 928
#define P_A1 0
#define P_A2 9
#define P_GAIN 18
#define P_BSHIFT 19
#define P_VIB 20
#define P_SAT 21
#define TONE_OFF 32
#define TONE_N 512
#define TONE_U0 (-16.0f)
#define TONE_U1 (6.0f)
#define LOCAL_OFF 544
#define LOCAL_N 192
#define LOCAL_B0 (-16.0f)
#define LOCAL_B1 (8.0f)
#define ENDS_OFF 736

#define GRID_PAD 2
#define GRID_Z0 (-14.0f)
#define GRID_Z1 (1.0f)
#define GRID_DZ (0.75f)
#define GRID_PRIOR (0.001f)

#define LUMA_BINS 1024
#define CHROMA_BINS 256
#define CHROMA_MAX (0.4f)
#define N_COUNTS 8
#define N_SUMS 10
#define PART_U32 (LUMA_BINS + CHROMA_BINS + N_COUNTS)
#define STATS_THREADS 256

#define C_PX 0
#define C_HI 1
#define C_HI_NEW 2
#define C_LO 3
#define C_OOG 4
#define C_CAST 5
#define C_SBLK 6
#define C_LCBLK 7

#define S_LUMA 0
#define S_RG 1
#define S_YB 2
#define S_RG2 3
#define S_YB2 4
#define S_CHROMA 5
#define S_CA 6
#define S_CB 7
#define S_STRUCT 8
#define S_LC 9

#define LOG2_GREY (-2.4739311883f)
#define Y_R 0.2627002120f
#define Y_G 0.6779980715f
#define Y_B 0.0593017165f

// Pixels whose darkest channel is this close to the clip point carry no
// colour information, so a white-balance change must not tint them.
#define CLIP_KNEE 0.92f
// A pixel is "clipped in the RAW" for clip_hi_new when any channel reaches this.
#define RAW_CLIP 0.995f
// Vibrance: chroma at which the boost halves, and the skin-hue window
// (Oklab hue 45 +- 25 degrees) that keeps only 30% of it.
#define VIB_C0 0.1f
#define SKIN_COS 0.70710678f
#define SKIN_SIN 0.70710678f
#define SKIN_EDGE 0.90630779f
#define SKIN_KEEP 0.7f
// Gamut: chroma below KNEE of the sRGB boundary is untouched; above it the
// chroma rolls off smoothly so out-of-gamut colours land just inside.
#define KNEE 0.8f
#define KNEE_INV 1.25f
#define OOG_EPS 1e-3f
// The structure statistic's scene log luminance floor (relative to raw
// clip): read noise below it is not structure.
#define SLUM_EPS 0.000244140625f
// Visibility thresholds of local contrast: EV of scene log luminance and
// sRGB-encoded luma.
#define VIS_X 0.08f
#define VIS_Y 0.01f
#define STRUCT_C1 0.01f
#define STRUCT_C2 0.0001f

struct Lab {
    float L, a, b;
};

__device__ __forceinline__ float clamp01(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }

__device__ __forceinline__ float smooth01(float t) {
    t = clamp01(t);
    return t * t * (3.0f - 2.0f * t);
}

// NaN and negative samples become 0 and absurd ones are capped, so a bad
// pixel cannot poison the statistics.
__device__ __forceinline__ float sane(float v) { return fminf(fmaxf(v, 0.0f), 64.0f); }

// Table sampled uniformly on [x0, x1], linearly interpolated, clamped at the ends.
__device__ __forceinline__ float lut(const float* t, int n, float x0, float x1, float x) {
    float f = (x - x0) * ((float)(n - 1) / (x1 - x0));
    f = fminf(fmaxf(f, 0.0f), (float)(n - 1));
    int i = min((int)f, n - 2);
    float a = f - (float)i;
    return t[i] + a * (t[i + 1] - t[i]);
}

// Global tone curve: scene-linear channel value -> display-linear [0, 1].
// Below the table the curve is proportional, which it already is there.
__device__ __forceinline__ float tone(const float* t, float s) {
    if (!(s > 0.0f)) return 0.0f;
    float u = __log2f(s) - LOG2_GREY;
    if (u <= TONE_U0) return t[0] * exp2f(u - TONE_U0);
    return lut(t, TONE_N, TONE_U0, TONE_U1, u);
}

__device__ __forceinline__ void rec2020_to_oklab(float r, float g, float b, Lab& o) {
    float l = 0.6167557872f * r + 0.3601983994f * g + 0.0230458134f * b;
    float m = 0.2651330640f * r + 0.6358393641f * g + 0.0990275718f * b;
    float s = 0.1001026342f * r + 0.2039065194f * g + 0.6959908464f * b;
    l = cbrtf(l);
    m = cbrtf(m);
    s = cbrtf(s);
    o.L = 0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s;
    o.a = 1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s;
    o.b = 0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s;
}

__device__ __forceinline__ void oklab_to_srgb_lin(float L, float a, float b, float& r, float& g, float& bl) {
    float l = L + 0.3963377774f * a + 0.2158037573f * b;
    float m = L - 0.1055613458f * a - 0.0638541728f * b;
    float s = L - 0.0894841775f * a - 1.2914855480f * b;
    l = l * l * l;
    m = m * m * m;
    s = s * s * s;
    r = 4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
    g = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
    bl = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
}

__device__ __forceinline__ bool in_gamut(float r, float g, float b, float eps) {
    return r >= -eps && g >= -eps && b >= -eps && r <= 1.0f + eps && g <= 1.0f + eps && b <= 1.0f + eps;
}

__device__ __forceinline__ float oetf(float v) {
    return v <= 0.0031308f ? 12.92f * v : 1.055f * exp2f(__log2f(v) * (1.0f / 2.4f)) - 0.055f;
}

__device__ __forceinline__ float hash01(unsigned x, unsigned y, unsigned c) {
    unsigned h = x * 0x8da6b343u ^ y * 0xd8163841u ^ c * 0xcb1ab31fu;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return (float)(h >> 8) * (1.0f / 16777216.0f);
}

// Rounds with a random threshold: unbiased, and smooth gradients do not band.
__device__ __forceinline__ unsigned char quant8(float e, unsigned x, unsigned y, unsigned c) {
    return (unsigned char)fminf(floorf(clamp01(e) * 255.0f + hash01(x, y, c)), 255.0f);
}

__device__ __forceinline__ float log2_lum(float Y) {
    return fminf(fmaxf(__log2f(fmaxf(Y, 1e-30f)), GRID_Z0), GRID_Z1);
}

// Base layer at grid coordinates (gx, gy) for log2 luminance l: trilinear
// read of the blurred bilateral grid, normalized by its weights. The tiny
// prior on the pixel's own value covers luminances the grid never saw.
__device__ float slice_base(const float* __restrict__ grid, int nx, int ny, int nz, float gx, float gy, float l) {
    float gz = (l - GRID_Z0) * (1.0f / GRID_DZ) + (float)GRID_PAD;
    gx += (float)GRID_PAD;
    gy += (float)GRID_PAD;
    int ix = min(max((int)floorf(gx), 0), nx - 2);
    int iy = min(max((int)floorf(gy), 0), ny - 2);
    int iz = min(max((int)floorf(gz), 0), nz - 2);
    float fx = clamp01(gx - (float)ix), fy = clamp01(gy - (float)iy), fz = clamp01(gz - (float)iz);
    float s = 0.0f, w = 0.0f;
    for (int dz = 0; dz < 2; dz++) {
        float wz = dz ? fz : 1.0f - fz;
        for (int dy = 0; dy < 2; dy++) {
            float wy = wz * (dy ? fy : 1.0f - fy);
            const float* row = grid + 2 * (((iz + dz) * ny + (iy + dy)) * nx + ix);
            s += wy * ((1.0f - fx) * row[0] + fx * row[2]);
            w += wy * ((1.0f - fx) * row[1] + fx * row[3]);
        }
    }
    return (s + GRID_PRIOR * l) / (w + GRID_PRIOR);
}

// One pixel from camera RGB (as-shot white balance, 1 = clip) to Oklab after
// vibrance and saturation, before gamut mapping. `base` is the pixel's base
// layer: smoothed log2 luminance at exposure 0 and as-shot white balance.
__device__ Lab render_core(const float* __restrict__ P, float r, float g, float b, float base) {
    r = sane(r);
    g = sane(g);
    b = sane(b);
    float x = P[P_A1 + 0] * r + P[P_A1 + 1] * g + P[P_A1 + 2] * b;
    float y = P[P_A1 + 3] * r + P[P_A1 + 4] * g + P[P_A1 + 5] * b;
    float z = P[P_A1 + 6] * r + P[P_A1 + 7] * g + P[P_A1 + 8] * b;
    float wc = smooth01((fminf(r, fminf(g, b)) - CLIP_KNEE) * (1.0f / (1.0f - CLIP_KNEE)));
    if (wc > 0.0f) {
        float x2 = P[P_A2 + 0] * r + P[P_A2 + 1] * g + P[P_A2 + 2] * b;
        float y2 = P[P_A2 + 3] * r + P[P_A2 + 4] * g + P[P_A2 + 5] * b;
        float z2 = P[P_A2 + 6] * r + P[P_A2 + 7] * g + P[P_A2 + 8] * b;
        x += wc * (x2 - x);
        y += wc * (y2 - y);
        z += wc * (z2 - z);
    }
    // Colours outside Rec.2020 move toward grey at constant luminance until
    // no channel is negative.
    float Y = Y_R * x + Y_G * y + Y_B * z;
    float mn = fminf(x, fminf(y, z));
    if (mn < 0.0f) {
        if (Y > 0.0f) {
            float k = Y / (Y - mn);
            x = Y + (x - Y) * k;
            y = Y + (y - Y) * k;
            z = Y + (z - Y) * k;
        } else {
            x = y = z = 0.0f;
        }
    }
    // Exposure, then highlights/shadows as a gain driven by the base layer:
    // detail (log luminance minus base) passes through untouched.
    float gain = P[P_GAIN] * exp2f(lut(P + LOCAL_OFF, LOCAL_N, LOCAL_B0, LOCAL_B1, base + P[P_BSHIFT]));
    x *= gain;
    y *= gain;
    z *= gain;
    // Whites and Blacks, by the pixel's own luminance and on all channels
    // alike: brightness decides, not the brightest channel.
    float Yp = Y_R * x + Y_G * y + Y_B * z;
    if (Yp > 0.0f) {
        float e = exp2f(lut(P + ENDS_OFF, LOCAL_N, LOCAL_B0, LOCAL_B1, __log2f(Yp) - LOG2_GREY));
        x *= e;
        y *= e;
        z *= e;
    }
    // Global tone on the largest and smallest channel, the middle one keeping
    // its relative place: hue holds, and colours fade toward white as their
    // brightest channel reaches it instead of clipping.
    float hi = fmaxf(x, fmaxf(y, z)), lo = fminf(x, fminf(y, z));
    float thi = tone(P + TONE_OFF, hi);
    if (hi - lo > 1e-20f) {
        float tlo = tone(P + TONE_OFF, lo);
        float k = (thi - tlo) / (hi - lo);
        x = tlo + (x - lo) * k;
        y = tlo + (y - lo) * k;
        z = tlo + (z - lo) * k;
    } else {
        x = y = z = thi;
    }
    Lab o;
    rec2020_to_oklab(x, y, z, o);
    float C = sqrtf(o.a * o.a + o.b * o.b);
    if (C > 1e-9f) {
        float f = P[P_SAT];
        float vib = P[P_VIB];
        if (vib != 0.0f) {
            float q = C * (1.0f / VIB_C0);
            float cosd = (o.a * SKIN_COS + o.b * SKIN_SIN) / C;
            float skin = smooth01((cosd - SKIN_EDGE) * (1.0f / (1.0f - SKIN_EDGE)));
            f *= 1.0f + vib * (1.0f - SKIN_KEEP * skin) / (1.0f + q * q);
        }
        o.a *= f;
        o.b *= f;
    }
    return o;
}

// Oklab -> linear sRGB in [0, 1]. Chroma is compressed toward the sRGB
// boundary at constant lightness and hue: untouched below KNEE of the
// boundary, rolling off smoothly above it, so out-of-gamut colours keep their
// gradations instead of flattening at the edge. `oog` reports whether the
// colour was outside the gamut before mapping.
__device__ void to_srgb_gamut(Lab c, float& r, float& g, float& b, bool& oog) {
    float L = clamp01(c.L);
    oklab_to_srgb_lin(L, c.a, c.b, r, g, b);
    oog = !in_gamut(r, g, b, OOG_EPS);
    float r2, g2, b2;
    oklab_to_srgb_lin(L, c.a * KNEE_INV, c.b * KNEE_INV, r2, g2, b2);
    if (!in_gamut(r2, g2, b2, 0.0f)) {
        // The boundary chroma as a fraction of the current one, by bisection.
        float lo, hi;
        if (in_gamut(r, g, b, 0.0f)) {
            lo = 1.0f;
            hi = KNEE_INV;
        } else {
            lo = 0.0f;
            hi = 1.0f;
        }
        for (int i = 0; i < 12; i++) {
            float mid = 0.5f * (lo + hi);
            oklab_to_srgb_lin(L, c.a * mid, c.b * mid, r2, g2, b2);
            if (in_gamut(r2, g2, b2, 0.0f)) lo = mid;
            else hi = mid;
        }
        float k = 0.0f;
        if (lo > 0.0f) {
            float x = 1.0f / lo;
            float xs = x <= KNEE ? x : KNEE + (1.0f - KNEE) * tanhf((x - KNEE) * (1.0f / (1.0f - KNEE)));
            k = xs * lo;
        }
        oklab_to_srgb_lin(L, c.a * k, c.b * k, r, g, b);
    }
    r = clamp01(r);
    g = clamp01(g);
    b = clamp01(b);
}

// Packs a preview image for rendering: RGB plus its base layer in w, and in
// `aux`: nothing (aux_mode 0), the scene log luminance the structure
// statistic reads (1), or the clamped log2 luminance that guides the
// full-resolution upsampling of the base layer (2).
extern "C" __global__ void prep_kernel(const float* __restrict__ rgb, int w, int h,
                                       const float* __restrict__ grid, int gnx, int gny, int gnz,
                                       float cells_per_px, float yr, float yg, float yb,
                                       float4* __restrict__ out, float* __restrict__ aux, int aux_mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h) return;
    int x = i % w, y = i / w;
    float r = sane(rgb[3 * i]), g = sane(rgb[3 * i + 1]), b = sane(rgb[3 * i + 2]);
    float Y = yr * r + yg * g + yb * b;
    float l = log2_lum(Y);
    float4 o;
    o.x = r;
    o.y = g;
    o.z = b;
    o.w = slice_base(grid, gnx, gny, gnz, ((float)x + 0.5f) * cells_per_px, ((float)y + 0.5f) * cells_per_px, l);
    out[i] = o;
    if (aux_mode == 1) aux[i] = __log2f(fmaxf(Y, 0.0f) + SLUM_EPS);
    else if (aux_mode == 2) aux[i] = l;
}

// The display image's base layer at a full-resolution pixel: joint bilateral
// upsampling (Kopf et al. 2007) over the 4x4 nearest display pixels, weighted
// by distance and by how close each one's luminance is to this pixel's. In
// smooth and finely textured regions this is the display base exactly, so the
// export matches the preview; at strong edges only the display pixels on this
// pixel's side count, so the base stays sharp.
#define UP_SIGMA 0.75f
#define UP_RANGE 0.5f
__device__ float upsample_base(const float4* __restrict__ dpx, const float* __restrict__ guide, int dw, int dh,
                               float xd, float yd, float l) {
    int x0 = (int)floorf(xd) - 1, y0 = (int)floorf(yd) - 1;
    float s = 0.f, w = 0.f, s_sp = 0.f, w_sp = 0.f;
    for (int j = 0; j < 4; j++) {
        int yy = min(max(y0 + j, 0), dh - 1);
        float dy = (float)(y0 + j) - yd;
        for (int i = 0; i < 4; i++) {
            int xx = min(max(x0 + i, 0), dw - 1);
            float dx = (float)(x0 + i) - xd;
            size_t k = (size_t)yy * dw + xx;
            float ws = __expf(-(dx * dx + dy * dy) * (0.5f / (UP_SIGMA * UP_SIGMA)));
            float dl = guide[k] - l;
            float wr = ws * __expf(-dl * dl * (0.5f / (UP_RANGE * UP_RANGE)));
            float b = dpx[k].w;
            s += wr * b;
            w += wr;
            s_sp += ws * b;
            w_sp += ws;
        }
    }
    // No display pixel alike (a full-resolution specular): fall back smoothly
    // to plain spatial interpolation.
    return (s + 1e-3f * s_sp) / (w + 1e-3f * w_sp);
}

// Renders one candidate to 8-bit sRGB with dithering.
extern "C" __global__ void render8_kernel(const float4* __restrict__ px, int w, int h,
                                          const float* __restrict__ cand, unsigned char* __restrict__ out) {
    __shared__ float P[CAND_STRIDE];
    for (int i = threadIdx.x; i < CAND_STRIDE; i += blockDim.x) P[i] = cand[i];
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h) return;
    float4 p = px[i];
    Lab c = render_core(P, p.x, p.y, p.z, p.w);
    float r, g, b;
    bool oog;
    to_srgb_gamut(c, r, g, b, oog);
    unsigned x = (unsigned)(i % w), y = (unsigned)(i / w);
    out[3 * i] = quant8(oetf(r), x, y, 0);
    out[3 * i + 1] = quant8(oetf(g), x, y, 1);
    out[3 * i + 2] = quant8(oetf(b), x, y, 2);
}

__device__ __forceinline__ float warp_sum(float v, unsigned mask_bits) {
    for (int m = 16; m >= 1; m >>= 1)
        if (mask_bits & m) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ unsigned warp_sum_u(unsigned v) {
    for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// Renders candidates on the search image and measures them. Grid: x = chunk
// of the image, y = candidate. Each block walks 16x16 tiles (an 8x8 block per
// pair of warps, a 4x4 window per half warp) and writes one partial record:
// histograms and counts as u32, sums as f64, all reduced in a fixed order so
// the result is deterministic. The host adds the chunks and finishes.
extern "C" __global__ void __launch_bounds__(STATS_THREADS)
stats_kernel(const float4* __restrict__ px, const float* __restrict__ slum, int w, int h,
             const float* __restrict__ cands, int chunks,
             unsigned* __restrict__ out_u32, double* __restrict__ out_f64) {
    __shared__ float P[CAND_STRIDE];
    __shared__ unsigned s_luma[LUMA_BINS];
    __shared__ unsigned s_chroma[CHROMA_BINS];
    __shared__ float s_half[STATS_THREADS / 32][6];
    __shared__ double s_red[N_SUMS][STATS_THREADS / 32];
    __shared__ unsigned s_cnt[N_COUNTS][STATS_THREADS / 32];

    const int cand = blockIdx.y, chunk = blockIdx.x, tid = threadIdx.x;
    const float* src = cands + (size_t)cand * CAND_STRIDE;
    for (int i = tid; i < CAND_STRIDE; i += STATS_THREADS) P[i] = src[i];
    for (int i = tid; i < LUMA_BINS; i += STATS_THREADS) s_luma[i] = 0;
    for (int i = tid; i < CHROMA_BINS; i += STATS_THREADS) s_chroma[i] = 0;
    __syncthreads();

    float a_luma = 0.f, a_rg = 0.f, a_yb = 0.f, a_rg2 = 0.f, a_yb2 = 0.f, a_chroma = 0.f, a_ca = 0.f, a_cb = 0.f,
          a_struct = 0.f, a_lc = 0.f;
    unsigned n_px = 0, n_hi = 0, n_hi_new = 0, n_lo = 0, n_oog = 0, n_cast = 0, n_sblk = 0, n_lcblk = 0;

    const int tiles_x = (w + 15) / 16, tiles_y = (h + 15) / 16, ntiles = tiles_x * tiles_y;
    const int blk = tid >> 6, l = tid & 63, warp = tid >> 5, lane = tid & 31;
    const int ox = (blk & 1) * 8 + (l & 7), oy = (blk >> 1) * 8 + (l >> 3);

    for (int t = chunk; t < ntiles; t += chunks) {
        const int x = (t % tiles_x) * 16 + ox, y = (t / tiles_x) * 16 + oy;
        const bool valid = x < w && y < h;
        float luma = 0.f, sx = 0.f;
        if (valid) {
            const size_t i = (size_t)y * w + x;
            const float4 p = px[i];
            Lab c = render_core(P, p.x, p.y, p.z, p.w);
            float r, g, b;
            bool oog;
            to_srgb_gamut(c, r, g, b, oog);
            const float R = oetf(r), G = oetf(g), B = oetf(b);
            luma = 0.2126f * R + 0.7152f * G + 0.0722f * B;
            // Oklab of what is actually shown (after gamut mapping).
            Lab s;
            {
                // linear sRGB -> LMS (Oklab M1), cube roots, M2
                float lm = 0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b;
                float mm = 0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b;
                float sm = 0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b;
                lm = cbrtf(lm);
                mm = cbrtf(mm);
                sm = cbrtf(sm);
                s.L = 0.2104542553f * lm + 0.7936177850f * mm - 0.0040720468f * sm;
                s.a = 1.9779984951f * lm - 2.4285922050f * mm + 0.4505937099f * sm;
                s.b = 0.0259040371f * lm + 0.7827717662f * mm - 0.8086757660f * sm;
            }
            const float chroma = sqrtf(s.a * s.a + s.b * s.b);
            const bool hi = fmaxf(R, fmaxf(G, B)) >= 254.0f / 255.0f;
            const bool raw_clipped = fmaxf(p.x, fmaxf(p.y, p.z)) >= RAW_CLIP;
            n_px++;
            n_hi += hi;
            n_hi_new += hi && !raw_clipped;
            n_lo += luma <= 2.0f / 255.0f;
            n_oog += oog;
            a_luma += luma;
            const float rg = R - G, yb = 0.5f * (R + G) - B;
            a_rg += rg;
            a_yb += yb;
            a_rg2 += rg * rg;
            a_yb2 += yb * yb;
            a_chroma += chroma;
            if (chroma < 0.04f && s.L > 0.15f && s.L < 0.95f) {
                n_cast++;
                a_ca += s.a;
                a_cb += s.b;
            }
            atomicAdd(&s_luma[min((int)(luma * (float)LUMA_BINS), LUMA_BINS - 1)], 1u);
            atomicAdd(&s_chroma[min((int)(chroma * ((float)CHROMA_BINS / CHROMA_MAX)), CHROMA_BINS - 1)], 1u);
            sx = slum[i];
        }
        // Local contrast: std of luma in each 4x4 window (lanes r*8 + c of a
        // warp; xor over the row and low column bits stays inside a window).
        {
            const unsigned m = 1 | 2 | 8 | 16;
            float v1 = warp_sum(luma, m), v2 = warp_sum(luma * luma, m), vn = warp_sum(valid ? 1.f : 0.f, m);
            if ((lane & 27) == 0 && vn == 16.f) {
                float mean = v1 * (1.f / 16.f);
                a_lc += sqrtf(fmaxf(v2 * (1.f / 16.f) - mean * mean, 0.f));
                n_lcblk++;
            }
        }
        // Structure: each 8x8 block is two warps; reduce each warp, then the
        // block's first thread combines both halves.
        {
            const unsigned all = 31;
            float sv[6] = {sx, luma, sx * sx, luma * luma, sx * luma, valid ? 1.f : 0.f};
            for (int k = 0; k < 6; k++) sv[k] = warp_sum(sv[k], all);
            if (lane == 0)
                for (int k = 0; k < 6; k++) s_half[warp][k] = sv[k];
            __syncthreads();
            if (l == 0) {
                float q[6];
                for (int k = 0; k < 6; k++) q[k] = s_half[2 * blk][k] + s_half[2 * blk + 1][k];
                if (q[5] == 64.f) {
                    const float inv = 1.f / 64.f;
                    float mx = q[0] * inv, my = q[1] * inv;
                    float vx = fmaxf(q[2] * inv - mx * mx, 0.f), vy = fmaxf(q[3] * inv - my * my, 0.f);
                    float cxy = q[4] * inv - mx * my;
                    float sdx = sqrtf(vx), sdy = sqrtf(vy);
                    // Visibility of local contrast in each image (0 = flat, 1 = clearly visible).
                    float tx = vx / (vx + VIS_X * VIS_X), ty = vy / (vy + VIS_Y * VIS_Y);
                    float cterm = (2.f * tx * ty + STRUCT_C1) / (tx * tx + ty * ty + STRUCT_C1);
                    float sterm = fmaxf((cxy + STRUCT_C2) / (sdx * sdy + STRUCT_C2), 0.f);
                    a_struct += cterm * fminf(sterm, 1.f);
                    n_sblk++;
                }
            }
            __syncthreads();
        }
    }

    // Partial record: histograms, counts, sums.
    const size_t part = (size_t)cand * chunks + chunk;
    unsigned* ou = out_u32 + part * PART_U32;
    double* od = out_f64 + part * N_SUMS;
    const float sums[N_SUMS] = {a_luma, a_rg, a_yb, a_rg2, a_yb2, a_chroma, a_ca, a_cb, a_struct, a_lc};
    const unsigned counts[N_COUNTS] = {n_px, n_hi, n_hi_new, n_lo, n_oog, n_cast, n_sblk, n_lcblk};
    for (int k = 0; k < N_SUMS; k++) {
        float v = warp_sum(sums[k], 31);
        if (lane == 0) s_red[k][warp] = (double)v;
    }
    for (int k = 0; k < N_COUNTS; k++) {
        unsigned v = warp_sum_u(counts[k]);
        if (lane == 0) s_cnt[k][warp] = v;
    }
    __syncthreads();
    if (tid < N_SUMS) {
        double acc = 0.0;
        for (int k = 0; k < STATS_THREADS / 32; k++) acc += s_red[tid][k];
        od[tid] = acc;
    } else if (tid >= 32 && tid < 32 + N_COUNTS) {
        unsigned acc = 0;
        for (int k = 0; k < STATS_THREADS / 32; k++) acc += s_cnt[tid - 32][k];
        ou[LUMA_BINS + CHROMA_BINS + (tid - 32)] = acc;
    }
    for (int i = tid; i < LUMA_BINS; i += STATS_THREADS) ou[i] = s_luma[i];
    for (int i = tid; i < CHROMA_BINS; i += STATS_THREADS) ou[LUMA_BINS + i] = s_chroma[i];
}

// Full resolution, one horizontal strip: renders into Oklab planes (before
// gamut mapping) so sharpening and chroma noise reduction can run on L and
// a/b. `y0` is the strip's first row in the full image; `dpx`/`guide` are the
// prepared display image (`prep_kernel`, aux_mode 2) whose base layer is
// upsampled, and `sx`/`sy` map full-resolution to display pixels.
extern "C" __global__ void full_kernel(const float* __restrict__ rgb, int w, int rows, int y0,
                                       const float4* __restrict__ dpx, const float* __restrict__ guide, int dw, int dh,
                                       float sx, float sy, float yr, float yg, float yb,
                                       const float* __restrict__ cand, float* __restrict__ Lp, float* __restrict__ Ap,
                                       float* __restrict__ Bp) {
    __shared__ float P[CAND_STRIDE];
    for (int i = threadIdx.x; i < CAND_STRIDE; i += blockDim.x) P[i] = cand[i];
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * rows) return;
    int x = i % w, y = i / w + y0;
    float r = sane(rgb[3 * (size_t)i]), g = sane(rgb[3 * (size_t)i + 1]), b = sane(rgb[3 * (size_t)i + 2]);
    float base = upsample_base(dpx, guide, dw, dh, ((float)x + 0.5f) * sx - 0.5f, ((float)y + 0.5f) * sy - 0.5f,
                               log2_lum(yr * r + yg * g + yb * b));
    Lab c = render_core(P, r, g, b, base);
    Lp[i] = c.L;
    Ap[i] = c.a;
    Bp[i] = c.b;
}

// Horizontal pass: Gaussian of L (for the unsharp mask) and an edge-aware
// (bilateral, guided by L and a/b) blur of a/b for chroma noise reduction.
extern "C" __global__ void hpass_kernel(const float* __restrict__ Lp, const float* __restrict__ Ap,
                                        const float* __restrict__ Bp, int w, int rows,
                                        float sharp_sigma, int sharp_rad, float nr_sigma, int nr_rad,
                                        float nr_l, float nr_ab,
                                        float* __restrict__ Lh, float* __restrict__ Ah, float* __restrict__ Bh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * rows) return;
    int x = i % w, row = i - x;
    float ks = -0.5f / (sharp_sigma * sharp_sigma), kn = -0.5f / (nr_sigma * nr_sigma);
    float kl = -0.5f / (nr_l * nr_l), kab = -0.5f / (nr_ab * nr_ab);
    float sl = 0.f, wl = 0.f;
    for (int d = -sharp_rad; d <= sharp_rad; d++) {
        int xx = min(max(x + d, 0), w - 1);
        float wt = __expf(ks * (float)(d * d));
        sl += wt * Lp[row + xx];
        wl += wt;
    }
    Lh[i] = sl / wl;
    float L0 = Lp[i], a0 = Ap[i], b0 = Bp[i];
    float sa = 0.f, sb = 0.f, wab = 0.f;
    for (int d = -nr_rad; d <= nr_rad; d++) {
        int j = row + min(max(x + d, 0), w - 1);
        float dl = Lp[j] - L0, da = Ap[j] - a0, db = Bp[j] - b0;
        float wt = __expf(kn * (float)(d * d) + kl * dl * dl + kab * (da * da + db * db));
        sa += wt * Ap[j];
        sb += wt * Bp[j];
        wab += wt;
    }
    Ah[i] = sa / wab;
    Bh[i] = sb / wab;
}

// Vertical pass and output: finishes both blurs, applies the unsharp mask to
// L (soft threshold spares fine noise; the result stays inside the 3x3
// neighbourhood's range, so edges get crisper without halos), gamut-maps and
// writes 16-bit sRGB for rows [row0, row0 + out_rows) of the strip.
extern "C" __global__ void finish16_kernel(const float* __restrict__ Lp, const float* __restrict__ Lh,
                                           const float* __restrict__ Ah, const float* __restrict__ Bh, int w,
                                           int rows, int row0, int out_rows, float sharp_sigma, int sharp_rad,
                                           float amount, float threshold, float nr_sigma, int nr_rad, float nr_l,
                                           float nr_ab, unsigned short* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * out_rows) return;
    int x = i % w, y = i / w + row0;
    float ks = -0.5f / (sharp_sigma * sharp_sigma), kn = -0.5f / (nr_sigma * nr_sigma);
    float kl = -0.5f / (nr_l * nr_l), kab = -0.5f / (nr_ab * nr_ab);
    float sl = 0.f, wl = 0.f;
    for (int d = -sharp_rad; d <= sharp_rad; d++) {
        int yy = min(max(y + d, 0), rows - 1);
        float wt = __expf(ks * (float)(d * d));
        sl += wt * Lh[(size_t)yy * w + x];
        wl += wt;
    }
    size_t c = (size_t)y * w + x;
    float L0 = Lp[c], a0 = Ah[c], b0 = Bh[c];
    float sa = 0.f, sb = 0.f, wab = 0.f;
    for (int d = -nr_rad; d <= nr_rad; d++) {
        size_t j = (size_t)min(max(y + d, 0), rows - 1) * w + x;
        float dl = Lp[j] - L0, da = Ah[j] - a0, db = Bh[j] - b0;
        float wt = __expf(kn * (float)(d * d) + kl * dl * dl + kab * (da * da + db * db));
        sa += wt * Ah[j];
        sb += wt * Bh[j];
        wab += wt;
    }
    float detail = L0 - sl / wl;
    detail *= detail * detail / (detail * detail + threshold * threshold);
    float lo = L0, hi = L0;
    for (int dy = -1; dy <= 1; dy++) {
        const float* row = Lp + (size_t)min(max(y + dy, 0), rows - 1) * w;
        for (int dx = -1; dx <= 1; dx++) {
            float v = row[min(max(x + dx, 0), w - 1)];
            lo = fminf(lo, v);
            hi = fmaxf(hi, v);
        }
    }
    Lab o;
    o.L = fminf(fmaxf(L0 + amount * detail, lo), hi);
    o.a = sa / wab;
    o.b = sb / wab;
    float r, g, b;
    bool oog;
    to_srgb_gamut(o, r, g, b, oog);
    size_t q = 3 * (size_t)i;
    out[q] = (unsigned short)(oetf(r) * 65535.0f + 0.5f);
    out[q + 1] = (unsigned short)(oetf(g) * 65535.0f + 0.5f);
    out[q + 2] = (unsigned short)(oetf(b) * 65535.0f + 0.5f);
}
