// Texture, Clarity and Dehaze. Owner: presence (with src/render/presence.rs,
// which launches the plane kernels, packs the sliders and holds the scales).
//
// src/render.rs splices this file into cuda/render.cu at its HOOKS line, so
// render.cu's #defines and helpers above that line are in scope, and its
// kernels are part of the same module.
//
// Two parts:
// - Planes, computed once per image (the search and display previews when
//   the engine is built, every strip of a full-resolution export) by the
//   pres_* kernels below. A float4 per pixel, independent of the sliders:
//     x  Texture's band: the log2 luminance lightly blurred (sigma 0.04% of
//        the long edge, below which is mostly noise) minus its edge-aware
//        smoothing over 0.4% of the long edge (a self-guided filter, He et
//        al. 2010, whose epsilon keeps strong edges out of the band);
//     y  that smoothing itself (log2 luminance). Clarity's band is y minus
//        the pixel's base layer (bilateral grid, about 2.5% of the long
//        edge, edge-aware, so strong edges get no halo) plus half of x;
//     z  haze transmission: dark channel prior (He, Sun and Tang 2009) over
//        a patch of 1% of the long edge, refined by a guided filter on
//        luminance;
//     w  the airlight (camera RGB at exposure 0, 1 = raw clip; neutral).
//   Every radius is a fraction of the long edge, so the 768 px search image,
//   the 1536 px display image and the export agree at their sizes. The
//   export filters its own strips for x and y and takes z and w from the
//   display image (pres_strip_kernel); its base layer is the display one,
//   upsampled.
// - presence_apply, called by render_core per pixel on scene-linear Rec.2020
//   after exposure, Highlights/Shadows and Whites/Blacks, before the global
//   tone curve. The sliders' strengths are packed by presence.rs at
//   P[P_PRES + PR_*]; PR_ON is 0 when all three sliders are, and the stage
//   is then skipped.

#define PRESENCE_HOOK

// Candidate slots, after P_PRES (packed by presence.rs).
#define PR_ON 0
#define PR_TEXTURE 1
#define PR_CLARITY 2
#define PR_UNVEIL 3
#define PR_FOG 4
#define PR_HAZE_LC 5
#define PR_HAZE_SAT 6
// Band amplitudes beyond which the boost saturates (EV): a bright speck on
// a dark wall gets crisper, not blown out.
#define PR_TEX_SOFT 0.5f
#define PR_CLA_SOFT 1.25f
// Share of Texture's band that Clarity also boosts.
#define PR_CLA_FINE 0.5f
// Coring of boosted fine detail (EV), and the EV from middle grey below
// which Texture stops boosting (fully PR_BLACK_WIDTH above it).
#define PR_CORE 0.03f
#define PR_BLACK (-8.0f)
#define PR_BLACK_WIDTH 3.0f
// Clarity's midtone weight: full around PR_MID EV from middle grey, half at
// PR_MID +- PR_MID_WIDTH, fading beyond, so highlights and shadows are spared.
#define PR_MID (-0.5f)
#define PR_MID_WIDTH 3.0f
// Dehaze keeps at least this share of a channel the veil would erase.
#define PR_FLOOR 0.15f
// Negative Dehaze: share of the fog that lands on near (haze-free) pixels.
#define PR_FOG_NEAR 0.4f

__device__ __forceinline__ float pres_soft(float d, float k) { return d * rsqrtf(1.0f + d * d * (1.0f / (k * k))); }

// Removes the veil from one channel and rescales by the transmission:
// max(v - veil, PR_FLOOR * v), smoothed so nothing clips, times inv_t.
__device__ __forceinline__ float pres_unveil(float v, float veil, float inv_t) {
    float a = v - veil, b = PR_FLOOR * v, d = a - b, e = 0.1f * veil;
    return 0.5f * (a + b + sqrtf(d * d + e * e)) * inv_t;
}

__device__ __forceinline__ void presence_apply(const float* __restrict__ P, float4 q, float base, float& x, float& y,
                                               float& z) {
    const float* S = P + P_PRES;
    if (S[PR_ON] == 0.0f) return;
    float unveil = S[PR_UNVEIL], fog = S[PR_FOG];
    if (unveil > 0.0f || fog > 0.0f) {
        // The airlight as this candidate renders it: through exposure and
        // Highlights/Shadows like the pixel (Whites/Blacks aside).
        float gain = P[P_GAIN] * exp2f(lut(P + LOCAL_OFF, LOCAL_N, LOCAL_B0, LOCAL_B1, base + P[P_BSHIFT]));
        float A = q.w * gain, haze = 1.0f - q.z;
        if (unveil > 0.0f) {
            // I = J t + A (1 - t), solved for J with t raised to 1 - s (1 - t).
            float h = unveil * haze;
            float veil = A * h, inv_t = 1.0f / (1.0f - h);
            x = pres_unveil(x, veil, inv_t);
            y = pres_unveil(y, veil, inv_t);
            z = pres_unveil(z, veil, inv_t);
        } else {
            float f = fog * (PR_FOG_NEAR + (1.0f - PR_FOG_NEAR) * haze);
            x += f * (A - x);
            y += f * (A - y);
            z += f * (A - z);
        }
    }
    float Y = Y_R * x + Y_G * y + Y_B * z;
    if (!(Y > 0.0f)) return;
    float u = __log2f(Y) - LOG2_GREY;
    float m = (u - PR_MID) * (1.0f / PR_MID_WIDTH);
    m *= m;
    float mid = 1.0f / (1.0f + m * m);
    // Boosts core the fine band (wiggles under PR_CORE EV are mostly noise)
    // and leave the near-black alone; smoothing takes the band as it is.
    float tex = S[PR_TEXTURE], cla = S[PR_CLARITY] + S[PR_HAZE_LC];
    float fine = q.x, cored = q.x * q.x * q.x / (q.x * q.x + PR_CORE * PR_CORE);
    float lift = smooth01((u - PR_BLACK) * (1.0f / PR_BLACK_WIDTH));
    float e = tex * pres_soft(tex > 0.0f ? cored * lift : fine, PR_TEX_SOFT) +
              cla * mid * pres_soft(q.y - base + PR_CLA_FINE * (cla > 0.0f ? cored : fine), PR_CLA_SOFT);
    float k = exp2f(e);
    x *= k;
    y *= k;
    z *= k;
    float sat = S[PR_HAZE_SAT];
    if (sat != 1.0f) {
        // Chroma about luminance, short of pushing any channel below zero.
        float Yk = Y * k, lo = fminf(x, fminf(y, z));
        if (lo < Yk) sat = fminf(sat, 1.0f + 0.9f * fmaxf(lo, 0.0f) / (Yk - lo));
        x = Yk + (x - Yk) * sat;
        y = Yk + (y - Yk) * sat;
        z = Yk + (z - Yk) * sat;
    }
}

// ---------------------------------------------------------------------------
// Plane kernels (launched from presence.rs). Images are w x h, row-major.

__device__ __forceinline__ size_t pres_index() { return (size_t)blockIdx.x * blockDim.x + threadIdx.x; }

// log2 luminance exactly as the base layer sees it, and the darkest channel.
// `src` holds `stride` floats per pixel (a prepared float4 image or a strip's RGB).
extern "C" __global__ void pres_lum_kernel(const float* __restrict__ src, int stride, int n, float yr, float yg,
                                           float yb, float* __restrict__ l, float* __restrict__ dark, int want_dark) {
    size_t i = pres_index();
    if (i >= (size_t)n) return;
    const float* p = src + i * stride;
    float r = sane(p[0]), g = sane(p[1]), b = sane(p[2]);
    l[i] = log2_lum(yr * r + yg * g + yb * b);
    if (want_dark) dark[i] = fminf(r, fminf(g, b));
}

// Separable Gaussian pass (edge pixels repeat).
extern "C" __global__ void pres_gauss_kernel(const float* __restrict__ src, int w, int h, float sigma, int rad,
                                             int vertical, float* __restrict__ dst) {
    size_t i = pres_index();
    if (i >= (size_t)w * h) return;
    int x = (int)(i % w), y = (int)(i / w);
    float k = -0.5f / (sigma * sigma), s = 0.0f, ws = 0.0f;
    for (int d = -rad; d <= rad; d++) {
        size_t j = vertical ? (size_t)min(max(y + d, 0), h - 1) * w + x : (size_t)y * w + min(max(x + d, 0), w - 1);
        float wt = __expf(k * (float)(d * d));
        s += wt * src[j];
        ws += wt;
    }
    dst[i] = s / ws;
}

// Separable minimum pass over the taps inside the image.
extern "C" __global__ void pres_min_kernel(const float* __restrict__ src, int w, int h, int rad, int vertical,
                                           float* __restrict__ dst) {
    size_t i = pres_index();
    if (i >= (size_t)w * h) return;
    int x = (int)(i % w), y = (int)(i / w);
    int pos = vertical ? y : x, len = vertical ? h : w;
    size_t step = vertical ? (size_t)w : 1;
    int a = max(pos - rad, 0), b = min(pos + rad, len - 1);
    const float* p = src + i - (size_t)(pos - a) * step;
    float m = *p;
    for (int k = a + 1; k <= b; k++) {
        p += step;
        m = fminf(m, *p);
    }
    dst[i] = m;
}

// Separable box mean over the taps inside the image, on `comps`
// interleaved floats per pixel. One thread per run of PRES_RUN outputs
// along the axis (of one line and component) slides a running sum, so the
// cost does not grow with the radius. Neighbouring threads take
// neighbouring columns (vertical) or runs (horizontal).
#define PRES_RUN 32
extern "C" __global__ void pres_box_kernel(const float* __restrict__ src, int w, int h, int comps, int rad,
                                           int vertical, float* __restrict__ dst) {
    int len = vertical ? h : w, lines = vertical ? w : h, runs = (len + PRES_RUN - 1) / PRES_RUN;
    size_t t = pres_index();
    if (t >= (size_t)runs * lines * comps) return;
    int c = (int)(t % comps);
    size_t rest = t / comps;
    int line, run;
    if (vertical) {
        line = (int)(rest % lines);
        run = (int)(rest / lines);
    } else {
        run = (int)(rest % runs);
        line = (int)(rest / runs);
    }
    size_t step = (vertical ? (size_t)w : 1) * comps;
    size_t origin = (vertical ? (size_t)line : (size_t)line * w) * comps + c;
    const float* in = src + origin;
    float* out = dst + origin;
    int p0 = run * PRES_RUN, p1 = min(p0 + PRES_RUN, len);
    int a = max(p0 - rad, 0), b = min(p0 + rad, len - 1);
    float s = 0.0f;
    for (int k = a; k <= b; k++) s += in[(size_t)k * step];
    for (int p = p0; p < p1; p++) {
        out[(size_t)p * step] = s / (float)(b - a + 1);
        if (p + 1 + rad < len) {
            b = p + 1 + rad;
            s += in[(size_t)b * step];
        }
        if (p - rad >= 0) {
            s -= in[(size_t)(p - rad) * step];
            a = p - rad + 1;
        }
    }
}

// Self-guided filter inputs: (I, I^2) per pixel.
extern "C" __global__ void pres_self_prep_kernel(const float* __restrict__ I, int n, float* __restrict__ out2) {
    size_t i = pres_index();
    if (i >= (size_t)n) return;
    float v = I[i];
    out2[2 * i] = v;
    out2[2 * i + 1] = v * v;
}

// Self-guided filter coefficients (a, b) from the box means of (I, I^2).
extern "C" __global__ void pres_self_coef_kernel(const float* __restrict__ m2, float eps, int n,
                                                 float* __restrict__ ab2) {
    size_t i = pres_index();
    if (i >= (size_t)n) return;
    float mI = m2[2 * i], var = fmaxf(m2[2 * i + 1] - mI * mI, 0.0f);
    float a = var / (var + eps);
    ab2[2 * i] = a;
    ab2[2 * i + 1] = mI * (1.0f - a);
}

// The haze guide (square root of luminance over the airlight) at a pixel.
__device__ __forceinline__ float pres_haze_guide(float l, float inv_a) { return sqrtf(clamp01(exp2f(l) * inv_a)); }

// Guided filter inputs for the transmission: (I, I^2, p, I p), with p the
// raw dark-channel transmission 1 - dark / A.
extern "C" __global__ void pres_haze_prep_kernel(const float* __restrict__ l, const float* __restrict__ dark,
                                                 float inv_a, int n, float* __restrict__ out4) {
    size_t i = pres_index();
    if (i >= (size_t)n) return;
    float I = pres_haze_guide(l[i], inv_a), p = 1.0f - clamp01(dark[i] * inv_a);
    out4[4 * i] = I;
    out4[4 * i + 1] = I * I;
    out4[4 * i + 2] = p;
    out4[4 * i + 3] = I * p;
}

// Guided filter coefficients (a, b) from the box means of (I, I^2, p, I p).
extern "C" __global__ void pres_coef_kernel(const float* __restrict__ m4, float eps, int n, float* __restrict__ ab2) {
    size_t i = pres_index();
    if (i >= (size_t)n) return;
    float mI = m4[4 * i], mp = m4[4 * i + 2];
    float var = fmaxf(m4[4 * i + 1] - mI * mI, 0.0f), cov = m4[4 * i + 3] - mI * mp;
    float a = cov / (var + eps);
    ab2[2 * i] = a;
    ab2[2 * i + 1] = mp - a * mI;
}

// A preview's planes. `ga` is the lightly blurred log2 luminance, `tex_ab`
// its self-guided filter coefficients and `haze_ab` the box-averaged ones of
// the transmission's guided filter.
extern "C" __global__ void pres_planes_kernel(const float* __restrict__ l, const float* __restrict__ ga,
                                              const float* __restrict__ tex_ab, const float* __restrict__ haze_ab,
                                              float inv_a, float airlight, float t_min, int n,
                                              float4* __restrict__ out) {
    size_t i = pres_index();
    if (i >= (size_t)n) return;
    float g = ga[i];
    float smooth = tex_ab[2 * i] * g + tex_ab[2 * i + 1];
    float t = haze_ab[2 * i] * pres_haze_guide(l[i], inv_a) + haze_ab[2 * i + 1];
    out[i] = make_float4(g - smooth, smooth, fminf(fmaxf(t, t_min), 1.0f), airlight);
}

// One full-resolution strip's planes: Texture's band and smoothing from the
// strip itself, the transmission bilinear from the display planes, and
// their airlight. `sx`/`sy` map full-resolution to display pixels.
extern "C" __global__ void pres_strip_kernel(int w, int rows, int y0, const float* __restrict__ ga,
                                             const float* __restrict__ tex_ab, const float4* __restrict__ dplanes,
                                             int dw, int dh, float sx, float sy, float4* __restrict__ out) {
    size_t i = pres_index();
    if (i >= (size_t)w * rows) return;
    int x = (int)(i % w), y = (int)(i / w) + y0;
    float g = ga[i];
    float smooth = tex_ab[2 * i] * g + tex_ab[2 * i + 1];
    float fx = fminf(fmaxf(((float)x + 0.5f) * sx - 0.5f, 0.0f), (float)(dw - 1));
    float fy = fminf(fmaxf(((float)y + 0.5f) * sy - 0.5f, 0.0f), (float)(dh - 1));
    int ix = (int)fx, iy = (int)fy;
    int ix1 = min(ix + 1, dw - 1), iy1 = min(iy + 1, dh - 1);
    float ax = fx - (float)ix, ay = fy - (float)iy;
    const float4* r0 = dplanes + (size_t)iy * dw;
    const float4* r1 = dplanes + (size_t)iy1 * dw;
    float t0 = r0[ix].z + ax * (r0[ix1].z - r0[ix].z);
    float t1 = r1[ix].z + ax * (r1[ix1].z - r1[ix].z);
    out[i] = make_float4(g - smooth, smooth, t0 + ay * (t1 - t0), dplanes[0].w);
}
