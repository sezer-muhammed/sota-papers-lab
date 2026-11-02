// Helpers shared by raster_fwd.cu and raster_bwd.cu. Owner: raster (E3).
// Spec: spec-raster Sections 4-5.2. Both units concatenate common.cuh + splat_common.cuh +
// this file + their own source (src/kernels/raster.rs), so every decision the backward must
// repeat (skip, clamp, subtile cull) is the same inline code in both kernels and agrees bit
// for bit with the forward.

#define TILE_PX (TILE_W * TILE_H)
#define SUB_W 8
#define SUB_H 4
#define SUB_COLS (TILE_W / SUB_W)
#define SUB_ROWS (TILE_H / SUB_H)
#define FWD_WARPS (TILE_PX / 32)
#define CHAIN_LEN (BUCKET * CHAIN_K)
#define BWD_THREADS (BWD_WARPS * 32)
#define STAGE 64

// Checkpoint layout (one entry per chain and tile pixel px = ly * TILE_W + lx):
//   main plane  [c_cap][TILE_PX]: float4 (C.r, C.g, C.b, T), or with CKPT_F16 uint2 = half4;
//   extra plane [c_cap][EXTRA_CH][TILE_PX] f32 after the main plane (depth and later channels).
// Words per pixel: CKPT_MAIN_WORDS + EXTRA_CH (src/kernels/raster.rs ckpt_words).
#if CKPT_F16
#define CKPT_MAIN_WORDS 2
#else
#define CKPT_MAIN_WORDS 4
#endif

// Extra channels: channel 0 is the camera depth z (expected depth); channels 1..3 are reserved
// for normals (Stage D) and are 0 until a per-v source exists.
#define N_CH (3 + EXTRA_CH)

#if TILE_H % SUB_H != 0 || TILE_PX % 32 != 0
#error "TILE_H must be a multiple of the 4-row subtile"
#endif

SPL_INLINE float rs_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Exponent e = -log2(e) * sigma at (dx, dy) = mean - pixel centre, from the pre-scaled conic
// (ca, cb, cc) = (-0.5 log2e A, -log2e B, -0.5 log2e C). One fixed evaluation order (no
// contraction freedom left to the compiler) so the forward and the backward agree exactly.
SPL_INLINE float rs_power(float ca, float cb, float cc, float dx, float dy) {
    return fmaf(ca * dx, dx, fmaf(cb * dx, dy, (cc * dy) * dy));
}

// gsplat 1.5.3 blend semantics (RasterizeToPixels3DGSFwd.cu): skip when sigma < 0 or
// alpha < 1/255; alpha = min(ALPHA_MAX, o G). BLEND_FASTERGS: no clamp.
// Returns true when the Gaussian is blended at this pixel (before the termination test).
SPL_INLINE bool rs_alpha(float o, float e, float& G, float& a_raw, float& alpha) {
    G = rs_ex2(e);
    a_raw = o * G;
#if BLEND == BLEND_FASTERGS
    alpha = a_raw;
#else
    alpha = fminf(ALPHA_MAX, a_raw);
#endif
    return e <= 0.0f && alpha >= ALPHA_MIN;
}

// Subtile s = row * SUB_COLS + col of local pixel px (8 x 4 pixel blocks of the tile).
SPL_INLINE uint32_t rs_subtile(uint32_t px) {
    return (px / (TILE_W * SUB_H)) * SUB_COLS + (px % TILE_W) / SUB_W;
}

// Which 8 x 4 subtiles of the tile with origin (tx0, ty0) contain a pixel centre where the
// record can reach alpha >= 1/255 (bit = subtile). The forward skips a (Gaussian, warp) pair
// when its bit is clear and the backward applies the same mask, so the two kernels blend
// exactly the same pairs whatever the rounding; the margins only make the cull conservative
// against the blend's own evaluation of the exponent.
//
// Geometry (exact for convex sets): with Q = -e = a dx^2 + b dx dy + c dy^2 (a, c > 0) the
// ellipse {Q <= P} restricted to a band dy in [ya, yb] spans x from mx + xl(clamp(dyL)) to
// mx + xr(clamp(-dyL)), where dyL is the dy of its leftmost point (the left boundary is convex
// in dy, the right one concave). That is spec-raster 2.8's band_x_interval in exponent units.
SPL_INLINE uint32_t rs_subtile_mask(float mx, float my, float ca, float cb, float cc, float pthr, float tx0,
                                    float ty0) {
    const float a = -ca, b = -cb, c = -cc;
    // Q <= pthr  <=>  o 2^e >= 1/255. Relative + absolute margin as the binning (KBIN_*).
    float P = fmaf(pthr, 1.0f + KBIN_REL, KBIN_ABS);
    if (!(P > 0.0f)) return 0u;
    // 4ac - b^2 with Kahan's FMA determinant (accurate for thin ellipses).
    const float bb = b * b;
    const float D = fmaf(4.0f * a, c, -bb) - fmaf(b, b, -bb);
    if (!(D > 0.0f) || !(a > 0.0f)) return (1u << (SUB_ROWS * SUB_COLS)) - 1u;  // degenerate: keep
    // Rounding allowance: the blend evaluates Q with an error of a few ulp of
    // a dx^2 + |b dx dy| + c dy^2, which at the boundary is P * kappa with
    // kappa = 4 (2ac + |b| sqrt(ac)) / D (>= 2; large only for thin ellipses).
    const float kappa = 4.0f * fmaf(fabsf(b), sqrtf(a * c), 2.0f * a * c) / D;
    P = P * fmaf(4.0e-6f, kappa, 1.0f);
    const float inv_d = 1.0f / D;
    const float x_ext = sqrtf(4.0f * c * P * inv_d);
    const float y_ext = sqrtf(4.0f * a * P * inv_d);
    const float dy_l = b * x_ext * (0.5f / c);  // dy of the leftmost point; the rightmost is at -dy_l
    const float inv2a = 0.5f / a;
    const float disc0 = 4.0f * a * P;
    uint32_t mask = 0u;
#pragma unroll
    for (int r = 0; r < SUB_ROWS; ++r) {
        const float ya = (ty0 + (float)(r * SUB_H) + 0.5f) - my;
        const float yb = ya + (float)(SUB_H - 1);
        if (yb < -y_ext || ya > y_ext) continue;
        const float dl = fminf(fmaxf(dy_l, ya), yb);
        const float dr = fminf(fmaxf(-dy_l, ya), yb);
        const float xl = mx + (-b * dl - sqrtf(fmaxf(0.0f, disc0 - D * dl * dl))) * inv2a;
        const float xr = mx + (-b * dr + sqrtf(fmaxf(0.0f, disc0 - D * dr * dr))) * inv2a;
#pragma unroll
        for (int col = 0; col < SUB_COLS; ++col) {
            const float xa = tx0 + (float)(col * SUB_W) + 0.5f;
            const float xb = xa + (float)(SUB_W - 1);
            if (xl <= xb && xr >= xa) mask |= 1u << (r * SUB_COLS + col);
        }
    }
    return mask;
}

// Value of extra channel k of a record (0: depth).
SPL_INLINE float rs_extra(const SplatRec& r, int k) { return k == 0 ? r.z : 0.0f; }
