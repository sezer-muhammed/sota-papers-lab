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

// ---------------------------------------------------------------- subtile cull
// Which 8 x 4 subtiles contain a pixel centre where a record can reach alpha >= 1/255. The
// forward tests each (record, warp subtile) pair with rs_subtile_hit; the backward builds the
// whole mask with rs_subtile_mask and applies it per (Gaussian, pixel), so both kernels blend
// exactly the same pairs. That needs bit-identical decisions from two call sites, so every
// operation below is pinned (__f*_rn: no FMA contraction or reassociation; approximate sqrt and
// reciprocal through PTX, which are deterministic). The margins only make the cull
// conservative against the blend's own evaluation of the exponent.
//
// Geometry (exact for convex sets): with Q = -e = a dx^2 + b dx dy + c dy^2 (a, c > 0) the
// ellipse {Q <= P} restricted to a band dy in [ya, yb] spans x from mx + xl(clamp(dyL)) to
// mx + xr(clamp(-dyL)), where dyL is the dy of its leftmost point (the left boundary is convex
// in dy, the right one concave). That is spec-raster 2.8's band_x_interval in exponent units.

SPL_INLINE float rs_sqrt(float x) {
    float y;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

SPL_INLINE float rs_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

struct RsEllipse {
    float mx, my, b, D, inv2a, disc0, y_ext, dy_l;
    uint32_t kind;  // 0: test bands, 1: never visible, 2: degenerate (keep everywhere)
};

SPL_INLINE RsEllipse rs_ellipse(float mx, float my, float ca, float cb, float cc, float pthr) {
    RsEllipse el;
    el.mx = mx;
    el.my = my;
    const float a = -ca, b = -cb, c = -cc;
    // Q <= pthr  <=>  o 2^e >= 1/255. Relative + absolute margin as the binning (KBIN_*).
    float P = __fmaf_rn(pthr, 1.0f + KBIN_REL, KBIN_ABS);
    // 4ac - b^2 with Kahan's FMA determinant (accurate for thin ellipses).
    const float bb = __fmul_rn(b, b);
    const float D = __fsub_rn(__fmaf_rn(__fmul_rn(4.0f, a), c, -bb), __fmaf_rn(b, b, -bb));
    el.kind = !(P > 0.0f) ? 1u : (!(D > 0.0f) || !(a > 0.0f)) ? 2u : 0u;
    // Rounding allowance: the blend evaluates Q with an error of a few ulp of
    // a dx^2 + |b dx dy| + c dy^2, which at the boundary is P * kappa with
    // kappa = 4 (2ac + |b| sqrt(ac)) / D (>= 2; large only for thin ellipses).
    const float ac = __fmul_rn(a, c);
    const float inv_d = rs_rcp(D);
    const float kappa = __fmul_rn(__fmul_rn(4.0f, __fmaf_rn(fabsf(b), rs_sqrt(ac), __fmul_rn(2.0f, ac))), inv_d);
    P = __fmul_rn(P, __fmaf_rn(4.0e-6f, kappa, 1.0f));
    const float a4p = __fmul_rn(__fmul_rn(4.0f, a), P);
    const float x_ext = rs_sqrt(__fmul_rn(__fmul_rn(__fmul_rn(4.0f, c), P), inv_d));
    el.y_ext = rs_sqrt(__fmul_rn(a4p, inv_d));
    el.dy_l = __fmul_rn(__fmul_rn(b, x_ext), __fmul_rn(0.5f, rs_rcp(c)));  // the rightmost point is at -dy_l
    el.inv2a = __fmul_rn(0.5f, rs_rcp(a));
    el.disc0 = a4p;
    el.b = b;
    el.D = D;
    return el;
}

// x-range [xl, xr] of the ellipse over the pixel-centre rows y0 + 0.5 .. y0 + SUB_H - 0.5
// (y0: first pixel row of the band); false when the band misses the ellipse.
SPL_INLINE bool rs_band(const RsEllipse& el, uint32_t y0, float& xl, float& xr) {
    const float ya = __fsub_rn(__fadd_rn((float)y0, 0.5f), el.my);
    const float yb = __fadd_rn(ya, (float)(SUB_H - 1));
    if (yb < -el.y_ext || ya > el.y_ext) return false;
    const float dl = fminf(fmaxf(el.dy_l, ya), yb);
    const float dr = fminf(fmaxf(-el.dy_l, ya), yb);
    const float sl = rs_sqrt(fmaxf(0.0f, __fsub_rn(el.disc0, __fmul_rn(__fmul_rn(el.D, dl), dl))));
    const float sr = rs_sqrt(fmaxf(0.0f, __fsub_rn(el.disc0, __fmul_rn(__fmul_rn(el.D, dr), dr))));
    xl = __fadd_rn(el.mx, __fmul_rn(__fsub_rn(__fmul_rn(-el.b, dl), sl), el.inv2a));
    xr = __fadd_rn(el.mx, __fmul_rn(__fadd_rn(__fmul_rn(-el.b, dr), sr), el.inv2a));
    return true;
}

// The band's x-range overlaps the pixel-centre columns x0 + 0.5 .. x0 + SUB_W - 0.5.
SPL_INLINE bool rs_cols(float xl, float xr, uint32_t x0) {
    return xl <= __fadd_rn((float)x0, (float)SUB_W - 0.5f) && xr >= __fadd_rn((float)x0, 0.5f);
}

// One subtile with pixel origin (x0, y0) (forward, per warp).
SPL_INLINE bool rs_subtile_hit(const RsEllipse& el, uint32_t x0, uint32_t y0) {
    if (el.kind != 0u) return el.kind == 2u;
    float xl, xr;
    return rs_band(el, y0, xl, xr) && rs_cols(xl, xr, x0);
}

// All subtiles of the tile with pixel origin (tx0, ty0), bit = row * SUB_COLS + col (backward).
SPL_INLINE uint32_t rs_subtile_mask(const RsEllipse& el, uint32_t tx0, uint32_t ty0) {
    if (el.kind != 0u) return el.kind == 2u ? (1u << (SUB_ROWS * SUB_COLS)) - 1u : 0u;
    uint32_t mask = 0u;
#pragma unroll
    for (int r = 0; r < SUB_ROWS; ++r) {
        float xl, xr;
        if (!rs_band(el, ty0 + r * SUB_H, xl, xr)) continue;
#pragma unroll
        for (int col = 0; col < SUB_COLS; ++col)
            if (rs_cols(xl, xr, tx0 + col * SUB_W)) mask |= 1u << (r * SUB_COLS + col);
    }
    return mask;
}

