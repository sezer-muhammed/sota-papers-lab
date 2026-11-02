// Splat types, constants and shared math. Owner: preprocess (E1); every other kernel
// owner reads it and requests changes. Spec: reports/image-to-3d/spec-raster.md
// Sections 1 and 6.1 (constants), spec-train.md Sections 2.1 and 12 (IterParams).
//
// The structs here are the device half of the contract; their Rust mirrors live in
// src/frame.rs (FrameParams, FrameCam, CamBase, Intr, IterParams), src/kernels/adam.rs
// (AdamIterPlan) and src/kernels/mod.rs (SplatRec). Sizes and offsets are checked on the
// GPU by the selftest layout probe, the #defines by src/kernels/layout.rs.
//
// Shared math (end of this file, E1): quat_to_rotmat (+ VJP), the SH basis for degree
// 0..3 in the Cartesian form of spec-raster 2.7 (+ VJP), and the tile-span functions of
// spec-raster 2.8 (span_ellipse, band_x_interval, span_pixel_box, span_tile_row) that
// preprocess_fwd, emit (binning.cu) and the per-warp subtile cull (raster_fwd.cu) call,
// so tile counts and emitted tiles match bit for bit.

// ---------------------------------------------------------------- compile-time variant (RasterVariant)
// One NVRTC module per distinct value; src/kernels/mod.rs passes them as -D options.

#ifndef TILE_H
#define TILE_H 16
#endif
#ifndef CHAIN_K
#define CHAIN_K 4
#endif
#ifndef CKPT_F16
#define CKPT_F16 0
#endif
#ifndef REC32
#define REC32 0
#endif
#ifndef EXTRA_CH
#define EXTRA_CH 0
#endif
#ifndef BLEND
#define BLEND 0
#endif
#ifndef DETERMINISTIC
#define DETERMINISTIC 0
#endif

#define BLEND_GSPLAT 0
#define BLEND_FASTERGS 1

// ---------------------------------------------------------------- geometry of tiles and passes

#define TILE_W 16
#define MAX_TILES 8192
#define BATCH 256
#define BUCKET 32
#define RASTER_THREADS 256
#define PRE_THREADS 256
#define PRE_BLOCKS_PER_SM 6
#define BWD_WARPS 4
#define SORT_THREADS 384
#define SORT_IPT 16
#define RADIX_BITS 8
#define RADIX_BINS 256
#define TILE_KEY_BITS 13
#define SH_REST 45
#define MAX_SH_DEGREE 3
#define CAM_GRAD_N 10
#define RING_SLOTS 8

// ---------------------------------------------------------------- numerics (spec-raster 6.1; gsplat defaults)

#define ALPHA_MIN (0.00392156862745098f)
#define ALPHA_MAX (0.99f)
#define T_MIN (0.0001f)
#define ONE_MINUS_ALPHA_MIN (0.000001f)
#define MIN_COMPENSATION (0.005f)
#define EPS2D_DEFAULT (0.3f)
#define NEAR_DEFAULT (0.01f)
#define FAR_DEFAULT (10000.0f)
#define CLIP_MARGIN (0.15f)
#define QUAT_MIN_NORM2 (0.00000001f)
#define KBIN_REL (0.0001f)
#define KBIN_ABS (0.0001f)
#define LOG2E (1.4426950408889634f)

#define SH_C0 (0.28209479177387814f)
#define SH_C1 (0.4886025119029199f)

// ---------------------------------------------------------------- status words (FrameBuffers.status, u32[STATUS_LEN])
// Every kernel after scan_tiles starts with `if (status[ST_OVERFLOW]) return;` so an
// overflowed iteration has no side effect at all (spec-raster 1.5).

#define STATUS_LEN 16
#define ST_OVERFLOW 0
#define ST_N_VIS 1
#define ST_N_INST 2
#define ST_N_CHAINS 3
#define ST_ITER 4
#define ST_OVERFLOW_ITER 5
#define ST_N_TILES_HIT 6

// Bits of ST_OVERFLOW: which capacity was exceeded.
#define OVF_VISIBLE 1
#define OVF_INSTANCES 2
#define OVF_CHAINS 4

// ---------------------------------------------------------------- FrameParams.flags

#define FLAG_AA 1
#define FLAG_FILTER3D 2
#define FLAG_DEPTH 4
#define FLAG_ABS_GRAD 8
#define FLAG_DENSIFY_STATS 16
#define FLAG_CAM_GRAD 32
#define FLAG_RANDOM_BG 64
#define FLAG_AA_GRAD 128

// raster_fwd mode argument (RenderMode).
#define MODE_TRAIN 0
#define MODE_INFER 1
#define MODE_COUNT 2

// ---------------------------------------------------------------- structs

// camera_table[n_images]: base poses from COLMAP (or Stage A), never modified on device.
struct CamBase {                     // 64 B
    float R[9];
    float t[3];
    uint32_t intr_id;
    uint32_t width, height;          // size the intrinsics refer to (the cached image)
    uint32_t pad;
};

// intrinsics[n_intr]: one per physical camera, at CamBase.width x height.
struct Intr {                        // 16 B
    float fx, fy, cx, cy;
};

// Written by frame_begin from the ring slot; read by every other kernel of the iteration.
struct FrameParams {                 // 64 B
    uint32_t iteration, cam_index, n_gaussians, sh_degree;
    float bg[3];
    float eps2d;
    uint32_t flags;
    float adam_lr_scale, bias_c1, bias_c2;
    uint32_t render_w, render_h;     // render size; intrinsics scale per axis from CamBase size
    float box_beta;                  // FastGS compact box: k = box_beta * 2 ln(255 o); 1 = off
    uint32_t pad;
};

// The camera every kernel uses this iteration: W = exp(xi^) W0, intrinsics scaled to the
// render size. Written by frame_begin.
struct FrameCam {                    // 128 B
    float R[9], t[3];                // world -> camera, row-major
    float fx, fy, cx, cy;
    float campos[3];                 // -R^T t
    float near_z, far_z;
    float lim_x0, lim_x1, lim_y0, lim_y1;  // EWA clip of x/z, y/z (0.15 W margin)
    uint32_t width, height, grid_w, grid_h, n_tiles, image_index;
    uint32_t intr_id;
};

// Optimizer plan for one iteration (spec-train 4.4; computed on the host in f64).
struct AdamIterPlan {                // 84 B
    float step_size[6];
    float bc2_sqrt_rcp[6];
    uint32_t due_mask, overwrite_mask;
    uint32_t elem_prefix[7];
};

// One pinned ring slot (spec-raster 1.3 + spec-train 12). frame_begin copies it into
// device memory; the raster reads `frame`, training kernels read the rest.
struct IterParams {                  // 256 B
    FrameParams frame;               //   0
    uint64_t target_offset;          //  64  byte offset of the target in the u8 image pool
    uint32_t target_w, target_h;     //  72  cached target size (resample source)
    uint32_t res_scale;              //  80  integer warm-up scale, 1 = full resolution
    float means_lr;                  //  84
    float noise_scale;               //  88  MCMC position noise factor (0 = off)
    uint32_t seed[2];                //  92  Philox key of the run
    uint32_t pad0;                   // 100
    AdamIterPlan adam;               // 104
    uint32_t pad1[17];               // 188
};

// One visible Gaussian as the binning and raster kernels read it (spec-raster 1.4).
struct SplatRec {                    // 48 B
    float mx, my;                    // mean2d, pixels
    float ca, cb, cc;                // conic pre-scaled for exp2
    float o;                         // effective opacity after compensations
    float r, g, b;                   // clamped colour
    float pthr;                      // log2(255 o) >= 0
    float z;                         // camera depth
    uint32_t flags;                  // bit c: colour channel c was clamped
};

// ================================================================ shared math (E1)
// Conventions of spec-raster 1.2 and 2: row-major matrices, quaternions (w, x, y, z),
// pixel centres at i + 0.5, conic (A, B, C) with q(d) = A dx^2 + 2 B dx dy + C dy^2.

#define LN2 (0.6931471805599453f)
// SplatRec stores the conic pre-scaled for exp2: ca = CONIC_SCALE_AC A, cb = CONIC_SCALE_B B,
// cc = CONIC_SCALE_AC C, so that the blend exponent is ca dx^2 + cb dx dy + cc dy^2.
#define CONIC_SCALE_AC (-0.7213475204444817f)
#define CONIC_SCALE_B (-1.4426950408889634f)

// Real SH constants, 3DGS order and signs (spec-raster 2.7).
#define SH_C2_0 (1.0925484305920792f)
#define SH_C2_1 (-1.0925484305920792f)
#define SH_C2_2 (0.31539156525252005f)
#define SH_C2_3 (-1.0925484305920792f)
#define SH_C2_4 (0.5462742152960396f)
#define SH_C3_0 (-0.5900435899266435f)
#define SH_C3_1 (2.890611442640554f)
#define SH_C3_2 (-0.4570457994644658f)
#define SH_C3_3 (0.3731763325901154f)
#define SH_C3_4 (-0.4570457994644658f)
#define SH_C3_5 (1.445305721320277f)
#define SH_C3_6 (-0.5900435899266435f)

// SH-rest coefficients (basis functions beyond DC) of a degree: 0, 3, 8, 15.
SPL_INLINE uint32_t sh_rest_count(uint32_t deg) { return (deg + 1u) * (deg + 1u) - 1u; }

// Rotation matrix (row-major) of the unit quaternion (w, x, y, z).
SPL_INLINE void quat_to_rotmat(float w, float x, float y, float z, float R[9]) {
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;
    R[0] = 1.f - 2.f * (yy + zz);
    R[1] = 2.f * (xy - wz);
    R[2] = 2.f * (xz + wy);
    R[3] = 2.f * (xy + wz);
    R[4] = 1.f - 2.f * (xx + zz);
    R[5] = 2.f * (yz - wx);
    R[6] = 2.f * (xz - wy);
    R[7] = 2.f * (yz + wx);
    R[8] = 1.f - 2.f * (xx + yy);
}

// VJP of quat_to_rotmat at the unit quaternion (w, x, y, z): G = dL/dR (row-major) to
// dL/d(w, x, y, z) of the unit quaternion, returned as float4 (w, x, y, z) in (x, y, z, w)
// fields order: .x = d/dw, .y = d/dx, .z = d/dy, .w = d/dz (spec-raster 5.3 step 7).
// The caller applies the normalization Jacobian.
SPL_INLINE float4 quat_to_rotmat_vjp(float w, float x, float y, float z, const float G[9]) {
    float4 g;
    g.x = 2.f * (x * (G[7] - G[5]) + y * (G[2] - G[6]) + z * (G[3] - G[1]));
    g.y = 2.f * (-2.f * x * (G[4] + G[8]) + y * (G[1] + G[3]) + z * (G[2] + G[6]) + w * (G[7] - G[5]));
    g.z = 2.f * (x * (G[1] + G[3]) - 2.f * y * (G[0] + G[8]) + z * (G[5] + G[7]) + w * (G[2] - G[6]));
    g.w = 2.f * (x * (G[2] + G[6]) + y * (G[5] + G[7]) - 2.f * z * (G[0] + G[4]) + w * (G[3] - G[1]));
    return g;
}

// The 15 SH basis values beyond DC at direction (x, y, z), Cartesian polynomials of
// spec-raster 2.7 (exact on the unit sphere). Entries k >= sh_rest_count(deg) are 0.
// `deg` is uniform per launch, so the branches do not diverge.
SPL_INLINE void sh_basis(uint32_t deg, float x, float y, float z, float Y[15]) {
#pragma unroll
    for (int k = 0; k < 15; ++k) Y[k] = 0.f;
    if (deg >= 1) {
        Y[0] = -SH_C1 * y;
        Y[1] = SH_C1 * z;
        Y[2] = -SH_C1 * x;
    }
    if (deg >= 2) {
        const float xx = x * x, yy = y * y, zz = z * z;
        Y[3] = SH_C2_0 * x * y;
        Y[4] = SH_C2_1 * y * z;
        Y[5] = SH_C2_2 * (2.f * zz - xx - yy);
        Y[6] = SH_C2_3 * x * z;
        Y[7] = SH_C2_4 * (xx - yy);
        if (deg >= 3) {
            Y[8] = SH_C3_0 * y * (3.f * xx - yy);
            Y[9] = SH_C3_1 * x * y * z;
            Y[10] = SH_C3_2 * y * (4.f * zz - xx - yy);
            Y[11] = SH_C3_3 * z * (2.f * zz - 3.f * xx - 3.f * yy);
            Y[12] = SH_C3_4 * x * (4.f * zz - xx - yy);
            Y[13] = SH_C3_5 * z * (xx - yy);
            Y[14] = SH_C3_6 * x * (xx - 3.f * yy);
        }
    }
}

// sum_k w[k] dY_k/d(x, y, z) for the polynomials of sh_basis (not projected onto the
// sphere's tangent plane; the caller does that, which makes the result independent of the
// polynomial extension off the sphere).
SPL_INLINE float3 sh_basis_vjp(uint32_t deg, float x, float y, float z, const float w[15]) {
    float3 g = make_float3(0.f, 0.f, 0.f);
    if (deg >= 1) {
        g.x += -SH_C1 * w[2];
        g.y += -SH_C1 * w[0];
        g.z += SH_C1 * w[1];
    }
    if (deg >= 2) {
        const float xx = x * x, yy = y * y, zz = z * z;
        g.x += SH_C2_0 * y * w[3] - 2.f * SH_C2_2 * x * w[5] + SH_C2_3 * z * w[6] + 2.f * SH_C2_4 * x * w[7];
        g.y += SH_C2_0 * x * w[3] + SH_C2_1 * z * w[4] - 2.f * SH_C2_2 * y * w[5] - 2.f * SH_C2_4 * y * w[7];
        g.z += SH_C2_1 * y * w[4] + 4.f * SH_C2_2 * z * w[5] + SH_C2_3 * x * w[6];
        if (deg >= 3) {
            g.x += 6.f * SH_C3_0 * x * y * w[8] + SH_C3_1 * y * z * w[9] - 2.f * SH_C3_2 * x * y * w[10] -
                   6.f * SH_C3_3 * x * z * w[11] + SH_C3_4 * (4.f * zz - 3.f * xx - yy) * w[12] +
                   2.f * SH_C3_5 * x * z * w[13] + 3.f * SH_C3_6 * (xx - yy) * w[14];
            g.y += 3.f * SH_C3_0 * (xx - yy) * w[8] + SH_C3_1 * x * z * w[9] +
                   SH_C3_2 * (4.f * zz - xx - 3.f * yy) * w[10] - 6.f * SH_C3_3 * y * z * w[11] -
                   2.f * SH_C3_4 * x * y * w[12] - 2.f * SH_C3_5 * y * z * w[13] - 6.f * SH_C3_6 * x * y * w[14];
            g.z += SH_C3_1 * x * y * w[9] + 8.f * SH_C3_2 * y * z * w[10] +
                   3.f * SH_C3_3 * (2.f * zz - xx - yy) * w[11] + 8.f * SH_C3_4 * x * z * w[12] +
                   SH_C3_5 * (xx - yy) * w[13];
        }
    }
    return g;
}

// Basis function k (0..14, beyond DC) of sh_basis and its gradient d Y_k / d(x, y, z), for
// loops unrolled over k (k folds to a constant, so each call is a few FMAs and the caller
// needs no 15-entry arrays). xx = x^2 etc.
SPL_INLINE float sh_y(int k, float x, float y, float z, float xx, float yy, float zz) {
    switch (k) {
        case 0: return -SH_C1 * y;
        case 1: return SH_C1 * z;
        case 2: return -SH_C1 * x;
        case 3: return SH_C2_0 * x * y;
        case 4: return SH_C2_1 * y * z;
        case 5: return SH_C2_2 * (2.f * zz - xx - yy);
        case 6: return SH_C2_3 * x * z;
        case 7: return SH_C2_4 * (xx - yy);
        case 8: return SH_C3_0 * y * (3.f * xx - yy);
        case 9: return SH_C3_1 * x * y * z;
        case 10: return SH_C3_2 * y * (4.f * zz - xx - yy);
        case 11: return SH_C3_3 * z * (2.f * zz - 3.f * xx - 3.f * yy);
        case 12: return SH_C3_4 * x * (4.f * zz - xx - yy);
        case 13: return SH_C3_5 * z * (xx - yy);
        default: return SH_C3_6 * x * (xx - 3.f * yy);
    }
}

SPL_INLINE float3 sh_dy(int k, float x, float y, float z, float xx, float yy, float zz) {
    switch (k) {
        case 0: return make_float3(0.f, -SH_C1, 0.f);
        case 1: return make_float3(0.f, 0.f, SH_C1);
        case 2: return make_float3(-SH_C1, 0.f, 0.f);
        case 3: return make_float3(SH_C2_0 * y, SH_C2_0 * x, 0.f);
        case 4: return make_float3(0.f, SH_C2_1 * z, SH_C2_1 * y);
        case 5: return make_float3(-2.f * SH_C2_2 * x, -2.f * SH_C2_2 * y, 4.f * SH_C2_2 * z);
        case 6: return make_float3(SH_C2_3 * z, 0.f, SH_C2_3 * x);
        case 7: return make_float3(2.f * SH_C2_4 * x, -2.f * SH_C2_4 * y, 0.f);
        case 8: return make_float3(6.f * SH_C3_0 * x * y, 3.f * SH_C3_0 * (xx - yy), 0.f);
        case 9: return make_float3(SH_C3_1 * y * z, SH_C3_1 * x * z, SH_C3_1 * x * y);
        case 10:
            return make_float3(-2.f * SH_C3_2 * x * y, SH_C3_2 * (4.f * zz - xx - 3.f * yy), 8.f * SH_C3_2 * y * z);
        case 11:
            return make_float3(-6.f * SH_C3_3 * x * z, -6.f * SH_C3_3 * y * z, 3.f * SH_C3_3 * (2.f * zz - xx - yy));
        case 12:
            return make_float3(SH_C3_4 * (4.f * zz - 3.f * xx - yy), -2.f * SH_C3_4 * x * y, 8.f * SH_C3_4 * x * z);
        case 13: return make_float3(2.f * SH_C3_5 * x * z, -2.f * SH_C3_5 * y * z, SH_C3_5 * (xx - yy));
        default: return make_float3(3.f * SH_C3_6 * (xx - yy), -6.f * SH_C3_6 * x * y, 0.f);
    }
}

// ---------------------------------------------------------------- tile span (spec-raster 2.8)
// Every operation below uses explicit round-to-nearest intrinsics (no FMA contraction, no
// fast-math approximations), so preprocess_fwd (count), emit (tiles) and raster_fwd
// (subtile cull) compute bit-identical spans from the same SplatRec in any unit.

// The bounding ellipse {q(d) <= k} of a record: (A, B, C) recovered from the stored
// pre-scaled conic, k = box_beta 2 ln(255 o_eff) (from pthr) plus the rounding margin.
struct SpanEllipse {
    float mx, my;        // centre, pixels
    float A, B, C;       // quadratic form, scalar-B convention
    float det;           // A C - B^2
    float k;             // Mahalanobis^2 cut-off incl. margin (k_bin)
    float x_ext, y_ext;  // half extents of the ellipse's bounding box
};

SPL_INLINE SpanEllipse span_ellipse(const SplatRec& r, float box_beta) {
    SpanEllipse e;
    e.mx = r.mx;
    e.my = r.my;
    e.A = __fdiv_rn(r.ca, CONIC_SCALE_AC);
    e.B = __fdiv_rn(r.cb, CONIC_SCALE_B);
    e.C = __fdiv_rn(r.cc, CONIC_SCALE_AC);
    e.det = __fsub_rn(__fmul_rn(e.A, e.C), __fmul_rn(e.B, e.B));
    const float k = __fmul_rn(__fmul_rn(box_beta, 2.f * LN2), fmaxf(r.pthr, 0.f));
    e.k = __fadd_rn(__fmul_rn(k, 1.f + KBIN_REL), KBIN_ABS);
    const float kd = __fdiv_rn(e.k, e.det);
    e.x_ext = __fsqrt_rn(__fmul_rn(kd, e.C));
    e.y_ext = __fsqrt_rn(__fmul_rn(kd, e.A));
    return e;
}

// x-range [xl, xr] of the ellipse restricted to the horizontal band y in [y0, y1]
// (AccuTile adapted to pixel centres): the extreme x over the band is reached at the band
// point closest to the y of the ellipse's leftmost (rightmost) point.
SPL_INLINE void band_x_interval(const SpanEllipse& e, float y0, float y1, float& xl, float& xr) {
    const float sh = __fdiv_rn(__fmul_rn(e.B, e.x_ext), e.C);
    const float yl = fminf(fmaxf(__fadd_rn(e.my, sh), y0), y1);
    const float yr = fminf(fmaxf(__fsub_rn(e.my, sh), y0), y1);
    const float ak = __fmul_rn(e.A, e.k);
    const float dl = __fsub_rn(yl, e.my);
    const float dr = __fsub_rn(yr, e.my);
    const float sl = __fsqrt_rn(fmaxf(__fsub_rn(ak, __fmul_rn(e.det, __fmul_rn(dl, dl))), 0.f));
    const float sr = __fsqrt_rn(fmaxf(__fsub_rn(ak, __fmul_rn(e.det, __fmul_rn(dr, dr))), 0.f));
    xl = __fadd_rn(e.mx, __fdiv_rn(__fsub_rn(-__fmul_rn(e.B, dl), sl), e.A));
    xr = __fadd_rn(e.mx, __fdiv_rn(__fadd_rn(-__fmul_rn(e.B, dr), sr), e.A));
}

// Pixel-centre index range [lo, hi] of the interval [c - ext, c + ext] clipped to
// [0, size - 1]; empty when lo > hi. Clamps in float first so huge values cannot overflow.
SPL_INLINE void span_axis(float c, float ext, uint32_t size, int& lo, int& hi) {
    const float fs = (float)size;
    lo = (int)fminf(fmaxf(ceilf(__fsub_rn(__fsub_rn(c, ext), 0.5f)), 0.f), fs);
    hi = (int)fminf(fmaxf(floorf(__fsub_rn(__fadd_rn(c, ext), 0.5f)), -1.f), fs - 1.f);
}

// Pixel-centre bounding box of the ellipse inside a W x H image. False when empty.
SPL_INLINE bool span_pixel_box(const SpanEllipse& e, uint32_t W, uint32_t H, int& px0, int& px1, int& py0,
                               int& py1) {
    span_axis(e.mx, e.x_ext, W, px0, px1);
    span_axis(e.my, e.y_ext, H, py0, py1);
    return px0 <= px1 && py0 <= py1;
}

// Tiles of tile row ty that contain a pixel centre of the ellipse's band: columns
// [tx0, tx1]; returns their number (0 when the row is empty). The box comes from
// span_pixel_box; rows ty outside [py0 / TILE_H, py1 / TILE_H] return 0.
SPL_INLINE uint32_t span_tile_row(const SpanEllipse& e, int ty, int px0, int px1, int py0, int py1, int& tx0,
                                  int& tx1) {
    const int r0 = max(ty * TILE_H, py0);
    const int r1 = min(ty * TILE_H + TILE_H - 1, py1);
    if (r0 > r1) return 0u;
    float xl, xr;
    band_x_interval(e, (float)r0 + 0.5f, (float)r1 + 0.5f, xl, xr);
    const int c0 = (int)fminf(fmaxf(ceilf(__fsub_rn(xl, 0.5f)), (float)px0), (float)px1 + 1.f);
    const int c1 = (int)fminf(fmaxf(floorf(__fsub_rn(xr, 0.5f)), (float)px0 - 1.f), (float)px1);
    if (c0 > c1) return 0u;
    tx0 = c0 / TILE_W;
    tx1 = c1 / TILE_W;
    return (uint32_t)(tx1 - tx0 + 1);
}
