// Splat types, constants and shared math. Owner: preprocess (E1); every other kernel
// owner reads it and requests changes. Spec: reports/image-to-3d/spec-raster.md
// Sections 1 and 6.1 (constants), spec-train.md Sections 2.1 and 12 (IterParams).
//
// The structs here are the device half of the contract; their Rust mirrors live in
// src/frame.rs (FrameParams, FrameCam, CamBase, Intr, IterParams), src/kernels/adam.rs
// (AdamIterPlan) and src/kernels/mod.rs (SplatRec). Sizes and offsets are checked on the
// GPU by the selftest layout probe, the #defines by src/kernels/layout.rs.
//
// Still to write here (E1): quat_to_rotmat (+ VJP), sh_eval / sh_eval_vjp for degree
// 0..3 in the Cartesian form of spec-raster 2.7, ewa_project (+ VJP), the opacity
// compensation, and band_x_interval (spec-raster 2.8), which emit (binning.cu) and the
// per-warp subtile cull (raster_fwd.cu) must call so counts match by construction.

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
