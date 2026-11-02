// Stage B initialization (Report section 3) and the depth-prior loss. Owner: init-b.
// Compiled as common.cuh + init.cu (src/init/mod.rs, SOURCE).
//
// Frames: F depth maps of one size H x W, planar per frame ([F][H][W]), with a
// confidence map of the same shape and an RGB u8 image ([F][H][W][3]) at the
// depth resolution. Per frame 16 floats (INIT_FRAME_FLOATS): world-to-camera
// rotation row-major (9), translation (3), fx, fy, cx, cy at the depth
// resolution. Depth is camera-space z; pixel (u, v) has its centre at
// (u + 0.5, v + 0.5) (COLMAP convention). Depth <= 0 or non-finite = no depth.
//
// Voxel hash: open addressing, linear probing, table size a power of two,
// key = three 21-bit voxel coordinates biased by 2^20, EMPTY = ~0. Each slot
// accumulates 8 floats: sum x, y, z, r, g, b, footprint (z / f), count.

#define INIT_FRAME_FLOATS 16
#define INIT_HIST_BINS 4096
#define INIT_HIST_LO (-16.0f)
#define INIT_HIST_PER_OCTAVE 128.0f
#define INIT_SLOT_FLOATS 8
#define INIT_MAX_PROBE 256
#define INIT_EMPTY 0xFFFFFFFFFFFFFFFFull
#define INIT_BIAS 1048576
#define INIT_SH_C0 0.28209479177387814f

// ctr[]: 0 = occupied slots, 1 = dropped (probe limit or out of range), 2 = points inserted,
// 3 = Gaussians written.

__device__ __forceinline__ bool init_valid_depth(float z) {
    return z > 0.0f && z < 3.0e38f;
}

__device__ __forceinline__ int init_hist_bin(float c) {
    float b = (__log2f(fmaxf(c, 1e-30f)) - INIT_HIST_LO) * INIT_HIST_PER_OCTAVE;
    return (int)fminf(fmaxf(b, 0.0f), (float)(INIT_HIST_BINS - 1));
}

// Histogram of log2(confidence) over valid-depth pixels, and per bin the sum of
// pixel footprints z / sqrt(fx fy) (the mean over kept bins seeds the voxel size).
extern "C" __global__ void init_conf_hist(const float* __restrict__ depth,
                                          const float* __restrict__ conf,
                                          const float* __restrict__ frames, uint32_t n_frames,
                                          uint32_t hw, uint32_t* __restrict__ hist,
                                          float* __restrict__ foot_hist) {
    __shared__ uint32_t sh_hist[INIT_HIST_BINS];
    __shared__ float sh_foot[INIT_HIST_BINS];
    for (int k = threadIdx.x; k < INIT_HIST_BINS; k += blockDim.x) {
        sh_hist[k] = 0;
        sh_foot[k] = 0.0f;
    }
    __syncthreads();
    uint64_t total = (uint64_t)n_frames * hw;
    for (uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; g < total;
         g += (uint64_t)gridDim.x * blockDim.x) {
        float z = depth[g];
        if (!init_valid_depth(z)) continue;
        const float* fr = frames + (g / hw) * INIT_FRAME_FLOATS;
        int b = init_hist_bin(conf[g]);
        atomicAdd(&sh_hist[b], 1u);
        atomicAdd(&sh_foot[b], z * rsqrtf(fr[12] * fr[13]));
    }
    __syncthreads();
    for (int k = threadIdx.x; k < INIT_HIST_BINS; k += blockDim.x)
        if (sh_hist[k]) {
            atomicAdd(&hist[k], sh_hist[k]);
            atomicAdd(&foot_hist[k], sh_foot[k]);
        }
}

extern "C" __global__ void init_table_clear(uint64_t* __restrict__ keys,
                                            float* __restrict__ vals, uint32_t table,
                                            uint32_t* __restrict__ ctr, uint32_t clear_vals) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < table;
         i += gridDim.x * blockDim.x) {
        keys[i] = INIT_EMPTY;
        if (clear_vals) {
            for (int k = 0; k < INIT_SLOT_FLOATS; ++k) vals[(uint64_t)i * INIT_SLOT_FLOATS + k] = 0.0f;
        }
    }
    if (blockIdx.x == 0 && threadIdx.x < 4) ctr[threadIdx.x] = 0;
}

__device__ __forceinline__ uint64_t init_hash(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ull;
    k ^= k >> 33;
    return k;
}

__device__ __forceinline__ bool init_key(int ix, int iy, int iz, uint64_t* key) {
    if (ix < -INIT_BIAS || ix >= INIT_BIAS || iy < -INIT_BIAS || iy >= INIT_BIAS ||
        iz < -INIT_BIAS || iz >= INIT_BIAS)
        return false;
    *key = ((uint64_t)(ix + INIT_BIAS) << 42) | ((uint64_t)(iy + INIT_BIAS) << 21) |
           (uint64_t)(iz + INIT_BIAS);
    return true;
}

// Slot of `key` (inserting it when `insert`), or -1 (absent / table full).
__device__ __forceinline__ int64_t init_slot(uint64_t* keys, uint32_t table, uint64_t key,
                                             bool insert, uint32_t* ctr) {
    uint32_t mask = table - 1;
    uint32_t s = (uint32_t)init_hash(key) & mask;
    for (int probe = 0; probe < INIT_MAX_PROBE; ++probe) {
        uint64_t cur = keys[s];
        if (cur == key) return s;
        if (cur == INIT_EMPTY) {
            if (!insert) return -1;
            uint64_t prev = atomicCAS((unsigned long long*)&keys[s], (unsigned long long)INIT_EMPTY,
                                      (unsigned long long)key);
            if (prev == INIT_EMPTY) {
                atomicAdd(&ctr[0], 1u);
                return s;
            }
            if (prev == key) return s;
        }
        s = (s + 1) & mask;
    }
    return -1;
}

// Unprojects every pixel with valid depth and confidence >= conf_thr, keys it
// into the voxel hash (origin + voxel size) and, when `accumulate`, adds its
// position, colour and footprint to the slot.
extern "C" __global__ void init_voxel_insert(
    const float* __restrict__ depth, const float* __restrict__ conf,
    const uint8_t* __restrict__ rgb, const float* __restrict__ frames, uint32_t n_frames,
    uint32_t width, uint32_t height, float conf_thr, float ox, float oy, float oz, float voxel,
    uint64_t* __restrict__ keys, float* __restrict__ vals, uint32_t table,
    uint32_t* __restrict__ ctr, uint32_t accumulate) {
    uint64_t hw = (uint64_t)width * height;
    uint64_t total = (uint64_t)n_frames * hw;
    float inv = 1.0f / voxel;
    for (uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; g < total;
         g += (uint64_t)gridDim.x * blockDim.x) {
        float z = depth[g];
        if (!init_valid_depth(z) || !(conf[g] >= conf_thr)) continue;
        uint64_t f = g / hw;
        uint32_t px = (uint32_t)(g - f * hw);
        float u = (float)(px % width) + 0.5f, v = (float)(px / width) + 0.5f;
        const float* fr = frames + f * INIT_FRAME_FLOATS;
        float xc = (u - fr[14]) / fr[12] * z - fr[9];
        float yc = (v - fr[15]) / fr[13] * z - fr[10];
        float zc = z - fr[11];
        // X = R^T (x_c - t)
        float x = fr[0] * xc + fr[3] * yc + fr[6] * zc;
        float y = fr[1] * xc + fr[4] * yc + fr[7] * zc;
        float w = fr[2] * xc + fr[5] * yc + fr[8] * zc;
        uint64_t key;
        if (!init_key(__float2int_rd((x - ox) * inv), __float2int_rd((y - oy) * inv),
                      __float2int_rd((w - oz) * inv), &key)) {
            atomicAdd(&ctr[1], 1u);
            continue;
        }
        int64_t s = init_slot(keys, table, key, true, ctr);
        if (s < 0) {
            atomicAdd(&ctr[1], 1u);
            continue;
        }
        atomicAdd(&ctr[2], 1u);
        if (!accumulate) continue;
        float* slot = vals + (uint64_t)s * INIT_SLOT_FLOATS;
        const uint8_t* c = rgb + g * 3;
        atomicAdd(&slot[0], x);
        atomicAdd(&slot[1], y);
        atomicAdd(&slot[2], w);
        atomicAdd(&slot[3], (float)c[0] * (1.0f / 255.0f));
        atomicAdd(&slot[4], (float)c[1] * (1.0f / 255.0f));
        atomicAdd(&slot[5], (float)c[2] * (1.0f / 255.0f));
        atomicAdd(&slot[6], z * rsqrtf(fr[12] * fr[13]));
        atomicAdd(&slot[7], 1.0f);
    }
}

// One Gaussian per occupied voxel, written at an atomically claimed index
// p < cap into the store layout (model.rs): mean [cap][3], log_scale [cap][3],
// quat [cap][4] (w, x, y, z), opacity [cap] logit, sh_dc [cap][3]. SH rest is
// not touched (the caller zeroes it).
//
// Scale: scale_mode 0 = footprint: s = max(footprint * foot_mult, min_frac * voxel),
// falling back to kNN when the footprint is not finite or not positive; 1 = kNN:
// s = sqrt(mean of the 3 smallest squared distances to the centroids of the 26
// neighbouring voxels), voxel size when fewer than 3 neighbours exist.
extern "C" __global__ void init_finalize(
    const uint64_t* __restrict__ keys, const float* __restrict__ vals, uint32_t table,
    uint32_t* __restrict__ ctr, uint32_t cap, float voxel, float foot_mult, float min_frac,
    uint32_t scale_mode, float opacity_logit, float* __restrict__ mean,
    float* __restrict__ log_scale, float* __restrict__ quat, float* __restrict__ opacity,
    float* __restrict__ sh_dc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= table) return;
    uint64_t key = keys[i];
    if (key == INIT_EMPTY) return;
    const float* sl = vals + (uint64_t)i * INIT_SLOT_FLOATS;
    float cnt = sl[7];
    if (!(cnt > 0.0f)) return;
    uint32_t p = atomicAdd(&ctr[3], 1u);
    if (p >= cap) return;
    float r = 1.0f / cnt;
    float m[3] = {sl[0] * r, sl[1] * r, sl[2] * r};
    float foot = sl[6] * r;

    float s;
    bool use_foot = scale_mode == 0 && foot > 0.0f && foot < 3.0e38f;
    if (use_foot) {
        s = fmaxf(foot * foot_mult, min_frac * voxel);
    } else {
        float best[3] = {3.0e38f, 3.0e38f, 3.0e38f};
        int found = 0;
        int ix = (int)((key >> 42) & 0x1FFFFF), iy = (int)((key >> 21) & 0x1FFFFF),
            iz = (int)(key & 0x1FFFFF);
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    uint64_t nk;
                    if (!init_key(ix + dx - INIT_BIAS, iy + dy - INIT_BIAS, iz + dz - INIT_BIAS, &nk))
                        continue;
                    int64_t ns = init_slot((uint64_t*)keys, table, nk, false, ctr);
                    if (ns < 0) continue;
                    const float* o = vals + (uint64_t)ns * INIT_SLOT_FLOATS;
                    float oc = o[7];
                    if (!(oc > 0.0f)) continue;
                    float ex = o[0] / oc - m[0], ey = o[1] / oc - m[1], ez = o[2] / oc - m[2];
                    float d2 = ex * ex + ey * ey + ez * ez;
                    found++;
                    if (d2 < best[2]) {
                        best[2] = d2;
                        if (best[2] < best[1]) { float t = best[1]; best[1] = best[2]; best[2] = t; }
                        if (best[1] < best[0]) { float t = best[0]; best[0] = best[1]; best[1] = t; }
                    }
                }
        s = found >= 3 ? sqrtf((best[0] + best[1] + best[2]) * (1.0f / 3.0f)) : voxel;
        s = fmaxf(s, 1e-7f);
    }
    float ls = __logf(s);
    for (int a = 0; a < 3; ++a) {
        mean[3 * p + a] = m[a];
        log_scale[3 * p + a] = ls;
        sh_dc[3 * p + a] = (sl[3 + a] * r - 0.5f) * (1.0f / INIT_SH_C0);
    }
    quat[4 * p + 0] = 1.0f;
    quat[4 * p + 1] = 0.0f;
    quat[4 * p + 2] = 0.0f;
    quat[4 * p + 3] = 0.0f;
    opacity[p] = opacity_logit;
}

// ------------------------------------------------------------------ depth prior
// Scale-and-shift-invariant loss of a rendered depth d against a prior D with
// per-pixel weight w (confidence or mask; 0 = ignore):
//   (s, t) = argmin sum w (s d + t - D)^2,   L = sum w (s d + t - D)^2 / sum w.
// At the optimum dL/ds = dL/dt = 0, so dL/dd_i = 2 w_i s (s d_i + t - D_i) / W.
// Three launches: per-block partial sums (6 per block), a one-block solve, the
// gradient. Deterministic (no float atomics).

#define SSI_BLOCK 256
#define SSI_SUMS 6

__device__ __forceinline__ double ssi_block_sum(double v, double* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    double r = 0.0;
    if (threadIdx.x == 0)
        for (int k = 0; k < (int)(blockDim.x >> 5); ++k) r += sh[k];
    return r;
}

// partial[b][6] = (W, Sd, Sdd, SD, SdD, SDD) over this block's grid-stride pixels.
extern "C" __global__ void ssi_partial(const float* __restrict__ d, const float* __restrict__ dp,
                                       const float* __restrict__ w, uint32_t n,
                                       double* __restrict__ partial) {
    __shared__ double sh[32];
    float a[SSI_SUMS] = {0, 0, 0, 0, 0, 0};
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float wi = w[i], di = d[i], Di = dp[i];
        if (!(wi > 0.0f) || !(Di == Di) || !(di == di)) continue;
        a[0] += wi;
        a[1] += wi * di;
        a[2] += wi * di * di;
        a[3] += wi * Di;
        a[4] += wi * di * Di;
        a[5] += wi * Di * Di;
    }
    for (int k = 0; k < SSI_SUMS; ++k) {
        double r = ssi_block_sum((double)a[k], sh);
        if (threadIdx.x == 0) partial[blockIdx.x * SSI_SUMS + k] = r;
    }
}

// out = (s, t, L, W) as floats, L unweighted by lambda.
extern "C" __global__ void ssi_solve(const double* __restrict__ partial, uint32_t blocks,
                                     float* __restrict__ out) {
    __shared__ double sh[32];
    __shared__ double tot[SSI_SUMS];
    for (int k = 0; k < SSI_SUMS; ++k) {
        double v = 0.0;
        for (uint32_t b = threadIdx.x; b < blocks; b += blockDim.x) v += partial[b * SSI_SUMS + k];
        double r = ssi_block_sum(v, sh);
        if (threadIdx.x == 0) tot[k] = r;
    }
    if (threadIdx.x != 0) return;
    double W = tot[0], Sd = tot[1], Sdd = tot[2], SD = tot[3], SdD = tot[4], SDD = tot[5];
    double s = 0.0, t = 0.0, L = 0.0;
    if (W > 0.0) {
        double var = Sdd * W - Sd * Sd;
        if (var > 1e-12 * Sdd * W) {
            s = (SdD * W - Sd * SD) / var;
            t = (SD - s * Sd) / W;
        } else {
            t = SD / W;  // constant render: no scale information, zero gradient
        }
        L = (s * s * Sdd + 2.0 * s * t * Sd + t * t * W - 2.0 * s * SdD - 2.0 * t * SD + SDD) / W;
        if (L < 0.0) L = 0.0;
    }
    out[0] = (float)s;
    out[1] = (float)t;
    out[2] = (float)L;
    out[3] = (float)W;
}

// grad[i] += lambda * 2 w s (s d + t - D) / W
extern "C" __global__ void ssi_backward(const float* __restrict__ d, const float* __restrict__ dp,
                                        const float* __restrict__ w, uint32_t n,
                                        const float* __restrict__ st, float lambda,
                                        float* __restrict__ grad) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float W = st[3];
    float wi = w[i], di = d[i], Di = dp[i];
    if (!(W > 0.0f) || !(wi > 0.0f) || !(Di == Di) || !(di == di)) return;
    float s = st[0], t = st[1];
    grad[i] += lambda * 2.0f * wi * s * (s * di + t - Di) / W;
}
