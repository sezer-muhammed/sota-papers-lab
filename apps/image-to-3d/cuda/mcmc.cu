// 3DGS-MCMC relocation, growth and position noise. Owner: densify (E6).
// Spec: spec-train Section 5.6. The relocation formula follows gsplat's
// `relocation_kernel` (Apache-2.0, gsplat/cuda/csrc/Relocation.cu) term for term, so the
// float cancellation of the alternating binomial sum matches it; 3DGS-MCMC itself is
// Inria-licensed and was read only.
//
// One relocation round on the device (no host sync except the dead count):
//   mcmc_weights   w = o (0 for dead when drawing from the alive set), dead flags
//   fscan_*        inclusive cdf of w
//   flag_* + scan_u32 (compact.cu)   the ordered dead list
//   mcmc_draw      sources by binary search, draws histogram
//   mcmc_adjust    o', s' at every drawn source (N = draws + 1, capped)
//   copy_rows (compact columns)      sources -> destinations, zero columns zeroed

#define MC_THREADS 256
#define MC_ITEMS 8
#define MC_TILE (MC_THREADS * MC_ITEMS)
#define NONE 0xffffffffu
#define COL_COPY 0
#define COL_ZERO 1

struct Column {
    uint64_t ptr;
    uint32_t words;
    uint32_t kind;
};

SPL_INLINE float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

// dead[i] = o <= min_opacity (logit compare) or |q|^2 < 1e-8; w[i] = sigmoid(o), or 0 for a
// dead Gaussian when `alive_only`.
extern "C" __global__ void mcmc_weights(const float* __restrict__ opacity, const float4* __restrict__ quat, uint32_t n,
                                        float dead_logit, uint32_t alive_only, float* __restrict__ w,
                                        uint32_t* __restrict__ dead) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float o = opacity[i];
    float4 q = quat[i];
    bool d = o <= dead_logit || q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w < 1e-8f;
    dead[i] = d;
    w[i] = (alive_only && d) ? 0.f : sigmoidf(o);
}

template <typename T>
SPL_INLINE T mc_warp_inscan(T v) {
    uint32_t lane = threadIdx.x & 31;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        T u = __shfl_up_sync(FULL_MASK, v, o);
        if (lane >= (uint32_t)o) v += u;
    }
    return v;
}

template <typename T>
SPL_INLINE T mc_block_exscan(T v, T* smem, T& total) {
    uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    T inc = mc_warp_inscan(v);
    if (lane == 31) smem[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        T s = lane < nw ? smem[lane] : (T)0;
        s = mc_warp_inscan(s);
        smem[lane] = s;
    }
    __syncthreads();
    T base = warp ? smem[warp - 1] : (T)0;
    total = smem[nw - 1];
    __syncthreads();
    return base + inc - v;
}

// Tile sums of w (blocked: thread t owns MC_ITEMS consecutive items).
extern "C" __global__ void fscan_reduce(const float* __restrict__ w, uint32_t n, float* __restrict__ part) {
    __shared__ float sm[32];
    uint32_t base = blockIdx.x * MC_TILE + threadIdx.x * MC_ITEMS;
    float s = 0.f;
    for (uint32_t j = 0; j < MC_ITEMS; j++)
        if (base + j < n) s += w[base + j];
    float tot;
    mc_block_exscan(s, sm, tot);
    if (threadIdx.x == 0) part[blockIdx.x] = tot;
}

// Single block (1024): exclusive scan of the tile sums in place.
extern "C" __global__ void fscan_partials(float* __restrict__ part, uint32_t count) {
    __shared__ float sm[32];
    uint32_t per = (count + blockDim.x - 1) / blockDim.x;
    uint32_t a = min(count, threadIdx.x * per), b = min(count, a + per);
    float s = 0.f;
    for (uint32_t i = a; i < b; i++) s += part[i];
    float tot;
    float run = mc_block_exscan(s, sm, tot);
    for (uint32_t i = a; i < b; i++) {
        float x = part[i];
        part[i] = run;
        run += x;
    }
}

// cdf[i] = inclusive prefix sum of w.
extern "C" __global__ void fscan_apply(const float* __restrict__ w, uint32_t n, const float* __restrict__ part, float* __restrict__ cdf) {
    __shared__ float sm[32];
    uint32_t base = blockIdx.x * MC_TILE + threadIdx.x * MC_ITEMS;
    float x[MC_ITEMS], s = 0.f;
    for (uint32_t j = 0; j < MC_ITEMS; j++) {
        x[j] = base + j < n ? w[base + j] : 0.f;
        s += x[j];
    }
    float tot;
    float run = part[blockIdx.x] + mc_block_exscan(s, sm, tot);
    for (uint32_t j = 0; j < MC_ITEMS; j++) {
        run += x[j];
        if (base + j < n) cdf[base + j] = run;
    }
}

// Tile counts of nonzero flags (blocked), for the ordered list of flagged indices.
extern "C" __global__ void flag_count(const uint32_t* __restrict__ flags, uint32_t n, uint32_t* __restrict__ part) {
    __shared__ uint32_t sm[32];
    uint32_t base = blockIdx.x * MC_TILE + threadIdx.x * MC_ITEMS;
    uint32_t s = 0;
    for (uint32_t j = 0; j < MC_ITEMS; j++)
        if (base + j < n) s += flags[base + j] != 0;
    uint32_t tot;
    mc_block_exscan(s, sm, tot);
    if (threadIdx.x == 0) part[blockIdx.x] = tot;
}

// list[rank] = i for every flagged i, in index order (part scanned by scan_u32).
extern "C" __global__ void flag_list(const uint32_t* __restrict__ flags, uint32_t n, const uint32_t* __restrict__ part,
              uint32_t* __restrict__ list) {
    __shared__ uint32_t sm[32];
    uint32_t base = blockIdx.x * MC_TILE + threadIdx.x * MC_ITEMS;
    uint32_t f[MC_ITEMS], s = 0;
    for (uint32_t j = 0; j < MC_ITEMS; j++) {
        f[j] = base + j < n && flags[base + j] != 0;
        s += f[j];
    }
    uint32_t tot;
    uint32_t run = part[blockIdx.x] + mc_block_exscan(s, sm, tot);
    for (uint32_t j = 0; j < MC_ITEMS; j++)
        if (f[j]) list[run++] = base + j;
}

// m draws with replacement, p ~ w: u = U(0,1) * cdf[n-1] from Philox (j, iteration,
// RNG_MCMC_DRAW) word `lane` (0 relocation, 1 growth); src[j] = first i with cdf[i] > u.
extern "C" __global__ void mcmc_draw(const float* __restrict__ cdf, uint32_t n, uint32_t m, uint2 seed,
                                     uint32_t iteration, uint32_t lane, uint32_t* __restrict__ src,
                                     uint32_t* __restrict__ draws) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    uint4 r = rng4(seed, j, iteration, RNG_MCMC_DRAW);
    float u = u01_open(lane ? r.y : r.x) * cdf[n - 1];
    uint32_t lo = 0, hi = n - 1;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        if (cdf[mid] > u) hi = mid;
        else lo = mid + 1;
    }
    src[j] = lo;
    atomicAdd(&draws[lo], 1u);
}

// At every source drawn N - 1 >= 1 times (N capped at n_max):
//   o' = 1 - (1 - o)^(1/N)
//   s' = s * o / sum_{i=1..N} sum_{k=0..i-1} C(i-1, k) (-1)^k o'^(k+1) / sqrt(k+1)
// o' clamped to [min_opacity, 1 - 2^-23] after s' (as 3DGS-MCMC). binoms[n][k], n_max x n_max.
extern "C" __global__ void mcmc_adjust(const uint32_t* __restrict__ draws, const float* __restrict__ binoms,
                                       uint32_t n_max, uint32_t n, float min_opacity, float* __restrict__ opacity,
                                       float* __restrict__ log_scale) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || draws[i] == 0) return;
    int nn = (int)min(draws[i] + 1, n_max);
    float o = sigmoidf(opacity[i]);
    float o_new = 1.f - powf(1.f - o, 1.f / (float)nn);
    float denom = 0.f;
    for (int a = 1; a <= nn; a++)
        for (int k = 0; k <= a - 1; k++) {
            float bin = binoms[(a - 1) * n_max + k];
            float term = ((k & 1) ? -1.f : 1.f) / sqrtf((float)(k + 1)) * powf(o_new, (float)(k + 1));
            denom += bin * term;
        }
    float coeff = o / denom;
    o_new = fminf(fmaxf(o_new, min_opacity), 1.f - 1.1920929e-7f);
    opacity[i] = logf(o_new / (1.f - o_new));
    for (int a = 0; a < 3; a++) log_scale[3 * i + a] += logf(coeff);
}

// For every column: row dst <- row src; COL_ZERO columns instead get dst <- 0 when
// `zero_dst` and src <- 0 when `zero_src` (resets the moments of the sources). dst = dst_list[j] or dst_base + j.
extern "C" __global__ void copy_rows(const Column* __restrict__ cols, const uint32_t* __restrict__ src_list,
                                     const uint32_t* __restrict__ dst_list, uint32_t dst_base, uint32_t m,
                                     uint32_t zero_src, uint32_t zero_dst) {
    Column col = cols[blockIdx.y];
    uint32_t* a = (uint32_t*)col.ptr;
    uint32_t w = col.words, len = m * w;
    for (uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < len; q += gridDim.x * blockDim.x) {
        uint32_t j = q / w, e = q - j * w;
        uint32_t s = src_list[j], d = dst_list ? dst_list[j] : dst_base + j;
        if (col.kind == COL_ZERO) {
            if (zero_dst) a[(uint64_t)d * w + e] = 0;
            if (zero_src) a[(uint64_t)s * w + e] = 0;
        } else {
            a[(uint64_t)d * w + e] = a[(uint64_t)s * w + e];
        }
    }
}

// After every optimizer step (captured in the iteration graph):
// mean += means_lr * noise_scale * sigmoid(-100 (o - 0.005)) * Sigma eps, Sigma = R S^2 R^T,
// eps ~ N(0, I) from Philox (i, iteration, RNG_MCMC_NOISE).
extern "C" __global__ void mcmc_noise(const IterParams* __restrict__ params, const float* __restrict__ log_scale,
                                      const float4* __restrict__ quat, const float* __restrict__ opacity,
                                      float* __restrict__ mean, const uint32_t* __restrict__ status) {
    if (status && status[ST_OVERFLOW]) return;
    const IterParams& P = *params;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P.frame.n_gaussians || P.noise_scale == 0.f) return;
    float o = sigmoidf(opacity[i]);
    float f = 1.f / (1.f + expf(100.f * (o - 0.005f)));
    float4 q = quat[i];
    float inv = rsqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    float w = q.x * inv, x = q.y * inv, y = q.z * inv, z = q.w * inv;
    float R[9] = {1.f - 2.f * (y * y + z * z), 2.f * (x * y - w * z),       2.f * (x * z + w * y),
                  2.f * (x * y + w * z),       1.f - 2.f * (x * x + z * z), 2.f * (y * z - w * x),
                  2.f * (x * z - w * y),       2.f * (y * z + w * x),       1.f - 2.f * (x * x + y * y)};
    uint2 seed = make_uint2(P.seed[0], P.seed[1]);
    uint4 r = rng4(seed, i, P.frame.iteration, RNG_MCMC_NOISE);
    float2 e01 = normal2(r.x, r.y), e23 = normal2(r.z, r.w);
    float e[3] = {e01.x, e01.y, e23.x};
    // t = R^T eps, t *= s^2, d = R t.
    float t[3];
    for (int a = 0; a < 3; a++) {
        float s = expf(log_scale[3 * i + a]);
        t[a] = (R[a] * e[0] + R[3 + a] * e[1] + R[6 + a] * e[2]) * s * s;
    }
    float k = P.means_lr * P.noise_scale * f;
    for (int a = 0; a < 3; a++) mean[3 * i + a] += k * (R[3 * a] * t[0] + R[3 * a + 1] * t[1] + R[3 * a + 2] * t[2]);
}
