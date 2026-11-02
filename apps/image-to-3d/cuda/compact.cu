// Order-preserving compaction with emission, Morton codes, a stable LSD radix sort and
// column gathers for every per-Gaussian array. Owner: densify (E6). Spec: spec-train 5.8-5.9.
//
// Every per-Gaussian buffer is a list of "columns": a contiguous run of `words` u32 per
// Gaussian (an interleaved array such as mean [cap][3], or one row of a planar array such
// as sh_rest row j, [cap]). Columns are moved in batches through the slab: slab_load
// copies the n live entries of each column of the batch into the slab, slab_scatter
// writes survivors, clones and split children from the slab to their final places. All
// reads of a batch happen before any of its writes, so the in-place move is safe for any
// destination order (survivors dst <= src, tails anywhere in [n_surv, n_new)), which the
// chunked dst <= src scheme of the first spec draft did not guarantee for the tail.
// The slab holds cap x 45 words, so a batch is any set of columns with sum(words) <= 45.

#define CT_THREADS 256
#define CT_ITEMS 8
#define CT_TILE (CT_THREADS * CT_ITEMS)
#define CT_WARPS (CT_THREADS / 32)
#define NONE 0xffffffffu

// Column kinds: what new (tail) entries receive.
#define COL_COPY 0  // a copy of the parent (parameters, last_visible)
#define COL_ZERO 1  // zero (moments, gradients, statistics)

struct Column {        // 16 B, mirrors model::Column
    uint64_t ptr;      // device address of entry 0
    uint32_t words;    // u32 words per Gaussian
    uint32_t kind;     // COL_*
};

// ---------------------------------------------------------------- scans

template <typename T>
SPL_INLINE T warp_inscan(T v) {
    uint32_t lane = threadIdx.x & 31;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        T u = __shfl_up_sync(FULL_MASK, v, o);
        if (lane >= (uint32_t)o) v += u;
    }
    return v;
}

// Exclusive block scan (blockDim.x a multiple of 32, <= 1024); `total` gets the block sum.
template <typename T>
SPL_INLINE T block_exscan(T v, T* smem, T& total) {
    uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    T inc = warp_inscan(v);
    if (lane == 31) smem[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        T s = lane < nw ? smem[lane] : (T)0;
        s = warp_inscan(s);
        smem[lane] = s;
    }
    __syncthreads();
    T base = warp ? smem[warp - 1] : (T)0;
    total = smem[nw - 1];
    __syncthreads();
    return base + inc - v;
}

// Single block (1024 threads): data[0..count) <- exclusive prefix sums; *total <- the sum.
// Each thread owns a contiguous run, so it reads the array twice and scans once.
extern "C" __global__ void scan_u32(uint32_t* __restrict__ data, uint32_t count,
                                                            uint32_t* __restrict__ total) {
    __shared__ uint32_t sm[32];
    uint32_t per = (count + blockDim.x - 1) / blockDim.x;
    uint32_t a = min(count, threadIdx.x * per), b = min(count, a + per);
    uint32_t s = 0;
    for (uint32_t i = a; i < b; i++) s += data[i];
    uint32_t tot;
    uint32_t run = block_exscan(s, sm, tot);
    for (uint32_t i = a; i < b; i++) {
        uint32_t x = data[i];
        data[i] = run;
        run += x;
    }
    if (threadIdx.x == 0 && total) *total = tot;
}

// ---------------------------------------------------------------- emission plan -> destinations

// keep = out_count != 0 (2 = FastGS prune candidate that VCP did not draw: kept).
SPL_INLINE void emit_item(const uint32_t* oc, const uint8_t* cf, const uint8_t* sf, uint32_t i, uint32_t& k,
                          uint32_t& c, uint32_t& s) {
    k = oc[i] != 0;
    c = cf ? cf[i] : 0;
    s = sf ? sf[i] : 0;
}

// Per tile of CT_TILE Gaussians: survivors, clones and splits -> part[0|1|2][tile].
extern "C" __global__ void emit_count(const uint32_t* __restrict__ out_count, const uint8_t* __restrict__ clone_flag,
               const uint8_t* __restrict__ split_flag, uint32_t n, uint32_t* __restrict__ part) {
    __shared__ uint32_t sm[3][CT_WARPS];
    uint32_t base = blockIdx.x * CT_TILE, k = 0, c = 0, s = 0;
    for (uint32_t j = 0; j < CT_ITEMS; j++) {
        uint32_t i = base + j * CT_THREADS + threadIdx.x;
        if (i < n) {
            uint32_t a, b, d;
            emit_item(out_count, clone_flag, split_flag, i, a, b, d);
            k += a, c += b, s += d;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        k += __shfl_xor_sync(FULL_MASK, k, o);
        c += __shfl_xor_sync(FULL_MASK, c, o);
        s += __shfl_xor_sync(FULL_MASK, s, o);
    }
    uint32_t w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) sm[0][w] = k, sm[1][w] = c, sm[2][w] = s;
    __syncthreads();
    if (threadIdx.x < 3) {
        uint32_t t = 0;
        for (int q = 0; q < CT_WARPS; q++) t += sm[threadIdx.x][q];
        part[threadIdx.x * gridDim.x + blockIdx.x] = t;
    }
}

// After scan_u32 over each third of `part` (totals[0..3] = survivors, clones, splits):
// dst[i] = survivor slot or NONE; tail_dst[i] = first slot of i's clone (after all
// survivors) or of its two split children (after all clones), else NONE.
extern "C" __global__ void emit_dst(const uint32_t* __restrict__ out_count, const uint8_t* __restrict__ clone_flag,
             const uint8_t* __restrict__ split_flag, uint32_t n, const uint32_t* __restrict__ part,
             const uint32_t* __restrict__ totals, uint32_t* __restrict__ dst, uint32_t* __restrict__ tail_dst) {
    __shared__ uint64_t sm[32];
    uint32_t base = blockIdx.x * CT_TILE;
    // Blocked arrangement: thread t owns items base + t*CT_ITEMS .. +CT_ITEMS (order-preserving).
    uint32_t k[CT_ITEMS], c[CT_ITEMS], s[CT_ITEMS];
    uint64_t sum = 0;
#pragma unroll
    for (uint32_t j = 0; j < CT_ITEMS; j++) {
        uint32_t i = base + threadIdx.x * CT_ITEMS + j;
        k[j] = c[j] = s[j] = 0;
        if (i < n) emit_item(out_count, clone_flag, split_flag, i, k[j], c[j], s[j]);
        sum += (uint64_t)k[j] | ((uint64_t)c[j] << 21) | ((uint64_t)s[j] << 42);
    }
    uint64_t tot;
    uint64_t run = block_exscan(sum, sm, tot);
    const uint32_t nb = gridDim.x;
    uint32_t rk = part[blockIdx.x] + (uint32_t)(run & 0x1fffff);
    uint32_t rc = part[nb + blockIdx.x] + (uint32_t)((run >> 21) & 0x1fffff);
    uint32_t rs = part[2 * nb + blockIdx.x] + (uint32_t)(run >> 42);
    uint32_t n_surv = totals[0], n_clone = totals[1];
#pragma unroll
    for (uint32_t j = 0; j < CT_ITEMS; j++) {
        uint32_t i = base + threadIdx.x * CT_ITEMS + j;
        if (i >= n) break;
        dst[i] = k[j] ? rk : NONE;
        tail_dst[i] = c[j] ? n_surv + rc : (s[j] ? n_surv + n_clone + 2 * rs : NONE);
        rk += k[j], rc += c[j], rs += s[j];
    }
}

// ---------------------------------------------------------------- column moves through the slab

// Batch layout in the slab: column q at word offset n * (sum of words of columns < q).
// blockIdx.y = column; grid-stride over its n * words words (contiguous both sides).
extern "C" __global__ void slab_load(const Column* __restrict__ cols, uint32_t n, uint32_t* __restrict__ slab) {
    Column col = cols[blockIdx.y];
    uint32_t off = 0;
    for (uint32_t q = 0; q < blockIdx.y; q++) off += cols[q].words;
    const uint32_t* src = (const uint32_t*)col.ptr;
    uint32_t* out = slab + (uint64_t)off * n;
    uint32_t len = n * col.words;
    for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < len; j += gridDim.x * blockDim.x) out[j] = src[j];
}

// Survivors to dst[i], clones / split children (both) to tail_dst[i] (+1): a copy or zero.
extern "C" __global__ void slab_scatter(const Column* __restrict__ cols, uint32_t n,
                                        const uint32_t* __restrict__ slab, const uint32_t* __restrict__ dst,
                                        const uint32_t* __restrict__ tail_dst,
                                        const uint8_t* __restrict__ split_flag) {
    Column col = cols[blockIdx.y];
    uint32_t off = 0;
    for (uint32_t q = 0; q < blockIdx.y; q++) off += cols[q].words;
    const uint32_t* src = slab + (uint64_t)off * n;
    uint32_t* out = (uint32_t*)col.ptr;
    uint32_t w = col.words, len = n * w;
    for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < len; j += gridDim.x * blockDim.x) {
        uint32_t i = j / w, e = j - i * w;
        uint32_t x = src[j];
        uint32_t d = dst[i], t = tail_dst[i];
        if (d != NONE) out[(uint64_t)d * w + e] = x;
        if (t != NONE) {
            uint32_t y = col.kind == COL_ZERO ? 0u : x;
            out[(uint64_t)t * w + e] = y;
            if (split_flag && split_flag[i]) out[(uint64_t)(t + 1) * w + e] = y;
        }
    }
}

// Morton gather: slab column q entry i <- column q entry perm[i].
extern "C" __global__ void slab_gather(const Column* __restrict__ cols, uint32_t n, const uint32_t* __restrict__ perm,
                                       uint32_t* __restrict__ slab) {
    Column col = cols[blockIdx.y];
    uint32_t off = 0;
    for (uint32_t q = 0; q < blockIdx.y; q++) off += cols[q].words;
    const uint32_t* src = (const uint32_t*)col.ptr;
    uint32_t* out = slab + (uint64_t)off * n;
    uint32_t w = col.words, len = n * w;
    for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < len; j += gridDim.x * blockDim.x) {
        uint32_t i = j / w, e = j - i * w;
        out[j] = src[(uint64_t)perm[i] * w + e];
    }
}

// Inverse of slab_load: column <- slab (after slab_gather).
extern "C" __global__ void slab_store(const Column* __restrict__ cols, uint32_t n, const uint32_t* __restrict__ slab) {
    Column col = cols[blockIdx.y];
    uint32_t off = 0;
    for (uint32_t q = 0; q < blockIdx.y; q++) off += cols[q].words;
    const uint32_t* src = slab + (uint64_t)off * n;
    uint32_t* out = (uint32_t*)col.ptr;
    uint32_t len = n * col.words;
    for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < len; j += gridDim.x * blockDim.x) out[j] = src[j];
}

// ---------------------------------------------------------------- split children

// Rotation matrix of a raw (w, x, y, z) quaternion, normalized here.
SPL_INLINE void e6_rotmat(float4 q, float R[9]) {
    float inv = rsqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    float w = q.x * inv, x = q.y * inv, y = q.z * inv, z = q.w * inv;
    R[0] = 1.f - 2.f * (y * y + z * z), R[1] = 2.f * (x * y - w * z), R[2] = 2.f * (x * z + w * y);
    R[3] = 2.f * (x * y + w * z), R[4] = 1.f - 2.f * (x * x + z * z), R[5] = 2.f * (y * z - w * x);
    R[6] = 2.f * (x * z - w * y), R[7] = 2.f * (y * z + w * x), R[8] = 1.f - 2.f * (x * x + y * y);
}

// ln 1.6 (the Inria split divides the scale by 0.8 * N with N = 2).
#define LN_1_6 0.47000363f

// After the moves: both children of split parent i (slots tail_dst[i], +1) hold a copy of
// the parent. Child c: mean += R(q) (s * eps), eps ~ N(0, I) from Philox
// (2i + c, iteration, RNG_SPLIT); log scale -= ln 1.6.
extern "C" __global__ void split_children(const uint8_t* __restrict__ split_flag, const uint32_t* __restrict__ tail_dst,
                                          uint32_t n, uint2 seed, uint32_t iteration, float* __restrict__ mean,
                                          float* __restrict__ log_scale, const float4* __restrict__ quat) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !split_flag[i]) return;
    uint32_t t = tail_dst[i];
    for (uint32_t c = 0; c < 2; c++) {
        uint32_t p = t + c;
        float R[9];
        e6_rotmat(quat[p], R);
        float s0 = expf(log_scale[3 * p]), s1 = expf(log_scale[3 * p + 1]), s2 = expf(log_scale[3 * p + 2]);
        uint4 r = rng4(seed, 2 * i + c, iteration, RNG_SPLIT);
        float2 e01 = normal2(r.x, r.y), e23 = normal2(r.z, r.w);
        float v0 = s0 * e01.x, v1 = s1 * e01.y, v2 = s2 * e23.x;
        mean[3 * p] += R[0] * v0 + R[1] * v1 + R[2] * v2;
        mean[3 * p + 1] += R[3] * v0 + R[4] * v1 + R[5] * v2;
        mean[3 * p + 2] += R[6] * v0 + R[7] * v1 + R[8] * v2;
        log_scale[3 * p] -= LN_1_6;
        log_scale[3 * p + 1] -= LN_1_6;
        log_scale[3 * p + 2] -= LN_1_6;
    }
}

// ---------------------------------------------------------------- Morton codes

// Order-preserving float <-> u32 (for atomicMin/Max and radix keys).
SPL_INLINE uint32_t f2key(float f) {
    uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
SPL_INLINE float key2f(uint32_t k) { return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k); }

// aabb[0..3] = min keys (init 0xffffffff), aabb[3..6] = max keys (init 0).
extern "C" __global__ void aabb_reduce(const float* __restrict__ mean, uint32_t n, uint32_t* __restrict__ aabb) {
    uint32_t lo[3] = {NONE, NONE, NONE}, hi[3] = {0, 0, 0};
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        for (int a = 0; a < 3; a++) {
            uint32_t k = f2key(mean[3 * i + a]);
            lo[a] = min(lo[a], k), hi[a] = max(hi[a], k);
        }
    for (int a = 0; a < 3; a++)
        for (int o = 16; o > 0; o >>= 1) {
            lo[a] = min(lo[a], __shfl_xor_sync(FULL_MASK, lo[a], o));
            hi[a] = max(hi[a], __shfl_xor_sync(FULL_MASK, hi[a], o));
        }
    if ((threadIdx.x & 31) == 0)
        for (int a = 0; a < 3; a++) atomicMin(&aabb[a], lo[a]), atomicMax(&aabb[3 + a], hi[a]);
}

SPL_INLINE uint32_t spread10(uint32_t v) {
    v &= 0x3ffu;
    v = (v | (v << 16)) & 0x030000ffu;
    v = (v | (v << 8)) & 0x0300f00fu;
    v = (v | (v << 4)) & 0x030c30c3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}

// q = floor(clamp((mu - min) / max_extent, 0, 1) * 1023) per axis, interleaved x | y << 1 | z << 2.
extern "C" __global__ void morton_codes(const float* __restrict__ mean, const uint32_t* __restrict__ aabb, uint32_t n,
                                        uint32_t* __restrict__ codes, uint32_t* __restrict__ idx) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float lo[3], ext = 0.f;
    for (int a = 0; a < 3; a++) {
        lo[a] = key2f(aabb[a]);
        ext = fmaxf(ext, key2f(aabb[3 + a]) - lo[a]);
    }
    float inv = ext > 0.f ? __fdiv_rn(1.f, ext) : 0.f;  // IEEE: the CPU mirror gets the same codes
    uint32_t c = 0;
    for (int a = 0; a < 3; a++) {
        float u = fminf(fmaxf(__fmul_rn(__fsub_rn(mean[3 * i + a], lo[a]), inv), 0.f), 1.f);
        c |= spread10((uint32_t)__fmul_rn(u, 1023.f)) << a;
    }
    codes[i] = c;
    idx[i] = i;
}

// ---------------------------------------------------------------- stable LSD radix sort, 8-bit digits
//
// Three kernels per pass over tiles of CT_TILE keys: radix_hist (per-tile digit counts,
// digit-major hist[d * tiles + tile]), scan_u32 over the whole table, radix_scatter (a
// stable rank per warp with __match_any_sync, warps own consecutive 256-key segments).

extern "C" __global__ void radix_hist(const uint32_t* __restrict__ keys, uint32_t n, uint32_t shift, uint32_t* __restrict__ hist) {
    __shared__ uint32_t h[256];
    h[threadIdx.x] = 0;
    __syncthreads();
    uint32_t base = blockIdx.x * CT_TILE;
#pragma unroll
    for (uint32_t j = 0; j < CT_ITEMS; j++) {
        uint32_t i = base + j * CT_THREADS + threadIdx.x;
        if (i < n) atomicAdd(&h[(keys[i] >> shift) & 255u], 1u);
    }
    __syncthreads();
    hist[threadIdx.x * gridDim.x + blockIdx.x] = h[threadIdx.x];
}

extern "C" __global__ void radix_scatter(const uint32_t* __restrict__ keys, const uint32_t* __restrict__ vals, uint32_t n, uint32_t shift,
                  const uint32_t* __restrict__ hist, uint32_t* __restrict__ keys_out, uint32_t* __restrict__ vals_out) {
    __shared__ uint32_t wh[CT_WARPS][257];
    for (uint32_t q = threadIdx.x; q < CT_WARPS * 257; q += CT_THREADS) (&wh[0][0])[q] = 0;
    __syncthreads();
    uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint32_t base = blockIdx.x * CT_TILE + warp * (CT_TILE / CT_WARPS);
    uint32_t k[CT_ITEMS], v[CT_ITEMS], rank[CT_ITEMS];
    uint32_t lt = lanemask_lt();
#pragma unroll
    for (uint32_t r = 0; r < CT_ITEMS; r++) {
        uint32_t i = base + r * 32 + lane;
        bool ok = i < n;
        k[r] = ok ? keys[i] : 0;
        v[r] = ok ? vals[i] : 0;
        uint32_t d = ok ? (k[r] >> shift) & 255u : 256u;
        uint32_t peers = __match_any_sync(FULL_MASK, d);
        uint32_t before = wh[warp][d];
        rank[r] = before + __popc(peers & lt);
        __syncwarp();
        if ((peers & lt) == 0) wh[warp][d] = before + __popc(peers);
        __syncwarp();
    }
    __syncthreads();
    {   // per digit: global offset of this tile + counts of earlier warps
        uint32_t d = threadIdx.x;
        uint32_t run = hist[d * gridDim.x + blockIdx.x];
        for (uint32_t w = 0; w < CT_WARPS; w++) {
            uint32_t c = wh[w][d];
            wh[w][d] = run;
            run += c;
        }
    }
    __syncthreads();
#pragma unroll
    for (uint32_t r = 0; r < CT_ITEMS; r++) {
        uint32_t i = base + r * 32 + lane;
        if (i < n) {
            uint32_t o = wh[warp][(k[r] >> shift) & 255u] + rank[r];
            keys_out[o] = k[r];
            vals_out[o] = v[r];
        }
    }
}
