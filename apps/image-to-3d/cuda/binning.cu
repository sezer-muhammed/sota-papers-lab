// Tile binning: instance offsets, emission, tile histogram, tile/chain scan.
// Owner: sort (E2). Spec: spec-raster Section 3.1 (kernels 6-9) and 2.8 (tile span).
//
// The span rule (spec-raster 2.8) lives in namespace `span`, written with explicit IEEE
// rounding intrinsics only (no fast-math approximations, no contraction), so every caller
// compiles it to the same instructions and the CPU mirror in src/kernels/sort/tests.rs
// reproduces it bit for bit. It reads only the first 32 bytes of a SplatRec
// (mx, my, ca, cb, cc, o): k = 2 ln(255 o) comes from o through span::ln, a deterministic
// IEEE-only logarithm, so a gather of a record costs 1.5 DRAM sectors on average, not 2.
//
// The training path is scan_emit: one kernel per 256 Gaussians in depth order that gathers
// each record once, counts its tiles, scans the counts (decoupled look-back), and writes
// the instances. Count and emission therefore come from the same code in the same thread
// (count == emit by construction) and binning needs no n_tiles from preprocess.
// scan_tiles + emit are the two-kernel form of spec-raster 3.1 (counts given in n_tiles);
// count_tiles writes the reference counts. They stay for the gsplat oracle test, for A/B
// timing and for callers that already have counts.
//
// No memsets: the scans' look-back words are epoch-tagged like the radix sort's (radix.cu),
// and bin_scan zeroes the histogram it consumed.

#define BIN_THREADS 256
#define SCAN_THREADS 256
#define SCAN_IPT 16
#define SCAN_PART (SCAN_THREADS * SCAN_IPT)
#define HIST_THREADS 512
#define BIN_SCAN_THREADS 1024
#define BIN_SCAN_IPT (MAX_TILES / BIN_SCAN_THREADS)
// A lane writes its own Gaussian when it has at most this many tile rows and tiles; larger
// Gaussians are shared by the whole warp, one row per lane (spec-raster 2.8, "load balance").
#define EMIT_LANE_ROWS 4
#define EMIT_LANE_TILES 16

// scan state words (RadixSorter::state after the radix words; layout mirrored in sort.rs)
#define SC_EPOCH 0
#define SC_PART 1
#define SC_DONE 2

#define LB_AGG 1u
#define LB_INC 2u

// Launch bounds sit on these declarations (`__global__ __launch_bounds__(...) void`), so
// each definition keeps the plain `extern "C" __global__ void` spelling the kernel-list test
// reads; the test gpu_kernels_carry_their_launch_bounds checks .maxntid in the PTX.
extern "C" __global__ __launch_bounds__(BIN_THREADS) void count_tiles(const FrameCam*, const SplatRec*, uint32_t*,
                                                                      const uint32_t*);
extern "C" __global__ __launch_bounds__(SCAN_THREADS) void scan_tiles(const uint32_t*, const uint32_t*, uint32_t,
                                                                      uint32_t*, uint64_t*, uint32_t*, uint32_t*);
extern "C" __global__ __launch_bounds__(BIN_THREADS) void emit(const FrameCam*, const uint32_t*, const SplatRec*,
                                                               const uint32_t*, uint16_t*, uint32_t*,
                                                               const uint32_t*);
extern "C" __global__ __launch_bounds__(BIN_THREADS) void scan_emit(const FrameCam*, const uint32_t*,
                                                                    const SplatRec*, uint32_t, uint32_t*, uint16_t*,
                                                                    uint32_t*, uint64_t*, uint32_t*, uint32_t*);
extern "C" __global__ __launch_bounds__(HIST_THREADS) void tile_hist(const uint16_t*, uint32_t*, const uint32_t*);
extern "C" __global__ __launch_bounds__(BIN_SCAN_THREADS) void bin_scan(const FrameCam*, uint32_t*, uint32_t,
                                                                        uint32_t, uint32_t*, uint32_t*, uint32_t*,
                                                                        uint32_t*);

// ---------------------------------------------------------------- tile span (spec-raster 2.8)

namespace span {

// -2 / log2(e) = -2 ln 2 and -1 / log2(e) = -ln 2: undo the exp2 pre-scaling of the conic.
#define SPAN_INV_HALF_LOG2E (-1.3862943611198906f)
#define SPAN_INV_LOG2E (-0.6931471805599453f)

// The record fields the span reads: the first 32 bytes of a SplatRec.
struct In {
    float mx, my, ca, cb, cc, o;
};

SPL_INLINE In load(const SplatRec* __restrict__ rec, uint32_t v) {
    const float4* p = reinterpret_cast<const float4*>(rec + v);
    const float4 a = p[0], b = p[1];
    return In{a.x, a.y, a.z, a.w, b.x, b.y};
}

// Natural logarithm of a normal x > 0 from IEEE operations only (bit-identical on every
// device and in the CPU mirror): x = 2^e m with m in [sqrt(1/2), sqrt(2)),
// ln m = 2 atanh(t), t = (m - 1) / (m + 1), odd series to t^9 (|t| < 0.172, truncation
// below 4e-10, far inside the k_bin margin).
SPL_INLINE float ln(float x) {
    const uint32_t b = __float_as_uint(x);
    int e = (int)((b >> 23) & 0xffu) - 127;
    float m = __uint_as_float((b & 0x007fffffu) | 0x3f800000u);
    if (m > __uint_as_float(0x3fb504f3u)) {  // sqrt(2)
        m = __fmul_rn(m, 0.5f);
        e += 1;
    }
    const float t = __fdiv_rn(__fsub_rn(m, 1.0f), __fadd_rn(m, 1.0f));
    const float t2 = __fmul_rn(t, t);
    float p = __fmaf_rn(t2, __uint_as_float(0x3de38e39u), __uint_as_float(0x3e124925u));  // 1/9, 1/7
    p = __fmaf_rn(t2, p, __uint_as_float(0x3e4ccccdu));                                   // 1/5
    p = __fmaf_rn(t2, p, __uint_as_float(0x3eaaaaabu));                                   // 1/3
    p = __fmaf_rn(t2, p, 1.0f);
    return __fmaf_rn((float)e, __uint_as_float(0x3f317218u), __fmul_rn(__fmul_rn(2.0f, t), p));  // ln 2
}

struct Span {
    float mx, my, A, B, inv_a, kb, det, ratio;  // ratio = B * x_ext / C
    int px0, px1, py0, py1;                     // pixel-centre bounds, clipped to the image
};

// Bounds of the ellipse {q(c - m) <= k_bin}, k_bin = k (1 + KBIN_REL) + KBIN_ABS with
// k = 2 ln(255 o), over pixel centres. False when no pixel centre of the image can be inside.
SPL_INLINE bool setup(const In& r, uint32_t width, uint32_t height, Span& s) {
    const float x = __fmul_rn(255.0f, r.o);
    if (!(x > 1.17549435e-38f) || !(x < 3.4e38f)) return false;  // normal and finite
    s.mx = r.mx;
    s.my = r.my;
    s.A = __fmul_rn(r.ca, SPAN_INV_HALF_LOG2E);
    s.B = __fmul_rn(r.cb, SPAN_INV_LOG2E);
    const float C = __fmul_rn(r.cc, SPAN_INV_HALF_LOG2E);
    const float k = __fmul_rn(2.0f, ln(x));
    s.kb = __fmaf_rn(k, 1.0f + KBIN_REL, KBIN_ABS);
    s.det = __fmaf_rn(s.A, C, -__fmul_rn(s.B, s.B));
    if (!(s.det > 0.0f) || !(s.kb > 0.0f) || !(s.A > 0.0f) || !(C > 0.0f)) return false;
    const float inv_det = __frcp_rn(s.det);
    const float x_ext = __fsqrt_rn(__fmul_rn(s.kb, __fmul_rn(C, inv_det)));
    const float y_ext = __fsqrt_rn(__fmul_rn(s.kb, __fmul_rn(s.A, inv_det)));
    s.inv_a = __frcp_rn(s.A);
    s.ratio = __fmul_rn(__fmul_rn(s.B, x_ext), __frcp_rn(C));
    const float fx0 = fmaxf(ceilf(__fsub_rn(__fsub_rn(s.mx, x_ext), 0.5f)), 0.0f);
    const float fx1 = fminf(floorf(__fsub_rn(__fadd_rn(s.mx, x_ext), 0.5f)), (float)width - 1.0f);
    const float fy0 = fmaxf(ceilf(__fsub_rn(__fsub_rn(s.my, y_ext), 0.5f)), 0.0f);
    const float fy1 = fminf(floorf(__fsub_rn(__fadd_rn(s.my, y_ext), 0.5f)), (float)height - 1.0f);
    if (!(fx0 <= fx1) || !(fy0 <= fy1)) return false;
    s.px0 = (int)fx0;
    s.px1 = (int)fx1;
    s.py0 = (int)fy0;
    s.py1 = (int)fy1;
    return true;
}

// x-range [xl, xr] of the ellipse restricted to y in [y0, y1] (continuous y).
SPL_INLINE void band_x_interval(const Span& s, float y0, float y1, float& xl, float& xr) {
    const float ak = __fmul_rn(s.A, s.kb);
    const float yl = fminf(fmaxf(__fadd_rn(s.my, s.ratio), y0), y1);
    const float dl = __fsub_rn(yl, s.my);
    const float ql = __fsqrt_rn(fmaxf(__fmaf_rn(-s.det, __fmul_rn(dl, dl), ak), 0.0f));
    xl = __fadd_rn(s.mx, __fmul_rn(__fsub_rn(-__fmul_rn(s.B, dl), ql), s.inv_a));
    const float yr = fminf(fmaxf(__fsub_rn(s.my, s.ratio), y0), y1);
    const float dr = __fsub_rn(yr, s.my);
    const float qr = __fsqrt_rn(fmaxf(__fmaf_rn(-s.det, __fmul_rn(dr, dr), ak), 0.0f));
    xr = __fadd_rn(s.mx, __fmul_rn(__fadd_rn(-__fmul_rn(s.B, dr), qr), s.inv_a));
}

// Tile rows the Gaussian can touch.
SPL_INLINE int row_lo(const Span& s) { return s.py0 / TILE_H; }
SPL_INLINE int row_hi(const Span& s) { return s.py1 / TILE_H; }

// Tile columns [tx0, tx1] of tile row ty; false when the row has no pixel centre inside.
SPL_INLINE bool row_tiles(const Span& s, int ty, int& tx0, int& tx1) {
    const float yb0 = (float)max(ty * TILE_H, s.py0) + 0.5f;
    const float yb1 = (float)min(ty * TILE_H + TILE_H - 1, s.py1) + 0.5f;
    float xl, xr;
    band_x_interval(s, yb0, yb1, xl, xr);
    const float c0 = fmaxf(ceilf(__fsub_rn(xl, 0.5f)), (float)s.px0);
    const float c1 = fminf(floorf(__fsub_rn(xr, 0.5f)), (float)s.px1);
    if (!(c0 <= c1)) return false;
    tx0 = (int)c0 / TILE_W;
    tx1 = (int)c1 / TILE_W;
    return true;
}

// Number of tiles of a set-up span (n_tiles of spec-raster 2.8).
SPL_INLINE uint32_t count(const Span& s) {
    uint32_t c = 0;
    for (int ty = row_lo(s); ty <= row_hi(s); ++ty) {
        int a, b;
        if (row_tiles(s, ty, a, b)) c += (uint32_t)(b - a + 1);
    }
    return c;
}

}  // namespace span

// ---------------------------------------------------------------- helpers

SPL_INLINE uint64_t bin_ld_relaxed(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

SPL_INLINE void bin_st_relaxed(uint64_t* p, uint64_t v) {
    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

SPL_INLINE uint64_t bin_pack(uint32_t tag, uint32_t flag, uint32_t value) {
    return ((uint64_t)((tag << 2) | flag) << 32) | value;
}

// Saturating add: instance totals are reported truthfully up to 2^32 - 1.
SPL_INLINE uint32_t sat_add(uint32_t a, uint32_t b) {
    const uint32_t s = a + b;
    return s < a ? 0xffffffffu : s;
}

SPL_INLINE uint32_t bin_warp_incl_scan(uint32_t x, uint32_t lane) {
#pragma unroll
    for (uint32_t o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(FULL_MASK, x, o);
        if (lane >= o) x = sat_add(x, y);
    }
    return x;
}

// Shared-memory index with one pad word per 32: thread-contiguous reads are conflict free.
SPL_INLINE uint32_t pad32(uint32_t i) { return i + (i >> 5); }

// Decoupled look-back of partition `part` with block total `total` (one thread): publishes
// the aggregate, sums predecessors until an inclusive prefix, publishes its own; returns the
// exclusive prefix.
SPL_INLINE uint32_t scan_lookback(uint64_t* __restrict__ lookback, uint32_t part, uint32_t tag, uint32_t total) {
    if (part == 0) {
        bin_st_relaxed(&lookback[0], bin_pack(tag, LB_INC, total));
        return 0;
    }
    bin_st_relaxed(&lookback[part], bin_pack(tag, LB_AGG, total));
    uint32_t excl = 0, q = part - 1;
    for (;;) {
        const uint64_t w = bin_ld_relaxed(&lookback[q]);
        const uint32_t hi = (uint32_t)(w >> 32);
        if ((hi >> 2) != tag) continue;  // not written in this scan yet
        excl = sat_add(excl, (uint32_t)w);
        if ((hi & 3u) == LB_INC) break;
        --q;
    }
    bin_st_relaxed(&lookback[part], bin_pack(tag, LB_INC, sat_add(excl, total)));
    return excl;
}

// Grid-wide check-out of a scan launch: the last block advances the epoch and resets the
// partition ticket (every block calls it once, at its end).
SPL_INLINE void scan_check_out(uint32_t* __restrict__ state) {
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(&state[SC_DONE], 1u) == gridDim.x - 1) {
            state[SC_EPOCH] += 1u;
            state[SC_PART] = 0u;
            state[SC_DONE] = 0u;
            __threadfence();
        }
    }
}

// Writes the instances of each lane's Gaussian into [off, end) (tile ids in row-major order,
// value v). `ok`: the span is set up (else the range, if any, is filled with tile 0). Small
// Gaussians are written by their lane; `big` ones by the whole warp, one at a time. If the
// span yields fewer tiles than the range holds (a count/emit mismatch, which the tests rule
// out) the rest repeats the last tile, so the range is always defined. Warp-collective.
SPL_INLINE void emit_warp(bool active, bool ok, const span::Span& s, uint32_t v, uint32_t off, uint32_t end,
                          uint32_t grid_w, uint16_t* __restrict__ tile_key, uint32_t* __restrict__ inst_v) {
    const uint32_t lane = threadIdx.x & 31;
    const bool big = ok && (span::row_hi(s) - span::row_lo(s) >= EMIT_LANE_ROWS || end - off > EMIT_LANE_TILES);
    if (active && end > off && !big) {
        uint32_t pos = off, last = 0;
        if (ok) {
            for (int ty = span::row_lo(s); ty <= span::row_hi(s); ++ty) {
                int a, b;
                if (!span::row_tiles(s, ty, a, b)) continue;
                const uint32_t row = (uint32_t)ty * grid_w;
                for (int tx = a; tx <= b && pos < end; ++tx, ++pos) {
                    last = row + (uint32_t)tx;
                    tile_key[pos] = (uint16_t)last;
                    inst_v[pos] = v;
                }
            }
        }
        for (; pos < end; ++pos) {
            tile_key[pos] = (uint16_t)last;
            inst_v[pos] = v;
        }
    }
    uint32_t todo = __ballot_sync(FULL_MASK, big);
    while (todo) {
        const uint32_t src = __ffs(todo) - 1;
        todo &= todo - 1;
        span::Span g;
        g.mx = __shfl_sync(FULL_MASK, s.mx, src);
        g.my = __shfl_sync(FULL_MASK, s.my, src);
        g.A = __shfl_sync(FULL_MASK, s.A, src);
        g.B = __shfl_sync(FULL_MASK, s.B, src);
        g.inv_a = __shfl_sync(FULL_MASK, s.inv_a, src);
        g.kb = __shfl_sync(FULL_MASK, s.kb, src);
        g.det = __shfl_sync(FULL_MASK, s.det, src);
        g.ratio = __shfl_sync(FULL_MASK, s.ratio, src);
        g.px0 = __shfl_sync(FULL_MASK, s.px0, src);
        g.px1 = __shfl_sync(FULL_MASK, s.px1, src);
        g.py0 = __shfl_sync(FULL_MASK, s.py0, src);
        g.py1 = __shfl_sync(FULL_MASK, s.py1, src);
        const uint32_t gv = __shfl_sync(FULL_MASK, v, src);
        const uint32_t gend = __shfl_sync(FULL_MASK, end, src);
        uint32_t gpos = __shfl_sync(FULL_MASK, off, src);
        uint32_t glast = 0;
        for (int r0 = span::row_lo(g); r0 <= span::row_hi(g); r0 += 32) {
            const int ty = r0 + (int)lane;
            int a = 0, b = -1;
            if (ty <= span::row_hi(g) && !span::row_tiles(g, ty, a, b)) b = a - 1;
            const uint32_t c = (uint32_t)(b - a + 1);
            const uint32_t incl = bin_warp_incl_scan(c, lane);
            const uint32_t first = (uint32_t)ty * grid_w + (uint32_t)a;  // the row's first tile id
            const uint32_t p0 = gpos + incl - c;
            uint32_t rows = __ballot_sync(FULL_MASK, c > 0);
            while (rows) {
                const uint32_t r = __ffs(rows) - 1;
                rows &= rows - 1;
                const uint32_t rc = __shfl_sync(FULL_MASK, c, r);
                const uint32_t rp = __shfl_sync(FULL_MASK, p0, r);
                const uint32_t rt = __shfl_sync(FULL_MASK, first, r);
                for (uint32_t j = lane; j < rc && rp + j < gend; j += 32) {
                    tile_key[rp + j] = (uint16_t)(rt + j);
                    inst_v[rp + j] = gv;
                }
                glast = rt + rc - 1;
            }
            gpos += __shfl_sync(FULL_MASK, incl, 31);
        }
        for (uint32_t p = gpos + lane; p < gend; p += 32) {
            tile_key[p] = (uint16_t)glast;
            inst_v[p] = gv;
        }
    }
}

// ---------------------------------------------------------------- kernels

// v in [0, V): n_tiles[v] = the span count of rec[v] (the reference count).
extern "C" __global__ void count_tiles(const FrameCam* __restrict__ cam, const SplatRec* __restrict__ rec,
                                       uint32_t* __restrict__ n_tiles, const uint32_t* __restrict__ status) {
    const uint32_t n = status[ST_N_VIS];
    const uint32_t v = blockIdx.x * BIN_THREADS + threadIdx.x;
    if (v >= n) return;
    span::Span s;
    n_tiles[v] = span::setup(span::load(rec, v), cam->width, cam->height, s) ? span::count(s) : 0u;
}

// k in [0, V): inst_off[k] = exclusive scan of n_tiles[sorted_v[k]] (single pass with
// decoupled look-back); inst_off[V] and status[ST_N_INST] get the true total, and
// OVF_INSTANCES is set when it exceeds i_cap.
extern "C" __global__ void scan_tiles(const uint32_t* __restrict__ sorted_v, const uint32_t* __restrict__ n_tiles,
                                      uint32_t i_cap, uint32_t* __restrict__ inst_off, uint64_t* __restrict__ lookback,
                                      uint32_t* __restrict__ state, uint32_t* __restrict__ status) {
    __shared__ uint32_t s_buf[SCAN_PART + SCAN_PART / 32];
    __shared__ uint32_t s_wsum[SCAN_THREADS / 32];
    __shared__ uint32_t s_part, s_tag, s_excl;
    if (status[ST_OVERFLOW]) return;
    const uint32_t n = status[ST_N_VIS];
    const uint32_t parts = min((n + SCAN_PART - 1) / SCAN_PART, gridDim.x);
    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (n == 0 && blockIdx.x == 0 && tid == 0) {
        inst_off[0] = 0;
        status[ST_N_INST] = 0;
    }
    if (blockIdx.x < parts) {
        if (tid == 0) {
            s_part = atomicAdd(&state[SC_PART], 1u);
            s_tag = state[SC_EPOCH] + 1u;
        }
        __syncthreads();
        const uint32_t part = s_part, tag = s_tag & 0x3fffffffu;
        const uint32_t base = part * SCAN_PART;
        // Coalesced loads of the sorted indices, gather of their counts.
#pragma unroll
        for (int i = 0; i < SCAN_IPT; ++i) {
            const uint32_t j = i * SCAN_THREADS + tid;
            const uint32_t k = base + j;
            s_buf[pad32(j)] = k < n ? n_tiles[sorted_v[k]] : 0u;
        }
        __syncthreads();
        uint32_t x[SCAN_IPT], sum = 0;
#pragma unroll
        for (int i = 0; i < SCAN_IPT; ++i) {
            x[i] = s_buf[pad32(tid * SCAN_IPT + i)];
            sum = sat_add(sum, x[i]);
        }
        // Block scan of the thread sums.
        const uint32_t incl = bin_warp_incl_scan(sum, lane);
        if (lane == 31) s_wsum[warp] = incl;
        __syncthreads();
        uint32_t woff = 0, total = 0;
#pragma unroll
        for (uint32_t w = 0; w < SCAN_THREADS / 32; ++w) {
            if (w < warp) woff = sat_add(woff, s_wsum[w]);
            total = sat_add(total, s_wsum[w]);
        }
        if (tid == 0) {
            const uint32_t excl = scan_lookback(lookback, part, tag, total);
            s_excl = excl;
            if (part == parts - 1) {
                const uint32_t all = sat_add(excl, total);
                inst_off[n] = all;
                status[ST_N_INST] = all;
                if (all > i_cap) atomicOr(&status[ST_OVERFLOW], OVF_INSTANCES);
            }
        }
        __syncthreads();
        uint32_t run = sat_add(s_excl, sat_add(woff, incl - sum));
#pragma unroll
        for (int i = 0; i < SCAN_IPT; ++i) {
            s_buf[pad32(tid * SCAN_IPT + i)] = run;
            run = sat_add(run, x[i]);
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < SCAN_IPT; ++i) {
            const uint32_t j = i * SCAN_THREADS + tid;
            if (base + j < n) inst_off[base + j] = s_buf[pad32(j)];
        }
    }
    scan_check_out(state);
}

// k in [0, V): v = sorted_v[k]; recompute the tile rows from rec[v]; write tile_key = t
// (u16) and inst_v = v at inst_off[k] + j, tiles in row-major order; exactly
// inst_off[k + 1] - inst_off[k] entries (see emit_warp).
extern "C" __global__ void emit(const FrameCam* __restrict__ cam, const uint32_t* __restrict__ sorted_v,
                                const SplatRec* __restrict__ rec, const uint32_t* __restrict__ inst_off,
                                uint16_t* __restrict__ tile_key, uint32_t* __restrict__ inst_v,
                                const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    const uint32_t n = status[ST_N_VIS];
    const uint32_t k = blockIdx.x * BIN_THREADS + threadIdx.x;
    const bool active = k < n;
    if (__ballot_sync(FULL_MASK, active) == 0) return;
    uint32_t v = 0, off = 0, end = 0;
    span::Span s{};
    bool ok = false;
    if (active) {
        v = sorted_v[k];
        off = inst_off[k];
        end = inst_off[k + 1];
        ok = end > off && span::setup(span::load(rec, v), cam->width, cam->height, s);
    }
    emit_warp(active, ok, s, v, off, end, cam->grid_w, tile_key, inst_v);
}

// scan_tiles and emit in one pass (the training path): partition p = Gaussians
// [256 p, 256 p + 256) in depth order. Each thread gathers its record once, counts its
// tiles, the block scans the counts and looks back for its global offset, then writes
// inst_off and the instances. A partition whose instances would end past i_cap writes
// none; the last partition reports the true total (status[ST_N_INST], inst_off[V]) and
// sets OVF_INSTANCES, so later kernels skip the iteration.
extern "C" __global__ void scan_emit(const FrameCam* __restrict__ cam, const uint32_t* __restrict__ sorted_v,
                                     const SplatRec* __restrict__ rec, uint32_t i_cap, uint32_t* __restrict__ inst_off,
                                     uint16_t* __restrict__ tile_key, uint32_t* __restrict__ inst_v,
                                     uint64_t* __restrict__ lookback, uint32_t* __restrict__ state,
                                     uint32_t* __restrict__ status) {
    __shared__ uint32_t s_wsum[BIN_THREADS / 32];
    __shared__ uint32_t s_part, s_tag, s_excl;
    if (status[ST_OVERFLOW]) return;
    const uint32_t n = status[ST_N_VIS];
    const uint32_t parts = min((n + BIN_THREADS - 1) / BIN_THREADS, gridDim.x);
    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (n == 0 && blockIdx.x == 0 && tid == 0) {
        inst_off[0] = 0;
        status[ST_N_INST] = 0;
    }
    if (blockIdx.x < parts) {
        if (tid == 0) {
            s_part = atomicAdd(&state[SC_PART], 1u);
            s_tag = state[SC_EPOCH] + 1u;
        }
        __syncthreads();
        const uint32_t part = s_part, tag = s_tag & 0x3fffffffu;
        const uint32_t k = part * BIN_THREADS + tid;
        const bool active = k < n;
        const uint32_t width = cam->width, height = cam->height, grid_w = cam->grid_w;
        uint32_t v = 0, c = 0;
        span::Span s{};
        bool ok = false;
        if (active) {
            v = sorted_v[k];
            ok = span::setup(span::load(rec, v), width, height, s);
            if (ok) c = span::count(s);
        }
        const uint32_t incl = bin_warp_incl_scan(c, lane);
        if (lane == 31) s_wsum[warp] = incl;
        __syncthreads();
        uint32_t woff = 0, total = 0;
#pragma unroll
        for (uint32_t w = 0; w < BIN_THREADS / 32; ++w) {
            if (w < warp) woff = sat_add(woff, s_wsum[w]);
            total = sat_add(total, s_wsum[w]);
        }
        if (tid == 0) {
            const uint32_t excl = scan_lookback(lookback, part, tag, total);
            s_excl = excl;
            if (part == parts - 1) {
                const uint32_t all = sat_add(excl, total);
                inst_off[n] = all;
                status[ST_N_INST] = all;
                if (all > i_cap) atomicOr(&status[ST_OVERFLOW], OVF_INSTANCES);
            }
        }
        __syncthreads();
        const uint32_t excl = s_excl;
        const uint32_t off = sat_add(excl, sat_add(woff, incl - c));
        if (active) inst_off[k] = off;
        // Uniform per block: write only when the whole partition fits.
        if (sat_add(excl, total) <= i_cap) emit_warp(active, ok, s, v, off, off + c, grid_w, tile_key, inst_v);
    }
    scan_check_out(state);
}

// One read of the I u16 keys: full MAX_TILES-bin histogram (smem, persistent blocks,
// flushed once per block) into hist[t], which bin_scan consumes and zeroes.
extern "C" __global__ void tile_hist(const uint16_t* __restrict__ tile_key, uint32_t* __restrict__ hist,
                                     const uint32_t* __restrict__ status) {
    __shared__ uint32_t sh[MAX_TILES];
    if (status[ST_OVERFLOW]) return;
    const uint32_t n = status[ST_N_INST];
    const uint32_t tid = threadIdx.x;
    for (uint32_t i = tid; i < MAX_TILES; i += HIST_THREADS) sh[i] = 0;
    __syncthreads();
    const uint32_t stride = gridDim.x * HIST_THREADS;
    const uint32_t n8 = n / 8;
    const uint4* k8 = reinterpret_cast<const uint4*>(tile_key);
    for (uint32_t i = blockIdx.x * HIST_THREADS + tid; i < n8; i += stride) {
        const uint4 q = k8[i];
        constexpr uint32_t M = MAX_TILES - 1;  // keys are < MAX_TILES; the mask keeps smem in bounds
        atomicAdd(&sh[q.x & M], 1u);
        atomicAdd(&sh[(q.x >> 16) & M], 1u);
        atomicAdd(&sh[q.y & M], 1u);
        atomicAdd(&sh[(q.y >> 16) & M], 1u);
        atomicAdd(&sh[q.z & M], 1u);
        atomicAdd(&sh[(q.z >> 16) & M], 1u);
        atomicAdd(&sh[q.w & M], 1u);
        atomicAdd(&sh[(q.w >> 16) & M], 1u);
    }
    for (uint32_t i = n8 * 8 + blockIdx.x * HIST_THREADS + tid; i < n; i += stride)
        atomicAdd(&sh[tile_key[i] & (MAX_TILES - 1)], 1u);
    __syncthreads();
    for (uint32_t i = tid; i < MAX_TILES; i += HIST_THREADS)
        if (sh[i]) atomicAdd(&hist[i], sh[i]);
}

// <<<1, 1024>>>: tile_start = exclusive scan of hist; buckets ceil(n_t / BUCKET), chains
// ceil(buckets / CHAIN_K) -> chain_off; total chains -> status[ST_N_CHAINS] (OVF_CHAINS
// over c_cap); tiles with instances -> status[ST_N_TILES_HIT]; both u16 radix passes'
// digit offsets from hist: digit_offsets[0][d] for the low lo_bits bits, [1][d] for the
// bits above them. Zeroes hist[0, n_tiles) for the next frame.
extern "C" __global__ void bin_scan(const FrameCam* __restrict__ cam, uint32_t* __restrict__ hist, uint32_t c_cap,
                                    uint32_t lo_bits, uint32_t* __restrict__ tile_start,
                                    uint32_t* __restrict__ chain_off, uint32_t* __restrict__ digit_offsets,
                                    uint32_t* __restrict__ status) {
    __shared__ uint32_t s_cnt[MAX_TILES];
    __shared__ uint32_t s_wi[BIN_SCAN_THREADS / 32], s_wc[BIN_SCAN_THREADS / 32], s_wh[BIN_SCAN_THREADS / 32];
    __shared__ uint32_t s_dig[RADIX_BINS];
    if (status[ST_OVERFLOW]) return;
    const uint32_t nt = min(cam->n_tiles, (uint32_t)MAX_TILES);
    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t t0 = tid * BIN_SCAN_IPT;
    uint32_t c[BIN_SCAN_IPT], inst = 0, chains = 0, hit = 0;
#pragma unroll
    for (int j = 0; j < BIN_SCAN_IPT; ++j) {
        const uint32_t t = t0 + j;
        c[j] = 0;
        if (t < nt) {
            c[j] = hist[t];
            hist[t] = 0;
        }
        s_cnt[t] = c[j];
        inst += c[j];
        chains += (c[j] + BUCKET * CHAIN_K - 1) / (BUCKET * CHAIN_K);
        hit += c[j] > 0;
    }
    const uint32_t ii = bin_warp_incl_scan(inst, lane), ic = bin_warp_incl_scan(chains, lane);
    const uint32_t ih = bin_warp_incl_scan(hit, lane);
    if (lane == 31) {
        s_wi[warp] = ii;
        s_wc[warp] = ic;
        s_wh[warp] = ih;
    }
    __syncthreads();
    uint32_t oi = 0, oc = 0, all_i = 0, all_c = 0, all_h = 0;
    for (uint32_t w = 0; w < BIN_SCAN_THREADS / 32; ++w) {
        if (w < warp) {
            oi += s_wi[w];
            oc += s_wc[w];
        }
        all_i += s_wi[w];
        all_c += s_wc[w];
        all_h += s_wh[w];
    }
    uint32_t ri = oi + ii - inst, rc = oc + ic - chains;
#pragma unroll
    for (int j = 0; j < BIN_SCAN_IPT; ++j) {
        const uint32_t t = t0 + j;
        if (t < nt) {
            tile_start[t] = ri;
            chain_off[t] = rc;
        }
        ri += c[j];
        rc += (c[j] + BUCKET * CHAIN_K - 1) / (BUCKET * CHAIN_K);
    }
    if (tid == 0) {
        tile_start[nt] = all_i;
        chain_off[nt] = all_c;
        status[ST_N_CHAINS] = all_c;
        status[ST_N_TILES_HIT] = all_h;
        if (all_c > c_cap) atomicOr(&status[ST_OVERFLOW], OVF_CHAINS);
    }
    __syncthreads();
    // Pass 0 digit counts: tiles with the same low bits; exclusive scan over the digits.
    const uint32_t lo_mask = (1u << lo_bits) - 1u;
    if (tid < RADIX_BINS) {
        uint32_t d = 0;
        if (tid <= lo_mask)
            for (uint32_t t = tid; t < nt; t += lo_mask + 1) d += s_cnt[t];
        s_dig[tid] = d;
    }
    __syncthreads();
    if (warp == 0) {
        uint32_t v[RADIX_BINS / 32], sum = 0;
#pragma unroll
        for (int j = 0; j < RADIX_BINS / 32; ++j) {
            v[j] = s_dig[lane * (RADIX_BINS / 32) + j];
            sum += v[j];
        }
        uint32_t run = bin_warp_incl_scan(sum, lane) - sum;
#pragma unroll
        for (int j = 0; j < RADIX_BINS / 32; ++j) {
            digit_offsets[lane * (RADIX_BINS / 32) + j] = run;
            run += v[j];
        }
    }
    // Pass 1: digit d covers tiles [d << lo_bits, (d + 1) << lo_bits), so its offset is a
    // tile_start entry (the stable first pass keeps that order within each digit).
    // (__syncthreads makes the block's global writes of tile_start visible to the block.)
    __syncthreads();
    if (tid < RADIX_BINS) digit_offsets[RADIX_BINS + tid] = tile_start[min(tid << lo_bits, nt)];
}
