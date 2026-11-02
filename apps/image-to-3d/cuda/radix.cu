// Onesweep LSD radix sort for NVRTC (no CUB): upfront digit histograms, stable passes
// with match-any ranking and decoupled look-back. Owner: sort (E2).
// Spec: spec-raster Section 3.2 (Adinets & Merrill, "Onesweep", 2022). Written from the
// paper and the spec; no CUB code is used (CUB is not reachable from NVRTC anyway).
//
// Used for the 32-bit depth sort over V, the 13-bit tile sort over I (digit offsets come
// from bin_scan, so no extra key read), and later the Morton re-sort over N.
// The item count is always read from device memory (`n`), never passed by value, so the
// launches can live in a captured CUDA graph.
//
// No memsets between sorts. Every launch of a pass kernel covers the capacity (one block
// per possible partition; blocks past the device count only check in and leave), and the
// scratch resets itself:
//   - look-back words are 64-bit: high word = (epoch tag << 2) | flag, low word = count.
//     A word written by an earlier pass carries an older tag and reads as "not ready", so
//     the look-back buffer is never cleared;
//   - the last block of a pass to check in (state[OS_DONE]) advances the epoch and resets
//     the partition ticket for the next pass;
//   - radix_hist32's last block scans the accumulated histograms into digit_offsets and
//     zeroes the accumulators.
// The sorter must be used from one stream at a time (RadixSorter takes &mut self).

// ---------------------------------------------------------------- configuration
// sort.rs (RadixConfig) passes these as -D defines; the fallbacks are the spec's values.

#ifndef OS_THREADS
#define OS_THREADS SORT_THREADS
#endif
#ifndef OS_IPT32
#define OS_IPT32 SORT_IPT
#endif
#ifndef OS_IPT16
#define OS_IPT16 SORT_IPT
#endif
#ifndef OS_MIN_BLOCKS
#define OS_MIN_BLOCKS 2
#endif
#ifndef OS_HIST_THREADS
#define OS_HIST_THREADS 512
#endif
// 0: rank with __match_any_sync; 1: rank with an 8-ballot multisplit.
#ifndef OS_RANK_BALLOT
#define OS_RANK_BALLOT 0
#endif

#define OS_WARPS (OS_THREADS / 32)

// state words (RadixSorter::state; layout mirrored in sort.rs)
#define OS_EPOCH 0
#define OS_PART 1
#define OS_DONE 2
#define OS_HIST_DONE 3

// look-back flags (bits 0-1 of the high word)
#define LB_AGG 1u
#define LB_INC 2u

// Launch bounds sit on these declarations (`__global__ __launch_bounds__(...) void`), so
// each definition keeps the plain `extern "C" __global__ void` spelling the kernel-list test
// reads; the test gpu_kernels_carry_their_launch_bounds checks .maxntid in the PTX.
extern "C" __global__ __launch_bounds__(OS_HIST_THREADS) void radix_hist32(const uint32_t*, const uint32_t*, uint32_t,
                                                                           uint32_t*, uint32_t*, uint32_t*,
                                                                           const uint32_t*);
extern "C" __global__ __launch_bounds__(OS_THREADS, OS_MIN_BLOCKS) void onesweep_u32(
    const uint32_t*, const uint32_t*, uint32_t*, uint32_t*, const uint32_t*, uint32_t, uint32_t, const uint32_t*,
    uint64_t*, uint32_t*, const uint32_t*);
extern "C" __global__ __launch_bounds__(OS_THREADS, OS_MIN_BLOCKS) void onesweep_u16(
    const uint16_t*, const uint32_t*, uint16_t*, uint32_t*, const uint32_t*, uint32_t, uint32_t, const uint32_t*,
    uint64_t*, uint32_t*, const uint32_t*);

// ---------------------------------------------------------------- helpers

SPL_INLINE uint64_t os_ld_relaxed(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

SPL_INLINE void os_st_relaxed(uint64_t* p, uint64_t v) {
    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

SPL_INLINE uint64_t lb_pack(uint32_t tag, uint32_t flag, uint32_t count) {
    return ((uint64_t)((tag << 2) | flag) << 32) | count;
}

SPL_INLINE uint32_t os_warp_incl_scan(uint32_t x, uint32_t lane) {
#pragma unroll
    for (uint32_t o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(FULL_MASK, x, o);
        if (lane >= o) x += y;
    }
    return x;
}

// Grid-wide check-out: the last block of the launch advances the epoch and resets the
// partition ticket. Every block calls it exactly once, at its very end.
SPL_INLINE void os_check_out(uint32_t* state) {
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(&state[OS_DONE], 1u) == gridDim.x - 1) {
            state[OS_EPOCH] += 1u;
            state[OS_PART] = 0u;
            state[OS_DONE] = 0u;
            __threadfence();
        }
    }
}

// Lanes of this warp holding the same digit (stable: callers rank by lane order).
SPL_INLINE uint32_t os_peers(uint32_t d) {
#if OS_RANK_BALLOT
    uint32_t peers = FULL_MASK;
#pragma unroll
    for (int b = 0; b < RADIX_BITS; ++b) {
        const uint32_t bit = (d >> b) & 1u;
        const uint32_t vote = __ballot_sync(FULL_MASK, bit);
        peers &= bit ? vote : ~vote;
    }
    return peers;
#else
    return __match_any_sync(FULL_MASK, d);
#endif
}

// ---------------------------------------------------------------- histogram

// One read of n keys: the digit histograms of every pass of a key_bits-bit sort (block
// histograms in shared memory, then global atomics into hist_acc). The last block to finish
// writes the exclusive scans to digit_offsets[pass][RADIX_BINS] and zeroes hist_acc.
extern "C" __global__ void radix_hist32(const uint32_t* __restrict__ keys, const uint32_t* __restrict__ n_ptr, uint32_t key_bits,
                 uint32_t* __restrict__ hist_acc, uint32_t* __restrict__ digit_offsets,
                 uint32_t* __restrict__ state, const uint32_t* __restrict__ status) {
    __shared__ uint32_t sh[4 * RADIX_BINS];
    __shared__ uint32_t s_last;
    if (status != nullptr && status[ST_OVERFLOW]) return;
    const uint32_t n = *n_ptr;
    const uint32_t passes = (key_bits + RADIX_BITS - 1) / RADIX_BITS;
    const uint32_t tid = threadIdx.x;
    for (uint32_t i = tid; i < 4 * RADIX_BINS; i += OS_HIST_THREADS) sh[i] = 0;
    __syncthreads();

    const uint32_t stride = gridDim.x * OS_HIST_THREADS;
    const uint32_t n4 = n / 4;
    const uint4* k4 = reinterpret_cast<const uint4*>(keys);
    for (uint32_t i = blockIdx.x * OS_HIST_THREADS + tid; i < n4; i += stride) {
        const uint4 q = k4[i];
        for (uint32_t p = 0; p < passes; ++p) {
            const uint32_t s = p * RADIX_BITS;
            atomicAdd(&sh[p * RADIX_BINS + ((q.x >> s) & 0xffu)], 1u);
            atomicAdd(&sh[p * RADIX_BINS + ((q.y >> s) & 0xffu)], 1u);
            atomicAdd(&sh[p * RADIX_BINS + ((q.z >> s) & 0xffu)], 1u);
            atomicAdd(&sh[p * RADIX_BINS + ((q.w >> s) & 0xffu)], 1u);
        }
    }
    for (uint32_t i = n4 * 4 + blockIdx.x * OS_HIST_THREADS + tid; i < n; i += stride) {
        const uint32_t k = keys[i];
        for (uint32_t p = 0; p < passes; ++p) atomicAdd(&sh[p * RADIX_BINS + ((k >> (p * RADIX_BITS)) & 0xffu)], 1u);
    }
    __syncthreads();
    for (uint32_t i = tid; i < passes * RADIX_BINS; i += OS_HIST_THREADS)
        if (sh[i]) atomicAdd(&hist_acc[i], sh[i]);

    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(&state[OS_HIST_DONE], 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    // Last block: one warp per pass, 8 bins per lane.
    const uint32_t warp = tid >> 5, lane = tid & 31;
    if (warp < passes) {
        uint32_t c[RADIX_BINS / 32];
        uint32_t sum = 0;
#pragma unroll
        for (int j = 0; j < RADIX_BINS / 32; ++j) {
            uint32_t* p = &hist_acc[warp * RADIX_BINS + lane * (RADIX_BINS / 32) + j];
            c[j] = ld_volatile(p);
            *p = 0u;
            sum += c[j];
        }
        uint32_t run = os_warp_incl_scan(sum, lane) - sum;
#pragma unroll
        for (int j = 0; j < RADIX_BINS / 32; ++j) {
            digit_offsets[warp * RADIX_BINS + lane * (RADIX_BINS / 32) + j] = run;
            run += c[j];
        }
    }
    if (tid == 0) state[OS_HIST_DONE] = 0u;
}

// ---------------------------------------------------------------- one onesweep pass

// Stable pass over (key, value) on digit (key >> shift) & ((1 << bits) - 1), bits <= 8.
// digit_base[d]: global start of digit d (exclusive scan of the pass histogram).
// Items: warp-striped, item i of lane l in warp w at base + w*32*IPT + i*32 + l, so
// ranking in (i, lane) order inside a warp and warps in order keeps input order.
template <typename K, int IPT>
SPL_INLINE void os_pass(const K* __restrict__ keys_in, const uint32_t* __restrict__ vals_in,
                        K* __restrict__ keys_out, uint32_t* __restrict__ vals_out,
                        const uint32_t* __restrict__ n_ptr, uint32_t shift, uint32_t bits,
                        const uint32_t* __restrict__ digit_base, uint64_t* __restrict__ lookback,
                        uint32_t* __restrict__ state, const uint32_t* __restrict__ status) {
    constexpr uint32_t NT = OS_THREADS, NW = OS_WARPS, PART = NT * IPT;
    __shared__ uint32_t s_wc[NW * RADIX_BINS];  // per-warp digit counters, then warp offsets
    __shared__ uint32_t s_start[RADIX_BINS];    // block-local exclusive digit start
    __shared__ uint32_t s_dst[RADIX_BINS];      // global start of the run minus s_start
    __shared__ uint32_t s_buf[PART];            // staging: keys, then values, in block order
    __shared__ uint32_t s_wsum[RADIX_BINS / 32];
    __shared__ uint32_t s_part, s_tag;

    if (status != nullptr && status[ST_OVERFLOW]) return;  // uniform: the state stays as it is
    const uint32_t n = *n_ptr;
    const uint32_t parts = min((n + PART - 1) / PART, gridDim.x);
    const uint32_t tid = threadIdx.x;

    if (blockIdx.x < parts) {
        if (tid == 0) {
            s_part = atomicAdd(&state[OS_PART], 1u);
            s_tag = state[OS_EPOCH] + 1u;
        }
        for (uint32_t i = tid; i < NW * RADIX_BINS; i += NT) s_wc[i] = 0;
        __syncthreads();
        const uint32_t part = s_part, tag = s_tag;
        const uint32_t lane = tid & 31, warp = tid >> 5;
        const uint32_t mask = (1u << bits) - 1u;
        const uint32_t nbins = 1u << bits;
        const uint32_t base = part * PART;
        const uint32_t first = base + warp * 32 * IPT + lane;

        // Out-of-range items get all-ones keys: the largest digit, after every valid item.
        uint32_t key[IPT], pos[IPT];
#pragma unroll
        for (int i = 0; i < IPT; ++i) {
            const uint32_t idx = first + i * 32;
            key[i] = idx < n ? (uint32_t)keys_in[idx] : 0xffffffffu;
        }

        // Warp-local stable ranks: peers of the same digit, counted in a warp-private row.
        uint32_t* wc = s_wc + warp * RADIX_BINS;
        const uint32_t lt = lanemask_lt();
#pragma unroll
        for (int i = 0; i < IPT; ++i) {
            const uint32_t d = (key[i] >> shift) & mask;
            const uint32_t peers = os_peers(d);
            const uint32_t leader = __ffs(peers) - 1;
            uint32_t c = 0;
            if (lane == leader) {
                c = wc[d];
                wc[d] = c + __popc(peers);
            }
            c = __shfl_sync(FULL_MASK, c, leader);
            pos[i] = c + __popc(peers & lt);
            __syncwarp();
        }
        __syncthreads();

        // Per digit: warp offsets and the block total; publish the aggregate at once.
        uint32_t total = 0;
        if (tid < RADIX_BINS) {
#pragma unroll
            for (uint32_t w = 0; w < NW; ++w) {
                const uint32_t c = s_wc[w * RADIX_BINS + tid];
                s_wc[w * RADIX_BINS + tid] = total;
                total += c;
            }
            if (tid < nbins)
                os_st_relaxed(&lookback[(uint64_t)part * RADIX_BINS + tid],
                              lb_pack(tag, part == 0 ? LB_INC : LB_AGG, total));
        }
        // Block-exclusive scan of the totals over digits (threads 0..255).
        const uint32_t incl = os_warp_incl_scan(total, lane);
        if (warp < RADIX_BINS / 32 && lane == 31) s_wsum[warp] = incl;
        __syncthreads();
        if (tid < RADIX_BINS) {
            uint32_t off = 0;
            for (uint32_t w = 0; w < warp; ++w) off += s_wsum[w];
            s_start[tid] = off + incl - total;
        }
        __syncthreads();

        // Block-sorted slot of every item; stage the keys.
#pragma unroll
        for (int i = 0; i < IPT; ++i) {
            const uint32_t d = (key[i] >> shift) & mask;
            pos[i] += s_start[d] + wc[d];
            s_buf[pos[i]] = key[i];
        }
        // Values are only needed after the look-back: load them now to hide the latency.
        uint32_t val[IPT];
#pragma unroll
        for (int i = 0; i < IPT; ++i) {
            const uint32_t idx = first + i * 32;
            val[i] = idx < n ? vals_in[idx] : 0u;
        }

        // Decoupled look-back: sum predecessors' aggregates until an inclusive prefix.
        if (tid < nbins) {
            uint32_t excl = 0;
            if (part > 0) {
                uint32_t q = part - 1;
                for (;;) {
                    const uint64_t w = os_ld_relaxed(&lookback[(uint64_t)q * RADIX_BINS + tid]);
                    const uint32_t hi = (uint32_t)(w >> 32);
                    if ((hi >> 2) != (tag & 0x3fffffffu)) continue;  // not written in this pass yet
                    excl += (uint32_t)w;
                    if ((hi & 3u) == LB_INC) break;
                    --q;
                }
                os_st_relaxed(&lookback[(uint64_t)part * RADIX_BINS + tid], lb_pack(tag, LB_INC, excl + total));
            }
            s_dst[tid] = digit_base[tid] + excl - s_start[tid];
        }
        __syncthreads();

        // Scatter from the staging buffer: consecutive slots, so runs of a digit coalesce.
        const uint32_t valid = min(PART, n - base);
        uint32_t dst[IPT];
#pragma unroll
        for (int j = 0; j < IPT; ++j) {
            const uint32_t s = tid + j * NT;
            dst[j] = 0;
            if (s < valid) {
                const uint32_t k = s_buf[s];
                dst[j] = s_dst[(k >> shift) & mask] + s;
                keys_out[dst[j]] = (K)k;
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < IPT; ++i) s_buf[pos[i]] = val[i];
        __syncthreads();
#pragma unroll
        for (int j = 0; j < IPT; ++j) {
            const uint32_t s = tid + j * NT;
            if (s < valid) vals_out[dst[j]] = s_buf[s];
        }
    }
    os_check_out(state);
}

// One stable pass over (u32 key, u32 value). OS_THREADS x OS_IPT32 items per partition;
// partitions taken in order from state[OS_PART]; look-back in lookback[partition][RADIX_BINS].
extern "C" __global__ void onesweep_u32(const uint32_t* __restrict__ keys_in, const uint32_t* __restrict__ vals_in,
                 uint32_t* __restrict__ keys_out, uint32_t* __restrict__ vals_out,
                 const uint32_t* __restrict__ n, uint32_t shift, uint32_t bits,
                 const uint32_t* __restrict__ digit_base, uint64_t* __restrict__ lookback,
                 uint32_t* __restrict__ state, const uint32_t* __restrict__ status) {
    os_pass<uint32_t, OS_IPT32>(keys_in, vals_in, keys_out, vals_out, n, shift, bits, digit_base, lookback,
                                state, status);
}

// The same for 16-bit keys (tile ids, TILE_KEY_BITS = 13: two passes).
extern "C" __global__ void onesweep_u16(const uint16_t* __restrict__ keys_in, const uint32_t* __restrict__ vals_in,
                 uint16_t* __restrict__ keys_out, uint32_t* __restrict__ vals_out,
                 const uint32_t* __restrict__ n, uint32_t shift, uint32_t bits,
                 const uint32_t* __restrict__ digit_base, uint64_t* __restrict__ lookback,
                 uint32_t* __restrict__ state, const uint32_t* __restrict__ status) {
    os_pass<uint16_t, OS_IPT16>(keys_in, vals_in, keys_out, vals_out, n, shift, bits, digit_base, lookback,
                                state, status);
}
