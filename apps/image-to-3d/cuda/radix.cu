// Onesweep LSD radix sort for NVRTC (no CUB): upfront digit histograms, stable passes
// with match-any ranking and decoupled look-back. Owner: sort (E2).
// Spec: spec-raster Section 3.2 (Adinets & Merrill, "Onesweep", 2022).
//
// Used for the 32-bit depth sort over V, the 13-bit tile sort over I (digit offsets come
// from bin_scan, so no extra key read), and later the Morton re-sort over N.
// The item count is always read from device memory (`n`), never passed by value, so the
// launches can live in a captured CUDA graph.
//
// STUBS: every kernel below has its final signature and an empty body.

// One read of n keys, RADIX_BINS-bin histograms for each of `passes` digits (warp-private
// smem histograms, then global atomics); the last block exclusive-scans them in place.
extern "C" __global__ void radix_hist32(const uint32_t* __restrict__ keys, const uint32_t* __restrict__ n,
                                        uint32_t passes, uint32_t* __restrict__ digit_hist,
                                        uint32_t* __restrict__ done_ctr, const uint32_t* __restrict__ status) {}

// One stable pass over (u32 key, u32 value) for digit `pass`. SORT_THREADS x SORT_IPT
// items per partition; partitions taken in order from part_ctr[pass]; look-back state in
// lookback[pass][partition][RADIX_BINS] (2-bit flag | 30-bit count).
extern "C" __global__ void onesweep_u32(const uint32_t* __restrict__ keys_in, const uint32_t* __restrict__ vals_in,
                                        uint32_t* __restrict__ keys_out, uint32_t* __restrict__ vals_out,
                                        const uint32_t* __restrict__ n, uint32_t pass,
                                        const uint32_t* __restrict__ digit_offsets, uint32_t* __restrict__ lookback,
                                        uint32_t* __restrict__ part_ctr, const uint32_t* __restrict__ status) {}

// The same for 16-bit keys (tile ids, TILE_KEY_BITS = 13: two passes).
extern "C" __global__ void onesweep_u16(const uint16_t* __restrict__ keys_in, const uint32_t* __restrict__ vals_in,
                                        uint16_t* __restrict__ keys_out, uint32_t* __restrict__ vals_out,
                                        const uint32_t* __restrict__ n, uint32_t pass,
                                        const uint32_t* __restrict__ digit_offsets, uint32_t* __restrict__ lookback,
                                        uint32_t* __restrict__ part_ctr, const uint32_t* __restrict__ status) {}
