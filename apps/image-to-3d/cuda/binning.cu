// Tile binning: instance offsets, emission, tile histogram, tile/chain scan.
// Owner: sort (E2). Spec: spec-raster Section 3.1 (kernels 6-9) and 2.8 (tile span).
//
// `emit` must reproduce preprocess_fwd's per-row tile count bit for bit: both call
// band_x_interval (splat_common.cuh, E1) on the stored record fields.
//
// STUBS: every kernel below has its final signature and an empty body.

// k in [0, V): inst_off[k] = exclusive scan of n_tiles[sorted_v[k]] (single pass with
// look-back); writes the true total to status[ST_N_INST] and sets OVF_INSTANCES when it
// exceeds i_cap.
extern "C" __global__ void scan_tiles(const uint32_t* __restrict__ sorted_v, const uint32_t* __restrict__ n_tiles,
                                      uint32_t i_cap, uint32_t* __restrict__ inst_off,
                                      uint32_t* __restrict__ lookback, uint32_t* __restrict__ part_ctr,
                                      uint32_t* __restrict__ status) {}

// k in [0, V): v = sorted_v[k]; recompute the tile rows from rec[v]; write tile_key = t
// (u16) and inst_v = v at inst_off[k] + j, tiles in row-major order.
extern "C" __global__ void emit(const FrameCam* __restrict__ cam, const uint32_t* __restrict__ sorted_v,
                                const SplatRec* __restrict__ rec, const uint32_t* __restrict__ inst_off,
                                uint16_t* __restrict__ tile_key, uint32_t* __restrict__ inst_v,
                                const uint32_t* __restrict__ status) {}

// One read of the I u16 keys: full MAX_TILES-bin histogram (smem, persistent blocks,
// flushed once per block).
extern "C" __global__ void tile_hist(const uint16_t* __restrict__ tile_key, uint32_t* __restrict__ hist,
                                     const uint32_t* __restrict__ status) {}

// <<<1, 1024>>>: tile_start = exclusive scan of hist; buckets ceil(n_t / BUCKET), chains
// ceil(buckets / CHAIN_K) -> chain_off; total chains -> status[ST_N_CHAINS] (OVF_CHAINS
// over c_cap); both u16 radix passes' digit offsets (low 8 bits, high 5 bits) from hist.
extern "C" __global__ void bin_scan(const FrameCam* __restrict__ cam, const uint32_t* __restrict__ hist,
                                    uint32_t c_cap, uint32_t* __restrict__ tile_start,
                                    uint32_t* __restrict__ chain_off, uint32_t* __restrict__ digit_offsets,
                                    uint32_t* __restrict__ status) {}
