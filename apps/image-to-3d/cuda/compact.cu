// Order-preserving compaction with emission, Morton codes and gathers for every
// per-Gaussian array. Owner: densify (E6). Spec: spec-train Sections 5.8-5.9.
//
// In-place chunked compaction: survivor moves have dst <= src, so for each chunk [a, b)
// in increasing order the chunk is copied to the scratch slab, then scattered from it.
// Tail entries (clones, split children) are generated from the slab copy of their parent.
//
// STUBS: every kernel below has its final signature and an empty body.

// Exclusive scans of out_count (survivors) and clone + 2 split (tail) -> destinations.
extern "C" __global__ void emit_scan(const uint32_t* __restrict__ out_count, const uint8_t* __restrict__ clone_flag,
                                     const uint8_t* __restrict__ split_flag, uint32_t n, uint32_t* __restrict__ dst,
                                     uint32_t* __restrict__ tail_dst, uint32_t* __restrict__ totals) {}

// Copies `words` u32 per Gaussian of chunk [a, b) from `array` into `slab`.
extern "C" __global__ void slab_load(const uint32_t* __restrict__ array, uint32_t words, uint32_t stride, uint32_t a,
                                     uint32_t b, uint32_t* __restrict__ slab) {}

// Scatters survivors of the chunk from the slab to dst[i]; writes clones and split
// children (new mean from Philox RNG_SPLIT, log scale - ln 1.6) to tail_dst[i].
extern "C" __global__ void slab_scatter(const uint32_t* __restrict__ slab, uint32_t words, uint32_t stride,
                                        uint32_t a, uint32_t b, const uint32_t* __restrict__ out_count,
                                        const uint32_t* __restrict__ dst, const uint32_t* __restrict__ tail_dst,
                                        uint32_t column, uint32_t* __restrict__ array) {}

// 30-bit Morton codes (10 bits per axis) over the means' AABB (minmax from a reduction).
extern "C" __global__ void morton_codes(const float* __restrict__ mean, const float* __restrict__ aabb, uint32_t n,
                                        uint32_t* __restrict__ codes, uint32_t* __restrict__ idx) {}

// out[i] = in[perm[i]] for `words` u32 per Gaussian (one column group of the slab).
extern "C" __global__ void gather_words(const uint32_t* __restrict__ in, const uint32_t* __restrict__ perm,
                                        uint32_t words, uint32_t stride, uint32_t n, uint32_t* __restrict__ out) {}
