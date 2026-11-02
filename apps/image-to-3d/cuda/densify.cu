// Densification decisions: VCD masks and view scores, clone/split/prune flags, VCP
// sampling keys, opacity clamp and reset. Owner: densify (E6).
// Spec: spec-train Sections 5.1-5.4, 5.7. FastGS is read-only (Inria-derived):
// reimplement from the spec; Faster-GS (Apache-2.0) may be derived with a provenance note.
//
// STUBS: every kernel below has its final signature and an empty body.

// mask[p] = (err[p] - min) / (max - min) > tau (u8), err and minmax from loss.cu error_map.
extern "C" __global__ void vcd_mask(const float* __restrict__ err, const float* __restrict__ minmax, float tau,
                                    uint32_t n_pixels, uint8_t* __restrict__ mask) {}

// After a count-mode raster pass: count_p[vis_idx[v]] += count_v[v],
// score_p[vis_idx[v]] += view_weight * count_v[v]; zeroes count_v.
extern "C" __global__ void scores_add(const uint32_t* __restrict__ count_v, const uint32_t* __restrict__ vis_idx,
                                      const uint32_t* __restrict__ status, float view_weight,
                                      uint32_t* __restrict__ count_p, float* __restrict__ score_p) {}

// Per Gaussian: clone / split / prune flags from the statistics, scores and parameters
// (Adc, FastGs rules; all decisions on the pre-step indexing). Writes out_count (0 or 1),
// clone_flag, split_flag and the densify statistic used for capping.
extern "C" __global__ void densify_plan(const float* __restrict__ log_scale, const float4* __restrict__ quat,
                                        const float* __restrict__ opacity, const float* __restrict__ vis_count,
                                        const float* __restrict__ grad_sum, const float* __restrict__ grad_abs_sum,
                                        const uint32_t* __restrict__ count_p, const float* __restrict__ score_p,
                                        const float* __restrict__ rules, uint32_t n, uint32_t* __restrict__ out_count,
                                        uint8_t* __restrict__ clone_flag, uint8_t* __restrict__ split_flag,
                                        float* __restrict__ priority) {}

// Gumbel-top-k keys for in-densify VCP: key = ln w - ln(-ln u), u from Philox
// (RNG_VCP_GUMBEL), w = 1 / (1e-6 + 1 - prune_score), as sortable u32.
extern "C" __global__ void gumbel_keys(const float* __restrict__ score_p, const float* __restrict__ minmax,
                                       uint2 seed, uint32_t iteration, uint32_t n, uint32_t* __restrict__ keys,
                                       uint32_t* __restrict__ idx) {}

// opacity <- min(opacity, logit_max) for all Gaussians and zero the opacity moments.
extern "C" __global__ void opacity_clamp(float* __restrict__ opacity, float2* __restrict__ moments, float logit_max,
                                         uint32_t n) {}
