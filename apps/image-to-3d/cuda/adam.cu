// Multi-group Adam over SoA parameters, invisible-Gaussian handling, LazyCatchUp.
// Owner: optimizer (E5). Spec: spec-train Section 4. Compiled as
// common.cuh + splat_common.cuh + adam.cuh + adam.cu.
//
// One launch per iteration covers every due group: a flat range over
// sum_g n * floats(g) elements, the group found from IterParams.adam.elem_prefix;
// 4 consecutive elements per thread. Starts by checking status[ST_OVERFLOW].
//
// STUBS: every kernel below has its final signature and an empty body.

// Overwrite mode reads g = (last_visible[i] == t) ? grad : 0, so no zeroing pass;
// Accumulate mode zeroes the gradient of every element it steps. MCMC regularizer
// gradients (opacity, scale) are added here (dense) when reg[0], reg[1] are non-zero.
extern "C" __global__ void adam_step(const IterParams* __restrict__ params, GroupPtrs ptrs,
                                     const uint32_t* __restrict__ last_visible, uint32_t n_cap, float2 reg,
                                     const uint32_t* __restrict__ status) {}

// Dense mode with fused Adam in the backward: steps only Gaussians with
// last_visible[i] != t (g = 0).
extern "C" __global__ void adam_invisible(const IterParams* __restrict__ params, GroupPtrs ptrs,
                                          const uint32_t* __restrict__ last_visible, uint32_t n_cap,
                                          const uint32_t* __restrict__ status) {}

// LazyCatchUp (4.6): applies the closed-form effect of the skipped zero-gradient steps
// since last_update[i] (one u32 per Gaussian per step plan), with the last 256 per-step
// learning rates in lr_hist. `flush` = 1 brings every Gaussian up to date.
extern "C" __global__ void adam_catchup(const IterParams* __restrict__ params, GroupPtrs ptrs,
                                        uint32_t* __restrict__ last_update, const float* __restrict__ lr_hist,
                                        uint32_t n_cap, uint32_t flush, const uint32_t* __restrict__ status) {}
