// 3DGS-MCMC relocation, growth and position noise. Owner: densify (E6).
// Spec: spec-train Section 5.6. Formulas after Faster-GS kernels_mcmc.cuh (Apache-2.0);
// 3DGS-MCMC itself is Inria-licensed and read only.
//
// STUBS: every kernel below has its final signature and an empty body.

// Dead: o <= min_opacity or |q|^2 < 1e-8. Sources drawn with replacement p ~ o (binary
// search in the inclusive scan `cdf`), N_i = draws_i + 1 (capped at 50): o' and s' from
// the binomial table `coeffs`, written to the source and copied to the dead slot.
extern "C" __global__ void mcmc_relocate(const float* __restrict__ cdf, const uint32_t* __restrict__ dead,
                                         uint32_t n_dead, const float* __restrict__ coeffs, uint2 seed,
                                         uint32_t iteration, float* __restrict__ opacity,
                                         float* __restrict__ log_scale, uint32_t* __restrict__ draws) {}

// Appends n_add copies of sources drawn p ~ o with the same opacity/scale adjustment.
extern "C" __global__ void mcmc_add(const float* __restrict__ cdf, uint32_t n, uint32_t n_add,
                                    const float* __restrict__ coeffs, uint2 seed, uint32_t iteration,
                                    uint32_t* __restrict__ src, float* __restrict__ opacity,
                                    float* __restrict__ log_scale) {}

// After every optimizer step (captured in the iteration graph):
// mean += noise_scale * sigmoid(-100 (o - 0.005)) * Sigma eps, eps ~ N(0, I) (Philox).
extern "C" __global__ void mcmc_noise(const IterParams* __restrict__ params, const float* __restrict__ log_scale,
                                      const float4* __restrict__ quat, const float* __restrict__ opacity,
                                      float* __restrict__ mean, const uint32_t* __restrict__ status) {}
