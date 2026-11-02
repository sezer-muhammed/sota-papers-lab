// Training loss 0.8 L1 + 0.2 (1 - SSIM), forward and backward in one pass over the
// render, plus target resampling and the VCD error map. Owner: loss (E4).
// Spec: spec-train Section 3. Design after fused-ssim (MIT, a7c48d6): write our own code
// and keep the note "design after fused-ssim (MIT)".
//
// Images are planar f32 [3][H][W] with H, W = IterParams.frame.render_h/w (the graph
// replays at a fixed size; sizes are never kernel arguments). Sums go to a 2-float device
// accumulator (sum |x - y|, sum S) read every ~100 iterations.
//
// STUBS: every kernel below has its final signature and an empty body.

// Cached u8 RGB (pool + IterParams.target_offset, target_w x target_h) -> f32 planar at
// the render size: copy at scale 1, separable antialiased tent filter otherwise (6.5).
extern "C" __global__ void resample_target(const IterParams* __restrict__ params, const uint8_t* __restrict__ pool,
                                           float* __restrict__ target, const uint32_t* __restrict__ status) {}

// v1 pass 1: five separable 11x11 Gaussian sums per channel, S and the three scaled
// partials a, b, d (planes [3][3][H][W]); block-reduced sums into `sums`.
extern "C" __global__ void ssim_fwd(const IterParams* __restrict__ params, const float* __restrict__ x,
                                    const float* __restrict__ y, float* __restrict__ partials,
                                    float* __restrict__ sums, const uint32_t* __restrict__ status) {}

// v1 pass 2: dL/dx = w*a + 2x (w*b) + y (w*d) + (l1/N) sign(x - y).
extern "C" __global__ void ssim_bwd(const IterParams* __restrict__ params, const float* __restrict__ x,
                                    const float* __restrict__ y, const float* __restrict__ partials,
                                    float* __restrict__ d_x, const uint32_t* __restrict__ status) {}

// v2: one kernel, 32x16 output tile, recomputes the forward over the halo in smem.
extern "C" __global__ void ssim_fused(const IterParams* __restrict__ params, const float* __restrict__ x,
                                      const float* __restrict__ y, float* __restrict__ d_x,
                                      float* __restrict__ sums, const uint32_t* __restrict__ status) {}

// Eval mode for VCD (5.4): e(p) = mean_c |x - y|, block min/max into minmax[2], and the
// view loss sums for E = 0.8 L1 + 0.2 (1 - SSIM).
extern "C" __global__ void error_map(const IterParams* __restrict__ params, const float* __restrict__ x,
                                     const float* __restrict__ y, float* __restrict__ err,
                                     float* __restrict__ minmax, float* __restrict__ sums) {}
