// Chained per-Gaussian raster backward. Owner: raster (E3).
// Spec: spec-raster Sections 5.1-5.2. Base: Taming's per-Gaussian backward as refined by
// Faster-GS (Apache-2.0); our change is chains of CHAIN_K buckets per warp with pixel
// state (A.rgb, T) in smem, active-pixel compaction and SoA atomics into g2d.
//
// STUBS: every kernel below has its final signature and an empty body.

// Persistent warps (BWD_WARPS per block) take chains from *chain_ctr. Per pixel constants
// come from the forward outputs and the loss: d_rgb[3][P], d_alpha[P], d_extra[E][P].
// Accumulates the 9 values (mx, my, A, B, C, o, r, g, b), plus abs (2) and extra (E),
// into g2d[9][v_cap], g2d_abs[2][v_cap], g2d_extra[E][v_cap].
extern "C" __global__ void raster_bwd(const IterParams* __restrict__ params, const FrameCam* __restrict__ cam,
                                      const SplatRec* __restrict__ rec, const uint32_t* __restrict__ inst_v,
                                      const uint32_t* __restrict__ tile_start, const uint32_t* __restrict__ chain_off,
                                      const uint32_t* __restrict__ tile_max_nc, const float* __restrict__ ckpt,
                                      const float* __restrict__ out_rgb, const float* __restrict__ out_T,
                                      const uint32_t* __restrict__ out_nc, const float* __restrict__ d_rgb,
                                      const float* __restrict__ d_alpha, const float* __restrict__ d_extra,
                                      uint32_t v_cap, float* __restrict__ g2d, float* __restrict__ g2d_abs,
                                      float* __restrict__ g2d_extra, uint32_t* __restrict__ chain_ctr,
                                      const uint32_t* __restrict__ status) {}
