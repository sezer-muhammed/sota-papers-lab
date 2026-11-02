// Per-iteration camera, preprocess forward and backward, camera-gradient reduction.
// Owner: preprocess (E1). Spec: spec-raster Sections 1.2-1.3, 2, 5.3-5.5.
// Derived code from Faster-GS (Apache-2.0) or gsplat (Apache-2.0) must carry
// `// Derived from <repo>@<commit> <file> (Apache-2.0)`. Never paste Inria-licensed code.
//
// STUBS: every kernel below has its final signature and an empty body.

// <<<1, 32>>>. Reads ring[*iter_counter % RING_SLOTS] over PCIe (mapped pinned memory),
// writes *params, composes *cam (W = exp(xi^) W0, intrinsics scaled to render size, clip
// limits, tile grid), zeroes status[ST_N_*] (ST_OVERFLOW stays sticky until the host clears it).
extern "C" __global__ void frame_begin(const IterParams* __restrict__ ring, const uint32_t* __restrict__ iter_counter,
                                       const CamBase* __restrict__ cams, const Intr* __restrict__ intr,
                                       const float* __restrict__ pose_delta, IterParams* __restrict__ params,
                                       FrameCam* __restrict__ cam, uint32_t* __restrict__ status) {}

// <<<1, 32>>>, the last kernel of an iteration: copies status into the mapped pinned mirror
// (ST_ITER = iteration) and increments *iter_counter.
extern "C" __global__ void frame_end(const IterParams* __restrict__ params, uint32_t* __restrict__ status,
                                     uint32_t* __restrict__ status_mirror, uint32_t* __restrict__ iter_counter) {}

// Persistent grid (SMs x PRE_BLOCKS_PER_SM x PRE_THREADS), chunks of 256 primitives from
// chunk_ctr. Culls, projects, evaluates SH, computes the tile span, compacts visible
// primitives in order into v-space (decoupled look-back over `lookback`), writes
// vis_idx, rec, depth_key0 = float bits of z, depth_val0 = v, n_tiles, zeroes g2d[.][v],
// sets vis_mask bit p and last_visible[p] = iteration, and status[ST_N_VIS].
extern "C" __global__ void preprocess_fwd(const IterParams* __restrict__ params, const FrameCam* __restrict__ cam,
                                          const float* __restrict__ mean, const float* __restrict__ log_scale,
                                          const float4* __restrict__ quat, const float* __restrict__ opacity,
                                          const float* __restrict__ sh_dc, const float* __restrict__ sh_rest,
                                          const float* __restrict__ filter3d, uint32_t n_cap, uint32_t v_cap,
                                          uint32_t* __restrict__ vis_mask, uint32_t* __restrict__ last_visible,
                                          uint32_t* __restrict__ vis_idx, SplatRec* __restrict__ rec,
                                          uint32_t* __restrict__ depth_key, uint32_t* __restrict__ depth_val,
                                          uint32_t* __restrict__ n_tiles, float* __restrict__ g2d,
                                          float* __restrict__ g2d_abs, float* __restrict__ g2d_extra,
                                          uint32_t* __restrict__ chunk_ctr, uint32_t* __restrict__ lookback,
                                          uint32_t* __restrict__ status) {}

// One thread per visible v (persistent grid). Recomputes forward quantities from the
// parameters and chains g2d back to mean, log_scale, quat, opacity, SH (5.3). `sink`
// selects Write (grad_* at p) or FusedAdam (moments + AdamIterPlan); pose/focal partials
// go to cam_partial[block][CAM_GRAD_N] (fixed block count: deterministic); densify
// statistics (vis_count, grad_sum, grad_abs_sum) are accumulated at p when non-null.
extern "C" __global__ void preprocess_bwd(const IterParams* __restrict__ params, const FrameCam* __restrict__ cam,
                                          const float* __restrict__ mean, const float* __restrict__ log_scale,
                                          const float4* __restrict__ quat, const float* __restrict__ opacity,
                                          const float* __restrict__ sh_dc, const float* __restrict__ sh_rest,
                                          const float* __restrict__ filter3d, uint32_t n_cap, uint32_t v_cap,
                                          const uint32_t* __restrict__ vis_idx, const SplatRec* __restrict__ rec,
                                          const float* __restrict__ g2d, const float* __restrict__ g2d_abs,
                                          const float* __restrict__ g2d_extra, uint32_t sink,
                                          float* __restrict__ grad_mean, float* __restrict__ grad_log_scale,
                                          float* __restrict__ grad_quat, float* __restrict__ grad_opacity,
                                          float* __restrict__ grad_sh_dc, float* __restrict__ grad_sh_rest,
                                          float* __restrict__ cam_partial, float* __restrict__ stat_vis_count,
                                          float* __restrict__ stat_grad_sum, float* __restrict__ stat_grad_abs_sum,
                                          const uint32_t* __restrict__ status) {}

// <<<1, 256>>>: sums cam_partial[n_blocks][CAM_GRAD_N] in fixed order into
// pose_grad[image][6] and intr_grad[intr_id][4] (5.4).
extern "C" __global__ void cam_reduce(const IterParams* __restrict__ params, const FrameCam* __restrict__ cam,
                                      const float* __restrict__ cam_partial, uint32_t n_blocks,
                                      float* __restrict__ pose_grad, float* __restrict__ intr_grad,
                                      const uint32_t* __restrict__ status) {}
