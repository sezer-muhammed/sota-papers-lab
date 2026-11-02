// Tile rasterizer forward: train, infer and count modes. Owner: raster (E3).
// Spec: spec-raster Section 4 (gsplat blend semantics by default, BLEND_FASTERGS as a flag).
//
// One block per tile (TILE_W x TILE_H, RASTER_THREADS threads); warp w owns an 8x4
// subtile; batches of BATCH records in smem; exact ellipse-vs-subtile ballot through
// band_x_interval; chained checkpoints every BUCKET * CHAIN_K Gaussians (train mode).
//
// STUBS: every kernel below has its final signature and an empty body.

// mode: MODE_TRAIN (ckpt, out_nc, tile_max_nc), MODE_INFER (colour and T only),
// MODE_COUNT (count[v] += masked pixels the Gaussian blends into; warp-aggregated).
// Outputs are planar CHW: out_rgb[3][P] (background composited), out_T[P], out_nc[P],
// out_extra[EXTRA_CH][P].
extern "C" __global__ void raster_fwd(const IterParams* __restrict__ params, const FrameCam* __restrict__ cam,
                                      const SplatRec* __restrict__ rec, const uint32_t* __restrict__ inst_v,
                                      const uint32_t* __restrict__ tile_start, const uint32_t* __restrict__ chain_off,
                                      uint32_t mode, const uint8_t* __restrict__ mask, float* __restrict__ out_rgb,
                                      float* __restrict__ out_T, uint32_t* __restrict__ out_nc,
                                      float* __restrict__ out_extra, float* __restrict__ ckpt,
                                      uint32_t* __restrict__ tile_max_nc, uint32_t* __restrict__ count,
                                      const uint32_t* __restrict__ status) {}
