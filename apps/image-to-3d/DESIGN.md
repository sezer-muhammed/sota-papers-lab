# image-to-3d: module contract

How the splat optimizer (Phase 3) is cut into files, who owns each, which buffers exist, who writes and reads them, and in what order one training iteration runs. Seven engineers work from this in parallel without editing each other's files.

Sources, in order of authority:

1. `reports/image-to-3d/spec-raster.md` (rasterizer: preprocess, sort, binning, blend, backward). Its names win for anything raster-owned.
2. `reports/image-to-3d/spec-train.md` (loss, Adam, densification, schedules, data, exports, Phase 4 math). Its semantics (Section 2.1) must hold.
3. `reports/Image to 3D.md` (design doc; git-excluded, read it from the main checkout).
4. This file: where the two specs disagree, the reconciliation in [Section 7](#7-reconciliations-between-the-two-specs) is binding.

Licences: Faster-GS, gsplat (Apache-2.0) and fused-ssim (MIT) may be adapted with `// Derived from <repo>@<commit> <file> (Apache-2.0)` (or MIT). Taming-3DGS, 3DGS, FastGS, StopThePop, Speedy-Splat, 3DGS-MCMC and DashGaussian are read-only: write our own code from the specs.

---

## 1. Owners

One owner per file. The header comment of every file names its owner. Change someone else's file only through them.

| Owner | Rust | CUDA | Delivers (spec section) |
|---|---|---|---|
| **E1 preprocess** | `kernels/mod.rs`, `kernels/layout.rs`, `kernels/preprocess.rs`, `kernels/selftest.rs` | `common.cuh`, `splat_common.cuh`, `preprocess.cu`, `selftest.cu` | Shared types and constants; `frame_begin`, `frame_end`; preprocess forward (raster 2) and backward with pose/focal gradients and sinks (raster 5.3-5.5); the shared math in `splat_common.cuh` (quaternions, SH, EWA, compensation, `band_x_interval`) |
| **E2 sort** | `kernels/sort.rs` | `radix.cu`, `binning.cu` | `RadixSorter` (onesweep, raster 3.2), `sort_depth`, `bin_tiles` (scan_tiles, emit, tile_hist, bin_scan; raster 3.1) |
| **E3 raster** | `kernels/raster.rs` | `raster_fwd.cu`, `raster_bwd.cu` | `raster_fwd` train/infer/count (raster 4), chained per-Gaussian `raster_bwd` (raster 5.1-5.2), `GpuRaster` (the `SplatRaster` impl), `count_masked` |
| **E4 loss** | `kernels/loss.rs`, `eval.rs`, `export/*` | `loss.cu`, `metrics.cu` | Fused L1 + D-SSIM v1 and v2, target resampling, VCD error map (train 3); PSNR/SSIM metrics and the eval protocol (train 7); PLY, SPZ v3/v4, COLMAP and transforms.json exports (train 9) |
| **E5 optimizer** | `kernels/adam.rs`, `reference.rs` | `adam.cuh`, `adam.cu` | Multi-group Adam with plans, sparsity modes, bf16 SR, LazyCatchUp, fused-Adam helpers (train 4); the CPU f64 reference rasterizer with finite differences and the committed gsplat fixtures (raster 6.3) |
| **E6 densify** | `kernels/densify.rs`, `model.rs` | `densify.cu`, `compact.cu`, `mcmc.cu` | ADC, FastGS VCD/VCP, MCMC (train 5); the store, compaction with emission, Morton re-sort, capacity (train 5.8-5.10) |
| **E7 trainer** | `gpu.rs`, `frame.rs`, `rng.rs`, `presets.rs`, `trainer.rs`, `ops.rs`, `refine.rs`, `io/*`, `testkit/*`, `lib.rs` | none | NVRTC cache, `FrameRing`, `PinnedStatus`, `IterationGraph`, overflow/replay, the loop and hooks, presets and schedules, kNN init, the ops, the test harness, `raster_bench` (raster 6.6) |

Every owner also writes the tests of their files (CPU tests anywhere, GPU tests `#[ignore]`).

Already implemented by the skeleton (owners maintain them): COLMAP bin/txt read and write, scene loading (split, intrinsics, extent, points), JPEG/PNG decode (libjpeg, PIL-exact) and area downscale, the GPU u8 image cache, the Gaussian store's allocation/upload/download, buffer allocation and the VRAM model, NVRTC compilation with a per-process cache, Philox, the struct layouts and their probes, the presets with the reference settings, the scaling rule, SH ramp and DashGaussian interpolation, step plans and LR schedules, densify schedules, the test harness, both ops' input handling. Everything else is a stub that returns `image-to-3d: <what> is not implemented yet (owner: <owner>; ...)`.

---

## 2. Files

```
apps/image-to-3d/
  DESIGN.md              this contract                                      E7
  README.md, paper.toml                                                     E7
  scripts/gpu-test.sh    build + run tests in the CUDA container            E7
  scripts/decode_parity.py  JPEG decode vs PIL                              E7
  cuda/
    common.cuh           ints, PTX (acquire/release, cp.async, f16/bf16), Philox, bf16 SR, warp, float3   E1
    splat_common.cuh     variant defaults, constants, status words, flags, structs, shared math          E1
    preprocess.cu        frame_begin, frame_end, preprocess_fwd, preprocess_bwd, cam_reduce             E1
    selftest.cu          layout/Philox/half/PTX probes (implemented)                                    E1
    radix.cu             radix_hist32, onesweep_u32, onesweep_u16                                       E2
    binning.cu           scan_tiles, emit, tile_hist, bin_scan                                          E2
    raster_fwd.cu        raster_fwd                                                                     E3
    raster_bwd.cu        raster_bwd                                                                     E3
    loss.cu              resample_target, ssim_fwd, ssim_bwd, ssim_fused, error_map                     E4
    metrics.cu           psnr_u8, ssim_u8                                                               E4
    adam.cuh             AdamGroupStep, GroupPtrs, adam_apply, grad_accumulate                          E5
    adam.cu              adam_step, adam_invisible, adam_catchup                                        E5
    densify.cu           vcd_mask, scores_add, densify_plan, gumbel_keys, opacity_clamp                 E6
    compact.cu           emit_scan, slab_load, slab_scatter, morton_codes, gather_words                 E6
    mcmc.cu              mcmc_relocate, mcmc_add, mcmc_noise                                            E6
  src/
    lib.rs ops.rs gpu.rs frame.rs rng.rs presets.rs trainer.rs refine.rs                                E7
    io/{colmap,scene,images,dataset_tests}.rs                                                           E7
    testkit/{npz,oracle,tol,synthetic,timing}.rs                                                        E7
    model.rs                                                                                            E6
    kernels/{mod,layout,preprocess,selftest}.rs                                                         E1
    kernels/sort.rs                                                                                     E2
    kernels/raster.rs                                                                                   E3
    kernels/loss.rs eval.rs export/{ply,spz,colmap}.rs                                                  E4
    kernels/adam.rs reference.rs                                                                        E5
    kernels/densify.rs                                                                                  E6
```

A compilation unit is `common.cuh + splat_common.cuh [+ adam.cuh] + <module>.cu` (`gpu::KernelSource`). Adding a kernel: write it in your `.cu`, add its name to your unit's `kernels` list, load it in your `*Kernels` struct. The test `every_listed_kernel_is_declared_in_its_source` keeps the lists and the sources in step; `gpu_every_unit_compiles_and_exports_its_kernels` compiles every unit and variant on the GPU.

---

## 3. Index spaces and buffers

| Space | Index | Capacity | Count lives in |
|---|---|---|---|
| Gaussians | p | `N_cap = Gaussians::cap` (multiple of 1024) | `Gaussians::n` (host), `FrameParams.n_gaussians` (device) |
| visible | v (order-preserving: `vis_idx[v]` increasing) | `V_cap = Capacity::visible` (= N_cap) | `status[ST_N_VIS]` |
| depth rank | k | V_cap | same |
| instances | i | `I_cap = Capacity::instances` | `status[ST_N_INST]` |
| tiles | t | `T_cap ≤ 8192` | `FrameCam.n_tiles` |
| chains | c | `C_cap = I_cap/(32K) + T_cap` (never overflows before I) | `status[ST_N_CHAINS]` |
| pixels | px, planar CHW | `P_cap` | `FrameParams.render_w × render_h` |

### 3.1 Persistent (N space)

| Buffer | Struct | Layout | Written by | Read by |
|---|---|---|---|---|
| `mean`, `log_scale`, `quat`, `opacity`, `sh_dc`, `sh_rest`, `filter3d` | `model::Gaussians` | `[cap][3]`, `[cap][3]`, `[cap][4]` (w,x,y,z raw), `[cap]` logit, `[cap][3]`, **`[45][cap]` planar: `(3k + c)·cap + p`**, `[cap]` | upload, Adam, compaction, MCMC | preprocess fwd/bwd, Adam, densify |
| `last_visible` | `Gaussians` | u32 `[cap]`: last iteration with ≥ 1 tile | **preprocess_fwd** | Adam (Overwrite mode, VisibleOnly), densify |
| gradients | `model::Grads` | same layouts as the parameters | preprocess_bwd (`GradSink::Write`) | Adam (zeroes Accumulate groups it steps) |
| moments | `adam::Moments` | per group `float2 (m, v)` at the parameter's element index (bf16 modes: 2 × bf16 in a u32) | Adam, fused Adam, compaction (zeroes new) | Adam |
| densify stats | `densify::DensifyStats` | f32 `[cap]` ×2 (+ abs) | preprocess_bwd when `FLAG_DENSIFY_STATS` | strategy; zeroed after each densify step |
| view scores | `densify::ViewScores` | u32 + f32 `[cap]` | `scores_add` after count-mode renders | FastGS strategy; zeroed per scoring round |

The host mirror `model::GaussiansHost` is compact (`n`) and stores SH rest per Gaussian as `[n][15][3]` (SPZ / Faster-GS order); `upload`/`download` transpose.

### 3.2 Per frame (`kernels::FrameBuffers` and friends)

| Buffer | Type, shape | Written by | Read by |
|---|---|---|---|
| `iter.params` | `IterParams` ×1 | frame_begin (from the ring) or `IterParamsDev::upload` (eager) | every kernel |
| `iter.cam` | `FrameCam` ×1 | frame_begin | preprocess, binning, raster, cam_reduce |
| `iter.status` | u32 `[16]`, `ST_*` | frame_begin zeroes counters; preprocess_fwd (N_VIS), scan_tiles (N_INST, OVERFLOW), bin_scan (N_CHAINS, OVERFLOW) | every kernel after scan_tiles (overflow guard), frame_end, host |
| `iter.iter_counter` | u32 | frame_end (+1) | frame_begin (ring slot) |
| `vis_mask` | u32 `[N_cap/32]` | preprocess_fwd | optional (Adam, densify) |
| `vis_idx` | u32 `[V_cap]`, v → p | preprocess_fwd | preprocess_bwd, scores_add |
| `rec` | `SplatRec` (48 B) `[V_cap]` | preprocess_fwd | emit, raster_fwd, raster_bwd, preprocess_bwd |
| `depth_key[0]`, `depth_val[0]` | u32 `[V_cap]` | preprocess_fwd (key = float bits of z, val = v) | sort_depth |
| `depth_key/val[depth_sorted]` | u32 `[V_cap]` | sort_depth | scan_tiles, emit (sorted v by rank k) |
| `n_tiles` | u32 `[V_cap]` | preprocess_fwd | scan_tiles |
| `inst_off` | u32 `[V_cap + 1]` by k | scan_tiles | emit |
| `tile_key[0]`, `inst_v[0]` | u16, u32 `[I_cap]` | emit | tile_hist, tile sort |
| `tile_key/inst_v[tile_sorted]` | u16, u32 `[I_cap]` | tile sort | raster_fwd, raster_bwd |
| `tile_hist`, `tile_start`, `chain_off` | u32 `[T_cap + 1]` | tile_hist, bin_scan | tile sort, raster_fwd, raster_bwd |
| `tile_digit_offsets` | u32 `[2][256]` | bin_scan | tile sort (no histogram pass) |
| `tile_max_nc` | u32 `[T_cap]` | raster_fwd (Train) | raster_bwd |
| `ckpt` | float4 `[C_cap][tile px]` (f16: half4) | raster_fwd (Train), one per chain | raster_bwd |
| `chain_ctr`, `chunk_ctr`, `chunk_lookback` | u32 | raster_bwd, preprocess_fwd | same (each stage memsets its own counters in its stage function, inside the graph) |
| `RenderOut::rgb, trans, n_contrib, extra` | f32 `[3][P]`, f32 `[P]`, u32 `[P]`, f32 `[E][P]` | raster_fwd | loss, raster_bwd, eval |
| `RenderOut::count` | u32 `[V_cap]` | raster_fwd (Count) | `scores_add` (densify.cu) |
| `PixelGrads::d_rgb, d_alpha, d_extra` | same shapes as RenderOut | loss | raster_bwd |
| `Grad2d::g` | f32 `[9][V_cap]` SoA: mx, my, A, B, C, o, r, g, b | preprocess_fwd zeroes rows v < V; raster_bwd accumulates | preprocess_bwd |
| `Grad2d::abs`, `extra` | f32 `[2][V_cap]`, `[E][V_cap]` | same | same |
| `CameraGrads::partial` → `pose`, `intr` | f32 `[blocks][10]` → `[n_images][6]`, `[n_intr][4]` | preprocess_bwd → cam_reduce | pose optimizer (Phase 4) |
| sorter scratch | `RadixSorter::{lookback, part_ctr, digit_hist, done_ctr}` | the sorts (memset per sort or epoch-tagged) | the sorts |
| target | f32 `[3][P]` | resample_target | loss |
| image pool | `io::images::ImageCache` u8 RGB interleaved + `CacheEntryDev` table | upload (once) | resample_target, eval |

Capacities and VRAM: `FrameBuffers::vram_bytes` models every per-frame buffer. Measured on the 4070 SUPER for N 1M, I 4M at 1297×840: model 367.6 MB, cudaMalloc'd 369.1 MB. For spec-raster 1.5's garden budget (N 6.5M, I 24M, K = 4) the model gives 0.78 GB per-v, 0.29 GB per-instance, 0.80 GB checkpoints (3.1 GB at K = 1), as the spec's table.

---

## 4. One training iteration

### 4.1 Host side (`Trainer::step`, E7)

```
t = last + 1
view      = ViewOrder::next_view()            Philox epoch shuffle of the training views
res_scale = schedule.res_scale(t)              DashGaussian (1 when off)
sh_degree = schedule.sh_degree(t)              +1 per sh_interval, max 3
plan      = adam.plan(t)                       due groups, step sizes, bias corrections (f64 → f32)
params    = trainer.iter_params(...)           one IterParams (Section 5)
ring.push(params)                              blocks only while slot t % 8 is in flight
key = GraphKey { n, render size, degree, due_mask, stats, abs_grad }
if key changed: re-capture record_iteration    CUDA-graph capture point
launch the graph (eager until E7 lands capture)
hooks(t): strategy.after_step (densify, prune, opacity clamp/reset, final prune), Morton, snapshot, eval
every ~100 t: lagging read of the loss sums and PinnedStatus; overflow → grow, drop graphs, replay from the first failed t
```

### 4.2 Device side (`record_iteration`, captured; one stream, no host sync)

| # | Stage function (owner) | Kernels | Reads | Writes |
|---|---|---|---|---|
| 1 | `preprocess::frame_begin` (E1) | frame_begin | ring slot `iter_counter % 8`, `CameraTable` | `iter.params`, `iter.cam`, status counters |
| 2 | `preprocess::preprocess_fwd` (E1) | preprocess_fwd | params, cam, Gaussians | vis_mask, `last_visible`, vis_idx, rec, depth_key/val[0], n_tiles, Grad2d rows, `ST_N_VIS` |
| 3 | `sort::sort_depth` (E2) | radix_hist32, 4 × onesweep_u32 | depth_key/val[0], `ST_N_VIS` | depth_key/val[depth_sorted] |
| 4 | `sort::bin_tiles` (E2) | scan_tiles, emit, tile_hist, bin_scan, 2 × onesweep_u16 | sorted v, n_tiles, rec, cam | inst_off, tile_key/inst_v, tile_hist/start, chain_off, digit offsets, `ST_N_INST`, `ST_N_CHAINS`, `ST_OVERFLOW` |
| 5 | `raster::raster_fwd(Train)` (E3) | raster_fwd | rec, sorted instances, tile_start, chain_off | RenderOut, ckpt, tile_max_nc |
| 6 | `loss::resample_target` (E4) | resample_target | image pool, params | target (u8 → f32 copy at scale 1, tent filter in warm-up; E4 may fold the copy into the loss kernel) |
| 7 | `Loss::forward_backward` (E4) | ssim_fused (or ssim_fwd + ssim_bwd) | RenderOut.rgb, target | PixelGrads, loss sums |
| 8 | `raster::raster_bwd` (E3) | raster_bwd | forward outputs, ckpt, tile_max_nc, PixelGrads | Grad2d (atomics) |
| 9 | `preprocess::preprocess_bwd` (E1) | preprocess_bwd | Gaussians, vis_idx, rec, Grad2d | `GradSink` (Grads, or params + moments), CameraGrads.partial, DensifyStats |
| 10 | `preprocess::cam_reduce` (E1, Phase 4) | cam_reduce | partials | pose/intr grads |
| 11 | `Adam::step` (E5) | adam_step (or adam_invisible after fused Adam) | plan, Grads, last_visible | params, moments, Grads (Accumulate: zeroed) |
| 12 | MCMC noise (E6, MCMC only) | mcmc_noise | params, Gaussians | mean |
| 13 | `preprocess::frame_end` (E1) | frame_end | status | pinned status mirror, `iter_counter + 1` |

In code: `GpuRaster::forward` runs 1-5, `GpuRaster::backward` 8-10, `SplatRaster::end_frame` 13, `Trainer::record_iteration` the whole sequence.

### 4.3 Rules every stage follows

- Stage functions only enqueue on the given stream. No `synchronize`, no allocation, no host readback, no per-iteration kernel arguments: anything that changes between iterations comes from `iter.params` / `iter.cam` / device counts. Sizes that are fixed per capacity (`n_cap`, `v_cap`) may be arguments.
- **Overflow:** `scan_tiles` and `bin_scan` write true totals and set `status[ST_OVERFLOW]` (`OVF_*` bits, sticky until the host clears it); **every kernel after them, including loss, Adam, densify statistics and MCMC noise, starts with `if (status[ST_OVERFLOW]) return;`** so the iteration has no side effect. The host grows the capacity to 1.25x the need, re-captures, rewinds and replays (view order and Philox draws are deterministic).
- Zeroing: frame_begin zeroes the status counters; preprocess_fwd zeroes Grad2d rows v < V; each stage memsets its own work counters and look-back state at its start (captured as memset nodes); Overwrite gradient groups need no zeroing (Adam reads `last_visible[i] == t`); Accumulate groups are zeroed by Adam when it steps them.
- Kernel launches carry `// SAFETY:` comments naming the kernel signature (raw-retouch style). Derived code carries the provenance line.
- Determinism: forward outputs bitwise identical run to run; backward identical in the `DETERMINISTIC` variant.

---

## 5. Device structs and conventions

All in `cuda/splat_common.cuh` (device) and their `#[repr(C)]` mirrors (`frame.rs`, `kernels/mod.rs`, `kernels/adam.rs`); sizes asserted at compile time, offsets checked on the GPU by `gpu_layout_probe_matches_rust` (37 fields, all agree).

| Struct | Bytes | Notes |
|---|---|---|
| `CamBase` | 64 | world→camera `R` (row-major) and `t` per image, `intr_id`, the size the intrinsics refer to |
| `Intr` | 16 | fx, fy, cx, cy per (camera, image size) |
| `FrameParams` | 64 | iteration, cam_index, n_gaussians, sh_degree, bg, eps2d, flags, adam_lr_scale, bias_c1/c2, **render_w, render_h, box_beta** |
| `FrameCam` | 128 | composed camera: R, t (with the pose delta), intrinsics scaled per axis to the render size, campos, near/far, J clip limits, size, tile grid, image index, intr_id |
| `IterParams` | 256 | `FrameParams` + target offset/size (u8 pool), res_scale, means_lr, noise_scale, Philox seed, `AdamIterPlan` |
| `AdamIterPlan` | 84 | step_size[6], bc2_sqrt_rcp[6], due_mask, overwrite_mask, elem_prefix[7] |
| `SplatRec` | 48 | mx, my, ca, cb, cc (conic pre-scaled for exp2), o_eff, r, g, b, pthr, z, clamp flags |
| `GroupPtrs` | 144 | device pointers of the 6 groups (param, grad, moments), passed by value |
| `CacheEntryDev` | 16 | image pool offset, width, height |

Conventions (both specs agree):

- COLMAP frames: `X_c = R X_w + t`, x right, y down, z forward; pixel (i, j) centre at (i + 0.5, j + 0.5); `u = fx·X/Z + cx`.
- Parameters raw: log scales, logit opacity, unnormalized quaternion (w, x, y, z); SH DC separate, rest planar on the device.
- 2D gradients use gsplat's scalar-B conic convention; densification gradients are in NDC units (`∂L/∂u·W/2`).
- Blend semantics default to gsplat's (α ≤ 0.99, skip α < 1/255, stop before T(1−α) ≤ 1e-4); `BLEND=1` is Faster-GS's.
- Pose perturbation is left-multiplied in the camera frame; ξ = (ρ, φ).
- Randomness: Philox-4x32-10, key = run seed, counter = (element, iteration, purpose, 0); purposes `RNG_*` (`common.cuh` / `kernels::layout`). `rng.rs` is the bit-identical CPU mirror (Random123 known answers and GPU equality tested).
- Group order everywhere: means, scales, rotations, opacities, sh0, sh_rest (`model::Group`, `GROUP_*`).
- Flags (`FLAG_*`): AA compensation, 3D filter, depth channel, abs-grad, densify stats, camera grads, random background, gradient through compensation.

NVRTC rules: no `#include` (the runtime container has no CUDA headers); compile for the device's `compute_XY` with `--use_fast_math -std=c++17` and the variant's `-D` defines (`TILE_H`, `CHAIN_K`, `CKPT_F16`, `REC32`, `EXTRA_CH`, `BLEND`, `DETERMINISTIC`; Adam: `ADAM_MOMENTS_BF16`). `offsetof` is unavailable (the layout probe uses address differences). The app needs compute capability 8.0+ (`cp.async`, bf16). PTX is cached per (unit, defines, arch) and modules per context for the life of the process; `gpu::compile_log()` reports every compilation (about 16-24 ms each for the stubs). `SPL_NVRTC_LINEINFO=1` adds `-lineinfo` for ncu.

---

## 6. Testing

| Layer | Where | How |
|---|---|---|
| CPU unit tests | every module | `cargo test -p spl-app-image-to-3d` on the Mac |
| CPU f64 reference | `reference.rs` (E5) | finite differences; every kernel owner compares against it |
| gsplat oracle | `testkit::oracle` + `~/spl-work/bin/gs-oracle` | direct (`Oracle::run`, native on the host) or staged (`stage`/`stage_ours` under `/out/image-to-3d/oracle`, then `gs-oracle batch` and `compare` on the host). Ops: `sh`, `project`, `isect`, `rasterize`, `ssim`, `adam`, `render` (schemas in `benchmarks/image-to-3d/ORACLE.md`) |
| tolerances | `testkit::tol` | `Tol::{IMAGE, TRANSMITTANCE, MEANS2D, CONIC, GRAD_2D, GRAD_3D, POSE}` = spec-raster 6.4 |
| scenes | `testkit::synthetic` | deterministic tiny scenes (default 256 Gaussians, 64×48, degree 3, 4 ring cameras) and their `render` oracle inputs |
| timing | `testkit::timing` | CUDA-event `GpuTimer`, `bench` (warm-up + reps, median/p10/p90), `gpu_snapshot` (nvidia-smi) |
| GPU tests | `#[ignore]` | on the host: `apps/image-to-3d/scripts/gpu-test.sh <checkout> [filter]` (datasets at `/datasets`, output at `/out/image-to-3d`); `SPL_GPU_LOCK=1` for timing runs |
| datasets | `io::dataset_tests` | all 13 benchmark scenes parse; garden image cache; PIL decode parity |

Until `reference.rs` lands, kernel owners are not blocked: the oracle gives independent inputs and outputs for every stage (`project` for preprocess, `isect` for binning, `rasterize` for the blend forward/backward, `render` end to end), and stage functions can be fed hand-filled `FrameBuffers` through `clone_htod`.

---

## 7. Reconciliations between the two specs

| Topic | spec-raster | spec-train | Here |
|---|---|---|---|
| Gaussian store | `Gaussians { cap, mean, log_scale, quat, opacity, sh_dc, sh_rest planar, filter3d }` | `{ n, capacity, means, ..., sh_rest [C][15][3], last_visible }` | Raster names and planar SH (train allowed the change before Phase 3), plus `n` and `last_visible`. Host mirror keeps train's names. |
| Per-iteration parameters | `FrameParams` (64 B) + device `FrameCam` from `CameraTable` | `IterParams` with a `CameraGpu` | `IterParams` = `FrameParams` + training fields; the camera is `cam_index` + the device-composed `FrameCam` (train's `CameraGpu` fields all exist there). Render size and FastGS's compact-box β moved into `FrameParams`' padding. |
| Device params location | inside `FrameRing` | `IterParamsDev` | `FrameBuffers::iter: IterParamsDev` (params, cam, status, counter); the ring only holds pinned slots. |
| Gradient routing | `enum GradSink { Write, FusedAdam }` + separate `cam`, `stats` args | `struct GradSink { grads, fused_adam, densify, camera }` | Raster's enum; the `SplatRaster` trait takes `GradTargets { sink, camera, stats }`. `FusedAdam` carries moments + options only. |
| `preprocess_fwd` signature | `(k, s, g: &Gaussians, f)` | forward writes `last_visible` | `(k, s, g: &mut Gaussians, f, g2d: &mut Grad2d)`: it stamps `last_visible` and zeroes the Grad2d rows the table says it zeroes. |
| `preprocess_bwd` signature | `g: &Gaussians` + `FusedAdam { params: &mut Gaussians, .. }` (cannot borrow both) | | `g: &mut Gaussians`; the fused sink updates it in place. |
| `RadixSorter` | `&self`, count as `&CudaSlice<u32>` | | `&mut self` (scratch is mutated), owns its radix kernels, count as a `CudaView` into `status`. |
| Camera gradients | `CameraGrads` (device buffers) | `CameraGrads { d_xi, d_f, d_c }` (10 floats) | Raster's name for the buffers; train's struct is `CameraGradSum`. |
| Count mode | `RenderOut::count[v]` | `count_masked(..., acc: &mut ViewScores)` in p-space | Raster writes v-space counts; `scores_add` (densify.cu) scatters them into `ViewScores` with the view weight. |
| Iteration end | "the last kernel increments iter_counter" | | an explicit `frame_end` kernel (status → pinned mirror, counter + 1). |
| `SplatRaster` | | `forward(gpu, g, params, opts)`, `backward(gpu, g, params, d_image, sink)` | `forward(gpu, g, cams, FrameSource::{Ring, Eager}, RenderMode)`, `params()`, `image()`, `output()`, `pixel_grads()`, `loss_io()` (split borrow), `backward(gpu, g, GradTargets)`, `count_masked`, `overflow_flag`, `end_frame`. `RenderOpts` folds into `IterParams`. |
| Densify windows | | "start 600 … end 14900" (ADC), "t > 500 ∧ t % 500 == 0 ∧ t < 15000" (FastGS) | One rule for all: `start < t < end ∧ t % every == 0` with start 500 and end 15000 (25000 MCMC); reproduces every reference set (tested: 144, 28, 144, 244 steps). |
| Train op | | `image_to_3d.train_colmap` | `image_to_3d.train` (dataset folder or uploaded zip; `dry_run` validates only) and `image_to_3d.render`, as the Phase 0/3 task asked. Later ops per the design doc (`pose`, `reconstruct`, `mesh`, `generate`, `bench`). |
| JPEG decode | | zune-jpeg unless PIL parity fails | Measured: zune-jpeg differs from PIL 11.3 (libjpeg-turbo) in 23.7% of samples, 9.07% by > 1 LSB (max 5) on 10 garden images. JPEGs therefore decode with libjpeg (mozjpeg, ISLOW + fancy upsampling): 0 differing samples of 32.7M. |

---

## 8. Measured so far (RTX 4070 SUPER, 1-2 Nov 2026)

| What | Result |
|---|---|
| NVRTC, all 12 units + 10 raster variants (stubs) | 53 compilations, 948 ms total, 15-36 ms each; 41 kernels load |
| Layout probe | 37 sizes/offsets agree Rust ↔ CUDA |
| VRAM model vs cudaMalloc (N 1M, I 4M, 1297×840) | 367.6 MB vs 369.1 MB |
| COLMAP garden parse (185 images, 1.96M 2D points, 138,766 points) | 23 ms; rewriting reproduces all three binary files byte for byte |
| Scene load, 13 benchmark scenes | 16-153 ms each (headers only); principal point at the image centre on all |
| Garden decode, 161 train images (16 threads) | 0.25 s libjpeg (0.19 s zune), 526 MB u8 |
| Pool upload (pageable) | 0.36 s, 1.5 GB/s (pinned staging is an E7 optimization) |
| D2D copy 512 MiB, CUDA events (`gpu_timer_measures_copy_bandwidth`) | 222-386 GB/s read+write, measured while a gsplat baseline held the GPU at 99%; spec-raster 0.1 measured 428 GB/s idle. Re-run with `SPL_GPU_LOCK=1` for a clean number |
| gsplat oracle, synthetic 256 Gaussians 64×48 | renders through our npz files; SSIM matches its float64 reference |
