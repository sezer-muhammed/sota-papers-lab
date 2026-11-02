# image_to_3d.reconstruct: Phase 4 (COLMAP-free) results

Branch `w/mvp-pipeline` (agent mvp-pipeline). Op `image_to_3d.reconstruct`
(`apps/image-to-3d/src/reconstruct/`): photos or a video → A0 ingest → native
VGGT-Ω (poses, intrinsics, depth, confidence; chunked registration above 200
frames) → init (engine points, kNN scales; the Stage B voxel init is still
available) and optional prune-first VCP → joint optimization (Phase 3 trainer +
per-image SE(3) and shared log-focal Adam from the preprocess-backward camera
gradients, PPISP, scale-shift-invariant depth prior 0.1 → 0) → `.spz`, `.ply`,
`cameras.json`, COLMAP text, `transforms.json`. The cameras and the engine
point cloud are announced once the engine finishes; splat snapshots every 8 s.

All numbers: RTX 4070 SUPER (12 GB), `~/spl-work/gpu.lock` held and the card
otherwise idle at start (`quiet_wait`: < 1.2 GB used by other processes;
382 MiB were the resident `spl serve` and an idle ComfyUI). Harness:
`reconstruct::gpu_tests::gpu_reconstruct_protocol`; logs and `result.json`
in `~/spl-work/out/image-to-3d/reconstruct/<tag>/` and
`~/spl-work/wt/mvp-run/<tag>.log` on the GPU host.

## Protocol (nvs-colmapfree, Report section 6)

- Mip-NeRF 360 `images_4` (outdoor) / `images_2` (indoor), T&T truck/train
  (Inria `tandt_db`, 979×546); every 8th name-sorted image held out.
- Every image (train and test) goes through the engine together; held-out
  images are never trained on. After training, held-out cameras are carried
  along with their nearest training camera's refinement (training information
  only), then 200 iterations of test-time pose refinement on frozen Gaussians
  (pose gradients only, rotation LR 1e-3 → 1e-5, translation the same × the
  scene extent). **Both before and after** are reported.
- End to end = from reading the JPEG files to `splat.spz` on disk (ingest,
  engine weight load from the page cache, engine, init, training, export).
  Test-time refinement and evaluation are not inside it.
- PSNR / SSIM: our CPU metrics on the 8-bit renders (3DGS convention).
  LPIPS-VGG via `gs-metrics` where listed.

## Results (defaults at commit 71ba782; test-time refinement LR 1e-3)

| Scene | Preset | End to end | Held-out PSNR before → after TTR | SSIM / LPIPS-VGG after | Training views PSNR | Gaussians | Run |
|---|---|---|---|---|---|---|---|
| garden | Fast | 94.3 s | 19.88 → **24.68** | 0.713 / 0.284 | 25.31 | 432k | C |
| garden | Quality | 294.7 s | 19.60 → **26.09** | 0.812 / 0.156 | 28.57 | 1.66M | C |
| bicycle | Fast | 84.9 s | 16.81 → **21.47** | 0.492 / 0.437 | 21.78 | 385k | C |
| bicycle | Quality | 314.5 s | 16.31 → **20.48** | 0.458 / 0.358 | 25.04 | 2.60M | C |
| room | Quality | 242.2 s | 21.73 → **29.98** | 0.895 / 0.243 | 32.76 | 672k | F |
| truck | Fast | 83.5 s | 17.56 → **22.79** | 0.806 / 0.238 | 23.74 | 192k | C |
| truck | Quality | 174.6 s | 17.49 → **23.21** | 0.838 / 0.172 | 26.03 | 728k | C |
| train | Fast | 80.8 s | 15.80 → **19.55** | 0.685 / 0.372 | 20.99 | 103k | F |
| train | Quality | 167.6 s | 15.49 → **20.15** | 0.745 / 0.265 | 24.16 | 563k | F |

SSIM and LPIPS-VGG from `gs-metrics` (3DGS convention) on the saved renders;
its PSNR equals ours to the third decimal. Engine time: 19-21 s for 185-194
frames in one pass; 33.5 s (truck, 251) / 33.6 s (train, 301) / 43.3 s (room,
311) in two registered chunks; weight load 7.1 s each run (page cache warm,
fresh process). Run-to-run spread of identical configs: up to ±0.4 dB
(bicycle Fast 21.47 / 21.87 / 22.25 with TTR LR 1e-3 / 3e-3 / 3e-3).

With test-time refinement LR 3e-3 (`E-*`, `D-*`): garden 24.63 / 25.45,
bicycle 21.87-22.25 / 20.89, truck 22.62 / 23.10, train — / 20.05 (another
agent's unlocked 5 GB job started during this run), room — / 29.93 (Fast /
Quality). 1e-3 stays the default.

Phone-video stand-in (`B-video-garden`): the 185 garden `images_4` frames as a
62 s 1280×828 H.264 30 fps clip with a slow brightness (±4 %) and saturation
(±8 %) drift (ffmpeg `eq`, `eval=frame`), Fast preset with PPISP: 617 frames
decoded at 10 fps, 150 sharpest keyframes, **90.2 s to `.spz`** (ingest
2.7 s, engine 13.6 s + 7.1 s load, training 59.2 s). Engine cameras and the
point cloud announced at 24.0 s, splat snapshots at 38, 46, 55, 63, 71, 79
and 87 s. Held-out keyframes (every 8th): 21.11 → 23.57 dB.

### Against the targets and baselines

| Target (Report §9) | Measured | Met |
|---|---|---|
| Mip360 ≥ 27.4 dB in ≤ 5 min end to end | garden 26.09 dB in 295 s; bicycle 20.48 dB in 315 s; room 29.98 dB in 242 s | time: garden and room yes, bicycle 15 s over; PSNR no (garden −1.3 dB) |
| Fast ≥ 26.5 dB in ≤ 90 s | garden 24.68 dB in 94 s; bicycle 21.47 dB in 85 s; truck 22.79 dB in 84 s | time: bicycle and truck yes, garden 4 s over; PSNR no |
| T&T ≥ 24.9 dB | truck 23.21, train 20.15 (Quality) | no |
| 60 s phone video → clean splat | 62 s clip → 90 s, 23.6 dB held-out keyframes | yes (stand-in video) |

- No VGGT-X or InstantSplat row finished in `~/spl-work/bench/results.jsonl`
  on this GPU (all four attempts `failed`), so the comparison is with the
  **published** VGGT-X Mip-NeRF 360 figure of 26.40 dB (paper, different
  hardware, its own protocol details): garden Quality is 0.31 dB below that
  number, bicycle, room and the T&T scenes further below their COLMAP-pose
  numbers.
- Same GPU, COLMAP poses (`w/integrate3d` trainer; the COLMAP run time is
  not included): garden Fast 26.43 dB with 49.9 s of training
  (`mvp-colmap-fast`, this build); Quality garden 27.61, bicycle 25.15, room
  32.14 dB (ledger). FastGS garden 27.30 dB in 265 s, bicycle 24.82 dB in
  190 s; Brush room 32.42 dB in 722 s (`results.jsonl`). Our COLMAP-free
  numbers are 1.5 (garden) to 4.7 (bicycle) dB below the COLMAP-pose trainer,
  but need no SfM: garden goes from JPEG files to `.spz` in 295 s.

## Where the gap comes from (garden, Fast)

| Run | Held-out after TTR | Training views | Note |
|---|---|---|---|
| Engine poses, no pose refinement | 21.73 | 22.07 | `gs1-2` |
| first default: pose LR 1e-4 from 500 to the densify end | 23.03 | — | `g-fast1` |
| pose LR 1e-3 to the densify end | 24.48 | 25.16 | `gs1-0`; 3e-3: 24.41 |
| + whole run, decay to 0.1× | 24.52 | 25.32 | `gs2-0` |
| + sparse kNN init (150k engine points) | 24.60 | 25.28 | `rs1-0` (current Fast) |
| + pose bootstrap 5000 it, LR 3e-3 | 25.10 | 25.67 | `bs1-2`, +17 s (current Quality) |
| depth prior off | 24.47 | 25.18 | `gs1-3`: neutral |
| **COLMAP poses**, Sim(3)-aligned into the engine frame, dense init | 25.47 (before TTR) | 26.51 | `gt2-0` |
| **COLMAP poses**, sparse kNN init | 25.72 (before TTR) | 26.67 | `gt4-0` |
| COLMAP trainer (COLMAP points) | 26.43 | — | `mvp-colmap-fast` |

Pose accuracy against COLMAP (161 training frames, `posecmp.py`): pairwise
relative rotation error, median, **1.14°** raw engine → 0.89° after joint
refinement → 0.75° with the bootstrap (COLMAP poses through the same path:
0.005°). One frame stays 11° off in every run. Focal is accurate (964.7 px vs
COLMAP's 961.4 px). So about 1 dB of the Fast gap is pose accuracy that
photometric refinement does not remove, and about 0.7 dB is the
initialization (engine points vs SfM points, with perfect poses).

Other findings:
- `FrameBuffers::ckpt` was sized without the extra (expected-depth) channel:
  any run with `extra_ch = 1` failed in `raster_bwd`. Fixed in
  `kernels/mod.rs` (one line; raster owner please review).
- The camera gradients need `FLAG_CAM_GRAD` in the iteration's flags as well
  as a `CameraGrads` target; without the flag they are silently zero.
- `ingest_video`'s future is not `Send` (a std mpsc receiver is held across
  awaits): the op drives it on a blocking thread with its own runtime.
- A world rescale to COLMAP-like units changes nothing (25.48 vs 25.47).
- Eager iterations (needed while PPISP, pose refinement or the depth prior
  run) cost little: 15k eager iterations ran at 255 it/s versus 265 it/s on
  CUDA graphs at the same Gaussian count.

## Next steps (not done here)

1. Pose accuracy: global bundle adjustment on engine correspondences (tracks
   from the dense head or a matcher) before training, as VGGT-X does;
   photometric refinement alone plateaus near 0.75°.
2. Initialization: triangulated (multi-view consistent) engine points instead
   of per-frame depth unprojection.
3. Engine: the Phase 2 attention schedule (frame / merge4 layers) is not wired
   into this trunk yet (`InterAttn::Merge4` has no kernel path); with it the
   185-frame forward would drop from 18 s.
4. Memory: 200 frames per engine pass on an idle 12 GB card; larger inputs
   use chunked registration (32 anchor frames, Sim(3) on camera centres),
   about 2x the engine time.
