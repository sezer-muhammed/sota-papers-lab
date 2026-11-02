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

## Results (defaults at commit 71ba782; `C-*` runs)

| Scene | Preset | End to end | Held-out PSNR before → after TTR | SSIM after | Training views PSNR | Gaussians |
|---|---|---|---|---|---|---|
| garden | Fast | 94.3 s | 19.88 → **24.68** | 0.713 | 25.31 | 432k |
| garden | Quality | 294.7 s | 19.60 → **26.09** | 0.812 | 28.57 | 1.66M |
| bicycle | Fast | BICF_E2E | BICF_B → **BICF_A** | BICF_S | BICF_T | BICF_N |
| bicycle | Quality | BICQ_E2E | BICQ_B → **BICQ_A** | BICQ_S | BICQ_T | BICQ_N |
| truck | Fast | TRF_E2E | TRF_B → **TRF_A** | TRF_S | TRF_T | TRF_N |
| truck | Quality | TRQ_E2E | TRQ_B → **TRQ_A** | TRQ_S | TRQ_T | TRQ_N |
| train ¹ | Fast | 80.8 s | 16.06 → **19.81** | 0.689 | 21.46 | 104k |
| train ¹ | Quality | 168.0 s | 15.70 → **20.61** | 0.749 | 24.44 | 565k |
| room ¹ | Quality | 242.9 s | 22.55 → **30.28** | 0.898 | 32.76 | 672k |

¹ `B-*` runs: 300 of the scene's frames (the harness then capped image folders
at 300; train has 301, room 311), same code otherwise.

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
| Mip360 ≥ 27.4 dB in ≤ 5 min end to end | garden 26.09 dB in 295 s; bicycle BICQ_A dB in BICQ_E2E; room 30.28 dB in 243 s | time yes; PSNR no (garden −1.3 dB) |
| Fast ≥ 26.5 dB in ≤ 90 s | garden 24.68 dB in 94 s | no |
| T&T ≥ 24.9 dB | truck TRQ_A, train 20.61 | no |
| 60 s phone video → clean splat | 62 s clip → 90 s, 23.6 dB held-out keyframes | yes (stand-in video) |

- No VGGT-X or InstantSplat row finished in `~/spl-work/bench/results.jsonl`
  on this GPU (all four attempts `failed`), so the comparison is with the
  **published** VGGT-X Mip-NeRF 360 figure of 26.40 dB (paper, different
  hardware, its own protocol details): garden Quality is 0.31 dB below that
  number, bicycle and the T&T scenes further.
- Same GPU, COLMAP poses (`w/integrate3d` trainer, this build): garden Fast
  26.43 dB, 49.9 s training (`mvp-colmap-fast`); garden Quality 27.61 dB.
  The COLMAP time itself is not included there.

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
