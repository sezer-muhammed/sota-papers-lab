# image_to_3d.reconstruct: quality pass (mvp-quality)

Branch `w/mvp-quality` (merges `w/mvp-pipeline`). RTX 4070 SUPER. **All runs
below are UNLOCKED** (another agent's jobs were using 7-9 GB of the card;
runs waited for < 1.2 GB used before starting, but could overlap later):
PSNR numbers are valid, times are preliminary. Logs:
`~/spl-work/wt/mq-run/*.log` on the GPU host.

## 1. Coarse-to-fine test-time pose refinement (evaluation protocol)

Same trained model, held-out views (every 8th), TTR at 1/4 → 1/2 → full
resolution (150 / 150 / 200 iterations, LR 1e-3 decaying) instead of 200
full-resolution iterations. Baseline: mvp-pipeline's numbers (run C, same defaults).

| Scene / preset | old TTR (200 full-res) | 1/4 | +1/2 | +full (c2f, 500 it) | +300 more full | Δ |
|---|---|---|---|---|---|---|
| bicycle Fast | 21.47 | 21.82 | 22.42 | **22.51** | 22.52 | **+1.04** |
| bicycle Quality | 20.48 | 19.89 | 21.33 | **21.50** | 21.50 | **+1.02** |
| garden Fast | 24.68 | 22.73 | 24.59 | 24.67 | 24.67 | ±0 |

Converged after the coarse-to-fine stages (300 extra iterations add ≤ 0.01
dB). Now the harness default (`SPL_RECON_TTR_SCHED`). Bicycle's held-out
poses were stuck in a local minimum at full resolution; garden's were not.

## 2. Geometric (depth-consistency) pose refinement — `geom.rs`

Before training, every training frame's pose (and depth scale) is refined so
its engine depth agrees with its 8 best-overlapping neighbours' depth maps
(relative depth residual, Huber, block Jacobi Gauss-Newton, CPU, 4-6 s).
Engine-only accuracy against COLMAP (`gpu_geo_pose_accuracy`, median pairwise
relative rotation error, training frames):

| Scene | raw engine | geo 10 it | geo 10 it, no scale | geo 20 it tight | geo 10 it, 6k samples / 12 nbrs |
|---|---|---|---|---|---|
| bicycle (169 frames) | 0.591° | **0.501°** (6.4 s) | 0.503° | 0.526° | 0.501° (16 s) |

End to end on garden Fast (c2f TTR in every row):

| Config | Held-out PSNR | Gaussians | E2E (prelim.) |
|---|---|---|---|
| defaults (a1) | 24.67 | 431k | 94.9 s |
| + geo 10 it | 24.68 | 433k | 97.7 s |
| + geo + consistent init (≥ 2 views, 150k pts) | 24.65 | 432k | 86.3 s* |
| + geo + consistent init, 400k pts | 24.58 | 496k | 98.3 s |

\* engine load 1.2 s (resident weights); not comparable.

Pose accuracy improves by 15 % but photometric pose refinement during
training already removes that much: **no PSNR gain on garden**. The geo step
and the consistent init stay off by default (`geo_iters` = 0,
`init_min_views` = 0).

## 3. Quality worse than Fast on bicycle

Still true with c2f TTR: Quality 21.50 vs Fast 22.51 dB, although Quality
fits the training views far better (25.0 vs 21.8 dB). Quality grows 2.6M
Gaussians that absorb residual pose error (view-dependent overfit); its
held-out poses also start further off (Quality 1/4 stage 19.89 vs Fast
21.82). Not fixed here.

## Against the targets

| Target | Best measured here | Met |
|---|---|---|
| garden Fast ≥ 26.5 dB, ≤ 90 s | 24.68 dB, ~95 s | no |
| garden Quality ≥ 27.4 dB, ≤ 5 min | not re-run (26.09 at mvp-pipeline) | no |

## Open

- Room / truck / train and locked timing runs were not done (card busy).
- A real BA needs correspondences (a matcher or a track head); depth
  consistency alone is too weak a signal against photometric refinement.
- Quality regression: try Fast's densification budget for 30k iterations,
  or pose refinement with a Gaussian cap until poses converge.

---

# Round 2 (mvp-quality-2)

Branch `w/mvp-quality-2` (merges `w/mvp-quality`). RTX 4070 SUPER. Ablation
runs (sections 1-4) were **unlocked** (other agents' jobs on the card; each
run waited for < 3 GB used for 30 s): PSNR valid, times preliminary. The
final table (section 4) ran under `flock -w 300 ~/spl-work/gpu.lock` with the
card idle at the start of each run. Logs: `~/spl-work/wt/mq2-run/*.log`;
outputs `~/spl-work/out/image-to-3d/reconstruct/<tag>/result.json`. PSNR is
held-out (every 8th frame) after the coarse-to-fine test-time pose refinement
(1/4 → 1/2 → full, 500 iterations) unless stated. "RRE" = median pairwise
relative rotation error of the refined training cameras against the
dataset's COLMAP model (new in `gpu_reconstruct_protocol`).

## 1. Pipeline gap (COLMAP poses through `reconstruct`): it was the init

Garden Fast, COLMAP cameras injected (`SPL_RECON_GT_POSES=1`), pose
refinement and depth prior off, no test-time refinement. Baseline: the
trainer alone on COLMAP (poses and SfM points), Fast: **26.43 dB**.

| Init | Held-out PSNR |
|---|---|
| 150k engine points, 20 % low-confidence pixels dropped (old default) | 25.72 (mvp-quality gt4) |
| engine depth re-unprojected with the COLMAP cameras | 25.63 |
| + depth-consistent fused points (`init_min_views` 2) | 25.71 |
| voxel init (`init_points` 0) | 25.49 |
| **the COLMAP SfM points** (`SPL_RECON_GT_POINTS=1`) | **26.43 / 26.41** (= trainer alone) |
| engine points, no confidence drop | 26.15 / 26.21 |
| + distance-adaptive voxel merge (0.01 extent) | 26.24 |
| + 300k points | 26.27 |
| SfM points + joint pose refinement + depth prior + c2f TTR | 26.53 |

The whole 0.7-0.8 dB pipeline gap was the init. With the SfM points the
pipeline matches the trainer alone exactly (26.43 vs 26.43). The dropped
"low-confidence" pixels are the far background, which SfM covers. Keeping
them and merging per voxel (the voxel doubles every extent beyond the camera
sphere) closes all but 0.15 dB.

## 2. Pose gap: photometric refinement does not fix engine poses; a BA does

Engine poses, garden Fast. Photometric joint refinement leaves the training
cameras **0.64-0.89° RRE** from COLMAP, while from COLMAP poses they stay
at 0.027°: the Gaussians absorb the error. New `reconstruct/ba.rs`:
Shi-Tomasi keypoints (half resolution, one per grid cell, with engine depth)
→ depth-predicted pyramidal KLT into overlapping neighbours with a
forward-backward check → Levenberg-Marquardt over every frame's pose (training
frame 0 is the gauge), one shared log-focal and per-track inverse depth,
Huber 1 px, Schur complement, dense Cholesky; then a second round without
residuals above 3 px. All frames take part, held-out ones included (as COLMAP
would; their images never train). CPU, 5 s for 185 frames.

| Garden Fast, engine poses | RRE | PSNR |
|---|---|---|
| old defaults (mvp-quality) | 0.894° | 24.65 |
| + no conf drop, voxel merge (section 1) | 0.762° | 25.41 |
| + pose LR 3e-3 (1e-3: −0.19; 6e-3: −0.05; 2e-3 decay 0.03: −0.13) | 0.671° | 25.60 |
| per-image focal | — | 25.23 |
| 3000-iteration 1/4-resolution pose bootstrap | 0.638° | 25.66 |
| **+ BA**, 1024 keypoints, 6 neighbours, 20 iterations (22 s) | 0.205° | 26.22 |
| BA only, no joint refinement | 0.198° | 25.91 |
| BA, 2048 keypoints | 0.201° | 26.20 |
| **BA, 512 keypoints, 4 neighbours, 10 + 5 iterations, reject 3 px (5 s)** | 0.213° | 26.22 |
| init from BA tracks only / merged with engine points | 0.176° / 0.212° | 26.00 / 26.23 |
| **+ square pixels** from the long-side engine focal | 0.067° | 26.35 |
| + diverse BA neighbours (2 best + 2 spread over the 16 best) | 0.060° | 26.39 |

Square pixels: the engine's two fovs disagree; its fy was 3.1-3.6 % off on
bicycle and room (fx within 0.5 % of COLMAP), and the BA's single focal
scale kept that error. Diverse neighbours: with the 4 best only, T&T train
drifted (RRE 0.90° raw → 1.07° after BA, 16.2 dB after training); the
spread-out pair closes the loop (0.53°, 19.3 dB).

## 3. Stage ablations on the new defaults (garden Fast; Quality where noted)

| Change | PSNR | Time | Decision |
|---|---|---|---|
| defaults | 26.25 | — | — |
| depth prior off (was 0.1 → 0 over 30 %) | 26.26 | −3 s | **off** |
| PPISP on (static photos) | 26.25 (SSIM +0.008) | +31 s | off for photos (still on for video) |
| joint pose refinement off | 26.10 | −0 s | on |
| prune-first | off since mvp-pipeline (`prune_fraction` 0) | — | off |
| BA rounds 2 (re-track from adjusted cameras) | bicycle 24.50 vs 24.43 | +7 s | 1 round |
| BA pose prior toward engine poses (σ 0.3°/1°) | train unchanged (data-dominated) | 0 | off |
| Quality, 5000-iteration pose bootstrap (old default) | 27.16 vs **27.43** without | +20 s | **off** |
| Quality vs Fast on bicycle | 25.41 vs 24.48 (section 4) | | fixed |

## 4. End to end, final defaults, GPU lock held (commit a42a973)

JPEG files → `.spz` on disk (`SPL_RECON_*` harness, first run per process
includes the 7-8 s cold engine-weight load). Every run took the GPU lock
(`flock -w 300`) and waited for an idle card (< 1.2 GB used for 30 s). Held-out:
every 8th frame, c2f test-time pose refinement (not counted in the time).
"Before": mvp-pipeline's verified rows (ee8f1bc / 71ba782, same protocol with
200 full-resolution test-time iterations) and mvp-quality's c2f re-scores.

| Scene | Preset | PSNR | SSIM | End to end | RRE | Before | Δ |
|---|---|---|---|---|---|---|---|
| garden | Fast | **26.38** | 0.798 | 91.9 s | 0.061° | 24.68 (94 s) | +1.70 |
| garden | Quality | **27.62** | 0.860 | 256.0 s | 0.049° | 26.09 (295 s) | +1.53 |
| bicycle | Fast | 24.48 | 0.680 | 89.6 s | 0.164° | 21.47 (c2f 22.51) | +3.01 |
| bicycle | Quality | 25.41 | 0.768 | 268.1 s | 0.174° | 20.48 (c2f 21.50) | +4.93 |
| room | Fast | 30.02 | 0.881 | 127.5 s | 0.289° | — | — |
| room | Quality | 31.64 | 0.913 | 227.9 s | 0.308° | 29.98 (242 s) | +1.66 |
| truck | Fast | 24.27 | 0.836 | 82.6 s | 0.214° | 22.79 (84 s) | +1.48 |
| truck | Quality | 25.68 | 0.883 | 159.4 s | 0.192° | 23.21 (175 s) | +2.47 |
| train | Fast | 19.03 | 0.642 | 83.6 s | 0.546° | 19.55 (81 s) | −0.52 |
| train | Quality | 19.41 | 0.694 | 157.9 s | 0.600° | 20.15 (168 s) | −0.74 |

Against the targets and named baselines (same GPU unless stated):

| Target / baseline | Ours | Met |
|---|---|---|
| Quality ≥ 27.4 dB in ≤ 5 min (garden) | 27.62 dB in 256 s | **yes** |
| Fast ≥ 26.5 dB in ≤ 90 s (garden) | 26.38 dB in 91.9 s (about 86 s with the engine weights resident) | no: −0.12 dB, +1.9 s |
| our trainer on COLMAP poses, garden Quality 27.61 dB (185 s training, SfM excluded) | 27.62 dB, no SfM, 256 s end to end | parity |
| our trainer on COLMAP, bicycle 25.15 / room 32.14 dB | 25.41 / 31.64 dB | +0.26 / −0.50 |
| FastGS (COLMAP poses), garden 27.30 dB in 265 s / bicycle 24.82 in 190 s | 27.62 in 256 s / 25.41 in 268 s, poses included | ahead on PSNR |
| Brush room 32.42 dB in 722 s | 31.64 dB in 228 s | −0.78 dB, 3.2x faster |
| VGGT-X published, Mip-NeRF 360 average 26.40 dB (paper, other hardware, 9 scenes) | 28.22 dB mean of garden, bicycle, room (3 of 9 scenes) | ahead on these 3; not a like-for-like average |

Caveat on the protocol: test views get a c2f test-time pose refinement
against the trained model (as InstantSplat / VGGT-X do for COLMAP-free
methods); the COLMAP-pose baselines use the COLMAP test poses unrefined. On
COLMAP poses the same refinement adds about +0.2 dB (section 1: 26.31 →
26.53).

## Open

- Garden Fast misses 26.5 dB by 0.12 and 90 s by 1.9 s (cold weight load
  7.4 s; about 86 s warm).
- T&T train is worse than before (19.0 / 19.4 vs 19.6 / 20.2 dB) and far from
  target: RRE stays 0.55-0.60°, the BA alone does not improve the engine's
  0.90° there, and training collapses to ~70k Gaussians in Fast.
- Room (311 frames, two engine chunks): RRE 0.29-0.31°, −0.5 dB vs COLMAP.
- LPIPS not computed this round (renders are in each run's `renders/`).
