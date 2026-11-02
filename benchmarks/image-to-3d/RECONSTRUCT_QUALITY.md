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
