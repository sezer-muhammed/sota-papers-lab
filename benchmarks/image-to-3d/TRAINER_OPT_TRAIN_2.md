# Trainer optimisation pass, round 2 (opt-train-2), RTX 4070 SUPER, 2 Nov 2026

Driver `trainer_gpu_tests.rs`, COLMAP poses, test = every 8th image, PSNR/SSIM from our
in-app metrics. One seed per row. Round 1: `TRAINER_OPT_TRAIN.md`.

Conditions: every run tried `flock -x -w 300 ~/spl-work/gpu.lock` (LOCKED unless marked).
Other agents (mesh-mode-2, mvp-pipeline) ran 9 GB GPU jobs without the lock during this
pass. The runner (`run.sh`) waits for ≥ 8.5-10.5 GB free VRAM, kills and restarts a run
whose capacity was squeezed below 2M Gaussians or that hit an OOM, but cannot stop another
process starting mid-run, so times can be contended even when LOCKED.

## What changed

- **DashGaussian resolution warm-up implemented** (`src/dash.rs`, `DashSchedule::from_spectra`;
  spec-train 6.5): spectra of ≤ 32 evenly spaced training images (radix-2 FFT, zero-padded),
  32 energy-matched levels, full resolution from the densify end, means LR held until the
  first level below scale 2. On in `fast` only (`fast` already asked for it; the schedule
  was a stub before, so round-1 `fast` ran at full resolution).
- **Quality unchanged** from round 1: a suite with SH-500 + warm-up in Quality was ~20 %
  faster but lost 0.2-0.9 dB (bicycle 24.68, flowers 20.74, garden 27.40, stump 25.99,
  room 31.84, drjohnson 28.12), so it was reverted.
- **New `big` preset**: Quality with grad_abs 3e-4, loss τ 0.06, SH every 500, no warm-up.
- New ablation overrides: `antialiased`, `sh_interval`, `densify_end`, `reset_every`.

## Ablations

Garden (images_4), 30k, Quality base = round-1 Quality (grad 4e-4, dense 0.001, τ 0.1, SH 1000).

| Setting | Train s | PSNR | SSIM | #G |
|---|---|---|---|---|
| Quality (round 1, suite) | 182 | 27.61 | 0.863 | 1.27M |
| grad 3e-4 | 204 | 27.626 | 0.866 | 1.49M |
| FastGS-Big parity (garden table) | 266 | 27.660 | 0.873 | 2.11M |
| grad 3e-4, τ 0.06 | 253 | 27.688 | 0.872 | 2.11M |
| + no opacity reset | 259 | 27.625 | 0.872 | 2.22M |
| + antialiased | 275 | 27.520 | 0.863 | 1.98M |
| + compact box β 1 | 254 | 27.719 | 0.873 | 2.11M |
| + SH every 250 | 251 | 27.658 | 0.872 | 2.07M |
| **+ SH every 500** | 252 | **27.755** | 0.873 | 2.09M |
| + SH 500, highfeature LR 0.04 | 264 | 27.581 | 0.871 | 2.12M |
| + DashGaussian warm-up (SH 1000) | **213** | 27.702 | 0.872 | 2.08M |
| grad 2e-4, τ 0.06, SH 500, warm-up (capacity squeezed to 3.1M) | 260 | 27.551 | 0.871 | 2.54M |
| 50k iterations, grad 3e-4 | 542 | 27.594 | 0.865 | 1.40M |
| MCMC cap 3M | 746 | 27.239 | 0.860 | 3.00M |
| grad 8e-4, dense 0.005 | 151 | 27.446 | 0.853 | 0.91M |
| grad 6e-4, dense 0.005, SH 500, warm-up | 134 | 27.399 | 0.852 | 1.01M |

Fast (15k) with the warm-up: garden 39.6 s / 26.319 dB / 0.799 vs round-1 quiet run 50 s /
26.40 dB (-21 % time, -0.08 dB).

Per-scene densification budget, Quality with grad 8e-4 + dense 0.005 ("d5g8") vs round-1 Quality:

| Scene | round-1 Quality s / PSNR | d5g8 s / PSNR | Δ PSNR | Baseline |
|---|---|---|---|---|
| bicycle | 189 / 25.16 | 150 / 25.38 | +0.22 | Faster-GS fused 25.27 / 619 s; FastGS 24.82 / 190 s |
| flowers | 201 / 21.07 | 144 / 21.46 | +0.39 | FastGS 21.35 / 182 s |
| flowers + warm-up | | 125 / 21.45 | +0.38 | |
| stump | 154 / 26.27 | 119 / 26.72 | +0.45 | none |
| treehill | 178 / 21.91 | 134 / 22.50 | +0.59 | (round-1 Fast 22.68) |
| treehill dense 0.01 | | 158 / 22.19 (g 4e-4) ; 136 / 22.49 (g 8e-4) | | |
| garden | 182 / 27.61 | 151 / 27.45 | -0.16 | FastGS 27.30 / 265 s |
| room | 144 / 32.07 | 120 / 31.95 | -0.12 | Brush 32.42 / 722 s |
| kitchen | 296 / 31.91 | 184 / 31.50 | -0.41 | gsplat-MCMC 31.52 / 1126 s |

flowers dense 0.005 alone (grad 4e-4): 180 s / 21.10 dB, so the gain comes from the higher
gradient threshold. Mid threshold grad 6e-4 + dense 0.005 + SH 500 + warm-up: flowers
133 s / 21.28 dB, garden 27.40 dB.

Reading: vegetation-heavy outdoor scenes (bicycle, flowers, stump, treehill) overgrow at the
generic grad 4e-4 and gain 0.2-0.6 dB with 8e-4; garden and the indoor scenes lose 0.1-0.4 dB.
This is the same split as FastGS's per-scene table. No single generic threshold wins both
groups, so Quality keeps 4e-4 and the higher threshold is left as a per-scene override (see
open issues).

## Suite

Commit of this file; all rows LOCKED (lock held; other unlocked GPU users were active on
the host for much of the pass, so times may be inflated). Quality = round-1 rows (settings
unchanged). Fast and Big re-run here.

| Scene | Quality s / PSNR (r1) | Big s | PSNR | SSIM | #G | Fast s | PSNR | SSIM | Fast r1 s / PSNR | Baselines (PSNR / train s) |
|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | 189 / 25.16 | 233 | 25.18 | 0.769 | 2.80M | 33 | 24.07 | 0.673 | 51 / 24.56 | FastGS 24.82/190; Faster-GS fused 25.27/619 |
| flowers | 201 / 21.07 | 275 | 21.05 | 0.616 | 3.05M | 34 | 20.78 | 0.533 | 91 / 20.97 | FastGS 21.35/182 |
| garden | 182 / 27.61 | 252 | 27.67 | 0.872 | 2.09M | 40 | 26.33 | 0.799 | 107 / 26.39 | FastGS 27.30/265 |
| stump | 154 / 26.27 | 227 | 26.39 | 0.765 | 2.67M | 28 | 25.95 | 0.720 | 44 / 26.13 | none |
| treehill | 178 / 21.91 | 252 | 21.70 | 0.598 | 3.36M | 34 | 22.63 | 0.578 | 42 / 22.68 | none |
| room | 144 / 32.07 | 180 | 32.19 | 0.928 | 0.98M | 36 | 30.82 | 0.898 | 51 / 30.73 | Brush 32.42/722 |
| counter | 282 / 29.51 | 202 | 29.65 | 0.916 | 0.76M | 41 | 27.78 | 0.876 | 59 / 27.91 | none |
| kitchen | 296 / 31.91 | 288 | 32.23 | 0.934 | 1.03M | 51 | 30.07 | 0.898 | 71 / 30.41 | gsplat-MCMC 31.52/1126 |
| bonsai | 380 / 32.78 | 172 | 32.89 | 0.948 | 0.82M | 39 | 30.49 | 0.923 | 53 / 30.58 | gsplat-MCMC 32.58/1223 |
| truck | 247 / 26.11 | | | | | 24 | 24.93 | 0.856 | 31 / 24.95 | none |
| train | 104 / 21.88 | | | | | 24 | 21.05 | 0.755 | 31 / 21.30 | none |
| playroom | 205 / 29.91 | | | | | 25 | 30.41 | 0.905 | 35 / 30.12 | none |
| drjohnson | 209 / 29.01 | | | | | 26 | 28.83 | 0.882 | 38 / 29.34 | none |

Mip-NeRF 360 means: Quality 27.59 dB, **Big 27.66 dB** (all 9 runs ≤ 288 s), Fast 26.53 dB
(round-1 Fast 26.71). Fast is 1.3-2.7x faster than round 1 for -0.0 to -0.5 dB (playroom
and room gain); the round-1 Fast times were partly contended.

## Open issues

- Big target NOT met: garden 27.67 dB (best ablation 27.755) vs ≥ 27.9; Mip360 mean 27.66 vs
  ≥ 27.9. More Gaussians (grad 2e-4), 50k iterations, MCMC (3M), antialiasing and no
  opacity reset all failed to raise garden. Next levers: densification that targets error
  (e.g. Taming-style score budgets) rather than thresholds, and checking our MCMC (27.24 at
  3M is below published MCMC numbers).
- Flowers/treehill gap: closed only with a per-scene override (grad 8e-4 + dense 0.005:
  flowers 21.46 > FastGS 21.35; treehill 22.50; also bicycle 25.38 at 150 s > Faster-GS fused
  25.27 at 619 s, stump +0.45); the same setting loses 0.1-0.4 dB on garden and indoor.
  A generic per-scene signal (e.g. DashGaussian spectrum s_max or early growth rate) to pick
  the threshold is untested.
- The DashGaussian warm-up hurts Quality (densification at low resolution); worth testing
  with `score_at_full_res` or warm-up ending before densification ramps.
- Per-iteration kernel work (raster backward SHFL/stalls, Adam) not attempted this round;
  note the profile's Adam 6 ms is the all-groups-due worst case, not the average.
- Host contention: other agents ran 9 GB jobs without the lock; several runs were killed
  and restarted (capacity squeeze/OOM). Times should be re-checked in a quiet pass.
