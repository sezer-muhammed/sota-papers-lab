# Trainer optimisation pass (opt-train), RTX 4070 SUPER, 2 Nov 2026

All runs under `flock ~/spl-work/gpu.lock` (other unlocked GPU users on the host may still
have been active), driver `trainer_gpu_tests.rs`, COLMAP poses, test = every 8th image,
PSNR/SSIM from our in-app metrics (match `gs-metrics` to 1e-4). One seed per row.

## Adam ablation, garden (images_4) Quality 30k

| Setting | Train s | PSNR | SSIM | Peak device VRAM |
|---|---|---|---|---|
| f32 moments, dense (old default) | 193.0 | 27.491 | 0.8627 | 12.1 GB |
| **bf16-SR moments (new default)** | **178.1** | **27.490** | **0.8624** | **10.0 GB** |
| VisibleOnly | 178.5 | 27.488 | 0.8610 | 12.4 GB |
| bf16 + VisibleOnly | 173.3 | 27.360 | 0.8596 | 9.6 GB |
| bf16 + LazyCatchUp | 389.6 | 27.422 | 0.8619 | 12.4 GB |

bicycle Quality bf16 + VisibleOnly: 415.5 s / 25.112 dB (peak 3.2M Gaussians; densification
changes), so sparsity stays off.

## Suite, presets at 4a65394 (bf16 moments)

Baselines: best full-length `ok` row in `~/spl-work/bench/results.jsonl` on this GPU.

| Scene | Quality s | PSNR | SSIM | #G | Fast s | PSNR | SSIM | Baselines (PSNR / train s) |
|---|---|---|---|---|---|---|---|---|
| bicycle | 189 | 25.16 | 0.762 | 2.13M | 51 | 24.56 | 0.703 | FastGS 24.82/190; Faster-GS fused 25.27/619 |
| flowers | 201 | 21.07 | 0.614 | 1.91M | 91 | 20.97 | 0.546 | FastGS 21.35/182 |
| garden | 182 | 27.61 | 0.863 | 1.27M | 107 | 26.39 | 0.806 | FastGS 27.30/265 |
| stump | 154 | 26.27 | 0.766 | 1.50M | 44 | 26.13 | 0.733 | none |
| treehill | 178 | 21.91 | 0.600 | 2.10M | 42 | 22.68 | 0.595 | none |
| room | 144 | 32.07 | 0.922 | 0.60M | 51 | 30.73 | 0.900 | Brush 32.42/722 |
| counter | 282 | 29.51 | 0.912 | 0.49M | 59 | 27.91 | 0.886 | none |
| kitchen | 296 | 31.91 | 0.931 | 0.74M | 71 | 30.41 | 0.912 | gsplat-MCMC 31.52/1126 |
| bonsai | 380 | 32.78 | 0.946 | 0.53M | 53 | 30.58 | 0.925 | gsplat-MCMC 32.58/1223 |
| truck | 247 | 26.11 | 0.890 | 0.61M | 31 | 24.95 | 0.862 | none |
| train | 104 | 21.88 | 0.825 | 0.61M | 31 | 21.30 | 0.780 | none |
| playroom | 205 | 29.91 | 0.908 | 0.73M | 35 | 30.12 | 0.904 | none |
| drjohnson | 209 | 29.01 | 0.894 | 1.09M | 38 | 29.34 | 0.897 | none |

Notes: Fast garden (107 s) and flowers (91 s) are about 2x the earlier 50 s Fast garden run;
likely contention from unlocked jobs, re-time in a quiet pass. Phase 3 exit test status:
bicycle Quality is 0.11 dB below Faster-GS fused at 3.3x its speed; other scenes lack a
Faster-GS 30k row. Quality loses to FastGS on flowers (-0.28 dB) and treehill Fast beats
Quality (+0.77 dB): the generic Quality densify thresholds overgrow those scenes.
