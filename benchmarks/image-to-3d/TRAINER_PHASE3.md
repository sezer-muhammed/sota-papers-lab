# Phase 3 trainer: first measured runs (RTX 4070 SUPER, 2 Nov 2026)

COLMAP poses, test = every 8th image, metrics from `gs-metrics` (Inria convention;
our CPU PSNR/SSIM matched it to 1e-4). Under the GPU lock; driver `trainer_gpu_tests.rs`.
`train_s` excludes load, init and eval. VRAM: process footprint with the fixed
capacity allocated up front (spec-train 5.10) / estimated need at the peak count.

| Scene | Preset | Train s | PSNR | SSIM | LPIPS | #G final (peak) | VRAM MiB alloc / need |
|---|---|---|---|---|---|---|---|
| garden (images_4) | quality 30k | 185.0 | 27.61 | 0.863 | 0.121 | 1.25M (1.41M) | 11083 / 2612 |
| garden | fast 15k | 50.4 | 26.40 | 0.806 | 0.229 | 0.39M (0.47M) | 8514 / 1369 |
| bicycle (images_4) | quality 30k | 201.2 | 25.15 | 0.761 | 0.209 | 2.13M (2.50M) | 8482 / 4034 |
| bicycle | fast 15k | 42.7 | 24.54 | 0.703 | 0.320 | 0.43M (0.60M) | 8482 / 1536 |
| room (images_2) | quality 30k | 147.6 | 32.14 | 0.924 | 0.211 | 0.60M (0.75M) | 11269 / 1766 |
| room | fast 15k | 50.8 | 30.79 | 0.901 | 0.263 | 0.15M (0.21M) | 7284 / 1056 |

Baselines on the same card (`results.jsonl`): FastGS garden 27.30 / 0.847 / 0.157 in
265 s; FastGS bicycle 24.82 / 0.722 / 0.297 in 190 s; Faster-GS fused bicycle 25.27 /
0.769 / 0.203 in 619 s; Brush room 32.42 / 0.929 / 0.199 in 722 s.

Profile, garden quality final model (1.25M Gaussians, 463k visible), eager dense
iteration (all Adam groups due), ms: adam 6.02, raster_bwd 2.32, preprocess_bwd 0.87,
preprocess_fwd 0.49, raster_fwd 0.48, loss 0.36, bin_tiles 0.26, sort_depth 0.08,
resample 0.04, frame_begin 0.01; total 10.94.
