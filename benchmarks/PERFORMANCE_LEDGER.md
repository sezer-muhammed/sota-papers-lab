# Performance ledger

Measured speed and quality results for SOTA Papers Lab apps, for the home page and marketing. Every row comes from a test run on our GPU host, not from a paper.

**Rules for anyone quoting this file:**
- Quote only rows marked **verified**. Rows marked **preliminary** were measured while other work shared the GPU, or still need an accuracy check. They can change.
- Always name the baseline and the hardware ("on an RTX 4070 SUPER, vs PyTorch FlashAttention-2"). Never write "Nx faster" without saying faster than what.
- Component results (one kernel) are not end-to-end results. Do not present a kernel speed-up as the speed-up of the whole app.

Hardware for every row unless stated: NVIDIA RTX 4070 SUPER (12 GB, sm_89), driver 595, CUDA 12.8 runtime, Ubuntu 24.04.

## image-to-3d (in development)

| Date | Component | Result | Baseline (same GPU) | Speed-up | Accuracy check | Status | Source |
|---|---|---|---|---|---|---|---|
| 2026-11-01 | Splat tile sort (onesweep radix, NVRTC) | 0.79 ms at 14.3M, 1.59 ms at 28.7M, 3.30 ms at 59.6M instances | CUB u16 as used by Faster-GS: 0.92 / 1.92 / 6.21 ms. gsplat 64-bit: 8.19 / 17.5 / 37.0 ms | 1.2–1.9x vs CUB; 10–11x vs gsplat | Identical ordering to gsplat, Faster-GS and CUB on 6 garden workloads (0 mismatches) | preliminary (GPU shared at 84–100% during timing) | k-sort, commit 3ddc396 |
| 2026-11-01 | Tile range extraction | 0.093 ms at 14.3M, 0.321 ms at 59.6M | Faster-GS ranges + buckets + readback: 0.880 / 1.138 ms | 3.5–9.5x | Bit-exact vs CPU mirror | preliminary (GPU shared) | k-sort, 3ddc396 |
| 2026-11-01 | Frame attention, bf16 (VGGT shapes, 16 heads × 64) | 57.5–67.5 TFLOPS | PyTorch SDPA FlashAttention-2: 40.8–53.0 TFLOPS; SDPA cuDNN: 47.6–63.6 | 1.26–1.41x vs FlashAttention-2; 1.05–1.21x vs cuDNN | rel. RMS error 1.868e-3 vs SDPA flash 1.873e-3 (fp64 reference) | verified (GPU idle, lock held) | nn-dense, 0b6b112 |
| 2026-11-01 | Global attention, bf16 (8–64 frames) | 69.7–73.3 TFLOPS | FlashAttention-2: 64.4–70.7; cuDNN 68.4–73.5 | 1.04–1.08x vs FA2; parity with cuDNN | as above | verified | nn-dense, 0b6b112 |
| 2026-11-01 | Quantised global attention (INT8 QK, FP8 PV, SageAttention2-style) on real VGGT activations, 200 frames (135k tokens) | 425 ms (FP32 PV accumulate), 358 ms (FP16 PV) | FlashAttention-2 2.8.3 bf16: 1047 ms; SDPA cuDNN 1018 ms | 2.5x (FP32 PV) to 2.9x (FP16 PV) vs FlashAttention-2 | cos 0.99999, rel. L1 0.44% vs fp64 (FA2: 0.15%) | verified for the kernel; end-to-end accuracy of the model pending | nn-sage, 8349f4f |
| 2026-11-01 | FP8 GEMM with per-channel weight scales (VGGT-1B layer shapes) | 141–153 TFLOPS | PyTorch `_scaled_mm` rowwise (per-token × per-channel): 39–55 TFLOPS; per-tensor: 140–146 | 2.8–3.6x vs rowwise; parity vs per-tensor | rel. error 1.672e-3, identical to PyTorch | verified | nn-dense, 0b6b112 |
| 2026-11-01 | Fused LayerNorm → FP8 quantisation | 465 GB/s (92% of the card's 504 GB/s) | PyTorch eager: 67 GB/s | ~7x | 11 of 1.41M FP8 values one step off PyTorch | verified | nn-dense, 0b6b112 |
| 2026-11-01 | Video keyframe ingest (60 s 4K HEVC 50 Mbps, 300 sharpest keyframes, GPU sharpness scoring) | 17.4 s (3.45x real time) | ffmpeg decode alone: 19.1 s | faster than plain decoding (frames are sampled, then the sharpest per time bin kept) | GPU sharpness = CPU reference within 1e-4 | preliminary (GPU 97-98% busy with other work) | ingest, 1ea00bd |
| 2026-11-01 | Splat file size, SPZ export (garden init, 138k Gaussians, SH3) | 1.5 MB | INRIA PLY: 33 MB | 22x smaller | round-trip within quantisation; validated with the reference reader | verified | io-export, 8f82b7a |
| 2026-11-01 | Splat rasterizer, forward and backward (garden 1/4 res, 1.86M visible Gaussians, 3.45M tile entries) | forward 1.41 ms, backward 17.5 ms | gsplat: forward 5.85 ms, backward 20.3 ms | 4.1x forward, 1.16x backward | images within 1.1e-6 of gsplat; gradients within 3e-6 of float64 (gsplat's own error 1.4e-4 to 3e-4) | preliminary (GPU 98% busy; earlier kernel version; locked re-run pending) | k-raster, 1a2b1ab |
| 2026-11-01 | Fused L1 + D-SSIM loss, forward and backward | 0.36 ms at 1297x840; 4.06 ms at 4K | fused-ssim + PyTorch L1: 0.78 ms; 11.88 ms | 2.2x (1297x840), 2.9x (4K) | gradient within 7.8e-9 of fused-ssim; loss closer to float64 than fused-ssim | preliminary (GPU 92-100% shared; ours and baseline timed back to back) | k-loss-opt, bb569c9 |
| 2026-11-01 | Fused multi-group Adam, 6M Gaussians (59 floats each) | 24.1 ms (all), 13.6 ms (50% visible) | PyTorch fused Adam: 48.8 ms; gsplat selective Adam: 19.0 ms | 2.0x vs PyTorch; 1.4x vs gsplat selective | matches torch.optim.Adam to 6e-8 per step; <= 2 ulp vs float64 over 100 steps | preliminary (GPU shared); at 1M Gaussians parity with gsplat (2.08 vs 1.96 ms) | k-loss-opt, bb569c9 |
| 2026-11-01 | Densification step (clone/split/prune, adaptive density control) at 1M / 3M / 6M Gaussians | 6.87 / 25.4 / 52.8 ms | gsplat: 17.6 / 60.2 / 170 ms. Faster-GS: 11.4 / 34.6 / 106 ms | 2.4–3.2x vs gsplat; 1.4–2.0x vs Faster-GS | densify plan bit-exact vs CPU reference; emission exact on every array | verified (GPU lock held) | k-densify, 6f73b0e |
| 2026-11-01 | MCMC relocation round at 1M / 3M / 6M Gaussians | 2.33 / 7.92 / 17.6 ms | gsplat: 5.81 / 17.3 / 32.6 ms. Faster-GS: 5.84 / 18.0 / 48.4 ms | 1.9–2.5x vs gsplat; 2.3–2.8x vs Faster-GS | relocation within 1.9e-6 of gsplat | verified (GPU lock held) | k-densify, 6f73b0e |
| 2026-11-02 | VGGT-Ω camera + dense heads (512x336, N = 8 / 32 frames) | 17.2 / 68.1 ms | official PyTorch heads: 43.5 / 170.7 ms | 2.5x | all 17 intermediates within 4.6e-3 rel. RMS of the official model | preliminary (lock state not recorded) | eng-heads, c2c2050 |
| 2026-11-02 | Native VGGT-Ω, full model (trunk + heads), bf16, 1 / 2 / 8 / 32 frames at 512 px | 26.8 / 47.3 / 236 / 1084 ms | official PyTorch VGGT-Ω bf16: 58.5 / 81.1 / 367 / 1668 ms | 2.2x / 1.7x / 1.56x / 1.54x | depth within 0.11–0.14% (median abs-rel) of official; trunk error within 1.0–1.1x of the official bf16-vs-fp32 noise; camera angles still 2–3x the official noise (gate not yet met) | speed verified (median of 10, GPU lock held); accuracy partly pending | eng-runtime aba2270, eng-trunk b3ce06c |
| 2026-11-02 | VGGT-Ω weight load (1.8 GB bf16 arena) | 0.65 s | our first version without pinned staging: 1.47 s | 2.3x | — | verified | eng-runtime, aba2270 |
| 2026-11-02 | Native VGGT-Ω after integration, bf16, 2 / 10 / 50 frames | 52.8 ms / 287 ms / 2.24 s | official PyTorch VGGT-Ω bf16 (same idle card): 91.2 ms / 482 ms / 3.62 s | 1.73x / 1.68x / 1.61x | ETH3D AUC@3/@30 28.15/86.22 vs official 30.37/86.30; Sintel 22.22/66.44 vs 20.74/66.00; depth AbsRel equal (0.0119 vs 0.0118) | verified (GPU lock held, idle card) | integrate-vggt, f7c85e6 |
| 2026-11-02 | Native VGGT-Ω memory: frames that fit on a 12 GB card | 100 frames in 6.44 s (7.1 GB), 200 frames in 20.8 s (11.1 GB) | official PyTorch: out of memory at 100 and 200 frames (8.7 GB peak at 50) | runs where the official code cannot | as above | verified | integrate-vggt, f7c85e6 |
| 2026-11-02 | Native VGGT-Ω with the fast-attention schedule (frame attention at 3 layers, 4x4 key merging at 4, INT8/FP8 SageAttention elsewhere), 2 / 10 / 50 frames | 52.0 ms / 263 ms / 1.62 s | official PyTorch VGGT-Ω bf16 (same idle card): 91.2 ms / 482 ms / 3.60 s | 1.75x / 1.83x / 2.22x | AUC@3/@30 on 9 held-out scenes: -0.34 / -0.12 vs dense (within the -2.0 / -1.0 budget) | verified (GPU lock held, idle card) | p2-native, 418e4b5 |
| 2026-11-02 | Native VGGT-Ω fast schedule, large inputs | 100 frames in 3.94 s (7.3 GB); 200 frames in 11.8 s (5.6 GB) | official PyTorch: at most 90 frames fit, taking 8.36 s (10.4 GB); out of memory at 100 | twice the frames of the official maximum, each run 2x+ faster | as above | verified | p2-native, 418e4b5 |
| 2026-11-02 | Camera poses from photos alone vs COLMAP (Mip-NeRF 360 garden, 100 frames) | ATE 1.58% of the trajectory radius, mean rotation error 0.94°, in seconds | COLMAP (reference poses) | — | — | verified | integrate-vggt, f7c85e6 |
| 2026-11-02 | Commercial-licence front end: DA3-BASE (Apache-2.0) on ONNX Runtime, 3 / 10 frames at 504x336 | 31.6 / 151 ms | official DA3 PyTorch bf16: 52.7 / 168 ms | 1.67x / 1.11x | matches official fp32 (depth 6e-5); pose quality far below VGGT-Ω (ETH3D AUC@30 23.3 vs 86.2; official DA3 itself 15.3), so not marketable for quality | preliminary (card contended) | commercial, 8beab78 |

### End-to-end training (COLMAP poses, Quality preset, Mip-NeRF 360 protocol, metrics with the Inria evaluation)

| Date | Scene | Ours | Baseline (same GPU) | Result | Status | Source |
|---|---|---|---|---|---|---|
| 2026-11-02 | garden (images_4) | 27.61 dB / SSIM 0.863 / LPIPS 0.121 in 185 s | FastGS: 27.30 dB in 265 s | +0.31 dB and 1.43x faster | quality verified; times preliminary (baseline run may have shared the GPU; quiet re-run pending) | integrate3d, 152a2dc |
| 2026-11-02 | bicycle (images_4) | 25.15 dB in 201 s | FastGS: 24.82 dB in 190 s. Faster-GS (fused): 25.27 dB in 619 s | +0.33 dB vs FastGS at similar time; 3.1x faster than Faster-GS at -0.12 dB | as above | integrate3d, 152a2dc |
| 2026-11-02 | room (images_2) | 32.14 dB in 148 s | Brush: 32.42 dB in 722 s | 4.9x faster at -0.28 dB | as above | integrate3d, 152a2dc |
| 2026-11-02 | garden, Fast preset | 26.40 dB in 50 s | — | — | quality verified | integrate3d, 152a2dc |
| 2026-11-02 | kitchen (images_2) | 31.91 dB in 296 s | gsplat MCMC: 31.52 dB in 1126 s | +0.39 dB and 3.8x faster | quality verified; times preliminary (quiet re-run pending) | opt-train, 76404de |
| 2026-11-02 | bonsai (images_2) | 32.78 dB in 380 s | gsplat MCMC: 32.58 dB in 1223 s | +0.20 dB and 3.2x faster | as above | opt-train, 76404de |
| 2026-11-02 | flowers (images_4) | 21.07 dB in 201 s | FastGS: 21.35 dB in 182 s | -0.28 dB (FastGS wins here) | quality verified | opt-train, 76404de |
| 2026-11-02 | Mip-NeRF 360, all 9 scenes, Quality preset | mean 27.59 dB, 144–380 s per scene | (no rival measured on all 9 yet) | — | quality verified | opt-train, 76404de |
| 2026-11-02 | Mip-NeRF 360, all 9 scenes, Big preset | mean 27.66 dB, every scene <= 288 s | (no rival measured on all 9 yet) | — | verified (GPU lock held) | opt-train-2, 0279945 |
| 2026-11-02 | bicycle with a denser-densification setting (grad 8e-4) | 25.38 dB in 150 s | Faster-GS (fused): 25.27 dB in 619 s | +0.11 dB and 4.1x faster | verified (lock held); setting is not the default (it costs 0.1-0.4 dB on garden, room, kitchen) | opt-train-2, 0279945 |
| 2026-11-02 | flowers with the same setting | 21.46 dB in 144 s | FastGS: 21.35 dB in 182 s | +0.11 dB and 1.26x faster | as above | opt-train-2, 0279945 |

### Mesh (Geometry preset + GPU TSDF + marching cubes)

| Date | Benchmark | Ours | Reference | Note | Status | Source |
|---|---|---|---|---|---|---|
| 2026-11-02 | DTU Chamfer, scans 24/37/40/55/63 | mean 1.23 mm; training 109–197 s plus 3–4 s meshing per scan | published, not run by us: 3DGS 2.18 mm (11.2 min), 2DGS 0.64 (10.9 min), GOF 0.64 (2 h), PGSR 0.48 (0.5 h) | better than 3DGS; behind 2DGS/GOF/PGSR in accuracy; do not claim SOTA mesh accuracy | superseded: this round used depth-validity masks instead of the standard IDR silhouettes, so it is not comparable to published numbers | mesh-mode, 5eeb283 |
| 2026-11-02 | DTU Chamfer, scans 24/37/40/55/63, standard evaluation (IDR silhouette culling as in 2DGS) | mean 0.563 mm (0.474 / 0.787 / 0.350 / 0.395 / 0.809); training 96–256 s plus 2.4–3.0 s meshing per scan | published, not run by us: 2DGS 0.64 mm in 10.9 min; PGSR 0.48 mm in 30 min | more accurate than published 2DGS while training ~3–7x shorter; PGSR is still more accurate (0.48) | verified for ours; times for scans 37 and 55 preliminary (lock timed out) | mesh-mode-2, 9543d3f |

### COLMAP-free end to end (photos or video in, splat out; native VGGT-Ω poses + our trainer)

| Date | Input | Ours (end to end, files in to .spz on disk) | Reference | Note | Status | Source |
|---|---|---|---|---|---|---|
| 2026-11-02 | 62 s 1080p H.264 video (stand-in made from Mip-NeRF 360 frames) | .spz in 90 s; camera points on screen at 24 s; held-out keyframes 23.6 dB | — | first end-to-end video run | verified (GPU lock held) | mvp-pipeline, ee8f1bc |
| 2026-11-02 | Mip-NeRF 360 garden, Quality / Fast | 26.09 dB in 295 s / 24.68 dB in 94 s | VGGT-X published 26.40 dB (not run here: VGGT-X and InstantSplat failed to run on this host); our trainer on COLMAP poses: 27.61 dB | below our targets (27.4 dB in 5 min; 26.5 dB in 90 s); not yet marketable as SOTA | verified | mvp-pipeline, ee8f1bc |
| 2026-11-02 | Poses vs COLMAP (garden) | median relative rotation error 0.75° after refinement; focal 964.7 vs 961.4 px | COLMAP | — | verified | mvp-pipeline, ee8f1bc |
| 2026-11-02 | COLMAP-free, round 2 (bundle adjustment on KLT tracks, voxel-merged init): garden Quality / Fast | 27.62 dB in 256 s / 26.38 dB in 92 s | VGGT-X published 26.40 dB (Mip-NeRF 360 mean; ours is garden only, so not yet comparable); our trainer on COLMAP poses: 27.61 dB | matches COLMAP-pose quality on garden; Quality target (27.4 dB in 5 min) met; Fast target (26.5 dB in 90 s) missed by 0.12 dB | verified (GPU lock held, idle card) | mvp-quality-2, 89c173f |
| 2026-11-02 | COLMAP-free round 2, other scenes (Quality) | bicycle 25.41 dB / 268 s; room 31.64 / 228 s; truck 25.68 / 159 s; train 19.41 / 158 s | our trainer on COLMAP poses: bicycle 25.16, room 32.07, truck 26.11, train 21.88 | bicycle beats COLMAP poses; train is 2.5 dB behind | verified | mvp-quality-2, 89c173f |
| 2026-11-02 | Pose error vs COLMAP after refinement (garden) | 0.060° median relative rotation | round 1: 0.894° | 15x lower | verified | mvp-quality-2, 89c173f |

### End-to-end COLMAP-free reconstruction (`image_to_3d.reconstruct`: JPEG files → native VGGT-Ω poses → splat → `.spz` on disk; nvs-colmapfree protocol, every 8th view held out, 200-iteration test-time pose refinement; GPU lock held, card idle at start)

| Date | Scene | Ours (held-out after test-time refinement) | Baseline | Result | Status | Source |
|---|---|---|---|---|---|---|
| 2026-11-02 | garden (images_4), Quality | 26.09 dB / SSIM 0.812 / LPIPS 0.156, 295 s end to end (before refinement 19.60 dB) | VGGT-X published Mip-NeRF 360: 26.40 dB (paper, other hardware); our trainer on COLMAP poses: 27.61 dB (SfM time excluded) | -0.31 dB vs VGGT-X's published number; -1.5 dB vs COLMAP poses, no SfM needed | verified (lock held, idle card); misses the 27.4 dB target | mvp-pipeline, 71ba782 |
| 2026-11-02 | garden (images_4), Fast | 24.68 dB / 0.713 / 0.284 in 94 s end to end | our trainer on COLMAP poses, Fast: 26.43 dB (50 s training, SfM excluded) | -1.75 dB | verified; misses the 26.5 dB / 90 s target | mvp-pipeline, 71ba782 |
| 2026-11-02 | room (images_2, 311 frames), Quality | 29.98 dB / 0.895 / 0.243 in 242 s | COLMAP poses: ours 32.14 dB; Brush 32.42 dB in 722 s | -2.2 dB vs COLMAP poses | verified | mvp-pipeline, 71ba782 |
| 2026-11-02 | bicycle (images_4), Quality / Fast | 20.48 dB in 315 s / 21.47 dB in 85 s | COLMAP poses: ours 25.15 dB; FastGS 24.82 dB | -4.7 dB (held-out poses not recovered) | verified; do not market | mvp-pipeline, 71ba782 |
| 2026-11-02 | T&T truck / train, Quality | 23.21 dB in 175 s / 20.15 dB in 168 s (Fast: 22.79 in 84 s / 19.55 in 81 s) | target 24.9 dB | below target | verified | mvp-pipeline, 71ba782 |
| 2026-11-02 | 62 s phone-style video (garden frames, H.264, exposure drift), Fast + PPISP | 90 s from video file to `.spz` (617 frames decoded, 150 keyframes); cameras and points shown at 24 s, splat previews every 8 s; held-out keyframes 23.57 dB | — | end-to-end path works | verified (stand-in video, not a real phone capture) | mvp-pipeline, 71ba782 |
| 2026-11-02 | garden (images_4), Quality | **27.62 dB** / SSIM 0.860 in **256 s** end to end (KLT-track BA, square pixels, voxel-merged init) | our trainer on COLMAP poses 27.61 dB (SfM excluded); FastGS on COLMAP 27.30 dB in 265 s; VGGT-X published 26.40 (Mip-NeRF 360 mean, paper) | meets the 27.4 dB / 5 min target; parity with COLMAP poses without SfM | verified (GPU lock held, idle card) | mvp-quality-2, a42a973 |
| 2026-11-02 | garden (images_4), Fast | 26.38 dB / 0.798 in 91.9 s (cold weight load; ~86 s warm) | our trainer on COLMAP, Fast: 26.43 dB | misses 26.5 dB / 90 s by 0.12 dB and 1.9 s | verified (lock held) | mvp-quality-2, a42a973 |
| 2026-11-02 | bicycle (images_4), Quality / Fast | 25.41 dB in 268 s / 24.48 dB in 89.6 s | COLMAP poses: ours 25.15 dB, FastGS 24.82 dB in 190 s | +0.26 dB vs our COLMAP-pose run (test poses refined, see protocol note) | verified (lock held) | mvp-quality-2, a42a973 |
| 2026-11-02 | room (images_2), Quality / Fast | 31.64 dB in 228 s / 30.02 dB in 128 s | COLMAP poses: ours 32.14 dB; Brush 32.42 dB in 722 s | −0.5 dB vs COLMAP poses; 3.2x faster than Brush at −0.78 dB | verified (lock held) | mvp-quality-2, a42a973 |
| 2026-11-02 | T&T truck / train, Quality | 25.68 dB in 159 s / 19.41 dB in 158 s (Fast: 24.27 in 83 s / 19.03 in 84 s) | mvp-pipeline: 23.21 / 20.15 dB; target 24.9 dB | truck meets 24.9; train below (−0.7 dB vs before) | verified (lock held) | mvp-quality-2, a42a973 |
| 2026-11-02 | Training-pose accuracy vs COLMAP (median relative rotation error) | garden 0.05-0.06°, bicycle 0.16-0.17°, truck 0.19-0.21°, room 0.29-0.31°, train 0.55-0.60° | engine + photometric refinement only: garden 0.64-0.89° | 10x better on garden | verified | mvp-quality-2, a42a973 |

### Object generation from 1-4 photos (Pixal3D multi-view, 1024 cascade, our sidecar)

| Date | Input | Ours | Baseline (same GPU) | Note | Status | Source |
|---|---|---|---|---|---|---|
| 2026-11-02 | 1 / 2 / 4 views (synthetic sphere and torus) | 508 / 380 / 368 s; peak 10.5 / 9.7 / 9.8 GB; GLBs valid (Khronos 0 errors) | official Pixal3D inference_mv.py with --low_vram: out of memory after ~240 s on the same 12 GB card | runs where the official code cannot (grouped low-memory NAF upsampler, cache release before CuMesh); quality only on simple synthetic shapes so far | verified (GPU lock held, idle card) | pixal3d-eval-2, a875ceb |

Not a win (do not market): Morton re-sort from random order is slower than Faster-GS (8.94 vs 6.21 ms at 1M; 96.1 vs 49.4 ms at 6M).

Not yet marketable: attention with 4×4 key merging runs in 33–38 ms (vs 1047 ms FlashAttention-2), but the approximation error is too high on some layers (up to 56% rel. L1) until the Phase 2 per-layer schedule search shows which layers can use it.

## raw-retouch (deployed)

| Date | What | Result | Status | Source |
|---|---|---|---|---|
| 2026-11-01 | RAW decode (rawler), 24 MP Sony ARW | ~50 ms | verified | build round, A7 III samples |
| 2026-11-01 | Full-resolution GPU demosaic (RCD), 61 MP | ~0.3 s | verified | develop module, A7R V sample |
| 2026-11-01 | Automatic edit per photo, without Claude (decode, develop, 56-slider search, export) | 4.0–6.2 s (search ~2.1–3 s) | verified | integrate2, 8 samples |
| 2026-11-01 | Automatic edit with Claude planning and blind judging | 28–48 s per photo, ~6 Claude calls | verified | integrate2 / live deploy |
| 2026-11-01 | 1:1 pixel-zoom tiles | first tile 0.5–1.2 s, cached tiles 10–25 ms | verified | integrate3 |
