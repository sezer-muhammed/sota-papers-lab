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
| 2026-11-02 | Splats → textured mesh, DTU (scans 24/37/40/55/63, 49 views at 800×600, 2DGS protocol: dilated-mask cull, 0.2 mm sampling, official ObsMask/plane, 20 mm cap) | Chamfer 1.41 / 1.06 / 1.09 / 1.01 / 1.59 mm, mean 1.23 mm. Train 171 / 170 / 161 / 197 / 109 s (Geometry preset, 30k iterations); mesh (median depth → TSDF 1 mm voxel → marching cubes → 300k triangles → UV → bake → GLB) 3.1–3.7 s | Published only, not run here: 3DGS 2.14/1.53/2.08/1.68/3.49 (mean 2.18; 11.2 min), 2DGS 0.48/0.91/0.39/0.39/1.01 (mean 0.64; 10.9 min), GOF 0.50/0.82/0.37/0.37/1.12 (mean 0.64; 2 h), PGSR 0.36/0.57/0.38/0.33/0.78 (mean 0.48; 0.5 h). Same GPU: our Quality preset without the regularisers on scan 24: 2.33 mm | Accuracy: 1.8x better than published 3DGS, 1.9x worse than published 2DGS and 2.6x worse than PGSR. Time: about 3 min vs 2DGS's published 10.9 min (their GPU, not this one) | GLBs pass the Khronos glTF Validator 2.0.0-dev.3.10 with 0 errors, 0 warnings; DTU eval is our port of the 2DGS/DTUeval-python protocol (`tools/dtu/eval.py`) | preliminary: training timed with the GPU lock held (only idle processes resident), meshing timed without the lock; Chamfer is a quality gap, not a win | mesh-mode |

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
