# VGGT-Ω native engine: Phase 2 (attention schedule, SageAttention, FP8)

RTX 4070 SUPER 12 GB, CUDA 12.8 container, measured 2026-11-02 (branch
`w/p2-native`). Schedule from `reports/image-to-3d/spec-attn-schedule.md`.
Tests: `src/vggt/phase2_tests.rs`; AUC scoring: `tools/vggt/p2_eval.py`.

## What is implemented

- Per-layer global-attention schedule (`AttnSchedule::named`): `dense`
  (default, unchanged), `sage`, `ours`, `ours+sage`, `wide`, `wide+sage`.
  `ours`: global blocks 3, 4, 5 → frame attention; 0, 1, 7, 19 → merge4;
  the rest dense. `+sage`: every non-frame global layer runs spl-nn
  SageAttention (INT8 QKᵀ, FP8 P·V, K/V smoothing).
- KV merge4: 4×4 mean-pooled patch K/V of frames 1.. with a +log(count)
  logit bias added in-register in the Sage kernel; the 17 special tokens and
  every token of frame 0 are never merged (new `FrameLayout::dense_frames`
  in spl-nn, kernel + reference + tests). Merge layers always use the Sage
  kernel (no bf16 merge path).
- `VggtConfig::sage_pv16`: SageAttention2++ FP16 P·V accumulation.
- FP8 GEMMs per block (`VggtConfig::fp8_blocks`, all four linears, per-token
  activation scales, per-channel weight scales applied by the consumer kernels;
  activations live in the idle DINO buffer). **Off by default**: see below.

## Accuracy (9 held-out scenes × 50 frames, COLMAP GT, official AUC)

Mip-NeRF 360 bonsai, counter, room, flowers, treehill; T&T train, truck; DB
drjohnson, playroom (the spec's eval split). All rows are the native engine.

| config | AUC@3 | AUC@30 | ΔAUC@3 | ΔAUC@30 | budget (−2.0 / −1.0) |
|---|---|---|---|---|---|
| dense | 67.85 | 95.24 | – | – | – |
| sage | 67.53 | 95.18 | −0.33 | −0.07 | yes |
| ours | 67.43 | 95.05 | −0.43 | −0.19 | yes |
| **ours+sage (FP32 P·V)** | 67.52 | 95.12 | **−0.34** | **−0.12** | **yes** |
| ours+sage, FP16 P·V | 67.32 | 94.97 | −0.54 | −0.27 | yes |
| ours+sage, FP16 P·V, FP8 blocks 16–23 | 65.51 | 94.79 | −2.34 | −0.45 | **no** |
| wide+sage | 66.09 | 94.90 | −1.77 | −0.34 | yes (marginal) |

ETH3D / Sintel subsets: predictions were dumped (`pred*/…/eth3d`, `sintel`)
but not scored yet (45 pairs each; too coarse for these deltas anyway).

### FP8 per-block error (`gpu_p2_fp8_sensitivity`)

Calibration scenes bicycle, garden, stump, kitchen × 30 frames, ours+sage with
FP16 P·V; one block in FP8 at a time vs the same config in bf16:

| blocks | pose_enc |Δ| | depth rel |
|---|---|---|
| frame / inter 0–15 (each) | 2.9–4.8e-3 | 1.0–1.5e-2 |
| frame / inter 16–23 (each) | 0.7–1.7e-3 | 0.13–0.37e-2 |
| all 43 | 6.5e-3 | 2.1e-2 |

Even the 16 least-sensitive blocks (16–23) cost 1.8 AUC@3 on top of FP16 P·V,
which breaks the budget, so FP8 GEMMs stay off. `P2_FP8_BLOCKS` keeps that
selection for later work (e.g. FP8 for MLPs only, or with FP32 P·V).

## Speed and memory (`gpu_p2_bench`, kitchen 512×336, GPU lock held, card otherwise idle)

Median GPU time of the whole forward (10 runs, 3 at N ≥ 100, after 2 warm-ups).
Memory: whole-card use while running (free-memory minimum); at N = 200 the
planner offloads the cache and releases trunk buffers before the dense head.

| N | official PyTorch bf16 | native dense | ours+sage | ours+sage FP16 P·V | vs dense | vs official | card memory (ours+sage) |
|---|---|---|---|---|---|---|---|
| 2 | 91.2 ms¹ | 52.5 ms | 52.0 ms | 51.9 ms | 1.01× | 1.75× | 3.0 GB |
| 10 | 482.3 ms¹ | 287.7 ms | 263.4 ms | 260.7 ms | 1.09× | 1.83× | 3.7 GB |
| 50 | 3604 ms | 2244 ms | 1623 ms | 1569 ms | 1.38× | 2.22× | 5.3 GB |
| 90 | 8364 ms (peak 10.4 GB) | – | – | – | – | – | – |
| 100 | OOM¹ | 6444 ms | 3938 ms | 3722 ms | 1.64× | (≥ 2.1× vs official at 90) | 7.3 GB |
| 200 | OOM | 21908 ms | 11835 ms | 10955 ms | 1.85× | – | 5.6 GB |

¹ from `INTEGRATION.md` §3 (same GPU, lock held). Official at 50–90:
`official_maxn.py` (full model, bf16 autocast, median of 3 after 2 warm-ups,
lock held); it fits at 90 frames (10.4 GB peak allocated) and OOMs at 100,
so the largest N it runs is 90–99. "vs official" uses ours+sage (FP32 P·V);
at 100 native frames vs 90 official frames.

## Exit test

| criterion | target | result | |
|---|---|---|---|
| AUC@30 drop | ≤ 1.0 | 0.12 | pass |
| AUC@3 drop | ≤ 2.0 | 0.34 | pass |
| 200 frames | ≤ 8 s | 11.8 s (11.0 s FP16 P·V) | **fail** |
| 200 frames memory | ≤ 12 GB | 5.6 GB card | pass |
| 2 views | ≤ 50 ms | 52.0 ms | **fail** (by 2 ms) |
| ≥ 4× official at 200 | (official OOMs) | 2.2× at 50; ≥ 2.1× at 90/100 | **fail** |

Where the time goes at 200 frames (ours+sage): trunk 10.2 s of 11.8 s. The 12
dense global layers dominate (the spec's FLOP count: −35% core FLOPs vs
dense); the measured 1.85× vs native dense already exceeds that because Sage
runs ~2.5× the bf16 flash rate. Reaching 8 s needs either merging layers 10–18
(the spec shows that breaks accuracy) or FP8 GEMMs that keep accuracy (per-block
data above says only a much smaller subset can). The 2-view case is not
attention-bound (frame blocks + heads) and is unchanged by Phase 2.
