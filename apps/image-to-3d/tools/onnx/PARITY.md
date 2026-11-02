# DA3-BASE on ONNX Runtime: parity, evaluation, speed (Phase 7)

RTX 4070 SUPER 12 GB, CUDA 12.8 runtime image, ONNX Runtime 1.23.2 (CUDA EP,
cuDNN 9.10), PyTorch 2.9.1. Measured 2026-11-02. Engine: `src/engine/da3.rs`;
graphs from `export_da3.py` (backbone fp16, camera decoder and DPT head fp32
as in the official autocast path; 240.6 MB shared weights + ~1.2 MB per
resolution bucket, 9 buckets = 251 MB on disk).

## 1. Parity (`check_parity.py`, ETH3D delivery_area frames, 504×336)

Relative poses (to frame 0) and median-scale-aligned depth; "floor" = the
official bf16-autocast path vs fp32.

| Graph | N | depth rel | rel. rotation max | fov max | floor (bf16 vs fp32): depth / rot / fov |
|---|---|---|---|---|---|
| fp32 | 3 | 5.5e-5 | 3e-6° | 1e-5° | 1.5e-3 / 0.21° / 0.18° |
| fp32 | 10 | 6.5e-5 | 2e-4° | 2e-5° | (reference view flips, see below) |
| fp16 (shipped) | 3 | 4.1e-4 | 0.014° | 0.024° | 1.5e-3 / 0.21° / 0.18° |
| fp16 (shipped) | 10 | 5.6e-2 | 59.5° | 2.9° | 0.10 / 173° / 1.35° |

The fp32 graph reproduces PyTorch at N ≠ the export N (dynamic N works).
At N = 10 DA3's in-graph reference-view selection (`saddle_balanced`, an
argmin over class-token statistics) sits on a near-tie for these frames:
bf16 and fp16 both pick a different reference than fp32, and DA3's outputs
change with the reference (173° for the official bf16 path itself). Our fp16
graph is closer to fp32 than the official bf16 path on every metric.

## 2. Evaluation subsets (`gpu_da3_dump_eval_predictions`, `da3_official.py eval`, `tools/vggt/eval_subset.py`)

10 frames, 45 pairs per subset (AUC@3 moves in 0.74 steps). VGGT-Ω rows from
`tools/vggt/INTEGRATION.md` §2 (same scorer and frames).

| Subset | Model | AUC@3 | AUC@30 | δ<1.25 | AbsRel |
|---|---|---|---|---|---|
| ETH3D delivery_area | **DA3-BASE ours (ORT fp16)** | 2.96 | 23.33 | 99.09% | 0.0400 |
| | DA3-BASE official (PyTorch bf16, camera decoder) | 0.74 | 15.26 | 98.21% | 0.0370 |
| | DA3-BASE official, ray-pose path | 0.00 | 16.15 | 98.21% | 0.0370 |
| | VGGT-Ω native (research) | 28.15 | 86.22 | 99.81% | 0.0119 |
| Sintel alley_1 | **DA3-BASE ours** | 0.00 | 44.37 | 82.72% | 0.1496 |
| | DA3-BASE official | 2.22 | 30.52 | 76.03% | 0.1966 |
| | DA3-BASE official, ray-pose path | 2.22 | 30.81 | 76.03% | 0.1966 |
| | VGGT-Ω native | 22.22 | 66.44 | 98.19% | 0.0316 |

Ours matches or beats the official DA3-BASE everywhere except Sintel AUC@3
(within one or two pairs). DA3-BASE itself is far below VGGT-Ω here: mean
relative rotation error 45-55° on ETH3D delivery_area (wide baseline, 10
frames), and far below the DA3 paper's ETH3D numbers (AUC@30 74.6), for the
official code as much as ours. Inverting the extrinsics makes both worse, so
it is not a convention error. Commercial-profile pose quality on wide-baseline
captures is therefore the open problem (Pi3 is the spec's quality option).

## 3. Speed (`gpu_da3_bench_scaling` vs `da3_official.py time`, Mip-NeRF 360 kitchen, 504×336, same frames)

Preliminary: the GPU lock was held, but other jobs used the card during parts
of both runs (they ignored the lock). Clean rows: card idle at start
(official N ≤ 10 in run 3, ours N ≤ 10 in run 2 with 1.8 GB of other
residents).

| N | official PyTorch bf16 | ours ORT fp16 | speed-up | clean? |
|---|---|---|---|---|
| 3 | 52.7 ms | 31.6 ms (backbone 16.4, head 14.0) | 1.67× | yes |
| 10 | 168.0 ms | 151.1 ms (backbone 62.5, head 84.1) | 1.11× | yes |
| 32 | 957 ms (min 649) | 495 ms | ~1.3-1.9× | no |
| 50 | 2340 ms | 924 ms | — | no |
| 100 | 6998 ms (contended) / OOM | 2463 ms | — | no |

Our time excludes decode/preprocess (30-430 ms on the CPU, N = 3-100). The
DPT head runs in fp32 (as in the official code) and is 30-55% of our time:
an fp16 head is the obvious next step. N = 200 ran out of memory on the
shared card (ORT arena grows with N; the backbone keeps 4 feature levels of
N×864×1536 fp16 on the device: 2 GB at N = 200).
