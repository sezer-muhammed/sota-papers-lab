# VGGT-Ω native engine: integration results (Phase 1)

The native Rust + CUDA engine (`src/vggt/`: trunk, heads, runtime) compared with
the official PyTorch model (`vggt_omega_1b_512.pt`, FAIR non-commercial research
licence: research use only). RTX 4070 SUPER 12 GB, driver 595, CUDA 12.8
container. Tests: `src/vggt/tests.rs` (parity), `src/vggt/integration.rs`
(scaling, evaluation dump, pose op vs COLMAP). Measured 2026-11-02.

## 1. Accuracy vs the oracle (`gpu_parity_with_oracle`)

Oracle dumps from `scripts/vggt_oracle.py` (official bf16 autocast and fp32).
"rel" = RMS(ours − ref) / RMS(ref); "floor" = the official bf16 output's own
distance from fp32.

| Set | N | depth rel vs fp32 | floor | depth abs-rel median vs bf16 | pose_enc rel vs fp32 | floor | rot / fov vs fp32 (deg) | official bf16 vs fp32 |
|---|---|---|---|---|---|---|---|---|
| t1 truck | 1 | 5.72e-3 | 5.71e-3 | 0.14% | 1.28e-3 | 4.55e-4 | 0.011 / 0.101 | 0.002 / 0.037 |
| t2 truck | 2 | 7.53e-3 | 5.62e-3 | 0.13% | 6.93e-4 | 4.64e-4 | 0.063 / 0.074 | 0.047 / 0.031 |
| g8 garden | 8 | 3.80e-3 | 3.28e-3 | 0.11% | 8.65e-4 | 6.01e-4 | 0.226 / 0.062 | 0.064 / 0.063 |
| b10 bonsai | 10 | 4.95e-3 | 4.82e-3 | 0.15% | 1.29e-3 | 1.67e-3 | 0.126 / 0.112 | 0.171 / 0.104 |
| t32 truck | 32 | 1.52e-2 | 1.78e-2 | 0.11% | 8.77e-4 | 4.88e-4 | 0.153 / 0.128 | 0.084 / 0.061 |
| k50 kitchen | 50 | 1.18e-2 | 1.19e-2 | 0.07% | 1.09e-3 | 6.87e-4 | 0.153 / 0.114 | 0.107 / 0.083 |

Depth is at the official bf16 path's own distance from fp32 on every set. The
two sets above 1e-2 (t32, k50) are above it only because the official bf16 output
is too; the RMS is dominated by a few depth-edge pixels (median 0.07-0.15%).
Layer by layer (`F_i`, `G_i`, i = 4, 23), ours is within 1.0-1.2× of the official
bf16-vs-fp32 distance. The camera gates (spec §8: rotation < 0.1°, fov < 0.05°, or
≤ 1.5× the official noise) still fail on t1, t2, g8, t32 and k50
(`gpu_parity_camera_gates`). The camera head is not the cause: on the official fp32
tokens its fov error is 0.047°. The cause is the trunk's special tokens, whose
late-layer error is 1.5× the noise floor (`gpu_trunk_matches_oracle_n2`).

## 2. Evaluation subsets (`gpu_dump_eval_predictions` + `eval_subset.py --predictions`)

| Subset | Model | AUC@3 | AUC@30 | δ<1.25 | AbsRel |
|---|---|---|---|---|---|
| ETH3D delivery_area (10 frames, 45 pairs) | official | 30.37 | 86.30 | 99.81% | 0.0118 |
| | native | 28.15 | 86.22 | 99.81% | 0.0119 |
| Sintel alley_1 (10 frames, 45 pairs) | official | 20.74 | 66.00 | 98.15% | 0.0336 |
| | native | 22.22 | 66.44 | 98.19% | 0.0316 |

With 45 pairs, AUC@3 moves in steps of 0.74 when one pair crosses a 1° bin
(0.074 for AUC@30), so the ±0.3 target is finer than these subsets resolve.
AUC@30 is within 0.08 on ETH3D and 0.44 better on Sintel. AUC@3 is 2.2 lower on
ETH3D (mean relative rotation error 1.37° vs 1.29°) and 1.5 higher on Sintel.
Depth metrics match.

## 3. Speed and memory (`gpu_bench_scaling`, GPU lock held, card idle)

Mip-NeRF 360 kitchen, evenly sampled frames, 512×336. Official: `vggt_oracle.py
--time` (bf16 autocast, median of 10 after warm-up) on the same frames. Native:
median GPU time (CUDA events) of 10 runs (3 at N ≥ 100) after 2 warm-ups. Native
memory: whole-card use while the engine runs (the card held 382 MiB before).

| N | official bf16 | native | speed-up | official peak | native card use |
|---|---|---|---|---|---|
| 2 | 91.2 ms | 52.8 ms | 1.73× | 6205 MiB | 3047 MiB |
| 10 | 482.3 ms | 287.0 ms | 1.68× | 6591 MiB | 3751 MiB |
| 50 | 3615 ms | 2241 ms | 1.61× | 8691 MiB | 5191 MiB |
| 100 | OOM | 6444 ms | — | — | 7079 MiB |
| 200 | OOM | 20776 ms | — | — | 11106 MiB |

Native stage split at N = 50: trunk 2121 ms, camera 9 ms, dense 101 ms. N ≤ 32
replays a CUDA graph of the trunk.

## 4. Per-kernel profile, N = 50 (nsys, Phase 2 input)

GPU time per forward (2237 ms of kernels):

| Kernel | ms / run | calls / run | share |
|---|---|---|---|
| `attn_w8_m1_n64_pv16` (global attention, 34,450 tokens) | 961.3 | 19 | 43.0% |
| cuBLASLt `s1688gemm 128x128` (QKV, proj, fc2) | 473.3 | 144 | 21.2% |
| CUTLASS `s16816gemm 256x128` (fc1) | 277.9 | 75 | 12.4% |
| `ln_f32_c4` (LayerNorm, residual+LN) | 122.7 | 144 | 5.5% |
| `row_c4` (fc1 bias + erf GELU) | 85.6 | 72 | 3.8% |
| `conv2d_bf16` (dense head) | 83.2 | 133 | 3.7% |
| cuBLASLt `s1688gemm relu` | 76.1 | 105 | 3.4% |
| `qkv_prologue` (QK-norm, RoPE) | 70.1 | 72 | 3.1% |
| `attn_w4_m1_n64_qa_pv16` (frame attention) | 56.8 | 53 | 2.5% |

Global attention runs at about 96 TFLOPS (fp16 P·V) and the GEMMs at about
70 TFLOPS (bf16 ceiling 73-77). The remaining wins are the Phase 2 levers:
fewer global keys (merge, register, frame schedules) for the 43%, and FP8 GEMMs
(141-153 TFLOPS in spl-nn) for the 37%.

## 5. End to end: `image_to_3d.pose` vs COLMAP (`gpu_pose_op_vs_colmap`)

Mip-NeRF 360 garden (`images_4`), evenly sampled frames, compared with the
scene's COLMAP model after a Sim(3) alignment of camera centres. ATE and the RPE
translation are relative to the RMS radius of the COLMAP trajectory.

| Frames | forward | wall (incl. load) | points | ATE | RPE rot | RPE trans | rotation error mean / max |
|---|---|---|---|---|---|---|---|
| 50 | 2377 ms | 16.0 s | 1.43 M | 1.52% | 0.73° | 1.72% | 0.95° / 1.39° |
| 100 | 6650 ms | 25.9 s | 1.72 M | 1.58% | 0.39° | 1.13% | 0.94° / 1.69° |
