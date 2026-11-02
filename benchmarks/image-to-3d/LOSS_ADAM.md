# image-to-3d: loss and optimizer kernels, measured

Fused L1 + D-SSIM (`apps/image-to-3d/cuda/loss.cu`, `src/kernels/loss.rs`) and the
multi-group sparse Adam (`cuda/adam.cu`, `cuda/adam.cuh`, `src/kernels/adam.rs`) on the
RTX 4070 SUPER (sm_89, 56 SMs, 504 GB/s), 2026-11-01. Accuracy against the oracle and
f64 references: [ORACLE.md, "Our kernels against the oracle"](ORACLE.md#our-kernels-against-the-oracle).

## How these were measured

- Ours: `apps/image-to-3d/scripts/gpu-test.sh <checkout> kernels::loss::tests::gpu_loss_bench`
  and `kernels::adam::tests::gpu_adam_bench` (CUDA events; loss: median of 200 after 20
  warm-ups; Adam: median of 50 after 5).
- References: `gs-ref --entry benchmarks/image-to-3d/tools/bench_loss_adam.py` (same
  event protocol in the gsplat image: torch 2.9.1, gsplat 1.5.3, fused-ssim a7c48d6).
- **Contention.** The GPU was shared with other jobs the whole time (nvidia-smi 92-100%
  utilization before and after every run; the `gpu.lock` queue was 16 runs deep with
  30-minute baselines), so the reference and our runs were taken back to back in the same
  window rather than under the lock. A 512 MiB D2D copy measured 223 GB/s median (424 GB/s
  p10) in that window versus 428 GB/s idle: medians are inflated by roughly the same factor
  for both sides; ratios are the meaningful number. Re-run both under `SPL_GPU_LOCK=1`
  for absolute numbers.

## Loss: 0.8 L1 + 0.2 (1 − SSIM), forward + backward

| Size | fused-ssim + torch L1 (the 3DGS / Faster-GS / FastGS path) | fused-ssim SSIM only | ours v1 (two kernels) | ours v2 (one kernel) | v2 vs the 3DGS path |
|---|---|---|---|---|---|
| 1297x840 (garden 1/4) | 0.780 ms | 0.556 ms | 0.423 ms | **0.362 ms** | 2.2x |
| 1237x822 | 0.710 | 0.505 | 0.396 | **0.338** | 2.1x |
| 1600x1063 | 1.315 | 0.900 | 0.640 | **0.568** | 2.3x |
| 3840x2160 | 11.88 | 7.43 | 4.42 | **4.06** | 2.9x |

Ours computes L1, SSIM, both gradients and the loss sums in one launch (v2) and writes
dL/dx directly; the reference path runs fused-ssim's two kernels plus four PyTorch
elementwise/reduction kernels and autograd. Target resampling at scale 1 (u8 → f32 planar
copy) adds 0.02 ms at 1297x840.

### Optimization steps (1297x840, same contended window)

| Step | v2 time | What changed |
|---|---|---|
| v1 (fused-ssim structure, 16x16 tiles, partials through DRAM) | 0.42 ms | reference |
| v2 first version (32x16 output tile, halo recompute, one element per thread per phase) | 0.52 ms | 4x less DRAM traffic but 2.6x the shared-memory loads: slower than v1 |
| v2 register sliding windows | 0.35-0.36 ms | horizontal passes 3 columns per thread (13 values in registers), vertical passes 3 / 2 rows per thread: shared loads per output 147 → 58 floats |
| integer clean-up, cp.async zero-fill tile load | 0.358 ms (p10 0.356) | instructions per output 1118 → 1033, integer ops 259 → 179; within noise |

Nsight Compute (`ssim_fused`, 1297x840, base clocks): 40-48 registers, 45-46 KB static
shared memory → 2 blocks (1024 threads) per SM, 63% achieved occupancy; IPC 2.2; FP32
1.7 G thread-instructions (548 per output); shared-memory pipe 73% busy with 41% extra
wavefronts from bank conflicts (phases 2 and 4 map 14 / 11 threads per row); top stall
`barrier` (3.3 cycles per issue: phases 3-5 keep 378 / 286 / 256 of 512 threads busy),
then `mio_throttle` 2.6. DRAM 12% (58 GB/s): the kernel is issue / shared-memory bound,
not memory bound. Remaining ideas: a 32x12 tile with the five moments split 2 + 3 to fit
3 blocks per SM, conflict-free row strides, balanced phase mappings (see open issues).

## Adam over the SoA store

Bytes per element and step: read p, g, m, v and write p, m, v = 28 B (bf16 moments 20 B).
"Effective" bandwidth uses that model (VisibleOnly: the visible half).

| Gaussians, configuration | torch.optim.Adam fused | gsplat selective Adam (Taming) | ours | ours vs torch |
|---|---|---|---|---|
| 1M, all 59 floats dense (Faster-GS parity) | 7.93 ms | | **4.11 ms** (402 GB/s) | 1.9x |
| 1M, main 14 floats (SH rest not due) | 1.64 (foreach 1.14) | | **0.90** (438 GB/s) | 1.8x (1.3x foreach) |
| 1M, 59 floats, 50% visible, skip invisible | | 1.96 | **2.08** (398 GB/s) | 0.94x gsplat |
| 1M, 59 floats, bf16-SR moments | | | **2.68** (440 GB/s) | 3.0x |
| 3M, all 59 dense | 25.5 | | **13.7** | 1.9x |
| 3M, 50% visible | | 8.65 | **6.52** | 1.3x gsplat |
| 6M, all 59 dense | 48.8 | | **24.1** (411 GB/s) | 2.0x |
| 6M, main 14 | 10.5 | | **5.54** (425 GB/s) | 1.9x |
| 6M, 50% visible | | 19.0 | **13.6** | 1.4x gsplat |
| 6M, all 59 bf16-SR | | | **17.3** | 2.8x |
| 6M, main 14 bf16-SR | | | **3.95** | 2.7x |

One launch covers every due group (persistent grid of 6 x 256-thread blocks per SM,
16-byte loads and stores per quad, the group found from a 6-entry prefix); Overwrite-mode
groups skip the gradient load of quads with no Gaussian rendered this iteration; skipped
(VisibleOnly) quads read only `last_visible`. Nsight Compute (`adam_step`, 1M, all
groups, 50% rendered): DRAM 82.6% of peak, 1.54 GB moved (model 1.65 GB; invisible
gradients are not read), 48 registers, 76% warps active, `long_scoreboard` the dominant
stall: memory bound at about 400 GB/s of the card's ~430 achievable.

What the step plans buy (per iteration at 1M, from the table): FastGS / Taming's SH rest
every 16 iterations turns 4.1 ms into 0.90 ms on the 15 iterations out of 16 when only the
main groups step (1.1 ms average); FastGS's every-32 / every-64 cadence after 15k
iterations makes the optimizer almost free.
