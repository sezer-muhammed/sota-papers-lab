# image-to-3d sort and binning: measurements (RTX 4070 SUPER, sm_89)

Garden (Mip-NeRF 360) 1/4-res workloads from `sort::bench::gpu_bench_sort_and_binning`,
CUDA events, median of 15 after 3 warm-ups. **The GPU was shared** (other agents' baselines
ran; nvidia-smi 84-100 % utilization around the runs; the lock queue was ~15 jobs deep), so
absolute numbers are upper bounds. Ours and the CUB reference ran back to back under the same
contention. The CUB reference is `cub_ref.cu` (nvcc 12.8, CUB 2.7.0) on dumps of the same
data; every reference result (depth order, tile order, tile ranges, gsplat 64-bit order,
gsplat offsets) was **identical** to ours on all six workloads.

Times in ms (2026-11-02, commit after 020b744).

| workload | V | I | depth sort ours / CUB | tile sort ours (2 × u16 onesweep) / CUB u16 (Faster-GS) / CUB u32 | gsplat 64-bit sort |
|---|---|---|---|---|---|
| n1m s4 | 0.55M | 2.3M | 0.078 / 0.078 | 0.140 / 0.130 / 0.131 | 0.540 |
| n3m s8 | 1.65M | 8.2M | 0.175 / 0.161 | 0.457 / 0.502 / 0.663 | 5.03 |
| n6m s10 | 3.29M | 14.3M | 0.328 / 0.339 | 0.794 / 0.923 / 1.200 | 8.19 |
| n6m s10 v40 | 2.76M | 14.0M | 0.263 / 0.244 | 0.773 / 0.900 / 1.165 | 8.30 |
| n6m s18 | 3.34M | 28.7M | 0.337 / 0.355 | 1.585 / 1.920 / 4.769 | 17.47 |
| n6m s30 | 3.42M | 59.6M | 0.351 / 0.368 | 3.295 / 6.210 / 8.300 | 36.99 |

Binning (instance emission and tile ranges), ours vs Faster-GS's post-sort range/bucket
extraction (which also does a host readback) and gsplat's offset encode:

| workload | scan_tiles + emit | scan_emit (fused, used by bin_tiles) | tile_hist + bin_scan (our ranges) | Faster-GS ranges+buckets+readback | gsplat encode |
|---|---|---|---|---|---|
| n1m s4 | 0.285 | 0.282 | 0.029 | 0.599 | 0.053 |
| n6m s10 | 1.743 | 1.706 | 0.093 | 0.880 | 0.298 |
| n6m s18 | 1.956 | 2.956 (p10 2.57) | 0.174 | 1.022 | 0.538 |
| n6m s30 | 4.550 (p10 3.04) | 3.751 | 0.321 | 1.138 | 1.048 |

Notes
- Emission dominates (1.5-3.7 ms). Faster-GS and gsplat emit inside their preprocess/isect
  kernels, which the CUB reference does not time, so this column has no like-for-like
  reference yet. It is far from the write-bandwidth bound (14.3M × 6 B ≈ 86 MB ≈ 0.17 ms):
  per-Gaussian work is serial per thread and warp-divergent. The next optimization is
  warp-cooperative emission (one warp per large Gaussian, as the spec's `emit_warp` lane-row
  split already does in part) and Nsight Compute profiling under the lock (`profile.sh`).
- scan_emit saves count_tiles (0.42 ms at 6M) and one V-length read; versus the two-kernel
  form it is break-even to 18 % faster except under contention at s18.
- gsplat's 64-bit single sort is 10-11× slower than our tile sort at every size.
- Raw logs: `~/spl-work/k-sort/bench_v2.txt`, `~/spl-work/k-sort/cub_v1.txt` on the GPU host.
