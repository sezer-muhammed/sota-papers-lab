# gs-oracle: gsplat reference ops for our kernel tests

`gs-oracle` runs gsplat's own CUDA ops (pinned: gsplat 1.5.3, commit 937e299,
built for sm_89 in the `spl-gs-ref` image) and fused-ssim (a7c48d6) on inputs a
Rust test saved as `.npz`, and writes their outputs (forward and backward) as
`.npz`, so a kernel can be compared element by element with the reference
everyone else benchmarks against. It also carries independent **float64**
references, so you can tell "we differ from gsplat" apart from "we are wrong".

```sh
gs-oracle <op> in.npz out.npz          # one op
gs-oracle batch DIR                    # DIR/<name>.<op>.in.npz -> DIR/<name>.<op>.out.npz, one process
gs-oracle check <op> in.npz out.npz    # any out.npz (ours or gsplat's) vs the float64 reference
gs-oracle compare ours.npz gsplat.npz  # per-key error table between two files
gs-oracle example <op> in.npz [--seed S --n N --width W --height H --cameras C]
gs-oracle selftest [--json report.json] [--keep DIR]
```

Paths can be anywhere on the host; the wrapper mounts them. Start-up (container
plus `import torch`) is about 3 s, so put many cases in one directory and use
`batch`.

A typical kernel test loop:

1. The Rust test (on the GPU host, `~/spl-work/bin/gpu-test`) writes
   `/out/oracle/raster_small.rasterize.in.npz` and its own result as
   `raster_small.rasterize.ours.npz` (`/out` is `~/spl-work/out`).
2. `gs-oracle batch ~/spl-work/out/oracle` writes the `*.out.npz` files.
3. `gs-oracle compare .../raster_small.rasterize.ours.npz .../raster_small.rasterize.out.npz`,
   or `gs-oracle check rasterize ... .in.npz ... .ours.npz` against float64, or
   the Rust test reads `out.npz` back and asserts tolerances itself.

`python benchmarks/image-to-3d/py/oracle_example.py <op> out.npz` (numpy only)
writes a valid random input for every op. Read it as the executable version of
the tables below.

## File format

- Standard `.npz`: a zip of `<key>.npy` files. `np.savez` (stored) and
  `np.savez_compressed` (deflate) are both read. Outputs are written stored
  (uncompressed), so a minimal reader only needs zip "stored" entries plus the
  `.npy` header.
- `.npy` v1.0/v2.0, little-endian, C order. Dtypes: `<f4` (float32), `<i4`
  (int32), `<i8` (int64), `|b1` or `|u1` (masks; non-zero is true). Float64
  inputs are accepted and cast to float32 (the CUDA ops are float32).
- Scalars are 0-d arrays (shape `()`); shape `(1,)` is accepted too.
- Unknown input keys are reported on stderr (`unused inputs`), which catches typos.
- Every output file has `_meta`: a `|u1` array holding UTF-8 JSON (op, run time
  in ms, gsplat / torch / CUDA / fused-ssim versions, GPU name, unused inputs).
- Upstream gradients are named `v_<output>`, gradients w.r.t. inputs
  `v_<input>` (gsplat's naming). Every backward input is optional; a missing
  one counts as zeros. Without any, only the forward runs.
- `C` = cameras, `N` = Gaussians, `K` = SH coefficients per channel
  (`(degree+1)^2`, 16 for degree 3), `D` = colour channels, `M` = intersections.

Conventions shared with gsplat: `viewmats` are world-to-camera 4x4 (OpenCV
axes: +x right, +y down, +z forward); `Ks` are pixel intrinsics; pixel (i, j)
has its centre at (j + 0.5, i + 0.5); quaternions are `wxyz` and need not be
normalised; `scales` and `opacities` are activated values (not log / logit);
`conics` are the upper triangle `[a, b, c]` of the inverse 2D covariance, and
the Gaussian falloff is `sigma = 0.5 (a dx^2 + c dy^2) + b dx dy`.

## Ops

### `sh`: spherical harmonics (gsplat `spherical_harmonics`)

| In | Shape, dtype | |
|---|---|---|
| `degree` | `()` i4 | degree to evaluate, 0..4 |
| `dirs` | `[N,3]` f4 | view directions, normalised inside |
| `coeffs` | `[N,K,3]` f4 | `K >= (degree+1)^2` |
| `masks` | `[N]` b1, optional | false: output and gradients are 0 |
| `v_colors` | `[N,3]` f4, optional | upstream gradient |

Out: `colors [N,3]` (raw SH sum: no +0.5, no clamp; `rasterization()` adds
those), `v_dirs [N,3]`, `v_coeffs [N,K,3]`.

### `project`: EWA projection (gsplat `fully_fused_projection`, dense)

| In | Shape, dtype | |
|---|---|---|
| `means` | `[N,3]` f4 | world space |
| `quats`, `scales` | `[N,4]`, `[N,3]` f4 | or `covars [N,6]` (upper triangle xx xy xz yy yz zz) instead |
| `viewmats`, `Ks` | `[C,4,4]`, `[C,3,3]` f4 | |
| `width`, `height` | `()` i4 | |
| `eps2d` | `()` f4, default 0.3 | added to the 2D covariance diagonal |
| `near_plane`, `far_plane` | `()` f4, 0.01 / 1e10 | |
| `radius_clip` | `()` f4, default 0 | |
| `calc_compensations` | `()` i4, default 0 | antialiasing factor `sqrt(det(S) / det(S + eps2d I))` |
| `opacities` | `[N]` f4, optional | opacity-aware (tighter) radii, as `rasterization()` passes them |
| `camera_model` | `()` i4, default 0 | 0 pinhole, 1 ortho, 2 fisheye |
| `v_means2d`, `v_depths`, `v_conics`, `v_compensations` | `[C,N,2]`, `[C,N]`, `[C,N,3]`, `[C,N]` f4 | upstream |

Out: `radii [C,N,2] i4` (x and y radius in pixels, `ceil(3.33 sqrt(S_ii))`, or
less with `opacities`; 0 means culled), `means2d [C,N,2]`, `depths [C,N]`
(camera z), `conics [C,N,3]`, `compensations [C,N]`; backward `v_means`,
`v_quats`, `v_scales` (or `v_covars`), `v_viewmats [C,4,4]`. Culled entries
(either radius 0) are written as 0 (gsplat leaves them uninitialised) and get
no gradient.

### `isect`: tile binning and sort (gsplat `isect_tiles` + `isect_offset_encode`)

| In | Shape, dtype | |
|---|---|---|
| `means2d`, `radii`, `depths` | `[C,N,2]` f4, `[C,N,2]` i4, `[C,N]` f4 | from `project` |
| `width`, `height` | `()` i4 | |
| `tile_size` | `()` i4, default 16 | |
| `tile_width`, `tile_height` | `()` i4, optional | default `ceil(width / tile_size)` etc. |

Out: `tiles_per_gauss [C,N] i4`, `isect_ids [M] i8`, `flatten_ids [M] i4`,
`isect_offsets [C,TH,TW] i4`, `tile_width`, `tile_height`.

- A Gaussian covers tiles `[floor(m/ts - r/ts), ceil(m/ts + r/ts))` per axis,
  clamped to the grid (a rectangle, not an ellipse test).
- `isect_id = camera << (32 + tile_bits) | tile_id << 32 | float_bits(depth)`
  with `tile_bits = bit_length(TW * TH)`, `tile_id = ty * TW + tx`. Sorted
  ascending: by camera, tile, then depth (positive float bits sort like the
  floats).
- `flatten_ids` holds `camera * N + gaussian` in that order.
- `isect_offsets[c,ty,tx]` is the start of the tile's run in the sorted list;
  the run ends at the next tile's offset (or `M`).

### `rasterize`: alpha compositing (gsplat `rasterize_to_pixels`, dense)

| In | Shape, dtype | |
|---|---|---|
| `means2d`, `conics` | `[C,N,2]`, `[C,N,3]` f4 | |
| `colors` | `[C,N,D]` f4 | D = 3 usually; any D works |
| `opacities` | `[C,N]` or `[N]` f4 | per view (after any compensation) |
| `width`, `height`, `tile_size` | `()` i4 | |
| `isect_offsets`, `flatten_ids` | `[C,TH,TW]` i4, `[M]` i4, optional | give both, or give `radii [C,N,2] i4` and `depths [C,N] f4` and the oracle bins them (and returns `isect_ids`, `flatten_ids`, `isect_offsets`) |
| `backgrounds` | `[C,D]` f4, optional | |
| `absgrad` | `()` i4, default 0 | also return `v_means2d_abs` (sum of per-pixel absolute gradients, AbsGS) |
| `v_render_colors`, `v_render_alphas` | `[C,H,W,D]`, `[C,H,W,1]` f4 | upstream |

Out: `render_colors [C,H,W,D]`, `render_alphas [C,H,W,1]`; backward
`v_means2d`, `v_conics`, `v_colors`, `v_opacities` (shape of the input),
`v_backgrounds`, `v_means2d_abs`.

Per pixel, over the tile's list front to back: `alpha = min(0.999, o exp(-sigma))`;
skip when `sigma < 0` or `alpha < 1/255`; stop *before* the Gaussian that would
make `T * (1 - alpha) <= 1e-4`; `C += c alpha T`, `T *= 1 - alpha`. Then
`render_alphas = 1 - T` and `render_colors += T * background`.

### `ssim`: fused SSIM and the 3DGS loss (fused-ssim)

| In | Shape, dtype | |
|---|---|---|
| `img1` | `[B,3,H,W]` f4 (or `[3,H,W]`) | the render: the gradient is taken w.r.t. this |
| `img2` | same | the target |
| `padding_valid` | `()` i4, default 0 | 0: zero-padded map averaged over all pixels (3DGS); 1: map cropped by 5 px (gsplat's trainer) |
| `lambda_dssim` | `()` f4, default 0.2 | |
| `v_ssim` | `()` f4, default 1 | upstream gradient of `ssim` |

Out: `ssim ()`, `v_img1` (= `v_ssim * dssim/dimg1`), `l1 ()` (mean |img1 - img2|),
`loss () = (1 - lambda) l1 + lambda (1 - ssim)`, `v_img1_loss` (gradient of
`loss`; the L1 part is `sign(img1 - img2) / numel` with sign(0) = 0), and the
float64 3DGS reference `ssim_map_ref [B,3,H,W] f8` and `ssim_ref () f8`.
Window: 11x11 Gaussian, sigma 1.5, `C1 = 0.01^2`, `C2 = 0.03^2`.

### `adam`: one optimizer step

| In | Shape, dtype | |
|---|---|---|
| `param`, `grad`, `exp_avg`, `exp_avg_sq` | `[N,...]` f4 | same shape |
| `step` | `()` i4, default 0 | steps already taken (this is step `step + 1`) |
| `lr`, `beta1`, `beta2`, `eps` | `()` f4: 1e-3, 0.9, 0.999, 1e-15 | 3DGS and gsplat use eps 1e-15 |
| `valid` | `[N]` b1, optional | per-row visibility for the selective variant |

Out: `param_out`, `exp_avg_out`, `exp_avg_sq_out` from `torch.optim.Adam`
(single-tensor path, bias-corrected); with `valid`, also `sel_param_out`,
`sel_exp_avg_out`, `sel_exp_avg_sq_out` from gsplat's fused selective Adam
kernel (Taming-3DGS): **no bias correction**, `p -= lr m / (sqrt(v) + eps)`,
rows with `valid = false` untouched (moments too).

### `relocation`: 3DGS-MCMC relocation (gsplat `compute_relocation`)

| In | Shape, dtype | |
|---|---|---|
| `opacities`, `scales` | `[N]`, `[N,3]` f4 | activated |
| `ratios` | `[N]` i4 | N_i = draws + 1, clamped to `[1, n_max]` |
| `n_max` | `()` i4, default 51 | binomial table size (ours: 50) |
| `min_opacity` | `()` f4, default 0.005 | clamp of the strategy |

Out: `new_opacities`, `new_scales` (the kernel's float sums), and
`new_opacities_clamped` (`clamp(o', min_opacity, 1 - 2^-23)`, what the MCMC
strategy writes).

### `multinomial`: sampling without replacement (`torch.multinomial`)

| In | Shape, dtype | |
|---|---|---|
| `weights` | `[n]` f4 | unnormalized |
| `k`, `trials`, `seed` | `()` i4 | k draws per trial, default 100000 trials |

Out: `counts [n] i4`, how often each index was drawn over all trials (the
inclusion frequencies FastGS's in-densify VCP relies on).

### `render`: the whole forward and backward (gsplat `rasterization`)

| In | Shape, dtype | |
|---|---|---|
| `means`, `quats`, `scales`, `opacities` | `[N,3]`, `[N,4]`, `[N,3]`, `[N]` f4 | activated |
| `colors` | `[N,K,3]` f4 SH coefficients, or `[N,D]` f4 colours | |
| `sh_degree` | `()` i4 | default: all of `K` |
| `viewmats`, `Ks`, `width`, `height` | as in `project` | |
| `tile_size`, `eps2d`, `near_plane`, `far_plane`, `radius_clip`, `camera_model` | as above | |
| `backgrounds` | `[C,3]` f4, optional | |
| `rasterize_mode` | `()` i4, default 0 | 0 classic, 1 antialiased (opacity times compensation) |
| `absgrad` | `()` i4, default 0 | |
| `grad_viewmats` | `()` i4, default 1 | return `v_viewmats` |
| `v_render_colors`, `v_render_alphas` | `[C,H,W,3]`, `[C,H,W,1]` f4 | upstream |

Out: `render_colors`, `render_alphas`, the intermediates `radii`, `means2d`,
`depths`, `conics`, `opacities_view [C,N]`, `tiles_per_gauss`, `isect_ids`,
`flatten_ids`, `isect_offsets`; backward `v_means`, `v_quats`, `v_scales`,
`v_opacities`, `v_colors`, `v_viewmats`, `v_means2d` (the screen-space gradient
the densifier reads) and `v_means2d_abs`. Colours from SH are
`max(SH(dir) + 0.5, 0)` with `dir = mean - camera centre`, evaluated only for
Gaussians with non-zero radii. Dense mode (`packed=False`).

## Accuracy you should expect

`gs-oracle selftest` runs every op on random scenes through the file path,
compares gsplat with the float64 references and runs each op twice to expose
non-determinism (gsplat's backward uses atomics). The numbers below are from
the selftest on the RTX 4070 SUPER; `rel` is `max |gsplat - fp64| / max |fp64|`
over the array, p99.9 the 99.9th percentile of the same ratio.

gsplat 1.5.3, torch 2.9.1+cu128, fused-ssim a7c48d6, 2026-11-01:

| Case | Forward (max rel) | Backward (max rel) | Discrete outputs | Second run |
|---|---|---|---|---|
| `sh`, N 512, degree 3 | colors 2.8e-7 | v_coeffs 4.4e-7, v_dirs 1.2e-7 | | v_dirs differs by 6e-8 |
| `project`, N 512, C 2, opacity-aware, compensations | means2d 1.0e-7, depths 7.6e-8, conics 6.4e-7, compensations 8.3e-7 | v_means 1.5e-7, v_quats 6.3e-7, v_scales 9.3e-7, v_viewmats 1.8e-7 | radii 0 / 2048 differ | v_viewmats differs (atomics) |
| `isect`, N 256, C 2, 64x48 | | | isect_ids, flatten_ids (1456), offsets, tiles_per_gauss all exact | identical |
| `rasterize`, N 128, C 2, 64x48, background, absgrad | colors 2.0e-7, alphas 1.9e-7 | v_means2d 1.6e-7, v_conics 4.2e-7, v_colors 4.3e-7, v_opacities 1.5e-7, v_backgrounds 9.9e-8 | binning exact | gradients differ, at most 3e-5 absolute (atomics) |
| `ssim`, 2x3x48x64, same | ssim 7.4e-8, l1 2.1e-9, loss 1.8e-7 | v_img1 3.3e-6, v_img1_loss 2.7e-6 | | identical |
| `ssim`, 1x3x48x64, valid | ssim 3.8e-8, loss 1.0e-7 | v_img1 1.4e-6, v_img1_loss 1.4e-6 | | identical |
| `adam`, 1000x3, eps 1e-15 | torch Adam: param 3.2e-8, moments 1.0e-7; selective: same | | | identical |
| `render`, N 200, 64x48, SH degree 3 | colors 3.4e-6, alphas 3.8e-6 | v_means 2.4e-6, v_quats 6.1e-6, v_scales 5.8e-6, v_opacities 1.2e-6, v_colors 2.0e-6, v_viewmats 4.1e-6 | radii 0 / 400 differ | gradients differ (atomics) |
| `render`, N 300, C 2, 80x60 | colors 3.1e-6, alphas 4.6e-6 | all at most 4.0e-6 | radii 0 / 1200 differ | gradients differ (atomics) |

The default pass thresholds of `check` (on p99.9; the max may be 20x that)
are 1e-5 for `sh` and `adam`, 1e-4 for the other float outputs, 2e-4 for
`render`, and exact for discrete outputs except 0.1% for `render`.

Read these as the tolerance a float32 kernel can meet: aim for errors against
float64 no worse than gsplat's, not for bitwise equality with gsplat. Discrete
outputs (radii, tile lists, intersection ids) should match exactly; a mismatch
there almost always means a different rounding or bound formula.

## Densification (E6) against the oracle and the CPU mirrors

`apps/image-to-3d/src/kernels/densify/gpu_tests.rs` (run with
`scripts/gpu-test.sh <checkout> densify::gpu_tests::gpu_ --skip gpu_bench`).
The cases are staged under `/out/image-to-3d/oracle-densify`; run
`gs-oracle batch ~/spl-work/out/image-to-3d/oracle-densify` (from a checkout
with these ops: `gs-ref --entry <checkout>/benchmarks/image-to-3d/py/oracle.py batch ...`)
and re-run the tests to include the gsplat comparison.

| Test | Reference | Tolerance | Measured (RTX 4070 SUPER, 2026-11-02) |
|---|---|---|---|
| `densify_plan`, ADC / FastGS / final prune, 40k Gaussians, 5 rule sets | `plan_cpu` (decisions in logit / log space, IEEE division) | bit exact (flags, out_count, priority, candidate count) | 0 mismatches |
| Emission, 5k and 30k Gaussians, every array (params, last_visible, 6 gradient groups, f32 moments, 3 stats) | host model `emission_cpu` + split rule | bit exact; split-child means 2e-5 rel (device Box-Muller uses fast-math) | exact; means 6.4e-7 |
| Capacity capping (tail 1.7k into 1.6k free) | `cap_tail_host` | bit exact | 54 dropped, store exact |
| Radix sort, 1M random 32-bit keys with duplicates, 300k 30-bit, 77 | stable CPU sort | exact keys and values | exact, stable |
| Morton re-sort, 200k Gaussians, every array | CPU codes (IEEE division), gather model | exact | 0 codes differ, every array permuted |
| Gumbel-top-k VCP, top-5 of 20, 2e5 trials | sequential sampling (1e6 trials); `multinomial` op (1e6 trials) | chi-square < 36.19 (df 19, p = 0.01) | 22.2; vs torch 15.8 |
| MCMC relocation, o in {0.006 .. 0.97}, N = 1..50 | `relocation` op (gsplat) | 1e-4 rel | 1.9e-6 |
| same | f64 closed form | 1e-5 rel for N <= 2, 1e-3 all N (the f32 alternating sum cancels; gsplat's too) | 6.9e-6 (N 2), 8.4e-5 (N 8), 2.9e-4 (N 32) |
| MCMC round, 20k with 2,244 dead | dead list, copies, zeroed moments, (o', s') vs f64 | exact / 1e-3 | exact; 7.2e-6 |
| MCMC noise, 2^20 draws | covariance (k Sigma)^2 | 1% of the largest variance | 0.30% |
| `vcd_mask`, `scores_add` (+ overflow guard) | CPU | exact | exact |


### Densification performance (RTX 4070 SUPER, exclusive `gpu.lock`, GPU idle before: 0 %, 382 MiB)

`gpu_bench_densify` (CUDA events, median of 10) against gsplat 1.5.3 `strategy.ops`
(`py/bench_densify_gsplat.py`) and Faster-GS's model ops (`py/bench_densify_fastergs.py`,
spl-bl-fastergs image). Same workloads: ADC step = 5 % clone, 5 % split, 10 % prune over
parameters + Adam moments; MCMC = relocate 2 % dead + grow 5 %. Times in ms.

| Op | 1M ours / gsplat / Faster-GS | 3M | 6M |
|---|---|---|---|
| ADC emission (178 words/G) | **6.87** / 17.6 / 11.4 | **25.4** / 60.2 / 34.6 | **52.8** / 170 / 106 |
| Morton re-sort, from random order | 8.94 / - / **6.21** | 34.4 / - / **21.6** | 96.1 / - / **49.4** |
| Morton re-sort, steady state (near-sorted) | 6.99 | 22.8 | 46.7 |
| MCMC round | **2.33** / 5.81 / 5.84 | **7.92** / 17.3 / 18.0 | **17.6** / 32.6 / 48.4 |
| MCMC noise | **0.035** / 1.52 / 0.114 | **0.366** / 5.00 / 0.450 | **0.722** / 10.2 / 0.826 |
| densify_plan (incl. 1 sync) | 0.05 | 0.29 | 0.56 |
| Radix sort 32-bit (k,v) | 0.109 | 0.255 | 1.03 |
| VCP Gumbel-top-k (keys+sort+mark) | 0.41 | 1.07 | 2.19 |

Before/after the Nsight Compute pass (unlocked, same contention both ways): the radix sort's
single-block scan was 60 % of the sort; per-digit row scan made it 5-6x faster (1M: 0.632 -> 0.123
ms, 3M: 1.92 -> 0.315 ms), Gumbel-top-k 2.2-3.7x. Slab moves run at ~89 % of peak memory
throughput (ncu), ~420 GB/s measured; emission is bound by its 4 passes over every word.
Morton is in-place (fixed capacity); Faster-GS's out-of-place gather does half the traffic but
allocates a second copy of every tensor.
