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

## Raster kernels (`raster_fwd`, `raster_bwd`) against the oracle

Owner: raster (E3). Test: `kernels::raster::tests::gpu_raster_matches_gsplat_oracle`
(apps/image-to-3d/src/kernels/raster/tests.rs). Three deterministic 2D scenes
(`raster_small` 300 Gaussians 64x48 with a background, `raster_odd` 1,200 on
67x53, `raster_dense` 20,000 on 160x120), opacities at most 0.98 so neither
clamp binds (gsplat 1.5.3 clamps alpha at **0.999**, our `ALPHA_MAX` is 0.99;
see below), radii opacity-aware + 1 px so gsplat's rectangle binning holds every
contributing pixel, random dL/dC and dL/dalpha.

Two passes on the GPU host:

```sh
apps/image-to-3d/scripts/gpu-test.sh <checkout> gsplat_oracle    # stages raster_*.rasterize.in.npz, fails
gs-oracle batch ~/spl-work/out/image-to-3d/oracle                 # gsplat's outputs
apps/image-to-3d/scripts/gpu-test.sh <checkout> gsplat_oracle    # compares, writes raster_*.rasterize.ours.npz
for c in raster_small raster_odd raster_dense; do                 # grade ours against float64
  gs-oracle check rasterize ~/spl-work/out/image-to-3d/oracle/$c.rasterize.{in,ours}.npz
done
```

The test runs our kernels twice per case, on gsplat's own tile lists
(`isect_offsets`, `flatten_ids`) and on our exact lists (spec-raster 2.8), and
asserts: image max abs ≤ 2e-5 and mean abs ≤ 1e-6, T max abs ≤ 1e-6 against
gsplat; gradients within spec-raster 6.4 (cosine ≥ 0.99999, rel-L2 ≤ 1e-4,
element rel ≤ 1e-3) against our CPU f64 reference; gradients against gsplat
within rel-L2 1e-3, because gsplat's backward recovers T back to front by
division and its own error against float64 reaches 3e-4 on these cases.

Measured (RTX 4070 SUPER, 2 Nov 2026), `gs-oracle check` against float64,
max |x - fp64| / max |fp64| over the array:

| Case | Output | gsplat 1.5.3 | ours (K = 4, 16x16) |
|---|---|---|---|
| raster_small | render_colors / alphas | 2.5e-7 / 1.9e-7 | 3.0e-7 / 2.1e-7 |
| | v_means2d / v_conics | 4.6e-6 / 9.8e-7 | 6.2e-7 / 4.7e-7 |
| | v_colors / v_opacities | 1.3e-6 / 9.1e-7 | 2.9e-7 / 2.7e-7 |
| raster_odd | render_colors / alphas | 7.9e-7 / 3.0e-8 | 1.2e-6 / 3.0e-8 |
| | v_means2d / v_conics | **1.4e-4 / 3.0e-4** (fails `check`) | 3.0e-6 / 9.8e-7 |
| | v_colors / v_opacities | 1.7e-4 / 2.6e-4 (fails) | 7.5e-7 / 2.3e-6 |
| raster_dense | render_colors / alphas | 1.1e-6 / 3.2e-8 | 9.9e-7 / 3.2e-8 |
| | v_means2d / v_conics | 1.3e-4 / 1.5e-4 | 8.9e-7 / 1.2e-6 |
| | v_colors / v_opacities | 9.7e-5 / 1.0e-4 | 2.7e-7 / 2.2e-6 |

Ours against gsplat directly: images within 1.1e-6 max abs, T within 2.2e-7,
gradients rel-L2 ≤ 1.9e-6 on `raster_small` and ≤ 3.3e-4 on the deep cases
(gsplat's error, per the table). `v_backgrounds` in our file is summed on the
host from our T; it is not a raster output.

The other raster GPU tests compare with the CPU f64 reference of
`kernels::raster::scene` (itself checked by finite differences with frozen
decisions: 534 gradients agree to 2.5e-8), over every implemented variant
(K = 1, 2, 4, 8; 16x16 and 16x12 tiles; Faster-GS blending; fp16 checkpoints;
1 and 4 extra channels), alpha clamping, terminating pixels, empty scenes,
1x1 and 17x33 images, a whole-image splat, and a tile listing 70,000
Gaussians: images within 1.5e-6, gradients rel-L2 within 4.2e-6 (fp16
checkpoints: 1.3e-3).

Constant to settle: gsplat 1.5.3 clamps alpha at 0.999 (`min(0.999f, ...)`
in `RasterizeToPixels3DGSFwd.cu`; 3DGS uses 0.99), while
`splat_common.cuh` sets `ALPHA_MAX 0.99` and spec-raster 0 calls that
"gsplat". The raster reads `ALPHA_MAX`, so parity with gsplat at the clamp is
a one-line change there (preprocess, E1) plus `layout.rs`.
