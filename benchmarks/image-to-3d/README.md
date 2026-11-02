# Image-to-3D benchmark harness (Phase 0)

Datasets, metrics, a gsplat oracle for kernel tests, and a baseline runner, all
on the GPU host. Design context: `reports/Image to 3D.md`, sections 1, 6 and 9.

| Path | What it is |
|---|---|
| `datasets.toml` | Manifest: URLs, hashes, licences, the protocol folder per scene |
| `fetch_datasets.py` | Fetches only what the protocol uses and verifies it (stdlib Python) |
| `docker/Dockerfile` | `spl-gs-ref`: official PyTorch 2.9.1 / CUDA 12.8 image (Python 3.11), gsplat 1.5.3 and fused-ssim built for sm_89, LPIPS weights |
| `py/metrics.py` | PSNR, SSIM, LPIPS as 3DGS and gsplat compute them (`gs-metrics`) |
| `py/oracle.py` | gsplat reference ops on npz files (`gs-oracle`), see [ORACLE.md](ORACLE.md) |
| `run_baseline.sh` | One tool on one scene under the GPU lock, one JSON line of results |
| `tools/<tool>.sh` | Baseline adapters: `gsplat` (3DGS densification), `gsplat-mcmc` (3DGS-MCMC) |
| `install.sh` | Host setup: wrappers, image, datasets |

## Setup on the GPU host

```sh
cd ~/spl-work/wt/<checkout>/benchmarks/image-to-3d
./install.sh                 # = --bin --image --datasets
./install.sh --freeze        # after a rebuild: refresh docker/constraints.txt, then commit it
```

`--image` builds with the local `default` buildx builder (other projects own the
other builders). The CUDA extensions compile in a throwaway stage that borrows
only nvcc and the headers from `spl-cuda-devel`; the image itself adds about
2.4 GB to the PyTorch base. Wrappers land in `~/spl-work/bin`: `gs-ref` (any
command in the image, on the GPU, as your uid, with host paths in the arguments
mounted at the same path), `gs-oracle` and `gs-metrics`.

Lint before committing: `ruff check . && ruff format --check .` (settings in
`ruff.toml`) and `shellcheck *.sh tools/*.sh bin/*`.

## Datasets

`~/spl-work/datasets/<dataset>/<scene>/`, 3.1 GB in total:

| Dataset | Scenes | Kept | Protocol folder (size) |
|---|---|---|---|
| `mipnerf360` | bicycle flowers garden stump treehill | `images_4/`, `sparse/0/`, `poses_bounds.npy` | `images_4` (about 1237x822) |
| `mipnerf360` | room counter kitchen bonsai | `images_2/`, `sparse/0/`, `poses_bounds.npy` | `images_2` (about 1557x1038) |
| `tandt` | truck train | `images/`, `sparse/0/` | `images` (979x546, 980x545) |
| `db` | playroom drjohnson | `images/`, `sparse/0/` | `images` (1264x832, 1332x876) |

- Mip-NeRF 360 is never downloaded whole: `fetch_datasets.py` reads the zip
  central directory over HTTP range requests and fetches only the kept members
  (2.2 GB of 17 GB). Every member is CRC32-checked on extraction. The 3DGS
  `tandt_db` package (683 MB) is downloaded, sha256-checked and deleted.
- Every protocol folder is under 1600 px wide, so 3DGS's automatic rescale
  never applies: all tools train and are scored on exactly these pixels.
- Each scene has a `SHA256SUMS`; `datasets.toml` pins the sha256 of that listing
  (`sha256_tree`). `python3 fetch_datasets.py --verify` re-hashes everything.
- Held-out views: sort the protocol folder's file names, every index with
  `i % 8 == 0` (3DGS `llffhold`, gsplat `test_every`).
- Licences are in the manifest. All of these are research-use only; do not
  redistribute.

## Metrics

```sh
gs-metrics RENDERS GT [--test-every 8] [--out m.json]
```

Pairs files by stem, refuses size mismatches, and reports per image and mean:
`psnr`, `ssim` (3DGS: 11x11 Gaussian, zero padding), `lpips` (LPIPS-VGG on
[0, 1] inputs exactly as 3DGS's lpipsPyTorch), plus gsplat's conventions
`ssim_tm` (torchmetrics) and `lpips_alex`. The protocol reports psnr / ssim /
lpips. `--test-every 8` lets GT be the scene's whole protocol folder.
`python /opt/spl-bench/metrics_check.py` (inside the image) checks it on known
pairs.

## Baselines

```sh
benchmarks/image-to-3d/run_baseline.sh gsplat garden              # full 30k-step run
benchmarks/image-to-3d/run_baseline.sh gsplat garden --steps 2000 --tag smoke
benchmarks/image-to-3d/run_baseline.sh gsplat-mcmc garden -- --strategy.cap-max 3000000
```

Each run takes `flock -x ~/spl-work/gpu.lock` (so does anyone timing on the
GPU), polls `nvidia-smi` every 200 ms, runs the adapter, scores the renders with
`gs-metrics` against the protocol folder, and appends one line to
`~/spl-work/bench/results.jsonl`:

| Field | Meaning |
|---|---|
| `wall_s` | Adapter wall time: container start, loading, training, rendering the test views |
| `train_s` | The tool's own training time |
| `psnr`, `ssim`, `lpips` | Our metrics on the 8-bit renders (3DGS convention) |
| `ssim_tm`, `lpips_alex` | gsplat convention, same renders |
| `tool_metrics` | What the tool itself reported, for cross-checking |
| `num_gaussians` | Final Gaussian count |
| `peak_vram_mib` | Peak `memory.used` minus the value before the run (other processes included in `peak_vram_total_mib`) |
| `util_mean`, `util_max`, `power_mean_w`, `sm_clock_mean_mhz` | nvidia-smi over the run |
| `gpu_apps_before`, `lock_wait_s` | Contention context |
| `image_id`, `harness`, `tool_version` | Exact image, harness content hash, tool version |

Run folders: `~/spl-work/bench/runs/<UTC>_<tool>_<scene>[_<tag>]/` with logs,
`smi.csv`, renders, `metrics.json` (per image) and `result.json`. Checkpoints and
PLY files are deleted unless `--keep`.

The `gsplat` adapter runs `simple_trainer.py default` (3DGS densification, 30k
steps) on a view of the scene whose `images/` is the protocol folder, with
`--data_factor 1`. That matters: with `--data_factor 4` gsplat 1.5.3 ignores
`images_4/` and re-downsamples the full-resolution JPEGs to PNG itself, a
different ground truth from the one 3DGS-family papers report on. Test views
are rendered from the final checkpoint by `py/gsplat_render.py` with the
trainer's own settings. `gsplat-mcmc` runs the `mcmc` preset (3DGS-MCMC, 1M
Gaussian cap unless `--strategy.cap-max` says otherwise) through the same
adapter. Arguments after `--` reach `simple_trainer.py` and are recorded in
`tool_version`.

Measured on garden (12 GB card): `gsplat default` does not finish. It
densifies to 7.2M Gaussians by step 11500 and runs out of the 11.6 GiB there,
even with `expandable_segments` (without it, step 7600). `--packed` does not
rescue it (torch peak 1.172 vs 1.186 GB at 2k steps: the parameters and Adam
state dominate, not the projection buffers). Use `gsplat-mcmc` with an explicit
cap for the large outdoor scenes, and say which cap in the report. Failed runs
stay in `results.jsonl` with `status: "failed"`.

### Adding a tool

Write `tools/<tool>.sh`. `run_baseline.sh` calls it with `SCENE_DIR`, `IMAGES`
(the protocol folder name), `TEST_EVERY`, `RUN_DIR`, `STEPS` (empty: the tool's
default) and `IMAGE` in the environment, and any arguments after `--`. It must:

1. train on the protocol folder of `SCENE_DIR` with COLMAP poses from
   `SCENE_DIR/sparse/0`, holding out sorted index `% TEST_EVERY == 0`;
2. write each held-out view to `RUN_DIR/renders/<image stem>.png`, 8-bit,
   rounded (`x * 255 + 0.5`), at the protocol folder's resolution;
3. write `RUN_DIR/tool.json` with `train_s`, `num_gaussians`, `tool_version`.

Run the tool in its own pinned container. Do not resize, crop or re-encode the
ground truth; if a tool insists on its own copy, render at the protocol size
anyway, so scores stay comparable.
