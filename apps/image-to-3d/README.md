# image-to-3d

Unposed images or video to Gaussian splats and meshes.

| | |
|---|---|
| Ops | `image_to_3d.train`, `image_to_3d.render` |
| Kernels | `cuda/*.cu` (NVRTC at runtime, header-free, compute capability 8.0+) |
| Design | `reports/Image to 3D.md`; specs `reports/image-to-3d/spec-raster.md`, `spec-train.md`; module contract [DESIGN.md](DESIGN.md) |
| Paper | see `paper.toml` |

Status: Phase 3 trainer assembled. `image_to_3d.train` trains on COLMAP poses (CUDA graphs per iteration, FastGS/ADC/MCMC densification, fused loss and Adam), evaluates the held-out views (PSNR, SSIM on the CPU) and exports PLY, SPZ and COLMAP; `image_to_3d.render` renders a PLY/SPZ. Not yet: pose refinement and PPISP (Phase 4), DashGaussian warm-up (`DashSchedule::from_spectra`; the Fast preset runs at full resolution), GPU metrics kernels, the fused-Adam sink. Full runs on a dataset: `trainer_gpu_tests.rs` (`SPL_TRAIN_*` variables, see its header).

## Run alone

Validate a COLMAP-posed dataset under the host's dataset root (`$SPL_DATASETS`, else `<data dir>/datasets`):

```sh
spl run image_to_3d.train --input '{"dataset": "mipnerf360/garden", "load": {"images_dir": "images_4"}, "dry_run": true}'
```

Or upload a zip holding `sparse/0/` and an image folder (`spl put scene.zip`) and pass `{"zip": "<artifact id>"}`.

Train (once the kernels land): drop `dry_run`; `train.preset` is one of `faster_gs`, `fastgs`, `fastgs_big`, `mcmc`, `quality` (default), `fast`; `train.iterations` rescales every schedule; `train.overrides.scene` applies FastGS's per-scene table.

## Tests

```sh
cargo test -p spl-app-image-to-3d                                   # CPU, anywhere
apps/image-to-3d/scripts/gpu-test.sh <checkout> [filter]            # GPU host: GPU + dataset tests in the CUDA container
SPL_GPU_LOCK=1 apps/image-to-3d/scripts/gpu-test.sh <checkout> bench # timing runs under ~/spl-work/gpu.lock
```

The gsplat oracle (`~/spl-work/bin/gs-oracle`) is used directly by tests running natively on the host, or through staged `.npz` cases (see `src/testkit/oracle.rs`).
