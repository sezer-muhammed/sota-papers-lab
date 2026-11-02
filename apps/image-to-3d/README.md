# image-to-3d

Unposed images or video to Gaussian splats and meshes.

| | |
|---|---|
| Ops | `image_to_3d.train`, `image_to_3d.render`, `image_to_3d.pose` (GPU-exclusive) |
| Kernels | `cuda/*.cu` (NVRTC at runtime, header-free, compute capability 8.0+) |
| Design | `reports/Image to 3D.md`; specs `reports/image-to-3d/spec-raster.md`, `spec-train.md`; module contract [DESIGN.md](DESIGN.md) |
| Paper | see `paper.toml` |

Status: Phase 3 skeleton. Dataset loading (COLMAP bin/txt, splits, intrinsics, PIL-exact JPEG decode, GPU image cache), the Gaussian store, buffer allocation, NVRTC compilation and the test harness work; the splat kernels are stubs owned per [DESIGN.md](DESIGN.md), so training and rendering return "not implemented (owner: ...)".

## Run alone

Validate a COLMAP-posed dataset under the host's dataset root (`$SPL_DATASETS`, else `<data dir>/datasets`):

```sh
spl run image_to_3d.train --input '{"dataset": "mipnerf360/garden", "load": {"images_dir": "images_4"}, "dry_run": true}'
```

Or upload a zip holding `sparse/0/` and an image folder (`spl put scene.zip`) and pass `{"zip": "<artifact id>"}`.

Train (once the kernels land): drop `dry_run`; `train.preset` is one of `faster_gs`, `fastgs`, `fastgs_big`, `mcmc`, `quality` (default), `fast`; `train.iterations` rescales every schedule; `train.overrides.scene` applies FastGS's per-scene table.

## Poses and points (native VGGT-Ω)

`image_to_3d.pose` runs our Rust + CUDA VGGT-Ω engine (`src/vggt/`, spec `reports/image-to-3d/spec-vggt.md`) on photos (`images`: artifact ids, `zip`, `dataset`: a folder under `$SPL_DATASETS`) or a `video` (sampled with ffmpeg) and stores `cameras.json` (w2c OpenCV, world = frame 0), a COLMAP text model (zip) and a coloured point cloud (PLY of small Gaussians, announced in slot `points` as soon as it exists).

```sh
spl run image_to_3d.pose --input '{"dataset": "tandt/truck/images", "max_frames": 32}'
```

**Weights licence.** VGGT-Ω checkpoints (`facebook/VGGT-Omega` on Hugging Face, gated) are under the FAIR non-commercial research licence: research use only. Never commit them, redistribute them or bake them into an image. Convert the downloaded `vggt_omega_1b_512.pt` once, then point the host at the result (`$SPL_VGGT_WEIGHTS`, else `<data dir>/models/vggt-omega/vggt_omega_1b_512.safetensors`):

```sh
docker run --rm -v ~/spl-work/models/vggt-omega:/m -v $PWD/apps/image-to-3d/scripts:/s:ro --entrypoint '' spl-vggt-ref \
  python -I /s/convert_vggt.py --input /m/vggt_omega_1b_512.pt --output /m/vggt_omega_1b_512.safetensors
```

Parity dumps of the official model: `scripts/vggt_oracle.py` (see its header); tests `vggt::tests::gpu_*` read them from `$SPL_VGGT_ORACLE`.

## Tests

```sh
cargo test -p spl-app-image-to-3d                                   # CPU, anywhere
apps/image-to-3d/scripts/gpu-test.sh <checkout> [filter]            # GPU host: GPU + dataset tests in the CUDA container
SPL_GPU_LOCK=1 apps/image-to-3d/scripts/gpu-test.sh <checkout> bench # timing runs under ~/spl-work/gpu.lock
```

The gsplat oracle (`~/spl-work/bin/gs-oracle`) is used directly by tests running natively on the host, or through staged `.npz` cases (see `src/testkit/oracle.rs`).
