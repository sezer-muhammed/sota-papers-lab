#!/usr/bin/env bash
# Usage: install.sh [--bin] [--image] [--datasets] [--freeze]   (no flags: --bin --image --datasets)
#
# Sets up the image-to-3d harness on the GPU host:
#   --bin       copy gs-ref, gs-oracle, gs-metrics to ~/spl-work/bin
#   --image     build spl-gs-ref on the official PyTorch 2.9.1 / CUDA 12.8 runtime image, compiling the
#               CUDA extensions with nvcc borrowed from spl-cuda-devel (both pulled if missing), with the
#               local `default` builder, so other projects' buildx builders stay untouched
#   --datasets  fetch + verify datasets.toml into ~/spl-work/datasets
#   --freeze    write the image's pip versions to docker/constraints.txt (commit it)
set -euo pipefail
HERE=$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)
WORK=${SPL_WORK:-$HOME/spl-work}
IMAGE=${SPL_GS_REF_IMAGE:-spl-gs-ref}
DEVEL=nvidia/cuda:12.8.1-devel-ubuntu24.04
RUNTIME=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

do_bin=0 do_image=0 do_data=0 do_freeze=0
[[ $# -gt 0 ]] || set -- --bin --image --datasets
for a in "$@"; do
  case $a in
    --bin) do_bin=1 ;;
    --image) do_image=1 ;;
    --datasets) do_data=1 ;;
    --freeze) do_freeze=1 ;;
    *) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 2 ;;
  esac
done

if [[ $do_bin == 1 ]]; then
  mkdir -p "$WORK/bin" "$WORK/bench"
  install -m 0755 "$HERE/bin/gs-ref" "$HERE/bin/gs-oracle" "$HERE/bin/gs-metrics" "$WORK/bin/"
  echo "installed gs-ref, gs-oracle, gs-metrics in $WORK/bin"
fi

if [[ $do_image == 1 ]]; then
  if ! docker image inspect spl-cuda-devel > /dev/null 2>&1; then
    docker pull "$DEVEL"
    docker tag "$DEVEL" spl-cuda-devel
  fi
  docker image inspect "$RUNTIME" > /dev/null 2>&1 || docker pull "$RUNTIME"
  docker buildx build --builder default --load -t "$IMAGE" -f "$HERE/docker/Dockerfile" "$HERE"
  docker image ls "$IMAGE"
fi

if [[ $do_data == 1 ]]; then
  python3 -I "$HERE/fetch_datasets.py"
fi

if [[ $do_freeze == 1 ]]; then
  {
    echo "# pip versions of the last good spl-gs-ref build (install.sh --freeze)."
    echo "# torch/torchvision, gsplat, fused-ssim and VCS packages are pinned in the Dockerfile."
    docker run --rm --entrypoint "" "$IMAGE" cat /opt/spl-bench/pip-freeze.txt \
      | grep -v -E ' @ |^(torch|torchvision|gsplat|fused[-_]ssim|pycolmap|nerfview)=='
  } > "$HERE/docker/constraints.txt"
  echo "wrote $HERE/docker/constraints.txt ($(grep -vc '^#' "$HERE/docker/constraints.txt") pins)"
fi
