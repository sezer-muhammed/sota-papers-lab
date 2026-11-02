#!/usr/bin/env bash
# Build extension wheels for the spl-pixal3d image.
# Run on the GPU host (needs CUDA toolkit from spl-cuda-devel).
# Products: ~/spl-work/p3d/wheels/*.whl
set -euo pipefail

WD="${HOME}/spl-work/p3d"
SRC="${WD}/src"
WHEELS="${WD}/wheels"
CUDA="${WD}/cuda"

mkdir -p "$SRC" "$WHEELS"

# 1. Extract CUDA toolkit from spl-cuda-devel (transient).
if [ ! -d "$CUDA" ]; then
  echo "==> Extracting CUDA from spl-cuda-devel..."
  cid=$(docker create spl-cuda-devel:latest)
  docker cp "$cid:/usr/local/cuda-12.8" "$CUDA"
  docker rm "$cid"
fi

# 2. Clone sources at pinned commits.
clone_if_missing() {
  local repo="$1" dir="$2" ref="${3:-main}"
  if [ ! -d "$dir" ]; then
    echo "==> Cloning $repo..."
    git clone --depth 1 --branch "$ref" "https://github.com/$repo.git" "$dir" 2>/dev/null || \
      git clone --depth 1 "https://github.com/$repo.git" "$dir"
  fi
}

clone_if_missing "TencentARC/Pixal3D" "$SRC/Pixal3D"
clone_if_missing "JeffreyXiang/FlexGEMM" "$SRC/FlexGEMM"
clone_if_missing "microsoft/TRELLIS.2" "$SRC/TRELLIS.2"
clone_if_missing "NVlabs/nvdiffrast" "$SRC/nvdiffrast" "v0.4.0"

# CuMesh: submodule of TRELLIS.2 or Pixal3D.
if [ ! -d "$SRC/CuMesh" ]; then
  clone_if_missing "xczz/CuMesh" "$SRC/CuMesh"
  (cd "$SRC/CuMesh" && git submodule update --init --recursive 2>/dev/null || true)
fi

# Eigen for o-voxel.
if [ ! -d "$SRC/TRELLIS.2/o-voxel/third_party/eigen" ]; then
  (cd "$SRC/TRELLIS.2" && git submodule update --init --recursive 2>/dev/null || true)
fi

# 3. Build wheels inside the pytorch container.
echo "==> Building wheels..."
docker run --rm --gpus all \
  -v "$CUDA:/usr/local/cuda:ro" \
  -v "$SRC:/src:ro" \
  -v "$WHEELS:/wheels" \
  -e TORCH_CUDA_ARCH_LIST=8.9 \
  -e MAX_JOBS=8 \
  pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime \
  bash -c '
    pip install ninja setuptools wheel
    for pkg in /src/FlexGEMM /src/CuMesh /src/TRELLIS.2/o-voxel /src/nvdiffrast; do
      echo "--- Building $(basename $pkg) ---"
      pip wheel --no-build-isolation --no-deps -w /wheels "$pkg" || echo "WARN: $pkg failed"
    done
    ls -lh /wheels/
  '

# 4. Clean up transient CUDA.
echo "==> Cleaning up..."
rm -rf "$CUDA"

echo "==> Wheels in $WHEELS:"
ls -lh "$WHEELS/"
