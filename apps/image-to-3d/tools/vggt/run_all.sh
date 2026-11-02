#!/usr/bin/env bash
# Master script: build the Docker image, convert weights, run benchmarks and evals.
#
# Usage (on the GPU host):
#   cd ~/spl-work/wt/vggt-ref/apps/image-to-3d/tools/vggt
#   bash run_all.sh [--skip-image] [--skip-convert] [--skip-benchmark] [--skip-eval]
set -euo pipefail

WORK=${SPL_WORK:-$HOME/spl-work}
IMAGE=spl-vggt-ref
CKPT=$WORK/models/vggt-omega/vggt_omega_1b_512.pt
CONVERTED=$WORK/models/vggt-omega/converted
RESULTS=$WORK/bench/vggt-ref
HERE=$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)

skip_image=0; skip_convert=0; skip_benchmark=0; skip_eval=0
for a in "$@"; do
  case $a in
    --skip-image)     skip_image=1 ;;
    --skip-convert)   skip_convert=1 ;;
    --skip-benchmark) skip_benchmark=1 ;;
    --skip-eval)      skip_eval=1 ;;
    *) echo "unknown flag: $a"; exit 2 ;;
  esac
done

# -- 1. Build Docker image --
if [[ $skip_image == 0 ]]; then
  echo "=== Building $IMAGE ==="
  docker buildx build --builder default --load -t "$IMAGE" -f "$HERE/Dockerfile" "$HERE"
  docker image ls "$IMAGE"
fi

# Helper to run inside the container
vggt_run() {
  docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null || true)" \
    -e HF_HUB_DISABLE_XET=1 \
    -v "$WORK:$WORK" \
    -v "$HERE:/workspace:ro" \
    -v "$HOME/.cache/huggingface:/tmp/.cache/huggingface:ro" \
    -w /workspace \
    "$IMAGE" "$@"
}

# -- 2. Verify checkpoint --
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found at $CKPT"
  echo "Download it first with: python -c 'from huggingface_hub import hf_hub_download; ...'"
  exit 1
fi
echo "=== Checkpoint: $(ls -lh "$CKPT" | awk '{print $5}')"

# -- 3. Convert weights --
if [[ $skip_convert == 0 ]]; then
  echo "=== Converting to safetensors + FP8 ==="
  vggt_run python -I /workspace/convert.py \
    --input "$CKPT" \
    --output-dir "$CONVERTED"
  echo "Converted files:"
  ls -lh "$CONVERTED"
fi

# -- 4. Benchmark --
if [[ $skip_benchmark == 0 ]]; then
  echo "=== Benchmarking (under GPU lock) ==="
  mkdir -p "$RESULTS"
  flock -x "$WORK/gpu.lock" \
    bash -c "nvidia-smi > '$RESULTS/nvidia-smi.txt' 2>&1; $(
      printf 'docker run --rm --gpus all '
      printf '--user %q:%q ' "$(id -u)" "$(id -g)"
      printf '-e HOME=/tmp '
      printf '-v %q:%q ' "$WORK" "$WORK"
      printf '-v %q:/workspace:ro ' "$HERE"
      printf '-w /workspace '
      printf '%q ' "$IMAGE"
      printf 'python -I /workspace/benchmark.py '
      printf '--checkpoint %q ' "$CKPT"
      printf '--output %q ' "$RESULTS/benchmark.json"
    )"
  echo "Benchmark results:"
  cat "$RESULTS/benchmark.json"
fi

# -- 5. Prepare eval datasets --
if [[ $skip_eval == 0 ]]; then
  echo "=== Preparing evaluation datasets ==="
  EVAL_DATA=$WORK/datasets/vggt-eval

  # ETH3D subset (3 scenes from HuggingFace)
  if [[ ! -d "$EVAL_DATA/eth3d/courtyard" ]]; then
    echo "--- ETH3D (3 scenes) ---"
    # Clone the official repo eval code into the container to use prepare.py
    vggt_run python -I -c "
import subprocess, sys
# Use the official eval/prepare.py from the cloned repo
subprocess.check_call([
    sys.executable, '-I', '/workspace/prepare_datasets.py',
    '--dataset', 'eth3d',
    '--output', '$EVAL_DATA/eth3d',
    '--scenes', 'courtyard', 'delivery_area', 'facade',
    '--source', 'hf',
])
"
  else
    echo "ETH3D data already prepared at $EVAL_DATA/eth3d"
  fi

  # Sintel subset (3 scenes from HuggingFace)
  if [[ ! -d "$EVAL_DATA/sintel/alley_1" ]]; then
    echo "--- Sintel (3 scenes) ---"
    vggt_run python -I -c "
import subprocess, sys
subprocess.check_call([
    sys.executable, '-I', '/workspace/prepare_datasets.py',
    '--dataset', 'sintel',
    '--output', '$EVAL_DATA/sintel',
    '--scenes', 'alley_1', 'bamboo_2', 'temple_2',
    '--source', 'hf',
])
"
  else
    echo "Sintel data already prepared at $EVAL_DATA/sintel"
  fi

  # -- 6. Run evaluation --
  echo "=== Running evaluation ==="
  mkdir -p "$RESULTS"

  # ETH3D
  echo "--- ETH3D eval ---"
  flock -x "$WORK/gpu.lock" \
    vggt_run python -I /workspace/eval_subset.py \
      --dataset eth3d \
      --data-root "$EVAL_DATA/eth3d" \
      --checkpoint "$CKPT" \
      --output "$RESULTS/eval_eth3d.json" \
      --frame-map /workspace/frames/eth3d_subset.json

  # Sintel
  echo "--- Sintel eval ---"
  flock -x "$WORK/gpu.lock" \
    vggt_run python -I /workspace/eval_subset.py \
      --dataset sintel \
      --data-root "$EVAL_DATA/sintel" \
      --checkpoint "$CKPT" \
      --output "$RESULTS/eval_sintel.json" \
      --frame-map /workspace/frames/sintel_subset.json

  echo "Evaluation results:"
  for f in "$RESULTS"/eval_*.json; do
    echo "--- $(basename "$f") ---"
    python3 -c "import json; d=json.load(open('$f')); print(json.dumps(d['metrics'], indent=2))"
  done
fi

echo "=== All done ==="
echo "Results in $RESULTS"
ls -la "$RESULTS"
