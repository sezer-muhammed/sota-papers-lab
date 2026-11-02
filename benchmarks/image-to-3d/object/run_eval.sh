#!/usr/bin/env bash
# Evaluate Pixal3D-mv object generation quality.
#
# Runs on the GPU host. Requires:
# - spl-pixal3d Docker image built
# - Weights fetched to ~/spl-work/models/pixal3d/
# - DoraBench/GSO evaluation subsets downloaded
#
# Usage: run_eval.sh [--views 1|2|4] [--subset dorabench|gso] [--limit N]
set -euo pipefail

VIEWS="${1:-4}"
SUBSET="${2:-gso}"
LIMIT="${3:-10}"

WORK="${HOME}/spl-work/eval/object"
WEIGHTS="${HOME}/spl-work/models/pixal3d"
RESULTS="${WORK}/results"

mkdir -p "$WORK" "$RESULTS"

echo "=== Object generation eval: ${SUBSET}, ${VIEWS} views, limit ${LIMIT} ==="

# 1. Render input views from GT meshes (if not already done).
INPUT_DIR="${WORK}/inputs/${SUBSET}_v${VIEWS}"
if [ ! -d "$INPUT_DIR" ]; then
  echo "==> Rendering input views..."
  docker run --rm --gpus all \
    -v "${WORK}:/work" \
    -v "${HOME}/spl-work/datasets:/data:ro" \
    spl-pixal3d python /opt/spl-p3d/render_inputs.py \
      --gt_dir "/data/${SUBSET}" \
      --output_dir "/work/inputs/${SUBSET}_v${VIEWS}" \
      --views "$VIEWS" \
      --limit "$LIMIT" \
      --resolution 1024
fi

# 2. Run Pixal3D-mv on each set of views.
PRED_DIR="${WORK}/predictions/${SUBSET}_v${VIEWS}"
mkdir -p "$PRED_DIR"
echo "==> Running Pixal3D-mv..."

for obj_dir in "$INPUT_DIR"/*/; do
  name=$(basename "$obj_dir")
  out_glb="${PRED_DIR}/${name}.glb"
  if [ -f "$out_glb" ]; then
    echo "  [skip] $name"
    continue
  fi
  echo "  [run] $name"
  t_start=$(date +%s)

  docker run --rm --gpus all \
    --shm-size 4g \
    -v "$obj_dir:/input:ro" \
    -v "$WEIGHTS:/weights:ro" \
    -v "$PRED_DIR:/output" \
    spl-pixal3d python /opt/spl-p3d/run_mv.py \
      --input_dir /input \
      --weights_dir /weights \
      --output_dir "/output/${name}_tmp" \
      --resolution 1024 \
      --low_vram 2>&1 | tail -5

  # Move GLB to final location.
  if [ -f "$PRED_DIR/${name}_tmp/output.glb" ]; then
    mv "$PRED_DIR/${name}_tmp/output.glb" "$out_glb"
    rm -rf "$PRED_DIR/${name}_tmp"
  fi

  t_end=$(date +%s)
  echo "  $name: $((t_end - t_start)) s"
done

# 3. Evaluate Chamfer + F-score.
echo "==> Evaluating..."
docker run --rm --gpus all \
  -v "$PRED_DIR:/pred:ro" \
  -v "${HOME}/spl-work/datasets/${SUBSET}:/gt:ro" \
  -v "$RESULTS:/results" \
  spl-pixal3d python /opt/spl-p3d/eval_object.py \
    --pred_dir /pred \
    --gt_dir /gt \
    --output "/results/${SUBSET}_v${VIEWS}.jsonl"

# 4. Validate GLBs with the Khronos validator.
echo "==> Validating GLBs..."
for glb in "$PRED_DIR"/*.glb; do
  name=$(basename "$glb")
  # Use npx if gltf-validator is not installed.
  if command -v gltf_validator &>/dev/null; then
    gltf_validator "$glb" 2>&1 | head -3
  elif command -v npx &>/dev/null; then
    npx --yes gltf-validator "$glb" 2>&1 | head -3
  else
    echo "  [warn] gltf-validator not available; skipping $name"
  fi
done

echo "=== Results in $RESULTS ==="
cat "$RESULTS/${SUBSET}_v${VIEWS}.jsonl" 2>/dev/null | tail -5
