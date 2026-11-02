#!/usr/bin/env bash
# Time the official Pixal3D inference_mv.py on evaluation inputs.
# Run with GPU lock held for fair comparison.
#
# Usage: time_official.sh [views] [object]
set -euo pipefail

VIEWS="${1:-4}"
OBJECT="${2:-sphere}"
EVAL_DIR="${HOME}/spl-work/eval/pixal3d"
WEIGHTS="${HOME}/spl-work/models/pixal3d"
INPUT_DIR="${EVAL_DIR}/inputs_v${VIEWS}/${OBJECT}"
OUTPUT_DIR="${EVAL_DIR}/official_v${VIEWS}"

mkdir -p "$OUTPUT_DIR"

echo "=== Official inference_mv.py: ${OBJECT} (${VIEWS} views) ==="
echo "Input: $INPUT_DIR"
echo "GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

t0=$(date +%s%N)

docker run --rm --gpus all --shm-size 4g \
    -v "${INPUT_DIR}:/input:ro" \
    -v "${WEIGHTS}:/weights:ro" \
    -v "${OUTPUT_DIR}:/output" \
    -e ATTN_BACKEND=sdpa \
    -e SPARSE_ATTN_BACKEND=sdpa \
    spl-pixal3d python /opt/pixal3d/inference_mv.py \
    --views_dir /input \
    --model_path /weights \
    --config_file pipeline_mv.json \
    --output "/output/${OBJECT}.glb" \
    --low_vram \
    --seed 42 2>&1 | tee "${OUTPUT_DIR}/${OBJECT}.log"

t1=$(date +%s%N)
elapsed_ms=$(( (t1 - t0) / 1000000 ))
echo "Wall time: ${elapsed_ms} ms ($((elapsed_ms / 1000)).$((elapsed_ms % 1000 / 100))s)"
