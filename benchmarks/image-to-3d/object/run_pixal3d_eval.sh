#!/usr/bin/env bash
# Full Pixal3D-mv evaluation: prepare data, generate, validate, evaluate.
# Runs on the GPU host with flock for timing.
#
# Usage: run_pixal3d_eval.sh [--resolution 1024|512] [--views 1 2 4]
set -euo pipefail

RESOLUTION="${RESOLUTION:-1024}"
EVAL_DIR="${HOME}/spl-work/eval/pixal3d"
WEIGHTS="${HOME}/spl-work/models/pixal3d"
IMAGE="spl-pixal3d"
LOCK="${HOME}/spl-work/gpu.lock"

mkdir -p "$EVAL_DIR"

echo "=== Pixal3D-mv Evaluation (resolution=${RESOLUTION}) ==="
echo "GPU status:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader

# 1. Prepare evaluation data (CPU only, no GPU lock needed).
if [ ! -d "${EVAL_DIR}/gt" ]; then
    echo "==> Preparing evaluation data..."
    docker run --rm \
        -v "${EVAL_DIR}:/work/eval" \
        ${IMAGE} python -c "
import sys; sys.path.insert(0, '/opt/spl-p3d')
exec(open('/opt/spl-p3d/prepare_eval.py').read() if __import__('os').path.exists('/opt/spl-p3d/prepare_eval.py') else '')
" 2>/dev/null || true

    # Fallback: mount the script from the host repo.
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    docker run --rm \
        -v "${EVAL_DIR}:/work/eval" \
        -v "${SCRIPT_DIR}/prepare_eval.py:/opt/spl-p3d/prepare_eval.py:ro" \
        ${IMAGE} python /opt/spl-p3d/prepare_eval.py \
        --output_dir /work/eval \
        --views 1 2 4 \
        --resolution "${RESOLUTION}"
fi

echo "==> Evaluation data:"
ls -la "${EVAL_DIR}/gt/" 2>/dev/null | tail -10
for nv in 1 2 4; do
    count=$(ls -d "${EVAL_DIR}/inputs_v${nv}"/*/ 2>/dev/null | wc -l)
    echo "  inputs_v${nv}: ${count} objects"
done

# 2. Generate with Pixal3D-mv (hold GPU lock for timing).
for nv in 1 2 4; do
    INPUT_DIR="${EVAL_DIR}/inputs_v${nv}"
    PRED_DIR="${EVAL_DIR}/predictions_v${nv}"
    mkdir -p "$PRED_DIR"

    echo ""
    echo "==> Generating ${nv}-view predictions (resolution=${RESOLUTION})..."

    for obj_dir in "${INPUT_DIR}"/*/; do
        [ -d "$obj_dir" ] || continue
        name=$(basename "$obj_dir")
        out_glb="${PRED_DIR}/${name}.glb"

        if [ -f "$out_glb" ]; then
            echo "  [skip] ${name}"
            continue
        fi

        echo -n "  [run] ${name}..."
        t_start=$(date +%s%N)
        tmp_out="${PRED_DIR}/${name}_tmp"
        mkdir -p "$tmp_out"

        flock -x -w 300 "$LOCK" docker run --rm --gpus all \
            --shm-size 4g \
            -v "${obj_dir}:/input:ro" \
            -v "${WEIGHTS}:/weights:ro" \
            -v "${tmp_out}:/output" \
            ${IMAGE} python /opt/spl-p3d/run_mv.py \
            --input_dir /input \
            --weights_dir /weights \
            --output_dir /output \
            --resolution "${RESOLUTION}" \
            --low_vram 2>&1 | tee "${PRED_DIR}/${name}.log" | tail -1

        t_end=$(date +%s%N)
        elapsed=$(( (t_end - t_start) / 1000000000 ))

        if [ -f "${tmp_out}/output.glb" ]; then
            mv "${tmp_out}/output.glb" "$out_glb"
            echo " ${elapsed}s"
        else
            echo " FAILED (${elapsed}s)"
        fi
        rm -rf "$tmp_out"
    done
done

# 3. Validate GLBs.
echo ""
echo "==> Validating GLBs..."
for nv in 1 2 4; do
    PRED_DIR="${EVAL_DIR}/predictions_v${nv}"
    for glb in "${PRED_DIR}"/*.glb; do
        [ -f "$glb" ] || continue
        name=$(basename "$glb")
        if command -v npx &>/dev/null; then
            result=$(npx --yes gltf-validator "$glb" 2>&1 | grep -c '"severity":0' || echo 0)
            if [ "$result" -eq 0 ]; then
                echo "  [PASS] v${nv}/${name}"
            else
                echo "  [FAIL] v${nv}/${name} (${result} errors)"
            fi
        else
            echo "  [skip] gltf-validator not available"
            break 2
        fi
    done
done

# 4. Evaluate Chamfer + F-score.
echo ""
echo "==> Computing metrics..."
RESULTS_DIR="${EVAL_DIR}/results"
mkdir -p "$RESULTS_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for nv in 1 2 4; do
    PRED_DIR="${EVAL_DIR}/predictions_v${nv}"
    [ -d "$PRED_DIR" ] || continue
    glb_count=$(ls "$PRED_DIR"/*.glb 2>/dev/null | wc -l)
    [ "$glb_count" -gt 0 ] || continue

    echo "  evaluating ${nv}-view predictions..."
    docker run --rm \
        -v "${PRED_DIR}:/pred:ro" \
        -v "${EVAL_DIR}/gt:/gt:ro" \
        -v "${RESULTS_DIR}:/results" \
        -v "${SCRIPT_DIR}/eval_object.py:/opt/spl-p3d/eval_object.py:ro" \
        ${IMAGE} python /opt/spl-p3d/eval_object.py \
        --pred_dir /pred \
        --gt_dir /gt \
        --output "/results/eval_v${nv}.jsonl"
done

# 5. Print results summary.
echo ""
echo "=== Results ==="
for nv in 1 2 4; do
    f="${RESULTS_DIR}/eval_v${nv}.jsonl"
    [ -f "$f" ] || continue
    echo "--- ${nv}-view ---"
    cat "$f"
done

echo ""
echo "=== Timing (from generation logs) ==="
for nv in 1 2 4; do
    PRED_DIR="${EVAL_DIR}/predictions_v${nv}"
    for log in "${PRED_DIR}"/*.log; do
        [ -f "$log" ] || continue
        name=$(basename "$log" .log)
        total=$(grep -o '"total_s":[0-9.]*' "$log" 2>/dev/null | head -1 | cut -d: -f2)
        [ -n "$total" ] && echo "  v${nv}/${name}: ${total}s"
    done
done
