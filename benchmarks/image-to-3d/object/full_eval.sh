#!/usr/bin/env bash
# Full Pixal3D-mv evaluation with timing, validation, and metrics.
# Runs on the GPU host. Requires GPU lock for timing.
set -euo pipefail

EVAL_DIR="${HOME}/spl-work/eval/pixal3d"
WEIGHTS="${HOME}/spl-work/models/pixal3d"
IMAGE="spl-pixal3d"
LOCK="${HOME}/spl-work/gpu.lock"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_PY="${SCRIPT_DIR}/../../../apps/image-to-3d/tools/pixal3d/sidecar/eval_object.py"
SIDECAR_PY="${SCRIPT_DIR}/../../../apps/image-to-3d/tools/pixal3d/sidecar/run_mv.py"
RESULTS="${EVAL_DIR}/results"

mkdir -p "$RESULTS"

echo "=== Pixal3D-mv Full Evaluation ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
date -u

# --- 1. Our sidecar (run_mv.py) with GPU lock ---
echo ""
echo "=== Part 1: Sidecar Generation (flock) ==="
for nv in 1 2 4; do
    INPUT_DIR="${EVAL_DIR}/inputs_v${nv}"
    PRED_DIR="${EVAL_DIR}/predictions_v${nv}"
    mkdir -p "$PRED_DIR"

    for obj_dir in "${INPUT_DIR}"/*/; do
        [ -d "$obj_dir" ] || continue
        name=$(basename "$obj_dir")
        out_glb="${PRED_DIR}/${name}.glb"
        [ -f "$out_glb" ] && { echo "  [skip] v${nv}/${name}"; continue; }

        echo -n "  [run] v${nv}/${name}..."
        tmp_out="${PRED_DIR}/${name}_tmp"
        mkdir -p "$tmp_out"

        t0=$(date +%s%N)
        flock -x -w 300 "$LOCK" docker run --rm --gpus all --shm-size 4g \
            -v "${obj_dir}:/input:ro" \
            -v "${WEIGHTS}:/weights:ro" \
            -v "${tmp_out}:/output" \
            -v "$(dirname "${SIDECAR_PY}"):/opt/spl-p3d:ro" \
            ${IMAGE} python /opt/spl-p3d/run_mv.py \
            --input_dir /input \
            --weights_dir /weights \
            --output_dir /output \
            --resolution 1024 \
            --low_vram \
            --max_tokens 32768 2>&1 | tee "${PRED_DIR}/${name}.log" | tail -1

        t1=$(date +%s%N)
        wall_ms=$(( (t1 - t0) / 1000000 ))

        if [ -f "${tmp_out}/output.glb" ]; then
            mv "${tmp_out}/output.glb" "$out_glb"
            size=$(du -h "$out_glb" | cut -f1)
            echo " ${wall_ms}ms (${size})"
            # Extract sidecar total_s
            total_s=$(grep -o '"total_s":[0-9.]*' "${PRED_DIR}/${name}.log" | head -1 | cut -d: -f2)
            peak=$(grep -o '"peak_vram_mb":[0-9.]*' "${PRED_DIR}/${name}.log" | head -1 | cut -d: -f2)
            echo "{\"method\":\"sidecar\",\"views\":${nv},\"object\":\"${name}\",\"wall_ms\":${wall_ms},\"sidecar_s\":${total_s:-0},\"peak_vram_mb\":${peak:-0},\"status\":\"ok\"}" >> "$RESULTS/timing.jsonl"
        else
            echo " FAILED"
            echo "{\"method\":\"sidecar\",\"views\":${nv},\"object\":\"${name}\",\"wall_ms\":${wall_ms},\"status\":\"failed\"}" >> "$RESULTS/timing.jsonl"
        fi
        rm -rf "$tmp_out"
    done
done

# --- 2. Official inference_mv.py timing (1 object, 4 views) ---
echo ""
echo "=== Part 2: Official inference_mv.py Timing ==="
OFF_DIR="${EVAL_DIR}/official"
mkdir -p "$OFF_DIR"

for obj_dir in "${EVAL_DIR}/inputs_v4"/*/; do
    [ -d "$obj_dir" ] || continue
    name=$(basename "$obj_dir")
    out_glb="${OFF_DIR}/${name}.glb"
    [ -f "$out_glb" ] && { echo "  [skip] ${name}"; continue; }

    echo -n "  [run] ${name}..."
    t0=$(date +%s%N)
    flock -x -w 300 "$LOCK" docker run --rm --gpus all --shm-size 4g \
        -v "${obj_dir}:/input:ro" \
        -v "${WEIGHTS}:/weights:ro" \
        -v "${OFF_DIR}:/output" \
        -e ATTN_BACKEND=sdpa \
        -e SPARSE_ATTN_BACKEND=sdpa \
        ${IMAGE} python /opt/pixal3d/inference_mv.py \
        --views_dir /input \
        --model_path /weights \
        --config_file pipeline_mv.json \
        --output "/output/${name}.glb" \
        --low_vram \
        --seed 42 2>&1 | tee "${OFF_DIR}/${name}.log" | tail -3

    t1=$(date +%s%N)
    wall_ms=$(( (t1 - t0) / 1000000 ))
    echo " ${wall_ms}ms"
    echo "{\"method\":\"official\",\"views\":4,\"object\":\"${name}\",\"wall_ms\":${wall_ms},\"status\":\"$([ -f \"$out_glb\" ] && echo ok || echo failed)\"}" >> "$RESULTS/timing.jsonl"
    break  # Just time one object for comparison
done

# --- 3. GLB Validation ---
echo ""
echo "=== Part 3: GLB Validation ==="
for nv in 1 2 4; do
    PRED_DIR="${EVAL_DIR}/predictions_v${nv}"
    for glb in "${PRED_DIR}"/*.glb; do
        [ -f "$glb" ] || continue
        name=$(basename "$glb")
        # Basic sanity: check file size > 1KB and starts with glTF magic
        size=$(stat --printf="%s" "$glb" 2>/dev/null || stat -f%z "$glb" 2>/dev/null)
        magic=$(xxd -l 4 -p "$glb" 2>/dev/null || echo "unknown")
        if [ "$size" -gt 1024 ] && [ "$magic" = "676c5446" ]; then
            echo "  [PASS] v${nv}/${name} (${size} bytes, glTF magic ok)"
        else
            echo "  [FAIL] v${nv}/${name} (size=${size}, magic=${magic})"
        fi
    done
done

# --- 4. Chamfer + F-score ---
echo ""
echo "=== Part 4: Chamfer / F-score ==="
for nv in 1 2 4; do
    PRED_DIR="${EVAL_DIR}/predictions_v${nv}"
    [ -d "$PRED_DIR" ] || continue
    glb_count=$(ls "$PRED_DIR"/*.glb 2>/dev/null | wc -l)
    [ "$glb_count" -gt 0 ] || continue

    echo "  evaluating ${nv}-view predictions..."
    docker run --rm \
        -v "${PRED_DIR}:/pred:ro" \
        -v "${EVAL_DIR}/gt:/gt:ro" \
        -v "${RESULTS}:/results" \
        -v "${EVAL_PY}:/eval.py:ro" \
        ${IMAGE} python /eval.py \
        --pred_dir /pred \
        --gt_dir /gt \
        --output "/results/metrics_v${nv}.jsonl"
done

# --- 5. Summary ---
echo ""
echo "=== Results Summary ==="
echo "--- Timing ---"
cat "$RESULTS/timing.jsonl" 2>/dev/null
echo ""
echo "--- Metrics ---"
for nv in 1 2 4; do
    f="${RESULTS}/metrics_v${nv}.jsonl"
    [ -f "$f" ] || continue
    echo "  ${nv}-view:"
    cat "$f"
done

echo ""
echo "--- Published Baselines (labeled as published) ---"
echo "  Pixal3D (Toys4K, multi-view, arXiv 2605.10922):"
echo "    2-view: CD 5.27e-4, F-score 64.94"
echo "    4-view: CD 4.73e-4, F-score 67.85"
echo "    6-view: CD 4.16e-4, F-score 69.04"
echo "  ReconViaGen (Dora-bench 300 objects, 4-view, arXiv 2510.23306):"
echo "    CD 0.090, F-score 0.953, PSNR 22.63"
echo "  TRELLIS (Toys4K, multi-view):"
echo "    4-view: CD 18.35e-4, F-score 45.90"
echo "  Note: our eval uses synthetic GT (6 primitives); published numbers use"
echo "  Toys4K or Dora-bench with real scans — not directly comparable."

echo ""
echo "Done. Results in: $RESULTS/"
