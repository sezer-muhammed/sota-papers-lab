#!/usr/bin/env bash
# Baseline adapter: gsplat examples/simple_trainer.py, pinned in the spl-gs-ref
# image. GSPLAT_CONFIG picks the preset: `default` (3DGS densification, the
# default here) or `mcmc` (3DGS-MCMC, 1M Gaussian cap; tools/gsplat-mcmc.sh).
# 30k steps unless STEPS is set. Extra args go to simple_trainer, e.g.
# `-- --packed` or `-- --strategy.absgrad --strategy.grow_grad2d 0.0008`.
#
# Adapter contract (env from run_baseline.sh): SCENE_DIR, IMAGES, TEST_EVERY,
# RUN_DIR, STEPS (may be empty), IMAGE. Must write RUN_DIR/renders/<stem>.png
# for every held-out view and RUN_DIR/tool.json with train_s, num_gaussians,
# tool_version (plus anything else worth keeping).
set -euo pipefail
: "${SCENE_DIR:?}" "${IMAGES:?}" "${TEST_EVERY:?}" "${RUN_DIR:?}" "${IMAGE:?}"
config=${GSPLAT_CONFIG:-default}
steps=${STEPS:-30000}
[[ -n $steps ]] || steps=30000

# The protocol folder is presented as data_factor 1: gsplat then reads exactly
# these pixels (its factor > 1 path would re-downsample full-res JPEGs instead).
view=$RUN_DIR/view
mkdir -p "$view"
ln -sfn "$SCENE_DIR/$IMAGES" "$view/images"
ln -sfn "$SCENE_DIR/sparse" "$view/sparse"

# expandable_segments avoids allocator fragmentation (2.7 GB reserved but
# unused when `default` OOMed on garden at step 7600); it changes no results.
# `default` on garden still passes 7.2M Gaussians by step 11500 and runs out of
# the 11.6 GiB there (`--packed` saves ~1%), so large outdoor scenes need mcmc.
run() {
  docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp -e TQDM_MININTERVAL=30 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$SCENE_DIR:$SCENE_DIR:ro" -v "$RUN_DIR:$RUN_DIR" -w /opt/gsplat/examples --entrypoint "" "$IMAGE" "$@"
}

run python simple_trainer.py "$config" \
  --data_dir "$view" --data_factor 1 --result_dir "$RUN_DIR/gsplat" --test_every "$TEST_EVERY" \
  --disable_viewer --disable_video --max_steps "$steps" \
  --eval_steps "$steps" --save_steps "$steps" --ply_steps "$steps" --save_ply "$@"

run python /opt/spl-bench/gsplat_render.py --result-dir "$RUN_DIR/gsplat" --data-dir "$view" --out "$RUN_DIR/renders"

python3 -I - "$RUN_DIR" "$steps" "$config" "$*" <<'EOF'
import json, os, sys
run, steps, config, extra = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
stats = os.path.join(run, "gsplat", "stats")
train = json.load(open(os.path.join(stats, f"train_step{steps - 1:04d}_rank0.json")))
val_p = os.path.join(stats, f"val_step{steps - 1:04d}.json")
val = json.load(open(val_p)) if os.path.exists(val_p) else {}
render = json.load(open(os.path.join(run, "render.json")))
out = {
    "train_s": round(train["ellipse_time"], 2),
    "num_gaussians": train["num_GS"],
    "tool_version": f"gsplat 1.5.3 (937e299) simple_trainer {config}" + (f" {extra}" if extra else ""),
    "torch_peak_alloc_gb": round(train["mem"], 3),
    "render_ms_mean": round(render["render_ms_mean"], 3),
    # gsplat's own eval on float renders: PSNR, torchmetrics SSIM, LPIPS-alex
    "tool_metrics": {k: val[k] for k in ("psnr", "ssim", "lpips") if k in val},
}
json.dump(out, open(os.path.join(run, "tool.json"), "w"), indent=2)
print(json.dumps(out))
EOF
