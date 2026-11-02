#!/usr/bin/env python3
"""Batch Pixal3D-mv evaluation: load pipeline once, process all inputs.

Designed to minimize GPU claim time by loading everything on CPU first,
then rapidly claiming GPU for inference. Waits for GPU to be free before
starting inference.

Usage (inside spl-pixal3d container):
  python /eval/batch_eval.py \
    --eval_dir /eval \
    --weights_dir /weights \
    --views 1 2 4 \
    --max_tokens 32768
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = "/tmp/autotune_cache.json"
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = "1"

sys.path.insert(0, "/opt/pixal3d")


def gpu_free_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip())
    except Exception:
        return 0


def gpu_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip())
    except Exception:
        return 99999


def wait_for_gpu(min_free_mb=10000, timeout_s=600):
    print(f"[Wait] Need {min_free_mb} MiB free GPU...", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        free = gpu_free_mb()
        if free >= min_free_mb:
            time.sleep(3)
            free2 = gpu_free_mb()
            if free2 >= min_free_mb:
                print(f"[Wait] GPU free: {free2} MiB (stable)", flush=True)
                return True
        time.sleep(5)
    print(f"[Wait] Timeout after {timeout_s}s", flush=True)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--views", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--objects", nargs="+",
                        default=["sphere", "torus", "box", "capsule", "cylinder", "cone"])
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    results_dir = eval_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: Load pipeline on CPU (no GPU needed) ===", flush=True)
    from inference_mv import init_pipeline, load_views, check_main_view
    import o_voxel

    pipeline = init_pipeline(
        model_path=args.weights_dir,
        config_file="pipeline_mv.json",
        device="cuda",
        low_vram=True,
    )
    print("[Pipeline] Loaded on CPU, ready for inference.", flush=True)

    timing_file = results_dir / "timing.jsonl"

    for nv in args.views:
        for obj in args.objects:
            input_dir = eval_dir / f"inputs_v{nv}" / obj
            pred_dir = eval_dir / f"predictions_v{nv}"
            pred_dir.mkdir(parents=True, exist_ok=True)
            out_glb = pred_dir / f"{obj}.glb"

            if out_glb.exists():
                print(f"[SKIP] v{nv}/{obj}", flush=True)
                continue

            if not input_dir.exists():
                print(f"[MISS] v{nv}/{obj} — no input dir", flush=True)
                continue

            if not wait_for_gpu(min_free_mb=10000, timeout_s=1800):
                print(f"[SKIP] v{nv}/{obj} — GPU timeout", flush=True)
                continue

            print(f"[RUN] v{nv}/{obj} max_tokens={args.max_tokens}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                views = load_views(str(input_dir), rembg=None)
                check_main_view(views)

                torch.manual_seed(42)
                mesh_list, (shape_slat, tex_slat, res) = pipeline.run_mv(
                    views,
                    seed=42,
                    return_latent=True,
                    pipeline_type="1024_cascade",
                    max_num_tokens=args.max_tokens,
                )

                mesh = mesh_list[0]
                glb = o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=mesh.attrs,
                    coords=mesh.coords,
                    attr_layout=pipeline.pbr_attr_layout,
                    grid_size=res,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=1000000,
                    texture_size=4096,
                    remesh=True,
                    remesh_band=1,
                    remesh_project=0,
                    use_tqdm=False,
                )

                rot = np.array([
                    [-1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1],
                ], dtype=np.float64)
                glb.apply_transform(rot)
                glb.export(str(out_glb), extension_webp=True)

                total_s = time.time() - t0
                peak_mb = torch.cuda.max_memory_allocated() / 1e6
                size_mb = out_glb.stat().st_size / 1e6

                record = {
                    "method": "sidecar",
                    "views": nv,
                    "object": obj,
                    "wall_s": round(total_s, 1),
                    "peak_vram_mb": round(peak_mb, 0),
                    "glb_mb": round(size_mb, 2),
                    "status": "ok",
                }
                print(f"[OK] v{nv}/{obj} {total_s:.1f}s peak={peak_mb:.0f}MB {size_mb:.1f}MB", flush=True)

            except torch.cuda.OutOfMemoryError as e:
                total_s = time.time() - t0
                record = {
                    "method": "sidecar",
                    "views": nv,
                    "object": obj,
                    "wall_s": round(total_s, 1),
                    "status": "oom",
                    "error": str(e)[:200],
                }
                print(f"[OOM] v{nv}/{obj} {total_s:.1f}s — {e}", flush=True)
                torch.cuda.empty_cache()

            except Exception as e:
                total_s = time.time() - t0
                record = {
                    "method": "sidecar",
                    "views": nv,
                    "object": obj,
                    "wall_s": round(total_s, 1),
                    "status": "error",
                    "error": str(e)[:200],
                }
                print(f"[ERR] v{nv}/{obj} {total_s:.1f}s — {e}", flush=True)
                torch.cuda.empty_cache()

            with open(timing_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            torch.cuda.empty_cache()

    print("=== Done ===", flush=True)
    if timing_file.exists():
        print(timing_file.read_text(), flush=True)


if __name__ == "__main__":
    main()
