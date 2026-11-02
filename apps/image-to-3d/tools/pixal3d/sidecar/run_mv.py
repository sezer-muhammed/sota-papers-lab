#!/usr/bin/env python3
"""Pixal3D-mv sidecar: reads transforms.json + RGBA PNGs, runs the
multi-view pipeline, writes output.glb. Progress and result as JSON on
stdout for the Rust op to parse.

Usage (inside the spl-pixal3d container):
  python /opt/spl-p3d/run_mv.py \
    --input_dir /input \
    --weights_dir /weights \
    --output_dir /output \
    --resolution 1024 \
    --low_vram
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def progress(stage: str, fraction: float, **kw):
    """Emit a JSON progress line on stdout."""
    d = {"stage": stage, "fraction": min(max(fraction, 0.0), 1.0)}
    d.update(kw)
    print(json.dumps(d), flush=True)


def result(glb_path: str, stages: list, peak_vram_mb: float, total_s: float):
    """Emit the final result JSON on stdout."""
    print(json.dumps({
        "glb_path": glb_path,
        "stages": stages,
        "peak_vram_mb": peak_vram_mb,
        "total_s": total_s,
    }), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=49152)
    args = parser.parse_args()

    t0 = time.time()
    stage_times = []

    input_dir = Path(args.input_dir)
    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    glb_path = str(output_dir / "output.glb")

    progress("loading", 0.0)

    sys.path.insert(0, "/opt/pixal3d")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = "/tmp/autotune_cache.json"
    os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = "1"

    from inference_mv import init_pipeline, load_views, check_main_view
    import o_voxel

    progress("loading_pipeline", 0.05)

    pipeline = init_pipeline(
        model_path=str(weights_dir),
        config_file="pipeline_mv.json",
        device="cuda",
        low_vram=args.low_vram,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import naf_lowmem
    naf_lowmem.patch(pipeline)
    naf_lowmem.patch_cumesh()

    torch.cuda.reset_peak_memory_stats()
    t_load = time.time() - t0
    stage_times.append({"name": "load", "seconds": t_load})
    progress("loaded", 0.1, vram_mb=torch.cuda.max_memory_allocated() / 1e6)

    progress("preprocessing", 0.15)
    views = load_views(str(input_dir), rembg=None)
    check_main_view(views)
    n_views = views["transform_matrix"].shape[1]
    progress("views_loaded", 0.2, views=n_views)

    t_run = time.time()
    torch.manual_seed(args.seed)

    pipeline_type = f"{args.resolution}_cascade"
    progress("inference", 0.25, pipeline_type=pipeline_type)

    mesh_list, (shape_slat, tex_slat, res) = pipeline.run_mv(
        views,
        seed=args.seed,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=args.max_tokens,
    )

    t_inference = time.time() - t_run
    stage_times.append({"name": "inference", "seconds": t_inference})
    progress("postprocessing", 0.85)

    t_post = time.time()
    mesh = mesh_list[0]
    peak_inference = torch.cuda.max_memory_allocated() / 1e6
    # CuMesh allocates with raw cudaMalloc, so blocks cached by PyTorch's
    # allocator are invisible to it: drop the latents and return the cache
    # before remeshing, or the 1024 remesh OOMs on a 12 GB card.
    del mesh_list, shape_slat, tex_slat
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    progress("postprocessing_mem", 0.86,
             allocated_mb=torch.cuda.memory_allocated() / 1e6,
             peak_inference_mb=peak_inference)

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
        use_tqdm=True,
    )

    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)
    glb.export(glb_path, extension_webp=True)

    t_postprocess = time.time() - t_post
    stage_times.append({"name": "postprocess", "seconds": t_postprocess})

    peak_vram = torch.cuda.max_memory_allocated() / 1e6
    total_s = time.time() - t0

    progress("done", 1.0, vram_mb=peak_vram)
    result(glb_path, stage_times, peak_vram, total_s)


if __name__ == "__main__":
    main()
