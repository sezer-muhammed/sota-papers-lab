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
    peak_vram = 0.0

    input_dir = Path(args.input_dir)
    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms = json.loads((input_dir / "transforms.json").read_text())

    progress("loading", 0.0)

    # Set up Pixal3D paths.
    sys.path.insert(0, "/opt/pixal3d")
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"

    # Import Pixal3D pipeline.
    from pixal3d.pipelines import Pixal3DMVImageTo3DPipeline

    progress("loading_pipeline", 0.05)

    # Load pipeline from weights directory.
    pipeline = Pixal3DMVImageTo3DPipeline.from_pretrained(
        str(weights_dir),
        low_vram=args.low_vram,
    )
    pipeline.to("cuda")

    torch.cuda.reset_peak_memory_stats()
    t_load = time.time() - t0
    stage_times.append({"name": "load", "seconds": t_load})
    progress("loaded", 0.1, vram_mb=torch.cuda.max_memory_allocated() / 1e6)

    # Load images from transforms.json.
    from PIL import Image
    images = []
    c2ws = []
    fovs = []

    for frame in transforms["frames"]:
        img_path = input_dir / frame["file_path"]
        img = Image.open(img_path).convert("RGBA")
        images.append(img)
        c2ws.append(frame["transform_matrix"])
        fovs.append(frame.get("camera_angle_x", 0.8))

    n_views = len(images)
    progress("preprocessing", 0.15, views=n_views)

    # Prepare camera input for the pipeline.
    import numpy as np
    c2w_matrices = [np.array(c, dtype=np.float32) for c in c2ws]

    # Run the pipeline.
    t_run = time.time()

    try:
        outputs = pipeline(
            images,
            camera_c2ws=c2w_matrices,
            camera_fovs=fovs,
            seed=args.seed,
            steps=12,
            cfg_strength=7.5,
            max_num_tokens=args.max_tokens,
        )
    except Exception as e:
        # If the multi-view API is different, try the inference_mv approach.
        sys.stderr.write(f"Pipeline call failed: {e}\nTrying inference_mv...\n")
        progress("inference_mv_fallback", 0.2)
        outputs = run_inference_mv(
            images, c2w_matrices, fovs, weights_dir, args, progress
        )

    t_inference = time.time() - t_run
    stage_times.append({"name": "inference", "seconds": t_inference})
    progress("postprocessing", 0.85)

    # Post-process to GLB.
    t_post = time.time()
    glb_path = str(output_dir / "output.glb")

    if hasattr(outputs, "meshes") and outputs.meshes:
        mesh = outputs.meshes[0]
        if hasattr(mesh, "export"):
            mesh.export(glb_path)
        else:
            # Use o-voxel postprocessing.
            from o_voxel.postprocess import to_glb
            to_glb(mesh, glb_path)
    elif isinstance(outputs, dict) and "mesh" in outputs:
        mesh = outputs["mesh"]
        if hasattr(mesh, "export"):
            mesh.export(glb_path)
    elif isinstance(outputs, dict) and "glb" in outputs:
        with open(glb_path, "wb") as f:
            f.write(outputs["glb"])
    else:
        # Try extracting from pipeline output.
        try:
            from o_voxel.postprocess import to_glb
            to_glb(outputs, glb_path, texture_size=2048)
        except Exception as e2:
            sys.stderr.write(f"Could not export GLB: {e2}\n")
            raise

    t_postprocess = time.time() - t_post
    stage_times.append({"name": "postprocess", "seconds": t_postprocess})

    peak_vram = torch.cuda.max_memory_allocated() / 1e6
    total_s = time.time() - t0

    progress("done", 1.0, vram_mb=peak_vram)
    result(glb_path, stage_times, peak_vram, total_s)


def run_inference_mv(images, c2w_matrices, fovs, weights_dir, args, progress_fn):
    """Fallback: use inference_mv.py directly."""
    import tempfile
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # Save images.
        for i, img in enumerate(images):
            img.save(tmpdir / f"view_{i:04d}.png")

        # Write transforms.
        transforms = {
            "mesh_scale": 1.0,
            "frames": [
                {
                    "file_path": f"view_{i:04d}.png",
                    "transform_matrix": c2w_matrices[i].tolist()
                    if hasattr(c2w_matrices[i], "tolist")
                    else c2w_matrices[i],
                    "camera_angle_x": fovs[i],
                }
                for i in range(len(images))
            ],
        }
        (tmpdir / "transforms.json").write_text(json.dumps(transforms))

        # Call inference_mv.py.
        cmd = [
            sys.executable,
            "/opt/pixal3d/inference_mv.py",
            "--input_dir", str(tmpdir),
            "--output_dir", str(tmpdir / "out"),
            "--pretrained_model_name_or_path", str(weights_dir),
            "--resolution", str(args.resolution),
            "--seed", str(args.seed),
        ]
        if args.low_vram:
            cmd.append("--low_vram")

        progress_fn("inference_mv", 0.3)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise RuntimeError(f"inference_mv.py failed: {proc.returncode}")

        # Find GLB.
        out_dir = tmpdir / "out"
        glbs = list(out_dir.glob("**/*.glb"))
        if not glbs:
            raise RuntimeError("No GLB produced by inference_mv.py")

        return {"glb": glbs[0].read_bytes()}


if __name__ == "__main__":
    main()
