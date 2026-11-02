#!/usr/bin/env python3
"""Render input views from GT meshes for evaluation.

For each GT mesh, renders N views at uniform azimuths with random
elevation and fov, saving RGBA PNGs + transforms.json.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import trimesh


def hammersley_sphere(n, seed=42):
    """N points on the upper hemisphere via Hammersley sequence."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(n):
        phi = 2.0 * math.pi * (i * 0.6180339887 % 1.0)
        cos_theta = 1.0 - (i + 0.5) / n * 0.6
        sin_theta = math.sqrt(max(0, 1.0 - cos_theta * cos_theta))
        x = sin_theta * math.cos(phi)
        y = sin_theta * math.sin(phi)
        z = cos_theta
        points.append([x, y, z])
    return np.array(points)


def look_at(eye, target, up=None):
    """Camera-to-world matrix (Blender convention: -Z forward, +Y up)."""
    if up is None:
        up = np.array([0, 0, 1], dtype=np.float64)
    forward = np.array(target, dtype=np.float64) - np.array(eye, dtype=np.float64)
    forward = forward / (np.linalg.norm(forward) + 1e-10)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-10)
    up = np.cross(right, forward)

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def render_mesh_views(mesh_path, output_dir, n_views=4, resolution=1024, seed=42):
    """Render n_views of a mesh and save as PNGs + transforms.json."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # Normalise to unit cube.
    bounds = mesh.bounds
    centre = (bounds[0] + bounds[1]) / 2.0
    extent = max((bounds[1] - bounds[0]).max(), 1e-6)
    mesh.apply_translation(-centre)
    mesh.apply_scale(1.0 / extent)

    rng = np.random.RandomState(seed)
    fov = rng.uniform(30, 50) * math.pi / 180.0
    bsphere_r = math.sqrt(3) / 2.0
    radius = bsphere_r / math.sin(fov / 2.0)

    # Camera positions on the sphere.
    positions = hammersley_sphere(n_views, seed)
    positions = positions * radius

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, pos in enumerate(positions):
        c2w = look_at(pos, [0, 0, 0])

        # Render with trimesh (simple depth-based render).
        try:
            scene = trimesh.Scene(mesh)
            scene.set_camera(
                resolution=(resolution, resolution),
                fov=(math.degrees(fov), math.degrees(fov)),
            )
            scene.camera_transform = c2w
            data = scene.save_image(resolution=(resolution, resolution))

            if data is not None:
                png_name = f"{i:04d}.png"
                with open(output_dir / png_name, "wb") as f:
                    f.write(data)
            else:
                # Create a placeholder.
                png_name = f"{i:04d}.png"
                img = np.zeros((resolution, resolution, 4), dtype=np.uint8)
                from PIL import Image
                Image.fromarray(img).save(output_dir / png_name)
        except Exception:
            png_name = f"{i:04d}.png"
            img = np.zeros((resolution, resolution, 4), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(img).save(output_dir / png_name)

        frames.append({
            "file_path": png_name,
            "transform_matrix": c2w.tolist(),
            "camera_angle_x": float(fov),
        })

    transforms = {
        "mesh_scale": 1.0,
        "frames": frames,
    }
    with open(output_dir / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)

    return len(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)

    # Find GT meshes.
    extensions = ["*.glb", "*.obj", "*.ply"]
    mesh_files = []
    for ext in extensions:
        mesh_files.extend(gt_dir.glob(ext))
    # Also check subdirectories.
    for ext in extensions:
        mesh_files.extend(gt_dir.glob(f"*/{ext}"))

    mesh_files = sorted(set(mesh_files))[:args.limit]
    print(f"Found {len(mesh_files)} GT meshes")

    for i, mesh_path in enumerate(mesh_files):
        name = mesh_path.stem
        obj_dir = output_dir / name
        if (obj_dir / "transforms.json").exists():
            print(f"  [{i+1}/{len(mesh_files)}] skip {name}")
            continue

        print(f"  [{i+1}/{len(mesh_files)}] rendering {name}...")
        try:
            n = render_mesh_views(
                mesh_path, obj_dir,
                n_views=args.views,
                resolution=args.resolution,
            )
            print(f"    {n} views rendered")
        except Exception as e:
            print(f"    ERROR: {e}")


if __name__ == "__main__":
    main()
