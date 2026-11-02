#!/usr/bin/env python3
"""Prepare a small evaluation dataset for Pixal3D-mv object generation.

Creates synthetic GT objects, renders multi-view RGBA inputs using a
numpy software rasterizer (no display/OpenGL needed), and organizes
them for eval_object.py.

Usage (inside spl-pixal3d container or any env with trimesh+numpy+PIL):
  python prepare_eval.py --output_dir /work/eval --views 1 2 4
"""
import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


OBJECTS = {
    "sphere": lambda: trimesh.creation.icosphere(subdivisions=4, radius=0.4),
    "torus": lambda: trimesh.creation.torus(
        major_radius=0.3, minor_radius=0.12, major_sections=48, minor_sections=24
    ),
    "capsule": lambda: trimesh.creation.capsule(height=0.5, radius=0.2, count=[16, 16]),
    "box": lambda: trimesh.creation.box(extents=[0.5, 0.6, 0.4]),
    "cylinder": lambda: trimesh.creation.cylinder(radius=0.25, height=0.6, sections=32),
    "cone": lambda: trimesh.creation.cone(radius=0.3, height=0.6, sections=32),
}


def hammersley_sphere(n, seed=42):
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


def look_at(eye, target):
    up = np.array([0, 0, 1], dtype=np.float64)
    fwd = np.array(target, dtype=np.float64) - np.array(eye, dtype=np.float64)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-10)
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-10)
    up = np.cross(right, fwd)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -fwd
    c2w[:3, 3] = eye
    return c2w


def software_render(mesh, c2w, fov_rad, resolution):
    """Render an RGBA image using a numpy software rasterizer."""
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    verts = mesh.vertices @ R.T + t
    faces = mesh.faces

    f = 0.5 * resolution / math.tan(fov_rad / 2.0)
    cx = cy = resolution / 2.0

    # Blender convention: camera looks -Z, so negate Z for projection.
    depth = -verts[:, 2]
    safe_depth = np.where(depth > 0.01, depth, 0.01)
    xs = verts[:, 0] / safe_depth * f + cx
    ys = -verts[:, 1] / safe_depth * f + cy
    zs = depth

    # Compute face normals in camera space for shading.
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    normals = normals / norms

    light_dir = np.array([0.3, -0.5, 0.8])
    light_dir = light_dir / np.linalg.norm(light_dir)
    diffuse = np.clip(np.abs(np.sum(normals * light_dir, axis=1)), 0.0, 1.0)
    ambient = 0.25
    shade = np.clip(ambient + 0.75 * diffuse, 0.0, 1.0)

    depth_buf = np.full((resolution, resolution), np.inf, dtype=np.float64)
    color_buf = np.zeros((resolution, resolution, 3), dtype=np.float64)
    alpha_buf = np.zeros((resolution, resolution), dtype=np.uint8)

    for fi in range(len(faces)):
        i0, i1, i2 = faces[fi]
        x0, y0, z0 = xs[i0], ys[i0], zs[i0]
        x1, y1, z1 = xs[i1], ys[i1], zs[i1]
        x2, y2, z2 = xs[i2], ys[i2], zs[i2]

        if z0 <= 0.01 or z1 <= 0.01 or z2 <= 0.01:
            continue

        xmin = max(int(min(x0, x1, x2)), 0)
        xmax = min(int(max(x0, x1, x2)) + 1, resolution)
        ymin = max(int(min(y0, y1, y2)), 0)
        ymax = min(int(max(y0, y1, y2)) + 1, resolution)

        if xmin >= xmax or ymin >= ymax:
            continue

        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-6:
            continue

        inv_area = 1.0 / area
        s = shade[fi]
        base_color = np.array([0.7, 0.65, 0.75])

        py, px = np.mgrid[ymin:ymax, xmin:xmax]
        w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) * inv_area
        w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) * inv_area
        w2 = 1.0 - w0 - w1

        mask = (w0 >= -0.001) & (w1 >= -0.001) & (w2 >= -0.001)
        if not np.any(mask):
            continue

        z_interp = w0 * z0 + w1 * z1 + w2 * z2
        closer = mask & (z_interp < depth_buf[ymin:ymax, xmin:xmax])

        if not np.any(closer):
            continue

        depth_buf[ymin:ymax, xmin:xmax][closer] = z_interp[closer]
        for c in range(3):
            color_buf[ymin:ymax, xmin:xmax, c][closer] = base_color[c] * s
        alpha_buf[ymin:ymax, xmin:xmax][closer] = 255

    rgba = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(color_buf * 255, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = alpha_buf
    return rgba


def render_views(mesh, n_views, resolution=1024, seed=42):
    rng = np.random.RandomState(seed)
    fov = rng.uniform(35, 45) * math.pi / 180.0
    bsphere_r = math.sqrt(3) / 2.0
    radius = bsphere_r / math.sin(fov / 2.0)

    positions = hammersley_sphere(n_views, seed) * radius
    images = []
    frames = []

    for i, pos in enumerate(positions):
        c2w = look_at(pos, [0, 0, 0])
        rgba = software_render(mesh, c2w, fov, resolution)

        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, "PNG")
        images.append(buf.getvalue())

        frames.append({
            "file_path": f"{i:04d}.png",
            "transform_matrix": c2w.tolist(),
            "camera_angle_x": float(fov),
        })

    return images, frames, fov


def prepare_object(name, mesh_fn, output_dir, view_counts, resolution=1024):
    mesh = mesh_fn()
    bounds = mesh.bounds
    centre = (bounds[0] + bounds[1]) / 2.0
    extent = max((bounds[1] - bounds[0]).max(), 1e-6)
    mesh.apply_translation(-centre)
    mesh.apply_scale(1.0 / extent)

    gt_dir = output_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(str(gt_dir / f"{name}.glb"))

    for nv in view_counts:
        input_dir = output_dir / f"inputs_v{nv}" / name
        input_dir.mkdir(parents=True, exist_ok=True)

        imgs, frames, fov = render_views(mesh, nv, resolution)
        transforms = {"mesh_scale": 1.0, "frames": frames}

        with open(input_dir / "transforms.json", "w") as f:
            json.dump(transforms, f, indent=2)
        for idx, img_data in enumerate(imgs):
            with open(input_dir / f"{idx:04d}.png", "wb") as f:
                f.write(img_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/work/eval")
    parser.add_argument("--views", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--objects", nargs="+", default=list(OBJECTS.keys()))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.objects:
        if name not in OBJECTS:
            print(f"  [skip] unknown: {name}")
            continue
        print(f"  preparing {name}...")
        prepare_object(name, OBJECTS[name], output_dir, args.views, args.resolution)

    for nv in args.views:
        d = output_dir / f"inputs_v{nv}"
        count = len(list(d.iterdir())) if d.exists() else 0
        print(f"  inputs_v{nv}: {count} objects")

    gt_dir = output_dir / "gt"
    gt_count = len(list(gt_dir.glob("*.glb"))) if gt_dir.exists() else 0
    print(f"  gt: {gt_count} meshes")
    print("Done.")


if __name__ == "__main__":
    main()
