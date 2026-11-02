#!/usr/bin/env python3
"""CPU contact sheet of a GLB: 4 azimuths, textured (base colour) and shaded.

Point-splat z-buffer over the mesh vertices (dense enough for Pixal3D's ~1M
triangle outputs), so it needs only trimesh + numpy + PIL, no GL context.

Usage: python render_glb.py mesh.glb out.png [--size 512]
"""
import argparse

import numpy as np
import trimesh
from PIL import Image


def load(path):
    scene = trimesh.load(path)
    meshes = scene.dump() if isinstance(scene, trimesh.Scene) else [scene]
    vs, ns, cs = [], [], []
    for m in meshes:
        try:
            col = m.visual.to_color().vertex_colors[:, :3]
        except Exception:
            col = np.full((len(m.vertices), 3), 200, np.uint8)
        vs.append(m.vertices)
        ns.append(m.vertex_normals)
        cs.append(col)
    v = np.concatenate(vs)
    lo, hi = v.min(0), v.max(0)
    v = (v - (lo + hi) / 2) / (hi - lo).max()
    return v, np.concatenate(ns), np.concatenate(cs).astype(np.float32)


def splat(v, n, c, azim, elev, size, shaded):
    a, e = np.radians(azim), np.radians(elev)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    r = rx @ ry
    p, nn = v @ r.T, n @ r.T
    x = ((p[:, 0] * 0.85 + 0.5) * (size - 1)).astype(int)
    y = ((0.5 - p[:, 1] * 0.85) * (size - 1)).astype(int)
    ok = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    x, y, z, nn, c = x[ok], y[ok], p[ok, 2], nn[ok], c[ok]
    order = np.argsort(z)  # far to near: later writes win
    img = np.full((size, size, 3), 255.0, np.float32)
    if shaded:
        light = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])
        val = 60 + 180 * np.clip(np.abs(nn @ light), 0, 1)
        col = np.repeat(val[:, None], 3, 1)
    else:
        col = c
    for dx in (0, 1):  # 2x2 splat closes pinholes
        for dy in (0, 1):
            xx, yy = np.clip(x + dx, 0, size - 1), np.clip(y + dy, 0, size - 1)
            img[yy[order], xx[order]] = col[order]
    return img.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()
    v, n, c = load(args.glb)
    rows = [np.concatenate([splat(v, n, c, az, 20, args.size, sh) for az in (0, 90, 180, 270)], 1)
            for sh in (False, True)]
    Image.fromarray(np.concatenate(rows, 0)).save(args.out)


if __name__ == "__main__":
    main()
