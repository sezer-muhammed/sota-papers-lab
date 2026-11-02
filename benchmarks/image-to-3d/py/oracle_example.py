#!/usr/bin/env python3
"""Write a random, physically plausible oracle input for one op (numpy only).

  python oracle_example.py <op> out.in.npz [--seed 0] [--n 256] [--width 64]
                           [--height 48] [--cameras 1]

This is the reference for how a Rust test should lay out an input file: the
key names, dtypes and shapes below are exactly what oracle.py reads (see
ORACLE.md). Everything is little-endian, C-order, float32 unless noted.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

OPS = ("sh", "project", "isect", "rasterize", "ssim", "adam", "render")


def _rotation(rng: np.random.Generator, max_angle: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = rng.uniform(-max_angle, max_angle)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)


def cameras(rng: np.random.Generator, c: int, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera 4x4 (OpenCV: +x right, +y down, +z forward) and pixel K."""
    viewmats = np.zeros((c, 4, 4), np.float32)
    ks = np.zeros((c, 3, 3), np.float32)
    for i in range(c):
        r = _rotation(rng, 0.15)
        t = rng.uniform(-0.2, 0.2, size=3)
        viewmats[i, :3, :3] = r
        viewmats[i, :3, 3] = t
        viewmats[i, 3, 3] = 1
        f = 1.1 * w * rng.uniform(0.9, 1.1)
        ks[i] = [[f, 0, w / 2 + rng.uniform(-2, 2)], [0, f, h / 2 + rng.uniform(-2, 2)], [0, 0, 1]]
    return viewmats, ks


def gaussians(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """3D Gaussians in front of the cameras, depth 2-6, activated parameters."""
    z = rng.uniform(2.0, 6.0, n)
    xy = rng.uniform(-0.45, 0.45, (n, 2)) * z[:, None]
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return {
        "means": np.concatenate([xy, z[:, None]], 1).astype(np.float32),
        "quats": q.astype(np.float32),  # wxyz, gsplat normalizes internally
        "scales": np.exp(rng.uniform(math.log(0.01), math.log(0.12), (n, 3))).astype(np.float32),
        "opacities": rng.uniform(0.05, 0.99, n).astype(np.float32),
    }


def splats_2d(rng: np.random.Generator, c: int, n: int, w: int, h: int, d: int) -> dict[str, np.ndarray]:
    """Projected 2D splats: what project() would hand to isect/rasterize."""
    means2d = np.stack([rng.uniform(-4, w + 4, (c, n)), rng.uniform(-4, h + 4, (c, n))], -1)
    sx = np.exp(rng.uniform(math.log(0.7), math.log(6.0), (c, n)))
    sy = np.exp(rng.uniform(math.log(0.7), math.log(6.0), (c, n)))
    rho = rng.uniform(-0.8, 0.8, (c, n))
    a, b, cc = sx * sx + 0.3, rho * sx * sy, sy * sy + 0.3  # covariance + eps2d
    det = a * cc - b * b
    conics = np.stack([cc / det, -b / det, a / det], -1)
    radii = np.stack([np.ceil(3.33 * np.sqrt(a)), np.ceil(3.33 * np.sqrt(cc))], -1)
    depths = rng.uniform(1.0, 10.0, (c, n))
    return {
        "means2d": means2d.astype(np.float32),
        "conics": conics.astype(np.float32),
        "radii": radii.astype(np.int32),
        "depths": depths.astype(np.float32),
        "colors": rng.uniform(0, 1, (c, n, d)).astype(np.float32),
        "opacities": rng.uniform(0.05, 0.99, (c, n)).astype(np.float32),
    }


def make(op: str, seed: int = 0, n: int = 256, w: int = 64, h: int = 48, c: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    i32 = np.int32
    if op == "sh":
        k = 16
        return {
            "degree": np.array(3, i32),
            "dirs": rng.normal(size=(n, 3)).astype(np.float32),
            "coeffs": (rng.normal(size=(n, k, 3)) * 0.3).astype(np.float32),
            "v_colors": rng.normal(size=(n, 3)).astype(np.float32),
        }
    if op == "project":
        g = gaussians(rng, n)
        viewmats, ks = cameras(rng, c, w, h)
        return {
            "means": g["means"],
            "quats": g["quats"],
            "scales": g["scales"],
            "viewmats": viewmats,
            "Ks": ks,
            "width": np.array(w, i32),
            "height": np.array(h, i32),
            "eps2d": np.array(0.3, np.float32),
            "near_plane": np.array(0.01, np.float32),
            "far_plane": np.array(1e10, np.float32),
            "calc_compensations": np.array(1, i32),
            "opacities": g["opacities"],  # opacity-aware radii, as rasterization() does
            "v_means2d": rng.normal(size=(c, n, 2)).astype(np.float32),
            "v_depths": rng.normal(size=(c, n)).astype(np.float32),
            "v_conics": rng.normal(size=(c, n, 3)).astype(np.float32),
            "v_compensations": rng.normal(size=(c, n)).astype(np.float32),
        }
    if op == "isect":
        s = splats_2d(rng, c, n, w, h, 3)
        return {
            "means2d": s["means2d"],
            "radii": s["radii"],
            "depths": s["depths"],
            "width": np.array(w, i32),
            "height": np.array(h, i32),
            "tile_size": np.array(16, i32),
        }
    if op == "rasterize":
        s = splats_2d(rng, c, n, w, h, 3)
        return {
            **s,
            "width": np.array(w, i32),
            "height": np.array(h, i32),
            "tile_size": np.array(16, i32),
            "backgrounds": rng.uniform(0, 1, (c, 3)).astype(np.float32),
            "absgrad": np.array(1, i32),
            "v_render_colors": rng.normal(size=(c, h, w, 3)).astype(np.float32),
            "v_render_alphas": rng.normal(size=(c, h, w, 1)).astype(np.float32),
        }
    if op == "ssim":
        base = rng.uniform(0, 1, (c, 3, h // 4 + 1, w // 4 + 1))
        img2 = np.kron(base, np.ones((1, 1, 4, 4)))[:, :, :h, :w]
        img1 = np.clip(img2 + rng.normal(scale=0.08, size=img2.shape), 0, 1)
        return {
            "img1": img1.astype(np.float32),
            "img2": img2.astype(np.float32),
            "padding_valid": np.array(0, i32),
            "lambda_dssim": np.array(0.2, np.float32),
            "v_ssim": np.array(1.0, np.float32),
        }
    if op == "adam":
        shape = (n, 3)
        return {
            "param": rng.normal(size=shape).astype(np.float32),
            "grad": (rng.normal(size=shape) * 1e-3).astype(np.float32),
            "exp_avg": (rng.normal(size=shape) * 1e-4).astype(np.float32),
            "exp_avg_sq": (rng.uniform(0, 1e-6, size=shape)).astype(np.float32),
            "step": np.array(10, i32),
            "lr": np.array(1.6e-4, np.float32),
            "beta1": np.array(0.9, np.float32),
            "beta2": np.array(0.999, np.float32),
            "eps": np.array(1e-15, np.float32),
            "valid": rng.uniform(size=n) < 0.7,
        }
    if op == "render":
        g = gaussians(rng, n)
        viewmats, ks = cameras(rng, c, w, h)
        sh = (rng.normal(size=(n, 16, 3)) * 0.1).astype(np.float32)
        sh[:, 0, :] = rng.uniform(-1.5, 1.5, (n, 3))
        return {
            **g,
            "colors": sh,
            "sh_degree": np.array(3, i32),
            "viewmats": viewmats,
            "Ks": ks,
            "width": np.array(w, i32),
            "height": np.array(h, i32),
            "tile_size": np.array(16, i32),
            "backgrounds": np.zeros((c, 3), np.float32),
            "v_render_colors": rng.normal(size=(c, h, w, 3)).astype(np.float32),
            "v_render_alphas": rng.normal(size=(c, h, w, 1)).astype(np.float32),
        }
    raise ValueError(f"unknown op {op!r}; ops: {', '.join(OPS)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("op", choices=OPS)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--cameras", type=int, default=1)
    a = ap.parse_args()
    np.savez(a.out, **make(a.op, a.seed, a.n, a.width, a.height, a.cameras))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
