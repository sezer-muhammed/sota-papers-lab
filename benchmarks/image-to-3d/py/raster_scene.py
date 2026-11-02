#!/usr/bin/env python3
"""Dump trained-model views for the raster benchmark and time gsplat's rasterizer on them.

  gs-ref --entry raster_scene.py --result-dir R --data-dir D --out DIR --tag T [--views 1,41,81]

Owner: raster (E3). For each chosen view of a gsplat simple_trainer checkpoint:
gsplat 1.5.3's projection exactly as rasterization() calls it (opacity-aware
radii, eps2d 0.3, classic mode, pinhole), SH colours (degree 3, +0.5, clamped
at 0), 16 px tile lists (isect_tiles + isect_offset_encode); then gsplat's
rasterize_to_pixels forward and backward on those inputs, timed per CUDA
kernel with torch.profiler (median over --reps) and as whole ops with CUDA
events. Writes <out>/<tag>_v<index>.npz with the visible Gaussians compacted
(v = rank among radii > 0) and flatten_ids remapped to v; the Rust benchmark
(apps/image-to-3d/src/kernels/raster/bench.rs) reads these files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/opt/gsplat/examples")
from datasets.colmap import Parser  # noqa: E402

from gsplat.cuda._wrapper import (  # noqa: E402
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    spherical_harmonics,
)

TILE = 16


def kernel_ms(fn, reps: int, key: str) -> list[float]:
    """Device time of every CUDA kernel whose name contains `key`, per call of fn."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    out = []
    for e in prof.events():
        if key in e.name and e.device_type.name == "CUDA":
            us = getattr(e, "device_time", None) or getattr(e, "cuda_time", 0.0)
            out.append(us / 1e3)
    return out


def event_ms(fn, reps: int) -> list[float]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--step", type=int, help="checkpoint step (default: latest)")
    ap.add_argument("--views", default="1,41,81,121", help="image indices (training views: index % 8 != 0)")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    cfg = yaml.load((a.result_dir / "cfg.yml").read_text(), Loader=yaml.UnsafeLoader)
    cfg = cfg if isinstance(cfg, dict) else vars(cfg)
    ckpts = sorted(a.result_dir.glob("ckpts/ckpt_*_rank0.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if a.step is not None:
        ckpts = [p for p in ckpts if int(p.stem.split("_")[1]) == a.step]
    if not ckpts:
        raise SystemExit(f"no checkpoint in {a.result_dir / 'ckpts'}")
    dev = torch.device("cuda")
    sp = torch.load(ckpts[-1], map_location=dev, weights_only=False)["splats"]
    means, quats = sp["means"], sp["quats"]
    scales, opacities = torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"])
    shs = torch.cat([sp["sh0"], sp["shN"]], 1)
    n_total = int(means.shape[0])
    parser = Parser(
        data_dir=str(a.data_dir), factor=1, normalize=cfg["normalize_world_space"], test_every=cfg["test_every"]
    )
    a.out.mkdir(parents=True, exist_ok=True)
    for idx in [int(s) for s in a.views.split(",") if s]:
        c2w = torch.from_numpy(parser.camtoworlds[idx]).float().to(dev)
        cam = parser.camera_ids[idx]
        K = torch.from_numpy(parser.Ks_dict[cam]).float().to(dev)[None]
        w, h = parser.imsize_dict[cam]
        viewmat = torch.linalg.inv(c2w)[None]
        with torch.no_grad():
            radii, means2d, depths, conics, _ = fully_fused_projection(
                means, None, quats, scales, viewmat, K, w, h,
                eps2d=0.3, near_plane=cfg["near_plane"], far_plane=cfg["far_plane"],
                packed=False, opacities=opacities,
            )  # fmt: skip
            vis = (radii > 0).all(-1)[0]
            dirs = means[None] - c2w[None, None, :3, 3]
            colors = spherical_harmonics(cfg["sh_degree"], dirs, shs[None].expand(1, -1, -1, -1), masks=vis[None])
            colors = torch.clamp_min(colors + 0.5, 0.0)
            tw, th = math.ceil(w / TILE), math.ceil(h / TILE)
            _, isect_ids, flat = isect_tiles(means2d, radii, depths, TILE, tw, th, sort=True, packed=False)
            offsets = isect_offset_encode(isect_ids, 1, tw, th)
        opac = opacities[None].expand(1, -1).contiguous()

        # gsplat raster forward + backward on exactly these inputs.
        m2 = means2d.detach().clone().requires_grad_(True)
        co = conics.detach().clone().requires_grad_(True)
        cl = colors.detach().clone().requires_grad_(True)
        op = opac.detach().clone().requires_grad_(True)
        g = torch.Generator(device=dev).manual_seed(1)
        v_rgb = (torch.rand((1, h, w, 3), device=dev, generator=g) - 0.5) * 4e-6

        def fwd():
            with torch.no_grad():
                return rasterize_to_pixels(m2, co, cl, op, w, h, TILE, offsets, flat, packed=False)

        def fwd_bwd():
            rc, ra = rasterize_to_pixels(m2, co, cl, op, w, h, TILE, offsets, flat, packed=False)
            (rc * v_rgb).sum().backward()

        f_k = kernel_ms(fwd, a.reps, "rasterize_to_pixels_3dgs_fwd")
        b_k = kernel_ms(fwd_bwd, a.reps, "rasterize_to_pixels_3dgs_bwd")
        f_op = event_ms(fwd, a.reps)
        fb_op = event_ms(fwd_bwd, a.reps)

        # Compact to the visible set (v = rank among radii > 0).
        v_of = torch.cumsum(vis.int(), 0) - 1
        flat_v = v_of[flat.long()].int()
        sel = vis.nonzero()[:, 0]
        out = {
            "width": np.array(w, np.int32),
            "height": np.array(h, np.int32),
            "means2d": means2d[0, sel].cpu().numpy(),
            "conics": conics[0, sel].cpu().numpy(),
            "colors": colors[0, sel].cpu().numpy(),
            "opacities": opacities[sel].cpu().numpy(),
            "depths": depths[0, sel].cpu().numpy(),
            "isect_offsets": offsets[0].cpu().numpy(),
            "flatten_ids": flat_v.cpu().numpy(),
            "viewmat": viewmat[0].cpu().numpy(),
            "K": K[0].cpu().numpy(),
            "gs_fwd_ms": np.array(float(np.median(f_k)), np.float32),
            "gs_bwd_ms": np.array(float(np.median(b_k)), np.float32),
            "gs_fwd_op_ms": np.array(float(np.median(f_op)), np.float32),
            "gs_fwd_bwd_op_ms": np.array(float(np.median(fb_op)), np.float32),
        }
        info = {
            "checkpoint": str(ckpts[-1]),
            "image": parser.image_names[idx],
            "n_gaussians": n_total,
            "visible": int(sel.numel()),
            "instances": int(flat.numel()),
            "width": w,
            "height": h,
            "gsplat_fwd_kernel_ms": float(np.median(f_k)),
            "gsplat_bwd_kernel_ms": float(np.median(b_k)),
            "gsplat_fwd_kernel_n": len(f_k),
            "gsplat_fwd_op_ms": float(np.median(f_op)),
            "gsplat_fwd_bwd_op_ms": float(np.median(fb_op)),
        }
        out["meta"] = np.frombuffer(json.dumps(info).encode(), np.uint8)
        path = a.out / f"{a.tag}_v{idx:03d}.npz"
        np.savez(path, **{k: np.ascontiguousarray(v) for k, v in out.items()})  # C order for the Rust reader
        print(json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
