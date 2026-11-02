#!/usr/bin/env python3
"""Render the held-out views of a gsplat simple_trainer run to <out>/<stem>.png.

  python gsplat_render.py --result-dir R --data-dir D --out DIR [--step S]

Mirrors simple_trainer's Runner.rasterize_splats / eval for the default
config (no appearance or pose optimisation, no bilateral grid): same parser
(factor 1 on the protocol image folder, normalize_world_space from cfg.yml),
same rasterization() arguments. Pixels are clamped and rounded to 8 bits
the way torchvision.utils.save_image does (x * 255 + 0.5), which is how 3DGS
render.py writes its renders. Also writes <out>/../render.json with timings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, "/opt/gsplat/examples")
from datasets.colmap import Dataset, Parser  # noqa: E402

from gsplat.rendering import rasterization  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--step", type=int, help="checkpoint step (default: latest)")
    a = ap.parse_args()

    cfg = yaml.load((a.result_dir / "cfg.yml").read_text(), Loader=yaml.UnsafeLoader)
    cfg = cfg if isinstance(cfg, dict) else vars(cfg)
    ckpts = sorted(a.result_dir.glob("ckpts/ckpt_*_rank0.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if a.step is not None:
        ckpts = [p for p in ckpts if int(p.stem.split("_")[1]) == a.step]
    if not ckpts:
        raise SystemExit(f"no checkpoint in {a.result_dir / 'ckpts'}")
    for key in ("app_opt", "pose_opt", "use_bilateral_grid", "with_ut", "with_eval3d"):
        if cfg.get(key):
            raise SystemExit(f"gsplat_render.py does not support {key}=True")

    dev = torch.device("cuda")
    ck = torch.load(ckpts[-1], map_location=dev, weights_only=False)
    sp = ck["splats"]
    parser = Parser(
        data_dir=str(a.data_dir), factor=1, normalize=cfg["normalize_world_space"], test_every=cfg["test_every"]
    )
    valset = Dataset(parser, split="val")
    a.out.mkdir(parents=True, exist_ok=True)

    means, quats = sp["means"], sp["quats"]
    scales, opacities = torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"])
    colors = torch.cat([sp["sh0"], sp["shN"]], 1)
    strategy = cfg.get("strategy")
    absgrad = bool(getattr(strategy, "absgrad", False)) if strategy is not None else False
    times = []
    for i in range(len(valset)):
        data = valset[i]
        c2w = data["camtoworld"][None].to(dev)
        K = data["K"][None].to(dev)
        h, w = data["image"].shape[:2]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            rgb, _, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(c2w),
                Ks=K,
                width=w,
                height=h,
                packed=cfg["packed"],
                absgrad=absgrad,
                sparse_grad=cfg["sparse_grad"],
                rasterize_mode="antialiased" if cfg["antialiased"] else "classic",
                camera_model=cfg["camera_model"],
                sh_degree=cfg["sh_degree"],
                near_plane=cfg["near_plane"],
                far_plane=cfg["far_plane"],
            )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        img = (rgb[0].clamp(0, 1) * 255 + 0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
        name = Path(parser.image_names[valset.indices[i]]).stem
        Image.fromarray(np.ascontiguousarray(img)).save(a.out / f"{name}.png")
    info = {
        "checkpoint": str(ckpts[-1]),
        "step": int(ck["step"]),
        "num_gaussians": int(means.shape[0]),
        "views": len(valset),
        "render_ms_mean": 1e3 * float(np.mean(times[1:] if len(times) > 1 else times)),
    }
    (a.out.parent / "render.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
