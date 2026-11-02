#!/usr/bin/env python3
"""Time Faster-GS's blend kernels on the raster benchmark's views.

  docker run ... spl-bl-fastergs python raster_fastergs.py --result-dir R --scenes DIR [--reps 20]

Owner: raster (E3). Runs in the spl-bl-fastergs image (Faster-GS 090e8ad,
Apache-2.0). Loads the gsplat checkpoint the views of `raster_scene.py` came
from (same parameters: log scales, wxyz quaternions, logit opacities, SH
degree 3), and for every `<tag>_v<index>.npz` in --scenes renders that camera
(viewmat, K, size stored in the file) with Faster-GS's `diff_rasterize`,
forward and backward, under torch.profiler. Prints one JSON line per view with
the median device time of every Faster-GS kernel (blend_cu is the forward
blend, blend_backward_kernel the per-Gaussian backward), and writes them next
to the scenes as `<tag>_v<index>.fastergs.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/opt/nerficg/src/Methods/FasterGS/FasterGSCudaBackend")
from FasterGSCudaBackend import _C  # noqa: E402
from FasterGSCudaBackend.torch_bindings.rasterization import RasterizerSettings, diff_rasterize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--scenes", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="scene files <tag>_v*.npz")
    ap.add_argument("--step", type=int)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    ckpts = sorted(a.result_dir.glob("ckpts/ckpt_*_rank0.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if a.step is not None:
        ckpts = [p for p in ckpts if int(p.stem.split("_")[1]) == a.step]
    dev = torch.device("cuda")
    sp = torch.load(ckpts[-1], map_location=dev, weights_only=False)["splats"]
    params = [
        sp["means"].contiguous(),
        sp["scales"].contiguous(),
        sp["quats"].contiguous(),
        sp["opacities"][:, None].contiguous(),
        sp["sh0"].contiguous(),
        sp["shN"].contiguous(),
    ]
    params = [p.detach().clone().requires_grad_(True) for p in params]
    for path in sorted(a.scenes.glob(f"{a.tag}_v*.npz")):
        with np.load(path) as z:
            viewmat = torch.from_numpy(z["viewmat"]).float().to(dev)
            K = z["K"]
            w, h = int(z["width"]), int(z["height"])
        c2w = torch.linalg.inv(viewmat)
        settings = RasterizerSettings(
            viewmat.contiguous(),
            c2w[:3, 3].contiguous(),
            torch.zeros(3, device=dev),
            16,
            w,
            h,
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
            0.01,
            1000.0,
            False,
        )
        g = torch.Generator(device=dev).manual_seed(1)
        grad = (torch.rand((3, h, w), device=dev, generator=g) - 0.5) * 4e-6

        def step():
            img = diff_rasterize(*params, torch.empty(0), settings)
            img.backward(grad)

        for _ in range(3):
            step()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(a.reps):
                step()
            torch.cuda.synchronize()
        per_kernel: dict[str, list[float]] = defaultdict(list)
        for e in prof.events():
            if e.device_type.name == "CUDA":
                us = getattr(e, "device_time", None) or getattr(e, "cuda_time", 0.0)
                per_kernel[e.name].append(us / 1e3)
        with torch.no_grad():
            out = _C.forward(*[p.detach() for p in params], *settings.as_tuple())
        n_instances, n_buckets = int(out[5]), int(out[6])
        kernels = {k: float(np.median(v)) for k, v in per_kernel.items() if len(v) >= a.reps}
        blend_f = sum(v for k, v in kernels.items() if "blend_cu" in k)
        blend_b = sum(v for k, v in kernels.items() if "blend_backward" in k)
        info = {
            "scene": path.name,
            "checkpoint": str(ckpts[-1]),
            "n_gaussians": int(params[0].shape[0]),
            "fastergs_instances": n_instances,
            "fastergs_buckets": n_buckets,
            "blend_fwd_ms": blend_f,
            "blend_bwd_ms": blend_b,
            "kernels_ms": dict(sorted(kernels.items(), key=lambda kv: -kv[1])),
            "gpu": torch.cuda.get_device_name(),
        }
        path.with_suffix(".fastergs.json").write_text(json.dumps(info, indent=1) + "\n")
        print(json.dumps({k: info[k] for k in info if k != "kernels_ms"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
