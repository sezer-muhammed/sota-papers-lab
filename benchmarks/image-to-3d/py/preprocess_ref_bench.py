#!/usr/bin/env python3
"""Preprocess kernels of gsplat 1.5.3 and Faster-GS on the garden benchmark inputs.

The counterpart of the Rust benchmark `gpu_preprocess_bench_garden`
(apps/image-to-3d/src/kernels/preprocess_tests.rs), which writes the timed
views and the Morton order to garden_views.npz. Same PLY, same Gaussian order,
same views and sizes (1M = every third Gaussian, 3M, 6M = each twice with a
small jitter), timed per kernel with the PyTorch profiler (CUPTI):

- gsplat: `fully_fused_projection` (opacity-aware radii, as `rasterization()`
  calls it) and `spherical_harmonics` on the visible set, forward and backward.
- Faster-GS (FasterGSCudaBackend): `preprocess_cu` and `preprocess_backward_cu`
  inside one `diff_rasterize` forward/backward.

Runs in the spl-bl-fastergs image (both libraries installed), e.g.
  docker run --rm --gpus all -v ~/spl-work/out:/out -v <this dir>:/bench:ro spl-bl-fastergs \
    python /bench/preprocess_ref_bench.py --ply /out/image-to-3d/garden_mcmc3m_10k.ply \
    --views /out/image-to-3d/bench/garden_views.npz --out /out/image-to-3d/bench/preprocess_ref.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile


def read_ply(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        names = [ln.split()[-1].decode() for ln in header.splitlines() if ln.startswith(b"property float")]
        n = int(next(ln.split()[-1] for ln in header.splitlines() if ln.startswith(b"element vertex")))
        data = np.fromfile(f, dtype=np.float32, count=n * len(names)).reshape(n, len(names))
    col = {k: i for i, k in enumerate(names)}

    def cols(prefix: str, k: int) -> np.ndarray:
        return data[:, [col[f"{prefix}{i}"] for i in range(k)]]

    rest = cols("f_rest_", 45).reshape(n, 3, 15).transpose(0, 2, 1)  # channel-major -> [N, 15, 3]
    return {
        "means": data[:, [col["x"], col["y"], col["z"]]],
        "sh0": cols("f_dc_", 3)[:, None, :],
        "rest": np.ascontiguousarray(rest),
        "opacity": data[:, col["opacity"]],
        "scales": cols("scale_", 3),
        "quats": cols("rot_", 4),
    }


def subset(m: dict[str, np.ndarray], order: np.ndarray, label: str) -> dict[str, np.ndarray]:
    if label == "1M":
        return {k: v[order[::3]] for k, v in m.items()}
    if label == "3M":
        return {k: v[order] for k, v in m.items()}
    # 6M: each Gaussian twice, the copy jittered by up to a quarter of its scale.
    out = {k: np.repeat(v[order], 2, axis=0) for k, v in m.items()}
    rng = np.random.default_rng(9)
    jitter = 0.5 * np.exp(out["scales"][1::2]) * (rng.random((len(order), 3), dtype=np.float32) - 0.5)
    out["means"][1::2] += jitter
    return out


def kernel_ms(fn, keys: list[str], reps: int) -> dict[str, float]:
    """Mean device time per call of every kernel whose name contains a key."""
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    out = {k: 0.0 for k in keys}
    for e in prof.key_averages():
        for k in keys:
            if k in e.key:
                out[k] += e.device_time_total / 1000.0 / reps
    return out


def event_ms(fn, reps: int) -> float:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return float(np.median(times))


def bench_gsplat(t: dict, views: dict, reps: int) -> dict:
    from gsplat.cuda._wrapper import fully_fused_projection, spherical_harmonics

    means, quats = t["means"].requires_grad_(True), t["quats"].requires_grad_(True)
    scales = t["scales"].exp().requires_grad_(True)
    opac = torch.sigmoid(t["opacity"])
    coeffs = torch.cat([t["sh0"], t["rest"]], 1).contiguous().requires_grad_(True)
    w, h = views["width"], views["height"]
    rows = []
    for vm, K in zip(views["viewmats"], views["Ks"], strict=True):
        vm, K = vm[None], K[None]
        campos = torch.linalg.inv(vm[0])[:3, 3]

        def fwd():
            m, q, s, c = means, quats, scales, coeffs
            radii, m2, depths, conics, _ = fully_fused_projection(
                m, None, q, s, vm, K, w, h, eps2d=0.3, near_plane=0.01, far_plane=1e4, packed=False, opacities=opac
            )
            vis = (radii > 0).all(-1)[0]
            colors = spherical_harmonics(3, m - campos, c, masks=vis)
            colors = torch.clamp_min(colors + 0.5, 0.0)
            return (m, q, s, c), (m2, conics, depths, colors)

        with torch.no_grad():
            _, outs = fwd()
            n_vis = int((outs[2][0] > 0).sum())
        inputs, outs = fwd()
        ups = [torch.randn_like(o) * 1e-3 for o in outs]

        def bwd():
            torch.autograd.grad(outs, inputs, ups, retain_graph=True)

        def fwd_only():
            with torch.no_grad():
                fwd()

        k_f = kernel_ms(fwd_only, ["projection", "spherical_harmonics"], reps)
        k_b = kernel_ms(bwd, ["projection", "spherical_harmonics"], reps)
        rows.append(
            {
                "visible": n_vis,
                "fwd_projection_kernel_ms": k_f["projection"],
                "fwd_sh_kernel_ms": k_f["spherical_harmonics"],
                "bwd_projection_kernel_ms": k_b["projection"],
                "bwd_sh_kernel_ms": k_b["spherical_harmonics"],
                "fwd_total_ms": event_ms(fwd_only, reps),
                "bwd_total_ms": event_ms(bwd, reps),
            }
        )
        del inputs, outs, ups
    return mean_rows(rows)


def bench_fastergs(t: dict, views: dict, reps: int) -> dict:
    sys.path.insert(0, "/opt/nerficg/src/Methods/FasterGS/FasterGSCudaBackend")
    from FasterGSCudaBackend.torch_bindings.rasterization import RasterizerSettings, diff_rasterize

    w, h = views["width"], views["height"]
    params = [
        t["means"],
        t["scales"],
        t["quats"],
        t["opacity"][:, None].contiguous(),
        t["sh0"],
        t["rest"],
    ]
    for p in params:
        p.requires_grad_(True)
    rows = []
    for vm, K in zip(views["viewmats"], views["Ks"], strict=True):
        settings = RasterizerSettings(
            vm.contiguous(),
            torch.linalg.inv(vm)[:3, 3].contiguous(),
            torch.zeros(3, device="cuda"),
            16,
            w,
            h,
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
            0.01,
            1e4,
            False,
        )
        image = diff_rasterize(*params, torch.empty(0), settings)
        up = torch.randn_like(image) * 1e-3

        def fwd():
            with torch.no_grad():
                diff_rasterize(*params, torch.empty(0), settings)

        def bwd():
            img = diff_rasterize(*params, torch.empty(0), settings)
            torch.autograd.grad(img, params, up)

        k_f = kernel_ms(fwd, ["preprocess_cu"], reps)
        k_fb = kernel_ms(bwd, ["preprocess_cu", "preprocess_backward_cu"], reps)
        rows.append(
            {
                "fwd_preprocess_kernel_ms": k_f["preprocess_cu"],
                "bwd_preprocess_kernel_ms": k_fb["preprocess_backward_cu"],
            }
        )
        del image, up
    return mean_rows(rows)


def mean_rows(rows: list[dict]) -> dict:
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | {"views": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ply", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--sizes", default="1M,3M,6M")
    ap.add_argument("--impls", default="gsplat,fastergs")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--max-views", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = read_ply(a.ply)
    z = np.load(a.views)
    nv = min(a.max_views, z["viewmats"].shape[0])
    views = {
        "viewmats": torch.from_numpy(z["viewmats"][:nv]).cuda(),
        "Ks": torch.from_numpy(z["Ks"][:nv]).cuda(),
        "width": int(z["width"]),
        "height": int(z["height"]),
    }
    order = z["order"].astype(np.int64)
    report = {"ply": a.ply, "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "sizes": {}}
    for label in a.sizes.split(","):
        sub = subset(model, order, label)
        t = {k: torch.from_numpy(np.ascontiguousarray(v)).cuda() for k, v in sub.items()}
        row = {"n": int(t["means"].shape[0])}
        for impl in a.impls.split(","):
            fn = {"gsplat": bench_gsplat, "fastergs": bench_fastergs}[impl]
            row[impl] = fn({k: v.detach().clone() for k, v in t.items()}, views, a.reps)
            torch.cuda.empty_cache()
        report["sizes"][label] = row
        print(label, json.dumps(row), flush=True)
        del t
        torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
