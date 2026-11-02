#!/usr/bin/env python3
"""Check metrics.py on known pairs (run inside spl-gs-ref).

  python metrics_check.py [--scene-image PATH] [--json report.json]

1. Analytic pair: GT in [0, 245], render = GT + 10 levels everywhere, so
   PSNR = 20 log10(255 / 10) = 28.130803608679106 dB (up to float32 rounding
   of x / 255, as in 3DGS); identical images give SSIM 1 and LPIPS 0.
2. A realistic pair (a scene image, or a synthetic one, vs. its JPEG q=60
   re-encode) checked across independent implementations:
   - metrics.ssim (3DGS conv, float32) vs a float64 conv reference and
     fused-ssim(padding="same", train=False)
   - metrics.lpips (torchmetrics vgg, normalize=False) vs the `lpips`
     package (vgg, v0.1, inputs left in [0, 1] like 3DGS's lpipsPyTorch)
   - metrics.lpips_alex (torchmetrics alex, normalize=True) vs lpips alex
     with normalize=True
   - metrics.ssim_tm vs a float64 re-implementation of torchmetrics' SSIM
   Tolerances are float32 accumulation error over a ~1 MP image.
3. The same pair through the file path (two folders -> gs-metrics JSON).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import metrics
import oracle_ref


def tm_ssim_f64(a: torch.Tensor, b: torch.Tensor) -> float:
    """torchmetrics 1.9 SSIM (gaussian 11, sigma 1.5, reflect pad 5, variances clamped at 0,
    mean over the full map; older releases cropped the 5 px border) in float64."""
    a, b = a.double(), b.double()
    ch = a.shape[1]
    w = oracle_ref.gaussian_window(dtype=torch.float64, device=a.device)[None, None].expand(ch, 1, 11, 11)
    pad = lambda x: F.pad(x, (5, 5, 5, 5), mode="reflect")  # noqa: E731
    conv = lambda x: F.conv2d(pad(x), w, groups=ch)  # noqa: E731
    mu1, mu2 = conv(a), conv(b)
    s11, s22 = (conv(a * a) - mu1**2).clamp(min=0), (conv(b * b) - mu2**2).clamp(min=0)
    s12 = conv(a * b) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1**2 + mu2**2 + c1) * (s11 + s22 + c2))
    return float(m.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-image", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    dev = torch.device("cuda")
    m = metrics.Metrics(dev)
    rep: dict = {}
    ok = True

    def expect(name: str, got: float, want: float, tol: float) -> None:
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        rep[name] = {"got": got, "want": want, "abs_err": abs(got - want), "tol": tol, "ok": good}
        print(f"{'ok ' if good else 'BAD'} {name:<34} got {got:.10f}  want {want:.10f}  |err| {abs(got - want):.2e}")

    # 1. analytic pair
    rng = np.random.default_rng(0)
    gt8 = (rng.uniform(0, 245, (1, 3, 1, 1)) + rng.uniform(-40, 40, (1, 3, 64, 96))).clip(0, 245).astype(np.uint8)
    r8 = gt8 + 10
    g = torch.from_numpy(gt8).to(dev).float() / 255
    r = torch.from_numpy(r8).to(dev).float() / 255
    v = m(r, g)
    expect("psnr(+10 levels)", v["psnr"], 20 * math.log10(25.5), 1e-5)
    same = m(g, g)
    expect("ssim(identical)", same["ssim"], 1.0, 1e-6)
    expect("lpips(identical)", same["lpips"], 0.0, 1e-6)

    # 2. realistic pair
    if a.scene_image:
        with Image.open(a.scene_image) as im:
            gt_img = im.convert("RGB")
    else:
        y, x = np.mgrid[0:240, 0:320]
        base = np.stack([np.sin(x / 9.0) * np.cos(y / 13.0), np.cos(x / 17.0 + y / 23.0), np.sin((x + y) / 31.0)], -1)
        noise = rng.normal(scale=0.15, size=base.shape)
        gt_img = Image.fromarray(((base * 0.4 + 0.5 + noise).clip(0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    gt_img.save(buf, format="JPEG", quality=60)
    rd_img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    with tempfile.TemporaryDirectory() as td:
        gdir, rdir = Path(td, "gt"), Path(td, "renders")
        gdir.mkdir()
        rdir.mkdir()
        gt_img.save(gdir / "view0.png")
        rd_img.save(rdir / "view0.png")
        res = metrics.evaluate(rdir, gdir, quiet=True)
    g = torch.from_numpy(np.asarray(gt_img)).to(dev).permute(2, 0, 1)[None].float() / 255
    r = torch.from_numpy(np.asarray(rd_img)).to(dev).permute(2, 0, 1)[None].float() / 255
    rep["pair"] = {k: res[k] for k in ("psnr", "ssim", "lpips", "ssim_tm", "lpips_alex")} | {
        "size": [res["width"], res["height"]]
    }
    print(f"pair: {rep['pair']}")
    expect("file path psnr == in-memory", res["psnr"], metrics.psnr(r, g), 1e-12)

    ssim64 = float(oracle_ref.ssim_map(r.double(), g.double()).mean())
    expect("ssim vs float64 conv", res["ssim"], ssim64, 1e-5)
    from fused_ssim import fused_ssim

    expect("ssim vs fused-ssim same", res["ssim"], float(fused_ssim(r, g, padding="same", train=False)), 1e-5)
    expect("ssim_tm vs float64 torchmetrics", res["ssim_tm"], tm_ssim_f64(r, g), 1e-5)

    import lpips as lp

    with torch.no_grad():
        vgg = lp.LPIPS(net="vgg", version="0.1", verbose=False).to(dev).eval()
        alex = lp.LPIPS(net="alex", version="0.1", verbose=False).to(dev).eval()
        expect("lpips vs lpips pkg vgg [0,1]", res["lpips"], float(vgg(r, g, normalize=False)), 1e-5)
        expect("lpips_alex vs lpips pkg alex", res["lpips_alex"], float(alex(r, g, normalize=True)), 1e-5)
        differs = float(abs(vgg(r, g, normalize=True) - res["lpips"]) > 1e-3)
        expect("lpips_vgg [-1,1] differs (sanity)", differs, 1.0, 0.0)

    rep["ok"] = ok
    if a.json:
        a.json.write_text(json.dumps(rep, indent=2) + "\n")
    print("metrics check", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
