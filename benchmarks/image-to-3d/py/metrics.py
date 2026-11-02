#!/usr/bin/env python3
"""Novel-view metrics for two image folders, as 3DGS and gsplat compute them.

  gs-metrics RENDERS GT [--test-every 8] [--out metrics.json] [--quiet]

Images are paired by file stem (render `00008.png` <-> GT `00008.JPG`). With
--test-every K, GT is a full scene image folder and only its sorted
index % K == 0 images are used (the 3DGS / gsplat hold-out). Every pair must
have the same size; nothing is resized.

Images load like 3DGS metrics.py: PIL, RGB channels of RGBA without
compositing, float32 / 255. Per-image values are averaged over images.

  psnr        20 log10(1 / sqrt(mse)) per image (3DGS image_utils.psnr; same as
              torchmetrics PeakSignalNoiseRatio(data_range=1) used by gsplat)
  ssim        3DGS utils/loss_utils.ssim: 11x11 Gaussian window, sigma 1.5,
              zero padding ("same"), C1 = 0.01^2, C2 = 0.03^2, mean of the map
  lpips       LPIPS-VGG v0.1 as 3DGS computes it: lpipsPyTorch on [0, 1]
              inputs, i.e. torchmetrics LPIPS(net_type="vgg", normalize=False)
  ssim_tm     gsplat simple_trainer: torchmetrics SSIM(data_range=1); with
              torchmetrics 1.9 that is reflect padding and the mean over the
              full map (older torchmetrics cropped the 5 px border)
  lpips_alex  gsplat simple_trainer default: LPIPS(net_type="alex", normalize=True)

The headline triple for the protocol is psnr / ssim / lpips (the 3DGS
convention, which Faster-GS, FastGS and VGGT-X report).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in EXTS)


def load(p: Path, dev: torch.device) -> torch.Tensor:
    """[1, 3, H, W] float32 in [0, 1], like torchvision to_tensor(PIL)[:, :3]."""
    with Image.open(p) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        a = np.array(im)[..., :3]
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev).permute(2, 0, 1)[None].float() / 255.0


def _window(dev: torch.device, ch: int = 3, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    g = torch.tensor([math.exp(-((x - size // 2) ** 2) / float(2 * sigma**2)) for x in range(size)])
    g = (g / g.sum()).unsqueeze(1)
    w2 = g.mm(g.t()).float()[None, None]
    return w2.expand(ch, 1, size, size).contiguous().to(dev)


def ssim_3dgs(a: torch.Tensor, b: torch.Tensor, window: torch.Tensor) -> float:
    ch = a.shape[1]
    mu1 = F.conv2d(a, window, padding=5, groups=ch)
    mu2 = F.conv2d(b, window, padding=5, groups=ch)
    mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.conv2d(a * a, window, padding=5, groups=ch) - mu1_sq
    s2 = F.conv2d(b * b, window, padding=5, groups=ch) - mu2_sq
    s12 = F.conv2d(a * b, window, padding=5, groups=ch) - mu12
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return float(m.mean())


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a.double() - b.double()) ** 2).mean())
    return float("inf") if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))


class Metrics:
    def __init__(self, dev: torch.device):
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

        self.dev = dev
        self.window = _window(dev)
        self.lpips_vgg = LPIPS(net_type="vgg", normalize=False).to(dev).eval()
        self.lpips_alex = LPIPS(net_type="alex", normalize=True).to(dev).eval()
        self.ssim_tm = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)

    @torch.no_grad()
    def __call__(self, r: torch.Tensor, g: torch.Tensor) -> dict[str, float]:
        out = {
            "psnr": psnr(r, g),
            "ssim": ssim_3dgs(r, g, self.window),
            "lpips": float(self.lpips_vgg(r, g)),
            "ssim_tm": float(self.ssim_tm(r, g)),
            "lpips_alex": float(self.lpips_alex(r, g)),
        }
        self.ssim_tm.reset()
        self.lpips_vgg.reset()
        self.lpips_alex.reset()
        return out


def pair(renders: Path, gt: Path, test_every: int | None) -> list[tuple[Path, Path]]:
    gts = list_images(gt)
    if test_every:
        gts = [p for i, p in enumerate(gts) if i % test_every == 0]
    rs = {p.stem: p for p in list_images(renders)}
    missing = [g.name for g in gts if g.stem not in rs]
    if missing:
        raise SystemExit(f"{len(missing)} GT image(s) have no render in {renders}: {missing[:5]}")
    extra = sorted(set(rs) - {g.stem for g in gts})
    if extra and not test_every:
        raise SystemExit(f"{len(extra)} render(s) have no GT in {gt}: {extra[:5]}")
    return [(rs[g.stem], g) for g in gts]


def evaluate(renders: Path, gt: Path, test_every: int | None = None, quiet: bool = False) -> dict:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = pair(renders, gt, test_every)
    if not pairs:
        raise SystemExit("no image pairs")
    m = Metrics(dev)
    t0 = time.time()
    per = []
    for rp, gp in pairs:
        r, g = load(rp, dev), load(gp, dev)
        if r.shape != g.shape:
            raise SystemExit(f"size mismatch: {rp.name} {tuple(r.shape[2:])} vs {gp.name} {tuple(g.shape[2:])}")
        v = m(r, g)
        per.append({"name": gp.stem, **v})
        if not quiet:
            print(f"{gp.stem:<24} psnr {v['psnr']:7.3f}  ssim {v['ssim']:.4f}  lpips {v['lpips']:.4f}", file=sys.stderr)
    keys = ["psnr", "ssim", "lpips", "ssim_tm", "lpips_alex"]
    mean = {k: float(np.mean([p[k] for p in per])) for k in keys}
    import torchmetrics

    return {
        **mean,
        "n": len(per),
        "width": int(r.shape[3]),
        "height": int(r.shape[2]),
        "renders": str(renders),
        "gt": str(gt),
        "test_every": test_every,
        "seconds": round(time.time() - t0, 2),
        "versions": {"torch": torch.__version__, "torchmetrics": torchmetrics.__version__, "pillow": Image.__version__},
        "per_image": per,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("renders", type=Path)
    ap.add_argument("gt", type=Path)
    ap.add_argument("--test-every", type=int, default=None, help="GT is a full image folder; use index %% K == 0")
    ap.add_argument("--out", type=Path, help="write the JSON here (default: stdout)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    res = evaluate(a.renders, a.gt, a.test_every, a.quiet)
    text = json.dumps(res, indent=2) + "\n"
    if a.out:
        a.out.write_text(text)
    else:
        sys.stdout.write(text)
    print(
        f"n={res['n']}  PSNR {res['psnr']:.3f}  SSIM {res['ssim']:.4f}  LPIPS {res['lpips']:.4f}  "
        f"(gsplat style: SSIM {res['ssim_tm']:.4f}, LPIPS-alex {res['lpips_alex']:.4f})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
