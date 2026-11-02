#!/usr/bin/env python3
"""Captures real global-attention Q/K/V for the sage benchmark (nn-sage).

VGGT-Ω weights are gated, so this runs the public VGGT-1B aggregator (same
family: alternating frame/global attention, QK LayerNorm, ViT-L heads of 64)
on video frames and records the inputs of chosen global attention layers
(after QK-norm and RoPE) in the run-directory format `torch_ref.py` and the
Rust `gpu_bench` read. At 448×294 a VGGT-1B frame is 5 special tokens plus a
21×32 patch grid, the grid VGGT-Ω has at 512×336.

    vggt_capture.py --frames-dir DIR --out OUT --layers 2,11,20 [--frames 200]

Needs the `vggt` package on PYTHONPATH (github.com/facebookresearch/vggt;
code and weights are for non-commercial research) and writes one
OUT/layerNN/ run directory per layer.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

W, H = 448, 294  # 32 × 21 patches of 14 px


class Done(Exception):
    pass


def load_frames(d, n):
    paths = sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))
    if len(paths) < n:
        raise SystemExit(f"{d} has {len(paths)} frames, need {n}")
    idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
    imgs = []
    for i in idx:
        im = Image.open(paths[i]).convert("RGB")
        # Center-crop to the 448:294 aspect, then resize.
        w, h = im.size
        target = W / H
        if w / h > target:
            nw = round(h * target)
            im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
        else:
            nh = round(w / target)
            im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
        im = im.resize((W, H), Image.BICUBIC)
        imgs.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0)
    return torch.stack(imgs)


def save_bf16(path, t):
    t.detach().to(torch.bfloat16).contiguous().cpu().view(torch.int16).numpy().tofile(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--layers", default="2,11,20")
    args = p.parse_args()
    layers = sorted(int(x) for x in args.layers.split(","))

    from vggt.models.vggt import VGGT
    import vggt.layers.attention as attn_mod

    model = VGGT.from_pretrained("facebook/VGGT-1B").eval()
    agg = model.aggregator.to("cuda", torch.bfloat16)
    del model
    torch.cuda.empty_cache()

    images = load_frames(args.frames_dir, args.frames).to("cuda", torch.bfloat16)[None]
    tokens_per_frame = 5 + (H // 14) * (W // 14)
    seen = {"global": 0}
    real_sdpa = F.scaled_dot_product_attention

    def sdpa(q, k, v, *a, **kw):
        if q.shape[2] > tokens_per_frame:  # a global layer: every frame in one sequence
            i = seen["global"]
            seen["global"] += 1
            if i in layers:
                d = os.path.join(args.out, f"layer{i:02d}")
                os.makedirs(d, exist_ok=True)
                for name, t in (("q", q), ("k", k), ("v", v)):
                    save_bf16(os.path.join(d, f"{name}.bin"), t[0].permute(1, 0, 2))
                with open(os.path.join(d, "meta.txt"), "w") as f:
                    f.write(f"frames={args.frames}\nheads={q.shape[1]}\nhead_dim={q.shape[3]}\n"
                             f"special=5\ngrid_h={H // 14}\ngrid_w={W // 14}\n"
                             f"source=vggt-1b global layer {i}, {args.frames_dir}\n")
                print(f"captured global layer {i}: q {tuple(q.shape)}", flush=True)
                if i == layers[-1]:
                    raise Done()
        return real_sdpa(q, k, v, *a, **kw)

    attn_mod.F.scaled_dot_product_attention = sdpa
    try:
        with torch.no_grad():
            agg(images)
    except Done:
        pass
    print("peak GiB", torch.cuda.max_memory_allocated() / 2**30)


if __name__ == "__main__":
    sys.exit(main())
