#!/usr/bin/env python3
"""Trunk oracle for the native VGGT-Ω engine (src/vggt/trunk, spec §8).

Runs the official aggregator (spl-vggt-ref image) and dumps, as fp32 .npy in
--out:
  images.npy             [N,3,H,W] normalised input (what the trunk reads)
  input.npy              [N,T,1024] tokens entering frame block 0
  frame_XX.npy / inter_XX.npy   [N,T,1024] stream after each listed layer
  pose_enc.npy           [N,9] (sanity)
--small also writes the fp32 side file the Rust parity tests load next to the
bf16 weights: every aggregator tensor of ≤ 1 dim or a token parameter
(LN weights, gammas, tokens, periods) in fp32, and the effective QKV biases
(bias ⊙ bias_mask) only where they are non-zero.

  python -I trunk_oracle.py --ckpt ... --out DIR --dtype bf16 --layers 0,1,2,4,23 img...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--layers", default="0,1,2,3,4,11,17,23")
    ap.add_argument("--small", default=None, help="write the fp32 side file here")
    a = ap.parse_args()

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    state = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))

    if a.small:
        from safetensors.torch import save_file

        small = {}
        for k, v in state.items():
            if not k.startswith("aggregator."):
                continue
            if k.endswith("qkv.bias"):
                eff = v * state[k.replace(".bias", ".bias_mask")]
                if eff.abs().max() > 0:
                    small[k] = eff.float().contiguous()
                continue
            if k.endswith("bias_mask"):
                continue
            if v.ndim <= 1 or "token" in k:
                small[k] = v.float().contiguous()
        save_file(small, a.small)
        print(f"wrote {len(small)} fp32 tensors to {a.small}")

    model = VGGTOmega(autocast=a.dtype != "fp32")
    model.load_state_dict(state, strict=True)
    del state
    model = model.eval().cuda()
    agg = model.aggregator
    layers = {int(x) for x in a.layers.split(",") if x}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    images = load_and_preprocess_images(a.images, mode="max_size", image_resolution=512, patch_size=16)
    n = images.shape[0]
    norm = (images - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)
    np.save(out / "images.npy", norm.float().numpy())

    dumps: dict[str, np.ndarray] = {}
    count = {"frame": 0, "inter": 0}

    def save(name, t):
        dumps[name] = t.detach().float().reshape(n, -1, t.shape[-1]).cpu().numpy()

    run_frame = agg._run_frame_block
    run_inter = agg._run_inter_frame_attention_block

    def frame_hook(tokens, *args):
        i = count["frame"]
        count["frame"] += 1
        if i == 0:
            save("input", tokens)
        res = run_frame(tokens, *args)
        if i in layers:
            save(f"frame_{i:02d}", res[0])
        return res

    def inter_hook(tokens, *args):
        i = count["inter"]
        count["inter"] += 1
        res = run_inter(tokens, *args)
        if i in layers:
            save(f"inter_{i:02d}", res)
        return res

    agg._run_frame_block = frame_hook
    agg._run_inter_frame_attention_block = inter_hook
    with torch.inference_mode():
        pred = model(images.cuda())
    dumps["pose_enc"] = pred["pose_enc"][0].float().cpu().numpy()
    for k, v in dumps.items():
        np.save(out / f"{k}.npy", v.astype(np.float32))
    print(f"N={n} images {tuple(images.shape)}; dumped {sorted(dumps)} to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
