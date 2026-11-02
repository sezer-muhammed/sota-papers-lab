#!/usr/bin/env python3
"""Parity dumps of the official VGGT-Omega model for the native engine's tests.

    python -I vggt_oracle.py --checkpoint vggt_omega_1b_512.pt --out DIR \
        --set NAME img1 img2 ... [--set NAME2 ...] [--layers] [--fp32] [--time]

Per set it writes `DIR/NAME/{bf16,fp32}/*.npy` (fp32 only with --fp32):

- `input.npy`      [N,3,H,W] f32, the preprocessed images in [0, 1] (max_size, 512)
- `dino.npy`       [N,P,1024] DINO output (patch tokens after the final LN)
- `F_i.npy`, `G_i.npy` [N,T,1024] frame / inter-frame block outputs, i in
                   {0, 2, 4, 23} (only with --layers)
- `pose_enc.npy`   [N,9], `depth.npy` / `depth_conf.npy` [N,H,W]
- `time.json`      median of 10 timed forwards after warm-up (with --time)

Layer dumps are f32 for N <= 2 and f16 otherwise. Research use only (FAIR
non-commercial licence); the dumps stay on the GPU host.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

LAYERS = (0, 2, 4, 23)


def load(ckpt: str, autocast: bool):
    from vggt_omega.models import VGGTOmega

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and ("model" in state or "state_dict" in state):
        state = state.get("model", state.get("state_dict"))
    m = VGGTOmega(autocast=autocast) if "autocast" in VGGTOmega.__init__.__code__.co_varnames else VGGTOmega()
    m.load_state_dict(state, strict=True)
    return m.eval().cuda()


def run_set(model, name: str, paths: list[str], out: Path, layers: bool, timing: bool) -> None:
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    images = load_and_preprocess_images(paths, mode="max_size", image_resolution=512, patch_size=16)
    n = images.shape[0]
    out.mkdir(parents=True, exist_ok=True)
    small = n <= 2
    dump: dict[str, np.ndarray] = {}

    def keep(key: str, t: torch.Tensor) -> None:
        a = t.detach().float().cpu().numpy()
        dump[key] = a if small else a.astype(np.float16)

    agg = model.aggregator
    hooks = []
    hooks.append(agg.patch_embed.register_forward_hook(
        lambda m, i, o: keep("dino", o["x_norm_patchtokens"] if isinstance(o, dict) else o)))
    if layers:
        for i in LAYERS:
            hooks.append(agg.frame_blocks[i].register_forward_hook(
                lambda m, inp, o, i=i: keep(f"F_{i}", o.reshape(n, -1, o.shape[-1]))))
            if i + 1 < len(agg.frame_blocks):
                hooks.append(agg.frame_blocks[i + 1].register_forward_pre_hook(
                    lambda m, inp, i=i: keep(f"G_{i}", inp[0].reshape(n, -1, inp[0].shape[-1]))))
        orig = agg.forward

        def wrapped(*a, **k):
            outs, start = orig(*a, **k)
            last = outs[23]
            keep("G_23", last[..., last.shape[-1] // 2:].reshape(n, -1, last.shape[-1] // 2))
            return outs, start

        agg.forward = wrapped
    x = images.cuda()
    with torch.inference_mode():
        pred = model(x[None] if x.dim() == 4 else x)
    for h in hooks:
        h.remove()
    if layers:
        agg.forward = orig
    np.save(out / "input.npy", images.numpy().astype(np.float32))
    for k, v in dump.items():
        np.save(out / f"{k}.npy", v)
    np.save(out / "pose_enc.npy", pred["pose_enc"][0].float().cpu().numpy())
    d = pred["depth"][0].float()
    np.save(out / "depth.npy", (d[..., 0] if d.dim() == 4 else d).cpu().numpy())
    np.save(out / "depth_conf.npy", pred["depth_conf"][0].float().cpu().numpy())
    if timing:
        ts = []
        with torch.inference_mode():
            for _ in range(3):
                model(x[None])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(10):
                t0 = time.perf_counter()
                model(x[None])
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        (out / "time.json").write_text(json.dumps({
            "frames": n, "hw": list(images.shape[-2:]), "median_s": float(np.median(ts)),
            "min_s": float(min(ts)), "peak_mib": torch.cuda.max_memory_allocated() / 2**20}))
    print(name, out, list(images.shape), {k: list(v.shape) for k, v in dump.items()})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--set", nargs="+", action="append", required=True, metavar=("NAME", "IMAGE"))
    p.add_argument("--layers", action="store_true")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--time", action="store_true")
    a = p.parse_args()
    for precision in ["bf16"] + (["fp32"] if a.fp32 else []):
        model = load(a.checkpoint, autocast=precision == "bf16")
        for s in a.set:
            run_set(model, s[0], s[1:], Path(a.out) / s[0] / precision, a.layers, a.time and precision == "bf16")
        # The forward patch makes a reference cycle: collect it, or the
        # bf16 model stays on the device while the fp32 one loads.
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
