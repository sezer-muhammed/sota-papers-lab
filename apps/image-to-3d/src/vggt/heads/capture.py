#!/usr/bin/env python3
"""Reference dumps for the native VGGT-Omega heads (owner: heads).

Runs the official model (spl-vggt-ref image, package vggt_omega @ 48b23c8),
captures the aggregator layers the heads read and every head intermediate,
and writes:

  <out>/heads_n2.npz   2 frames: L_4/L_11/L_17/L_23 (fp32, [N,T,2048]),
                       camera + dense intermediates (fp16 bits as uint16, `<name>_f16`), pose_enc,
                       extrinsics, intrinsics, depth, depth_conf, world_points
                       (fp32), and the same heads run on bf16-rounded inputs
                       (`*_bf16in`, the noise floor of a bf16 layer cache)
  <out>/heads_n9.npz   9 frames (two dense chunks: 8 + 1): L_* as raw bf16
                       (uint16) plus the final outputs
  --weights FILE       (optional) camera_head.* and dense_head.* in fp32 safetensors,
                       effective qkv bias (bias * bias_mask), no mask buffers

Usage (inside the container):
  python -I capture.py --checkpoint CKPT --images DIR --out DIR --weights FILE

The checkpoint is FAIR non-commercial research licensed: dumps and weights
stay on the GPU host and are never committed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

LAYERS = (4, 11, 17, 23)
DEV = "cuda"


def bf16_bits(x: torch.Tensor) -> np.ndarray:
    return x.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)


def unproject(depth, K, E):
    # depth [N,H,W]; K [N,3,3]; E [N,3,4] (w2c). Pixel coordinates are integer
    # indices (no +0.5), as in tools/vggt/oracle.py.
    N, H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32),
                          torch.arange(W, device=depth.device, dtype=torch.float32), indexing="ij")
    fx, fy = K[:, 0, 0].view(N, 1, 1), K[:, 1, 1].view(N, 1, 1)
    cx, cy = K[:, 0, 2].view(N, 1, 1), K[:, 1, 2].view(N, 1, 1)
    cam = torch.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], -1)
    R, t = E[:, :, :3], E[:, :, 3]
    return torch.einsum("nji,nhwj->nhwi", R, cam - t[:, None, None, :])


def run_heads(model, layers, images, patch_start, tap: dict | None):
    """Official heads on the given layer list (fp32, autocast off)."""
    from vggt_omega.utils.pose_enc import encoding_to_camera

    hooks = []
    if tap is not None:
        ch, dh = model.camera_head, model.dense_head

        def save(name):
            def f(_m, _i, o):
                # fp16 stored as raw uint16 bits under `<name>_f16` (the Rust npz
                # reader has no <f2).
                tap[f"{name}_f16"] = o.detach().float().cpu().numpy().astype(np.float16).view(np.uint16)
            return f

        hooks.append(ch.token_norm.register_forward_hook(save("cam_token_norm")))
        for i, b in enumerate(ch.trunk):
            hooks.append(b.register_forward_hook(save(f"cam_block{i}")))
        hooks.append(ch.camera_branch.register_forward_hook(save("cam_raw")))
        for k in range(4):
            hooks.append(dh.resize_layers[k].register_forward_hook(save(f"dense_resize{k}")))
        for k, name in enumerate(["layer1_rn", "layer2_rn", "layer3_rn", "layer4_rn"]):
            hooks.append(getattr(dh.scratch, name).register_forward_hook(save(f"dense_rn{k}")))
        for k in range(1, 5):
            hooks.append(getattr(dh.scratch, f"refinenet{k}").register_forward_hook(save(f"dense_refine{k}")))
        hooks.append(dh.proj.register_forward_hook(save("dense_proj")))
        hooks.append(dh.proj_conf.register_forward_hook(save("dense_proj_conf")))
    full = [None] * 24
    for i, x in zip(LAYERS, layers):
        full[i] = x
    with torch.inference_mode(), torch.autocast(device_type=DEV, enabled=False):
        pose_enc = model.camera_head(full, patch_token_start=patch_start)
        depth, conf = model.dense_head(full, images=images, patch_token_start=patch_start)
    for h in hooks:
        h.remove()
    E, K = encoding_to_camera(pose_enc, images.shape[-2:])
    depth = depth[0, ..., 0].float()
    world = unproject(depth, K[0].float(), E[0].float())
    return {
        "pose_enc": pose_enc[0].float().cpu().numpy(),
        "extrinsics": E[0].float().cpu().numpy(),
        "intrinsics": K[0].float().cpu().numpy(),
        "depth": depth.cpu().numpy(),
        "depth_conf": conf[0].float().cpu().numpy(),
        "world_points": world.cpu().numpy(),
    }


def capture(model, paths, n, out: Path, full_dump: bool):
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    images = load_and_preprocess_images(paths[:n], mode="max_size", image_resolution=512, patch_size=16)
    images = images.to(DEV)[None]
    with torch.inference_mode(), torch.autocast(device_type=DEV, dtype=torch.bfloat16):
        tokens, patch_start = model.aggregator(images)
    layers = [tokens[i].float() for i in LAYERS]
    assert patch_start == 17
    data: dict[str, np.ndarray] = {"image_hw": np.array(images.shape[-2:], dtype=np.int32)}
    tap = {} if full_dump else None
    data.update(run_heads(model, layers, images, patch_start, tap))
    if full_dump:
        data.update(tap)
        for i, x in zip(LAYERS, layers):
            data[f"L{i}"] = x[0].cpu().numpy()
        rounded = [x.to(torch.bfloat16).float() for x in layers]
        for k, v in run_heads(model, rounded, images, patch_start, None).items():
            data[f"{k}_bf16in"] = v
    else:
        for i, x in zip(LAYERS, layers):
            data[f"L{i}_bf16"] = bf16_bits(x[0])
    path = out / f"heads_n{n}.npz"
    np.savez(path, **data)
    print(f"{path}: {sum(v.nbytes for v in data.values()) / 1e6:.1f} MB, image {tuple(images.shape[-2:])}")


def save_heads(state, path):
    heads = {}
    for k, v in state.items():
        if not (k.startswith("camera_head.") or k.startswith("dense_head.")) or k.endswith("bias_mask"):
            continue
        if k.endswith("attn.qkv.bias"):
            v = v * state[k + "_mask"]
        heads[k] = v.float().contiguous()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(heads, path)
    print(f"{path}: {len(heads)} tensors")



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default="cuda", help="cuda or cpu (aggregator in CPU bf16 autocast)")
    args = ap.parse_args()
    global DEV
    DEV = args.device
    torch.set_num_threads(16)

    from vggt_omega.models import VGGTOmega

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = VGGTOmega()
    model.load_state_dict(state, strict=True)
    model = model.eval().to(DEV)

    if args.weights:
        save_heads(state, args.weights)

    paths = sorted(str(p) for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".png"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    capture(model, paths, 2, out, True)
    capture(model, paths, 9, out, False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
