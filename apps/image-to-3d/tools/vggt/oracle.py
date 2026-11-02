#!/usr/bin/env python3
"""VGGT-Ω oracle: run the official model and dump inputs, intermediates and outputs.

Usage:
    vggt-oracle img1.png img2.png --out result.npz
    vggt-oracle img1.png img2.png --out result.npz --dump-layers 0,4,11,17,23
    vggt-oracle img1.png img2.png --out result.npz --dtype bf16

Saves an .npz with:
    preprocessed    (N, 3, H, W)  float32   preprocessed input images
    pose_enc        (N, 9)        float32   raw pose encoding
    extrinsics      (N, 3, 4)     float64   camera-from-world
    intrinsics      (N, 3, 3)     float64   camera intrinsics
    depth           (N, H, W)     float32   predicted depth (single channel)
    depth_conf      (N, H, W)     float32   depth confidence
    world_points    (N, H, W, 3)  float32   unprojected world-space points
    block_XX        (N, T, D)     float32   intermediate block outputs (if --dump-layers)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


def _load_model(checkpoint: str, device: str, dtype: str):
    from vggt_omega.models import VGGTOmega

    state = torch.load(Path(checkpoint).expanduser().resolve(), map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    autocast = dtype != "fp32"
    model = VGGTOmega(autocast=autocast)
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)
    return model


def _unproject(depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> np.ndarray:
    """Unproject depth to world points. depth: (B,N,H,W), intrinsics: (B,N,3,3), extrinsics: (B,N,3,4)."""
    B, N, H, W = depth.shape
    device = depth.device
    v, u = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                          torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    fx = intrinsics[:, :, 0, 0].view(B, N, 1, 1)
    fy = intrinsics[:, :, 1, 1].view(B, N, 1, 1)
    cx = intrinsics[:, :, 0, 2].view(B, N, 1, 1)
    cy = intrinsics[:, :, 1, 2].view(B, N, 1, 1)
    x = (u.unsqueeze(0).unsqueeze(0) - cx) / fx * depth
    y = (v.unsqueeze(0).unsqueeze(0) - cy) / fy * depth
    z = depth
    cam_pts = torch.stack([x, y, z], dim=-1)  # (B,N,H,W,3)
    R = extrinsics[:, :, :3, :3]  # (B,N,3,3)
    t = extrinsics[:, :, :3, 3]   # (B,N,3)
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.einsum("bnij,bnj->bni", R_inv, t)
    world = torch.einsum("bnij,bnhwj->bnhwi", R_inv, cam_pts) + t_inv[:, :, None, None, :]
    return world[0].float().cpu().numpy()


def run_oracle(args: argparse.Namespace) -> int:
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device = args.device
    model = _load_model(args.checkpoint, device, args.dtype)

    images = load_and_preprocess_images(
        args.images,
        mode="max_size",
        image_resolution=512,
        patch_size=16,
    )
    preprocessed = images.numpy().copy()  # (N,3,H,W)

    images = images.to(device)

    dump_layers = set()
    if args.dump_layers:
        dump_layers = set(int(x) for x in args.dump_layers.split(","))

    block_outputs: dict[int, np.ndarray] = {}

    if dump_layers:
        hooks = []

        def _make_hook(block_idx):
            def hook_fn(module, input, output):
                block_outputs[block_idx] = output.detach().float().cpu().numpy()
            return hook_fn

        agg = model.aggregator
        for idx in dump_layers:
            if idx < len(agg.frame_blocks):
                hooks.append(agg.frame_blocks[idx].register_forward_hook(_make_hook(idx)))

    t0 = time.time()
    with torch.inference_mode():
        prediction = model(images)
    elapsed = time.time() - t0

    if dump_layers:
        for h in hooks:
            h.remove()

    image_hw = prediction["images"].shape[-2:]
    pose_enc = prediction["pose_enc"].detach().float().cpu().numpy()[0]
    extrinsics_t, intrinsics_t = encoding_to_camera(prediction["pose_enc"], image_hw)
    extrinsics = extrinsics_t.detach().double().cpu().numpy()[0]
    intrinsics = intrinsics_t.detach().double().cpu().numpy()[0]
    depth = prediction["depth"].detach().float().cpu().numpy()[0]
    if depth.ndim == 4:
        depth = depth[..., 0]
    depth_conf = prediction["depth_conf"].detach().float().cpu().numpy()[0]
    if depth_conf.ndim == 4:
        depth_conf = depth_conf[..., 0]

    depth_t = prediction["depth"].float()
    if depth_t.ndim == 5:
        depth_t = depth_t[..., 0]
    world_points = _unproject(depth_t, intrinsics_t.float(), extrinsics_t.float())

    output: dict[str, np.ndarray] = {
        "preprocessed": preprocessed.astype(np.float32),
        "pose_enc": pose_enc.astype(np.float32),
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "depth": depth.astype(np.float32),
        "depth_conf": depth_conf.astype(np.float32),
        "world_points": world_points.astype(np.float32),
    }

    for idx, arr in sorted(block_outputs.items()):
        output[f"block_{idx:02d}"] = arr.astype(np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), **output)
    sizes = {k: v.shape for k, v in output.items()}
    total_bytes = sum(v.nbytes for v in output.values())
    print(f"Saved {out_path} ({total_bytes / 1e6:.1f} MB, {elapsed:.2f}s inference)")
    for k, s in sizes.items():
        print(f"  {k}: {s}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="Input image paths")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--checkpoint", default="~/spl-work/models/vggt-omega/vggt_omega_1b_512.pt",
                        help="Path to .pt checkpoint")
    parser.add_argument("--dump-layers", default=None,
                        help="Comma-separated block indices to dump (e.g. 0,4,11,17,23)")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16",
                        help="Inference precision (default: bf16)")
    parser.add_argument("--device", default="cuda")
    return run_oracle(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
