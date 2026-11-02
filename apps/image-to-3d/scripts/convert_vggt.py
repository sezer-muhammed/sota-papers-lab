#!/usr/bin/env python3
"""Converts the official VGGT-Omega checkpoint into the native engine's weights.

    python -I convert_vggt.py --input vggt_omega_1b_512.pt \
        --output vggt_omega_1b_512.safetensors [--fp8 vggt_omega_1b_512_fp8.safetensors]

Runs in the PyTorch image (spl-vggt-ref or pytorch/pytorch). The weights are
under the FAIR non-commercial research licence: research use only, never
redistribute the output or bake it into an image.

Layout (spec-vggt.md section 9, read by apps/image-to-3d/src/vggt/weights.rs):

- `aggregator.*` first: the device region, uploaded to the GPU in one copy.
  2-D weights (Linear; the patch conv reshaped to [1024, 768] in (c, ky, kx)
  order) and their biases in BF16; LayerNorm parameters, LayerScale gammas,
  tokens and RoPE periods in F32.
- `camera_head.*`, `dense_head.*` after it, all F32: the heads read them on
  the host and upload their own layouts.
- Every tensor starts at a 256-byte offset of the data section; gaps are
  filled with `__pad.<n>` U8 tensors so the file stays standard safetensors.
- `qkv.bias_mask` buffers are folded in: the effective bias is
  `bias * bias_mask`. All aggregator masks are zero, so those biases are
  dropped; the camera head keeps [b_q, 0, b_v].
- Dropped: `mask_token`, `text_alignment_head.*`, every `bias_mask`.
- `__metadata__`: `format = spl-vggt-1`, `device_bytes` (length of the device
  region from the start of the data section).

`--fp8` writes a second file with FP8 E4M3 copies of the aggregator Linear
weights (`<name>` F8_E4M3, per-output-channel `<name>_scale` F32 = amax / 448).
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

ALIGN = 256
DEVICE_PREFIX = "aggregator."
HOST_PREFIXES = ("camera_head.", "dense_head.")


def effective(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, t in state.items():
        if name.endswith(".bias_mask") or name.endswith("mask_token"):
            continue
        if name.startswith("text_alignment_head."):
            continue
        if not (name.startswith(DEVICE_PREFIX) or name.startswith(HOST_PREFIXES)):
            raise SystemExit(f"unexpected tensor {name}")
        t = t.detach().float()
        mask = state.get(name + "_mask") if name.endswith(".qkv.bias") else None
        if mask is not None:
            t = t * mask.float()
            if name.startswith(DEVICE_PREFIX) and not torch.any(t != 0):
                continue
        if name == "aggregator.patch_embed.patch_embed.proj.weight":
            t = t.reshape(t.shape[0], -1)
        out[name] = t.contiguous()
    return out


def device_dtype(name: str, t: torch.Tensor, all_: dict[str, torch.Tensor]) -> str:
    if not name.startswith(DEVICE_PREFIX):
        return "F32"
    if name.endswith(".weight") and t.dim() == 2:
        return "BF16"
    if name.endswith(".bias"):
        w = all_.get(name[: -len(".bias")] + ".weight")
        if w is not None and w.dim() == 2:
            return "BF16"
    return "F32"


def to_bytes(t: torch.Tensor, dtype: str) -> bytes:
    if dtype == "BF16":
        return t.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
    if dtype == "F32":
        return t.float().numpy().astype("<f4").tobytes()
    if dtype == "F8_E4M3":
        return t.view(torch.uint8).numpy().tobytes()
    raise ValueError(dtype)


def write(path: Path, entries: list[tuple[str, str, list[int], bytes]], meta: dict[str, str]) -> None:
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    off = 0
    pad = 0
    for name, dtype, shape, data in entries:
        gap = (-off) % ALIGN
        if gap:
            header[f"__pad.{pad}"] = {"dtype": "U8", "shape": [gap], "data_offsets": [off, off + gap]}
            blobs.append(b"\0" * gap)
            off += gap
            pad += 1
        if name == "__device_end__":
            meta["device_bytes"] = str(off)
            continue
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    header["__metadata__"] = meta
    h = json.dumps(header, separators=(",", ":")).encode()
    # Pad the header so the data section starts 8-aligned in the file.
    h += b" " * ((-(8 + len(h))) % 8)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    tmp.rename(path)
    print(f"wrote {path}: {len(entries)} tensors, {off / 1e9:.3f} GB")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fp8", default=None, help="also write FP8 E4M3 Linear weights here")
    a = p.parse_args()

    state = torch.load(a.input, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(state, dict) and ("model" in state or "state_dict" in state):
        state = state.get("model", state.get("state_dict"))
    eff = effective(state)
    dev = [(n, t) for n, t in eff.items() if n.startswith(DEVICE_PREFIX)]
    host = [(n, t) for n, t in eff.items() if not n.startswith(DEVICE_PREFIX)]
    entries = []
    for n, t in dev:
        d = device_dtype(n, t, eff)
        entries.append((n, d, list(t.shape), to_bytes(t, d)))
    entries.append(("__device_end__", "", [], b""))
    for n, t in host:
        entries.append((n, "F32", list(t.shape), to_bytes(t, "F32")))
    write(Path(a.output), entries, {"format": "spl-vggt-1", "source": Path(a.input).name})

    if a.fp8:
        fp8 = []
        for n, t in dev:
            if not (n.endswith(".weight") and t.dim() == 2) or "patch_embed.patch_embed" in n:
                continue
            scale = (t.abs().amax(dim=1) / 448.0).clamp(min=1e-12)
            q = (t / scale[:, None]).to(torch.float8_e4m3fn)
            fp8.append((n, "F8_E4M3", list(t.shape), to_bytes(q, "F8_E4M3")))
            fp8.append((n + "_scale", "F32", [t.shape[0]], to_bytes(scale, "F32")))
        fp8.append(("__device_end__", "", [], b""))
        write(Path(a.fp8), fp8, {"format": "spl-vggt-1-fp8", "source": Path(a.input).name})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
