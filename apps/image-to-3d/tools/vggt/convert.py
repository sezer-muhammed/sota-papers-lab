#!/usr/bin/env python3
"""Convert VGGT-Ω .pt checkpoint to safetensors with bf16 master weights and FP8 E4M3 copies.

Usage:
    python convert.py --input vggt_omega_1b_512.pt --output-dir ./converted/

Produces:
    weights.safetensors       bf16 master weights (all parameters)
    weights_fp8.safetensors   FP8 E4M3 copies of every Linear weight, per-output-channel scaled
    manifest.json             name, shape, dtype, scale info for every tensor
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch


def _to_fp8_e4m3_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise a 2-D weight to FP8 E4M3 with one scale per output channel (row)."""
    assert weight.ndim == 2, f"expected 2-D weight, got {weight.shape}"
    fp32 = weight.float()
    amax = fp32.abs().amax(dim=1).clamp(min=1e-12)
    fp8_max = torch.tensor(448.0, dtype=torch.float32, device=weight.device)
    scale = amax / fp8_max
    scaled = fp32 / scale[:, None]
    quantised = scaled.to(torch.float8_e4m3fn)
    return quantised, scale


def convert(input_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} ...")
    state = torch.load(input_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))

    bf16_tensors: dict[str, torch.Tensor] = {}
    fp8_tensors: dict[str, torch.Tensor] = {}
    fp8_scales: dict[str, torch.Tensor] = {}
    manifest: dict[str, dict] = {}

    for name in sorted(state.keys()):
        param = state[name]
        bf16 = param.to(torch.bfloat16)
        bf16_tensors[name] = bf16
        entry: dict = {
            "shape": list(param.shape),
            "dtype": "bfloat16",
            "numel": param.numel(),
        }

        if param.ndim == 2 and "weight" in name:
            q, s = _to_fp8_e4m3_per_channel(param)
            fp8_name = name
            scale_name = name.replace(".weight", ".weight_scale")
            fp8_tensors[fp8_name] = q
            fp8_scales[scale_name] = s
            entry["fp8_e4m3"] = {
                "tensor": fp8_name,
                "scale_tensor": scale_name,
                "scale_shape": list(s.shape),
                "fp8_dtype": "float8_e4m3fn",
            }

        manifest[name] = entry

    try:
        from safetensors.torch import save_file
    except ImportError:
        print("safetensors not installed; saving as .pt instead", file=sys.stderr)
        torch.save(bf16_tensors, output_dir / "weights.pt")
        torch.save({**fp8_tensors, **fp8_scales}, output_dir / "weights_fp8.pt")
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    bf16_path = output_dir / "weights.safetensors"
    print(f"Saving bf16 weights to {bf16_path} ({len(bf16_tensors)} tensors) ...")
    save_file(bf16_tensors, str(bf16_path))

    fp8_all = {**fp8_tensors, **fp8_scales}
    fp8_path = output_dir / "weights_fp8.safetensors"
    print(f"Saving FP8 weights to {fp8_path} ({len(fp8_all)} tensors) ...")
    save_file(fp8_all, str(fp8_path))

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest at {manifest_path} ({len(manifest)} entries)")

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to vggt_omega_1b_512.pt")
    parser.add_argument("--output-dir", required=True, help="Directory for converted files")
    args = parser.parse_args()
    convert(Path(args.input), Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
