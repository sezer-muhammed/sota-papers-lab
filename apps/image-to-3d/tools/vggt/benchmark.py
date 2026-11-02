#!/usr/bin/env python3
"""Benchmark VGGT-Ω official inference: time and VRAM at various frame counts.

Usage:
    python benchmark.py --checkpoint ~/spl-work/models/vggt-omega/vggt_omega_1b_512.pt

Runs N = 2, 10, 50, 100, 200 frames (512 px synthetic inputs) in bf16 and fp32,
recording wall time, peak VRAM, and OOM status.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch


FRAME_COUNTS = [2, 10, 50, 100, 200]


def _make_dummy_images(n: int, resolution: int = 512) -> torch.Tensor:
    """Synthetic images: (1, N, 3, H, W) in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(1, n, 3, resolution, resolution)


def _run_one(checkpoint: str, n: int, dtype_str: str, device: str) -> dict:
    from vggt_omega.models import VGGTOmega

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    autocast = dtype_str != "fp32"
    model = VGGTOmega(autocast=autocast)
    state = torch.load(Path(checkpoint).expanduser().resolve(), map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)

    images = _make_dummy_images(n).to(device)

    # Warmup
    with torch.inference_mode():
        if n <= 10:
            _ = model(images[:, :2])
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = model(images)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    del model, images, state
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "n_frames": n,
        "dtype": dtype_str,
        "elapsed_s": round(elapsed, 3),
        "peak_vram_mb": round(peak_mb, 1),
        "oom": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    results = []
    for dtype_str in ["bf16", "fp32"]:
        for n in FRAME_COUNTS:
            print(f"\n--- N={n}, dtype={dtype_str} ---")
            try:
                r = _run_one(args.checkpoint, n, dtype_str, args.device)
                print(f"  time={r['elapsed_s']:.3f}s  peak_vram={r['peak_vram_mb']:.0f}MB")
                results.append(r)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM!")
                gc.collect()
                torch.cuda.empty_cache()
                results.append({
                    "n_frames": n,
                    "dtype": dtype_str,
                    "elapsed_s": None,
                    "peak_vram_mb": None,
                    "oom": True,
                })
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                results.append({
                    "n_frames": n,
                    "dtype": dtype_str,
                    "elapsed_s": None,
                    "peak_vram_mb": None,
                    "oom": False,
                    "error": str(e),
                })

    # GPU info
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    summary = {
        "gpu": gpu_name,
        "gpu_memory_gb": round(gpu_mem, 1),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "results": results,
    }

    out_path = Path(args.output) if args.output else Path("benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nResults written to {out_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
