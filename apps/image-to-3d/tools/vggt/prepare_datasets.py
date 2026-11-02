#!/usr/bin/env python3
"""Prepare small evaluation subsets for VGGT-Ω reference evaluation.

Uses the official eval data preparation logic from /opt/vggt-omega/eval/ inside
the spl-vggt-ref container.

Usage (inside spl-vggt-ref container):
    python prepare_datasets.py --dataset eth3d --output ~/spl-work/datasets/vggt-eval/eth3d \
        --scenes courtyard delivery_area facade
    python prepare_datasets.py --dataset sintel --output ~/spl-work/datasets/vggt-eval/sintel \
        --scenes alley_1 bamboo_2 temple_2
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> int:
    eval_dir = Path("/opt/vggt-omega/eval")
    if not eval_dir.is_dir():
        print(f"ERROR: {eval_dir} not found. Run inside spl-vggt-ref container.", file=sys.stderr)
        return 1

    repo_root = eval_dir.parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(eval_dir))

    spec = importlib.util.spec_from_file_location("prepare", eval_dir / "prepare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main()


if __name__ == "__main__":
    raise SystemExit(main())
