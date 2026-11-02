#!/usr/bin/env bash
# Fetch Pixal3D-mv weights and convert fp32 DiTs to bf16.
# Run inside the spl-pixal3d container (or any env with torch+safetensors).
#
# Usage: fetch_weights.sh <output_dir> [--profile geo|full]
#
# Downloads one file at a time to keep peak transient under 6 GB.
set -euo pipefail

OUT="${1:?usage: fetch_weights.sh <output_dir> [--profile geo|full]}"
PROFILE="${3:-full}"
mkdir -p "$OUT"

python3 - "$OUT" "$PROFILE" <<'PYEOF'
import sys, os, json
from pathlib import Path

out = Path(sys.argv[1])
profile = sys.argv[2] if len(sys.argv) > 2 else "full"

# Allow offline HF if weights already exist.
os.environ.setdefault("HF_HUB_CACHE", str(out / ".hf_cache"))

import torch
from safetensors.torch import load_file, save_file
from huggingface_hub import hf_hub_download

REPO = "TencentARC/Pixal3D"

# MV DiTs (fp32 on HF -> bf16).
MV_DITS = [
    "ckpts/ss_flow_img_dit_1_3B_64_bf16_mv",
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_mv",
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16_mv",
]
if profile == "full":
    MV_DITS.append("ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv")

# Decoders (fp16, keep as-is).
DECODERS = [
    "ckpts/ss_dec_conv3d_16l8_fp16",
    "ckpts/shape_dec_next_dc_f16c32_fp16",
]
if profile == "full":
    DECODERS.append("ckpts/tex_dec_next_dc_f16c32_fp16")

def fetch_and_convert(name, to_bf16=False):
    # Files on HF are flat: {name}.safetensors and {name}.json (no subdirectory).
    st_file = out / f"{name}.safetensors"
    json_file = out / f"{name}.json"
    if st_file.exists():
        print(f"[skip] {name} already exists")
        return

    st_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[dl] {name}.safetensors ...")
    src = hf_hub_download(REPO, f"{name}.safetensors",
                          local_dir=str(out), local_dir_use_symlinks=False)

    try:
        hf_hub_download(REPO, f"{name}.json",
                        local_dir=str(out), local_dir_use_symlinks=False)
    except Exception:
        pass

    if to_bf16:
        print(f"[convert] {name} -> bf16 ...")
        sd = load_file(src)
        bf = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd.items()}
        tmp = str(st_file) + ".tmp"
        save_file(bf, tmp)
        orig = os.path.getsize(src)
        os.replace(tmp, str(st_file))
        conv = os.path.getsize(str(st_file))
        print(f"  {orig/1e9:.2f} GB -> {conv/1e9:.2f} GB")
        del sd, bf
    else:
        print(f"[keep] {name} (fp16)")

# DINOv3 and NAF weights are auto-downloaded via transformers and
# torch.hub on first pipeline run (no HF_HUB_OFFLINE in the container).

# Pipeline config (strip rembg_model — we supply our own alpha masks).
def fetch_pipeline():
    cfg_path = out / "pipeline_mv.json"
    if cfg_path.exists():
        print("[skip] pipeline_mv.json already exists")
        return
    print("[dl] pipeline_mv.json ...")
    hf_hub_download(REPO, "pipeline_mv.json",
                    local_dir=str(out), local_dir_use_symlinks=False)
    with open(str(cfg_path)) as f:
        cfg = json.load(f)
    if "rembg_model" in cfg.get("args", {}):
        del cfg["args"]["rembg_model"]
        with open(str(cfg_path), "w") as f:
            json.dump(cfg, f, indent=2)
        print("[patch] removed rembg_model from pipeline_mv.json (own masks)")

print(f"=== Fetching Pixal3D-mv weights ({profile}) ===")
for name in MV_DITS:
    fetch_and_convert(name, to_bf16=True)
for name in DECODERS:
    fetch_and_convert(name, to_bf16=False)
fetch_pipeline()
print("=== Done ===")
PYEOF
