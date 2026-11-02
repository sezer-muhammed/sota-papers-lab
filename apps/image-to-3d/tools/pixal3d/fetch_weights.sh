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
    local = out / name
    safetensors = local / "model.safetensors"
    config_json = local / "config.json"
    if safetensors.exists():
        print(f"[skip] {name} already exists")
        return

    local.mkdir(parents=True, exist_ok=True)

    # Download safetensors.
    print(f"[dl] {name}/model.safetensors ...")
    src = hf_hub_download(REPO, f"{name}/model.safetensors",
                          local_dir=str(out), local_dir_use_symlinks=False)

    # Download config.json if it exists.
    try:
        hf_hub_download(REPO, f"{name}/config.json",
                        local_dir=str(out), local_dir_use_symlinks=False)
    except Exception:
        pass

    if to_bf16:
        print(f"[convert] {name} fp32 -> bf16 ...")
        sd = load_file(src)
        bf = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd.items()}
        tmp = str(safetensors) + ".tmp"
        save_file(bf, tmp)
        os.replace(tmp, str(safetensors))
        # Verify size roughly halved.
        orig = os.path.getsize(src)
        conv = os.path.getsize(str(safetensors))
        print(f"  {orig/1e9:.2f} GB -> {conv/1e9:.2f} GB")
        if src != str(safetensors):
            os.remove(src)
        del sd, bf
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print(f"[keep] {name} (fp16)")

# DINOv3 weights.
DINO_REPO = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
DINO_FILE = "dinov3_vitl16_pretrain_lvd1689m.pth"
def fetch_dino():
    dino_dir = out / "dinov3"
    dino_path = dino_dir / DINO_FILE
    if dino_path.exists():
        print("[skip] DINOv3 already exists")
        return
    dino_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dl] DINOv3 ViT-L/16 from {DINO_REPO} ...")
    src = hf_hub_download(DINO_REPO, DINO_FILE,
                          local_dir=str(dino_dir), local_dir_use_symlinks=False)
    # Convert to bf16.
    print("[convert] DINOv3 fp32 -> bf16 ...")
    sd = torch.load(src, map_location="cpu", weights_only=True)
    bf = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd.items()}
    tmp = str(dino_path) + ".tmp"
    torch.save(bf, tmp)
    os.replace(tmp, str(dino_path))
    if src != str(dino_path):
        os.remove(src)
    del sd, bf
    print(f"  DINOv3 saved to {dino_path}")

# NAF weights.
def fetch_naf():
    naf_dir = out / "naf"
    naf_path = naf_dir / "naf_release.pth"
    if naf_path.exists():
        print("[skip] NAF already exists")
        return
    naf_dir.mkdir(parents=True, exist_ok=True)
    print("[dl] NAF weights ...")
    hf_hub_download("valeoai/NAF", "naf_release.pth",
                    local_dir=str(naf_dir), local_dir_use_symlinks=False)

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
fetch_dino()
fetch_naf()
fetch_pipeline()
print("=== Done ===")
PYEOF
