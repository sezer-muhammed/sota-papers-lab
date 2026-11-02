# VGGT-Ω Oracle

Reference tools for running the official VGGT-Ω model and producing ground-truth
outputs for validating our native Rust + CUDA engine.

## Prerequisites

1. **Docker image**: `spl-vggt-ref` (PyTorch 2.9.1 / CUDA 12.8 with the official
   `vggt-omega` package at commit `48b23c8`).
2. **Checkpoint**: `~/spl-work/models/vggt-omega/vggt_omega_1b_512.pt` (4.6 GB,
   downloaded from `facebook/VGGT-Omega` on HuggingFace).

Build the image:

```sh
cd apps/image-to-3d/tools/vggt
docker build -t spl-vggt-ref -f Dockerfile .
```

## Oracle CLI

```sh
vggt-oracle img1.png img2.png --out result.npz [--dump-layers 0,4,11,17,23] [--dtype fp32|bf16]
```

Install the CLI by copying or symlinking `vggt-oracle` to `~/spl-work/bin/`.

### Output format

The `.npz` file contains:

| Key            | Shape              | Dtype   | Description |
|----------------|--------------------|---------|-------------|
| `preprocessed` | `(N, 3, H, W)`    | float32 | Preprocessed input (ImageNet-normalised) |
| `pose_enc`     | `(N, 9)`           | float32 | Raw 9-D pose encoding: T(3), quat(4), FoV(2) |
| `extrinsics`   | `(N, 3, 4)`        | float64 | Camera-from-world [R|t] in OpenCV coords |
| `intrinsics`   | `(N, 3, 3)`        | float64 | Camera intrinsics (fx, fy, cx, cy) |
| `depth`        | `(N, H, W)`        | float32 | Predicted metric depth |
| `depth_conf`   | `(N, H, W)`        | float32 | Depth confidence (1 + exp(logit)) |
| `world_points` | `(N, H, W, 3)`     | float32 | Unprojected world-space 3-D points |
| `block_XX`     | `(B*N, T, D)`      | float32 | Block output (only with `--dump-layers`) |

The coordinate convention is OpenCV (right-down-forward).

## Weight conversion

```sh
python convert.py --input vggt_omega_1b_512.pt --output-dir ./converted/
```

Produces:
- `weights.safetensors` — all parameters in bf16
- `weights_fp8.safetensors` — FP8 E4M3 copies of every Linear weight with
  per-output-channel scales
- `manifest.json` — tensor names, shapes, dtypes, and FP8 scale metadata

## Benchmarks

```sh
python benchmark.py --checkpoint ~/spl-work/models/vggt-omega/vggt_omega_1b_512.pt
```

Tests N = 2, 10, 50, 100, 200 frames at 512 px in bf16 and fp32. Records
wall time and peak VRAM. Results saved to `benchmark_results.json`.

### Reference results (RTX 4070 SUPER, 12 GB, ~2.5 GB occupied by other processes)

| N frames | dtype | Time (s) | Peak VRAM (MB) | Status |
|----------|-------|----------|----------------|--------|
| 2        | bf16  | 0.138    | 6232           | OK     |
| 10       | bf16  | 0.852    | 6840           | OK     |
| 50       | bf16  | —        | —              | OOM    |
| 100      | bf16  | —        | —              | OOM    |
| 200      | bf16  | —        | —              | OOM    |
| 2        | fp32  | 0.479    | 4715           | OK     |
| 10       | fp32  | 3.633    | 5782           | OK     |
| 50+      | fp32  | —        | —              | OOM    |

Note: ~2.5 GB was used by other GPU processes during benchmarking. With a
clean GPU, N=50 bf16 might fit (~9.9 GB peak in an earlier contended run).
N=50+ requires attention chunking or quantised attention (Phase 2).

## Evaluation

Small-subset evaluation (3 scenes each from ETH3D and Sintel):

```sh
# Prepare data (runs inside the Docker container)
bash run_all.sh

# Or run individual eval:
python eval_subset.py --dataset eth3d \
    --data-root ~/spl-work/datasets/vggt-eval/eth3d \
    --checkpoint ~/spl-work/models/vggt-omega/vggt_omega_1b_512.pt \
    --output results_eth3d.json \
    --frame-map frames/eth3d_subset.json
```

### Metrics

- **Pose**: AUC@3, AUC@30 (relative rotation and translation accuracy)
- **Depth**: AbsRel (absolute relative error), delta<1.25 (accuracy threshold)

The evaluation uses the same metrics as the official VGGT-Ω paper:
- Relative pose pairs from `torch.combinations`
- Depth aligned with median scale+shift
- Depth clipped to [0.1, 100.0] range

### Reference evaluation results (vggt_omega_1b_512.pt)

**ETH3D** (delivery_area, 10 frames):
- AUC@3: 30.37, AUC@30: 86.30
- delta<1.25: 99.81%, AbsRel: 0.0118

**Sintel** (alley_1, 10 frames):
- AUC@3: 20.74, AUC@30: 66.00
- delta<1.25: 98.15%, AbsRel: 0.0336

### Dataset subsets

| Dataset | Scenes | Frames/scene | Total frames |
|---------|--------|--------------|--------------|
| ETH3D   | delivery_area | 10 | 10 |
| Sintel  | alley_1 | 10 | 10 |

Frame indices match the official evaluation protocol. Full frame maps are in
`frames/eth3d.json` and `frames/sintel.json` (13 and 16 scenes respectively).

## Model architecture summary

- **Patch embed**: DINOv2-style ViT (24 blocks, 1024-dim, 16 heads)
- **Aggregator**: 24 alternating frame-attention + inter-frame-attention blocks
  - Inter-frame types: global (most) or register-only (blocks 2, 6, 9, 14, 20)
  - QK-norm, LayerScale (1e-5), 2-D RoPE (base=100, normalize="max")
  - Camera token (2 variants) + 16 register tokens per frame
- **Camera head**: 4 attention blocks on camera+register tokens, projects to 9-D
- **Dense head**: DPT-style multi-scale fusion from layers 4, 11, 17, 23
- **Parameters**: ~1B

## File listing

```
apps/image-to-3d/tools/vggt/
├── Dockerfile          spl-vggt-ref image definition
├── ORACLE.md           this file
├── benchmark.py        inference speed + VRAM benchmarks
├── convert.py          .pt → safetensors (bf16 + FP8 E4M3)
├── eval_subset.py      pose + depth evaluation on subsets
├── frames/
│   ├── eth3d.json          full ETH3D frame map (13 scenes)
│   ├── eth3d_subset.json   3-scene subset for quick eval
│   ├── sintel.json         full Sintel frame map (16 scenes)
│   └── sintel_subset.json  3-scene subset for quick eval
├── oracle.py           oracle inference script
├── prepare_datasets.py dataset preparation helper
├── remote_zip.py       HTTP range-request ZIP reader
├── run_all.sh          master orchestration script
└── vggt-oracle         CLI wrapper
```
