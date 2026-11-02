"""The official DA3-BASE PyTorch path (DA3 repo, Apache-2.0) as the
reference for the commercial engine:

- `eval`: depth and extrinsics on the vggt-ref evaluation subsets, written
  as `<out>/<dataset>/<scene>/{depth,extrinsics}.npy` for
  `tools/vggt/eval_subset.py --predictions`. Frames get the same aspect
  crop as the VGGT-Ω evaluation (into [0.5, 2], so the scored depth region
  matches), then DA3's own `InputProcessor` (504 long side, cv2 resize,
  sides rounded to multiples of 14) and the official forward
  (`DepthAnything3.forward`: bf16 autocast backbone, fp32 heads,
  camera-decoder poses).
- `time`: forward time (CUDA-synchronised wall clock, median after two
  warm-ups) and peak memory at the given N on evenly sampled frames of a
  scene, the same frames `gpu_da3_bench_scaling` uses.

    python da3_official.py eval --weights model.safetensors --data-root <vggt-eval> --frames-dir <tools/vggt/frames> --out <dir>
    python da3_official.py time --weights model.safetensors --images <dir> --n 3,10,32
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from export_da3 import build


RAY_POSE = False


def official_forward(net, x):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return net(x[None], None, None, [], False, RAY_POSE, "saddle_balanced")


def preprocess(images):
    from depth_anything_3.utils.io.input_processor import InputProcessor

    imgs, _, _ = InputProcessor()(images, None, None, process_res=504, process_res_method="upper_bound_resize",
                                  num_workers=1, print_progress=False)
    return imgs


def affine_inverse(m):
    r, t = m[..., :3, :3], m[..., :3, 3:]
    rt = r.transpose(-1, -2)
    return torch.cat([rt, -rt @ t], dim=-1)


def evaluate(a, net):
    sys.path.insert(0, str(Path(a.frames_dir).parent))
    from eval_subset import _crop_box

    for ds in ("eth3d", "sintel"):
        subset = json.loads((Path(a.frames_dir) / f"{ds}_subset.json").read_text())
        for scene, idx in subset.items():
            root = Path(a.data_root) / ds / scene
            if not root.is_dir():
                continue
            names = [l.strip() for l in (root / "frames.txt").read_text().splitlines()]
            pil = [Image.open(root / "images" / names[i]).convert("RGB") for i in idx]
            l, t, r, b = _crop_box(*pil[0].size)
            w, h = pil[0].size
            pil = [im.crop((round(l * w), round(t * h), round(r * w), round(b * h))) for im in pil]
            x = preprocess(pil).cuda()
            official_forward(net, x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = official_forward(net, x)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
            depth = out["depth"][0].float().cpu().numpy()
            ext = out["extrinsics"][0].float()  # w2c [N, 4, 4] or [N, 3, 4]
            ext = ext[:, :3, :]
            # World = frame 0 (scores are relative; this matches the engine).
            c2w0 = affine_inverse(ext[:1])[0]
            c2w0 = torch.cat([c2w0, torch.tensor([[0, 0, 0, 1.0]], device=c2w0.device)])
            ext = ext @ c2w0
            d = Path(a.out) / ds / scene
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "depth.npy", depth.astype(np.float32))
            np.save(d / "extrinsics.npy", ext.cpu().numpy().astype(np.float32))
            print(f"eval: {ds}/{scene} {len(idx)} frames {x.shape[-1]}x{x.shape[-2]} in {ms:.1f} ms -> {d}", flush=True)


def timing(a, net):
    files = sorted(f for f in os.listdir(a.images) if f.lower().endswith((".jpg", ".png")))
    rows = []
    for n in [int(v) for v in a.n.split(",")]:
        L = len(files)
        idx = list(range(L)) if L <= n else [(i * (L - 1) + (n - 1) // 2) // (n - 1) for i in range(n)]
        x = preprocess([os.path.join(a.images, files[i]) for i in idx]).cuda()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ts = []
        reps = 10 if n <= 50 else 3
        try:
            for k in range(reps + 2):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                official_forward(net, x)
                torch.cuda.synchronize()
                if k >= 2:
                    ts.append((time.perf_counter() - t0) * 1e3)
        except torch.OutOfMemoryError:
            print(json.dumps({"n": n, "oom": True}), flush=True)
            continue
        row = {"n": n, "w": x.shape[-1], "h": x.shape[-2], "median_ms": float(np.median(ts)), "min_ms": float(min(ts)),
               "runs": reps, "peak_mib": torch.cuda.max_memory_allocated() / 2**20}
        print(json.dumps(row), flush=True)
        rows.append(row)
        del x
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("eval", "time"))
    p.add_argument("--weights", required=True)
    p.add_argument("--data-root")
    p.add_argument("--frames-dir")
    p.add_argument("--images")
    p.add_argument("--n", default="3,10,32,50,100,200")
    p.add_argument("--out")
    p.add_argument("--ray-pose", action="store_true", help="DA3's ray-head poses instead of the camera decoder")
    a = p.parse_args()
    global RAY_POSE
    RAY_POSE = a.ray_pose
    net = build(a.weights).cuda().eval()
    (evaluate if a.mode == "eval" else timing)(a, net)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
