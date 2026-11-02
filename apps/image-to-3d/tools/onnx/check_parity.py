"""Parity of the exported DA3-BASE graphs (ONNX Runtime CUDA EP; backbone,
then the head 8 frames at a time, as the engine runs them) against the
PyTorch model in fp32, with the official bf16-autocast path as the noise
floor. Real frames, several N per bucket. Writes a JSON report; summarised
in `PARITY.md`.

    python check_parity.py --weights model.safetensors --dir da3-base \
        --images <dir> --out parity.json [--tag 504x336_fp32]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from export_da3 import Backbone, Head, build

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def frames(paths, n, h, w):
    out = []
    for p in paths[:n]:
        im = Image.open(p).convert("RGB").resize((w, h), Image.BICUBIC)
        out.append(((np.asarray(im, np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))
    return np.ascontiguousarray(np.stack(out))


def rel(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(b**2)), 1e-12))


def quat_to_mat(q):
    i, j, k, r = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_s = 2.0 / (q * q).sum(-1)
    m = np.stack([1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
                  two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
                  two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)], -1)
    return m.reshape(q.shape[:-1] + (3, 3))


def relative(pe):
    """Camera-to-world pose encodings -> poses relative to frame 0 (rotation,
    unit-scale translation): invariant to the reference view the graph picks."""
    pe = pe.astype(np.float64)
    r, t = quat_to_mat(pe[:, 3:7]), pe[:, :3]
    r_rel = np.einsum("ji,njk->nik", r[0], r)
    t_rel = np.einsum("ji,nj->ni", r[0], t - t[0])
    scale = np.linalg.norm(t_rel, axis=-1).max()
    return r_rel, t_rel / max(scale, 1e-12)


def angle_deg(a, b):
    c = (np.einsum("nij,nij->n", a, b) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def compare(got, ref):
    """Depth after a global median scale (depth is relative and its scale
    follows the reference view); poses relative to frame 0."""
    d, c, pe = got
    rd, rc, rpe = ref
    s = np.median(rd / d)
    ra, ta = relative(pe)
    rb, tb = relative(rpe)
    return {
        "depth_rel": rel(d * s, rd),
        "depth_absrel_median": float(np.median(np.abs(d * s - rd) / np.abs(rd))),
        "depth_scale": float(s),
        "conf_rel": rel(c, rc),
        "rel_rot_max_deg": float(angle_deg(ra, rb).max()),
        "rel_rot_mean_deg": float(angle_deg(ra, rb).mean()),
        "rel_trans_max": float(np.linalg.norm(ta - tb, axis=-1).max()),
        "fov_max_deg": float(np.degrees(np.abs(pe[:, 7:] - rpe[:, 7:])).max()),
        "raw_pose_rel": rel(pe, rpe),
    }


def run_ort(backbone, head, x, chunk=8):
    dtype = np.float16 if backbone.get_inputs()[0].type == "tensor(float16)" else np.float32
    *levels, pose = backbone.run(None, {"images": x.astype(dtype)})
    depth, conf = [], []
    for s in range(0, x.shape[0], chunk):
        d, c = head.run(None, {f"f{i}": levels[i][s:s + chunk] for i in range(4)})
        depth.append(d)
        conf.append(c)
    return [np.concatenate(depth).astype(np.float32), np.concatenate(conf).astype(np.float32), pose.astype(np.float32)]


def run_torch(net, x, bf16, chunk=8):
    bb, hd = Backbone(net), Head(net)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        *levels, pose = bb(torch.from_numpy(x).cuda())
    with torch.no_grad():
        outs = [hd(*[l[s:s + chunk].float() for l in levels]) for s in range(0, x.shape[0], chunk)]
    return [torch.cat([o[0] for o in outs]).float().cpu().numpy(), torch.cat([o[1] for o in outs]).float().cpu().numpy(),
            pose.float().cpu().numpy()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tags", default="504x336_fp32,504x336")
    p.add_argument("--frames", default="3,10,16")
    a = p.parse_args()
    import onnxruntime as ort

    ort.preload_dlls()
    paths = sorted(str(q) for q in Path(a.images).iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
    ns = [int(n) for n in a.frames.split(",")]
    got = {}
    for tag in a.tags.split(","):
        w, h = map(int, tag.split("_")[0].split("x"))
        prov = [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"]
        bb = ort.InferenceSession(str(Path(a.dir) / f"backbone_{tag}.onnx"), providers=prov)
        hd = ort.InferenceSession(str(Path(a.dir) / f"head_{tag}.onnx"), providers=prov)
        for n in ns:
            got[tag, n] = (w, h, run_ort(bb, hd, frames(paths, n, h, w)))
        del bb, hd
    net = build(a.weights).cuda()
    report = []
    for (tag, n), (w, h, g) in got.items():
        x = frames(paths, n, h, w)
        ref = run_torch(net, x, False)
        bf = run_torch(net, x, True)
        row = {"graph": tag, "n": n, "w": w, "h": h, "onnx_vs_fp32": compare(g, ref), "bf16_vs_fp32": compare(bf, ref)}
        print(json.dumps(row), flush=True)
        report.append(row)
    Path(a.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
