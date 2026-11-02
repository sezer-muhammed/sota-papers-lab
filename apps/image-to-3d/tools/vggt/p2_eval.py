#!/usr/bin/env python3
"""Phase 2 pose evaluation of the native engine on the held-out scenes of
reports/image-to-3d/spec-attn-schedule.md (50 evenly spaced frames, COLMAP
ground truth, the official eval/metrics.py AUC).

    p2_eval.py list  <datasets> <out>          # <out>/<scene>/frames.txt + gt.npy
    p2_eval.py score <list> <pred> <schedules>  # AUC per schedule vs dense

Run `score` where `vggt_omega/eval/metrics.py` is importable (the p2 container:
PYTHONPATH=~/spl-work/p2/src/vggt-omega/eval). Frame selection and the GT
reader follow the p2-schedule scripts (p2lib.scene_frames).
"""
import glob
import json
import os
import struct
import sys

import numpy as np

EVAL = ["mipnerf360/bonsai", "mipnerf360/counter", "mipnerf360/room", "mipnerf360/flowers",
        "mipnerf360/treehill", "tandt/train", "tandt/truck", "db/drjohnson", "db/playroom"]


def read_images_bin(path):
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _, qw, qx, qy, qz, tx, ty, tz, _ = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            np2 = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * np2, 1)
            w, x, y, z = np.array([qw, qx, qy, qz]) / np.linalg.norm([qw, qx, qy, qz])
            r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            out[name.decode()] = np.concatenate([r, [[tx], [ty], [tz]]], 1)
    return out


def scene_frames(root, n):
    imgdir = [d for d in glob.glob(root + "/images*") if os.path.isdir(d)][0]
    gt = read_images_bin(root + "/sparse/0/images.bin")
    names = sorted(x for x in gt if os.path.exists(os.path.join(imgdir, x)))
    idx = np.unique(np.linspace(0, len(names) - 1, n).round().astype(int))
    names = [names[i] for i in idx]
    return [os.path.join(imgdir, x) for x in names], np.stack([gt[x] for x in names])


def cmd_list(datasets, out, n=50):
    for s in EVAL:
        paths, gt = scene_frames(f"{datasets}/{s}", n)
        d = f"{out}/{s}"
        os.makedirs(d, exist_ok=True)
        open(f"{d}/frames.txt", "w").write("\n".join(paths) + "\n")
        np.save(f"{d}/gt.npy", gt)
        print(s, len(paths))


def cmd_score(lst, pred, schedules):
    from metrics import pose_errors, pose_metrics
    res = {}
    for sch in schedules:
        per = {}
        for s in EVAL:
            gt = np.load(f"{lst}/{s}/gt.npy")
            ext = np.load(f"{pred}/{sch}/{s}/extrinsics.npy").astype(np.float64)
            r, t = pose_errors(ext, gt)
            m = pose_metrics(r, t)
            per[s] = {"AUC@3": float(m["AUC@3"]), "AUC@30": float(m["AUC@30"])}
        res[sch] = {"AUC@3": float(np.mean([v["AUC@3"] for v in per.values()])),
                    "AUC@30": float(np.mean([v["AUC@30"] for v in per.values()])), "per": per}
    base = res.get("dense")
    for sch, r in res.items():
        d = "" if base is None else f"  Δ {r['AUC@3'] - base['AUC@3']:+.2f} / {r['AUC@30'] - base['AUC@30']:+.2f}"
        print(f"{sch:12s} AUC@3 {r['AUC@3']:.2f} AUC@30 {r['AUC@30']:.2f}{d}")
    json.dump(res, open(f"{pred}/scores.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "list":
        cmd_list(sys.argv[2], sys.argv[3])
    else:
        cmd_score(sys.argv[2], sys.argv[3], sys.argv[4].split(","))
