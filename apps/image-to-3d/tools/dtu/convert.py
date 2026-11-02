"""IDR/NeuS-format DTU scan (cameras_hd.npz, image_hd/, mask_hd/) -> COLMAP text
model in the normalised frame (world_mat @ scale_mat), as 2DGS trains on DTU.

usage: python -I convert.py <raw_scan_dir> <dataset_dir>
Moves image_hd/ and mask_hd/ into <dataset_dir>; writes sparse/0/*.txt with
100k random points in the unit sphere (no SfM points are shipped).
Our own code; the camera convention is IDR's (MIT).
"""
import os, shutil, sys

import cv2
import numpy as np

raw, out = sys.argv[1], sys.argv[2]
cams = np.load(os.path.join(raw, "cameras_hd.npz"))
os.makedirs(os.path.join(out, "sparse", "0"), exist_ok=True)
for d in ("image_hd", "mask_hd"):
    if not os.path.isdir(os.path.join(out, d)):
        shutil.move(os.path.join(raw, d), os.path.join(out, d))
names = sorted(os.listdir(os.path.join(out, "image_hd")))
h, w = cv2.imread(os.path.join(out, "image_hd", names[0])).shape[:2]


def rotmat_to_qvec(R):
    m = R
    t = np.trace(m)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
    q = np.zeros(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return q


with open(os.path.join(out, "sparse/0/cameras.txt"), "w") as fc, open(os.path.join(out, "sparse/0/images.txt"), "w") as fi:
    for i, name in enumerate(names):
        P = (cams[f"world_mat_{i}"] @ cams[f"scale_mat_{i}"])[:3, :4]
        K, R, C = cv2.decomposeProjectionMatrix(P)[:3]
        K = K / K[2, 2]
        C = (C[:3] / C[3]).reshape(3)
        t = -R @ C
        # OpenCV/IDR pixel centres sit at integers, COLMAP's at +0.5.
        fc.write(f"{i + 1} PINHOLE {w} {h} {K[0, 0]:.10f} {K[1, 1]:.10f} {K[0, 2] + 0.5:.10f} {K[1, 2] + 0.5:.10f}\n")
        q = rotmat_to_qvec(R)
        fi.write(f"{i + 1} {' '.join(f'{x:.12f}' for x in q)} {' '.join(f'{x:.12f}' for x in t)} {i + 1} {name}\n\n")
rng = np.random.default_rng(0)
p = rng.normal(size=(100_000, 3))
p *= (rng.random(100_000) ** (1 / 3) / np.linalg.norm(p, axis=1))[:, None]
with open(os.path.join(out, "sparse/0/points3D.txt"), "w") as f:
    for k, x in enumerate(p):
        f.write(f"{k + 1} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f} 128 128 128 0\n")
np.save(os.path.join(out, "scale_mat.npy"), cams["scale_mat_0"])
print(f"{len(names)} views {w}x{h} -> {out}")
