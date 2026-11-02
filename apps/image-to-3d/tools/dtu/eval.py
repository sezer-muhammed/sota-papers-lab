"""DTU Chamfer of a mesh in the normalised frame, following the 2DGS protocol:
cull with the dilated object masks (2DGS evaluate_single_scene.py), map back
with scale_mat, sample the surface at 0.2 mm, downsample, then the official
DTU distances (ObsMask / ground-plane filtering, 20 mm cap) as in
DTUeval-python. Our own implementation of that protocol.

usage: python -I eval.py <mesh.ply> <dataset_dir> <gt_dir> <scan> [out.json]
"""
import json, os, sys

import cv2
import numpy as np
import scipy.io
import trimesh
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

mesh_path, ds, gt, scan = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
out_json = sys.argv[5] if len(sys.argv) > 5 else None
mesh = trimesh.load(mesh_path, process=False)
V = np.asarray(mesh.vertices, np.float64)
F = np.asarray(mesh.faces)

# --- cull (normalised frame): keep vertices inside every dilated mask or outside the frame
cams = np.load(os.path.join(ds, "cameras_hd.npz")) if os.path.exists(os.path.join(ds, "cameras_hd.npz")) else None
names = sorted(os.listdir(os.path.join(ds, "image_hd")))
masks = sorted(os.listdir(os.path.join(ds, "mask_hd")))
# Cameras from the COLMAP text written by convert.py.
Ks, Rt = {}, {}
for line in open(os.path.join(ds, "sparse/0/cameras.txt")):
    a = line.split()
    if a:
        Ks[int(a[0])] = np.array([[float(a[4]), 0, float(a[6])], [0, float(a[5]), float(a[7])], [0, 0, 1]])
imgs = [l.split() for l in open(os.path.join(ds, "sparse/0/images.txt")) if l.strip()]
yy, xx = np.mgrid[-24:25, -24:25]
disk = xx * xx + yy * yy <= 24 * 24
keep = np.ones(len(V), bool)
for a in imgs:
    qw, qx, qy, qz = map(float, a[1:5])
    R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                  [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                  [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])
    t = np.array(list(map(float, a[5:8])))
    K = Ks[int(a[8])]
    m = cv2.imread(os.path.join(ds, "mask_hd", masks[names.index(a[9])]))[:, :, 0] > 127
    m = binary_dilation(m, disk)
    H, W = m.shape
    c = (K @ (R @ V.T + t[:, None])).T
    px = c[:, :2] / (c[:, 2:3] + 1e-6)
    # 2DGS samples with align_corners=True on [0, W-1] (nearest).
    inside = (px[:, 0] > 0) & (px[:, 0] < W - 1) & (px[:, 1] > 0) & (px[:, 1] < H - 1)
    xi = np.clip(np.round(px[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(px[:, 1]).astype(int), 0, H - 1)
    keep &= (~inside) | m[yi, xi]
fkeep = keep[F].all(1)
F = F[fkeep]
used = np.unique(F)
remap = -np.ones(len(V), int)
remap[used] = np.arange(len(used))
S = np.load(os.path.join(ds, "scale_mat.npy"))
V = V[used] * S[0, 0] + S[:3, 3][None]
F = remap[F]

# --- sample the triangles at 0.2 mm (DTUeval-python's scheme), then downsample
thresh = 0.2
tri = V[F]
v1, v2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
l1, l2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
area2 = np.linalg.norm(np.cross(v1, v2), axis=1)
nz = area2 > 0
tri, v1, v2, l1, l2, area2 = tri[nz], v1[nz], v2[nz], l1[nz], l2[nz], area2[nz]
thr = thresh * np.sqrt(l1 * l2 / area2)
n1, n2 = np.floor(l1 / thr), np.floor(l2 / thr)
a = n1.astype(np.int64)
b = n2.astype(np.int64)
ok = (a > 0) & (b > 0)
tri, v1, v2, a, b = tri[ok], v1[ok], v2[ok], a[ok], b[ok]
cells = (a + 1) * (b + 1)
pts = [V]
for lo in range(0, len(tri), 200_000):
    sl = slice(lo, lo + 200_000)
    cnt = cells[sl]
    owner = np.repeat(np.arange(len(cnt)), cnt)
    start = np.repeat(np.cumsum(cnt) - cnt, cnt)
    j = np.arange(cnt.sum()) - start
    bb = b[sl][owner] + 1
    c0 = (j // bb + 0.5) / (a[sl][owner] + 1)
    c1 = (j % bb + 0.5) / bb
    inside = c0 + c1 < 1
    o = owner[inside]
    pts.append(c0[inside, None] * v1[sl][o] + c1[inside, None] * v2[sl][o] + tri[sl][o, 0])
data = np.concatenate(pts)
rng = np.random.default_rng(0)
data = data[rng.permutation(len(data))]
tree = cKDTree(data)
mask = np.ones(len(data), bool)
for i, nb in enumerate(tree.query_ball_point(data, thresh, workers=-1)):
    if mask[i]:
        mask[nb] = False
        mask[i] = True
data = data[mask]

# --- official distances
obs = scipy.io.loadmat(os.path.join(gt, f"ObsMask{scan}_10.mat"))
ObsMask, BB, Res = obs["ObsMask"], obs["BB"].astype(np.float32), obs["Res"]
patch = 60
inb = ((data >= BB[:1] - patch) & (data < BB[1:] + patch * 2)).sum(-1) == 3
data_in = data[inb]
g = np.around((data_in - BB[:1]) / Res).astype(np.int32)
gin = ((g >= 0) & (g < np.expand_dims(ObsMask.shape, 0))).sum(-1) == 3
data_obs = data_in[gin][ObsMask[g[gin, 0], g[gin, 1], g[gin, 2]].astype(bool)]
stl = trimesh.load(os.path.join(gt, f"stl{scan:03}_total.ply")).vertices
d2s = cKDTree(stl).query(data_obs, workers=-1)[0]
plane = scipy.io.loadmat(os.path.join(gt, f"Plane{scan}.mat"))["P"].reshape(-1)
stl_above = stl[(np.concatenate([stl, np.ones((len(stl), 1))], 1) * plane).sum(-1) > 0]
s2d = cKDTree(data_in).query(stl_above, workers=-1)[0]
md = 20
acc, comp = d2s[d2s < md].mean(), s2d[s2d < md].mean()
res = {"scan": scan, "accuracy": float(acc), "completeness": float(comp), "chamfer": float((acc + comp) / 2),
       "kept_faces": int(len(F)), "data_points": int(len(data_obs))}
print(json.dumps(res))
if out_json:
    json.dump(res, open(out_json, "w"), indent=1)
