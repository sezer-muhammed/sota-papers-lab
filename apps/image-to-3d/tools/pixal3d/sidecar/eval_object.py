#!/usr/bin/env python3
"""Evaluate object reconstruction: Chamfer distance, F-score, and timing.

Compares predicted GLB meshes against GT meshes from DoraBench/GSO/Toys4K.
Reports JSON lines on stdout.

Usage:
  python eval_object.py \
    --pred_dir /output/glbs \
    --gt_dir /data/gt_meshes \
    --output /output/eval_results.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh


def sample_points(mesh, n=100000, seed=42):
    """Sample n points uniformly from a mesh surface."""
    rng = np.random.RandomState(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=rng.randint(0, 2**31))
    return pts.astype(np.float64)


def chamfer_l1(pts_a, pts_b):
    """L1 Chamfer distance between two point sets (unit-cube normalised)."""
    from scipy.spatial import cKDTree

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    d_a2b, _ = tree_b.query(pts_a, k=1)
    d_b2a, _ = tree_a.query(pts_b, k=1)

    return float(np.mean(d_a2b) + np.mean(d_b2a))


def f_score(pts_a, pts_b, threshold=0.05):
    """F-score at the given distance threshold."""
    from scipy.spatial import cKDTree

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    d_a2b, _ = tree_b.query(pts_a, k=1)
    d_b2a, _ = tree_a.query(pts_b, k=1)

    precision = float(np.mean(d_a2b < threshold))
    recall = float(np.mean(d_b2a < threshold))

    if precision + recall < 1e-10:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def normalise_to_unit_cube(pts):
    """Centre and scale points to [-0.5, 0.5]."""
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    centre = (lo + hi) / 2.0
    extent = max((hi - lo).max(), 1e-6)
    return (pts - centre) / extent


def load_mesh(path):
    """Load a mesh from GLB, OBJ, or PLY."""
    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def cube_rotations():
    """The 24 proper rotations of the cube (signed axis permutations, det +1)."""
    import itertools

    rots = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            r = np.zeros((3, 3))
            for i, (j, sg) in enumerate(zip(perm, signs)):
                r[i, j] = sg
            if np.linalg.det(r) > 0:
                rots.append(r)
    return rots


def icp(src, dst, iters=20):
    """Rigid ICP (Kabsch on nearest neighbours); returns src transformed."""
    from scipy.spatial import cKDTree

    tree = cKDTree(dst)
    cur = src.copy()
    for _ in range(iters):
        _, idx = tree.query(cur, k=1)
        m = dst[idx]
        mc, cc = m.mean(0), cur.mean(0)
        u, _, vt = np.linalg.svd((cur - cc).T @ (m - mc))
        d = np.sign(np.linalg.det(vt.T @ u.T))
        r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
        cur = (cur - cc) @ r.T + mc
    return cur


def align(pred_pts, gt_pts, n_sub=20000, seed=0):
    """Rigidly align pred to gt: best of 24 cube rotations, then ICP.

    Generated objects come out in the main view's frame (Pixal3D-mv poses the
    result in frame 0), not the GT's, so metrics are taken after alignment,
    as in the Pixal3D / TRELLIS object benchmarks.
    """
    rng = np.random.RandomState(seed)
    ps = pred_pts[rng.choice(len(pred_pts), min(n_sub, len(pred_pts)), replace=False)]
    gs = gt_pts[rng.choice(len(gt_pts), min(n_sub, len(gt_pts)), replace=False)]
    best = None
    for r in cube_rotations():
        cand = normalise_to_unit_cube(ps @ r.T)
        cand = icp(cand, gs, iters=10)
        cd = chamfer_l1(cand, gs)
        if best is None or cd < best[0]:
            best = (cd, r)
    full = normalise_to_unit_cube(pred_pts @ best[1].T)
    return icp(full, gt_pts, iters=20)


def evaluate_pair(pred_path, gt_path, n_points=100000, do_align=True):
    """Evaluate one predicted mesh against ground truth."""
    pred_mesh = load_mesh(pred_path)
    gt_mesh = load_mesh(gt_path)

    pred_pts = normalise_to_unit_cube(sample_points(pred_mesh, n_points))
    gt_pts = normalise_to_unit_cube(sample_points(gt_mesh, n_points))
    if do_align:
        pred_pts = align(pred_pts, gt_pts)

    cd = chamfer_l1(pred_pts, gt_pts)
    fs = f_score(pred_pts, gt_pts, threshold=0.05)

    return {"chamfer_l1": cd, "f_score_005": fs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--output", default="eval_results.jsonl")
    parser.add_argument("--n_points", type=int, default=100000)
    parser.add_argument("--no_align", action="store_true",
                        help="skip rigid alignment (24 cube rotations + ICP)")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    pred_files = sorted(pred_dir.glob("*.glb"))
    results = []

    for pred_path in pred_files:
        name = pred_path.stem
        gt_candidates = [
            gt_dir / f"{name}.glb",
            gt_dir / f"{name}.obj",
            gt_dir / f"{name}.ply",
            gt_dir / name / "mesh.obj",
            gt_dir / name / "model.obj",
        ]
        gt_path = next((p for p in gt_candidates if p.exists()), None)
        if gt_path is None:
            print(json.dumps({"name": name, "error": "no GT mesh found"}))
            continue

        try:
            t = time.time()
            metrics = evaluate_pair(pred_path, gt_path, args.n_points, not args.no_align)
            metrics["name"] = name
            metrics["eval_s"] = round(time.time() - t, 2)
            results.append(metrics)
            print(json.dumps(metrics))
        except Exception as e:
            print(json.dumps({"name": name, "error": str(e)}))

    # Summary.
    if results:
        avg_cd = np.mean([r["chamfer_l1"] for r in results])
        avg_fs = np.mean([r["f_score_005"] for r in results])
        summary = {
            "summary": True,
            "n_objects": len(results),
            "mean_chamfer_l1": round(float(avg_cd), 6),
            "mean_f_score_005": round(float(avg_fs), 4),
        }
        print(json.dumps(summary))

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
