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


def evaluate_pair(pred_path, gt_path, n_points=100000):
    """Evaluate one predicted mesh against ground truth."""
    pred_mesh = load_mesh(pred_path)
    gt_mesh = load_mesh(gt_path)

    pred_pts = normalise_to_unit_cube(sample_points(pred_mesh, n_points))
    gt_pts = normalise_to_unit_cube(sample_points(gt_mesh, n_points))

    cd = chamfer_l1(pred_pts, gt_pts)
    fs = f_score(pred_pts, gt_pts, threshold=0.05)

    return {"chamfer_l1": cd, "f_score_005": fs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--output", default="eval_results.jsonl")
    parser.add_argument("--n_points", type=int, default=100000)
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
            metrics = evaluate_pair(pred_path, gt_path, args.n_points)
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
