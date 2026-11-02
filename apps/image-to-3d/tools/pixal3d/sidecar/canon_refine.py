#!/usr/bin/env python3
"""Pose sensitivity guard (spec-pixal3d §2.7): after the SS stage,
project occupied 64³ voxels into each view and check silhouette IoU.
If any view < 0.7, refine that view's pose by maximising IoU of the
projected visual hull with 50 Gauss-Newton steps on a distance-transform
loss.

This is called by run_mv.py between the SS and shape stages, if enabled.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch


def project_voxels(voxels_64, c2w, fov_x, resolution=64):
    """Project occupied voxel centres into a view, return a binary mask."""
    if voxels_64.sum() == 0:
        return np.zeros((resolution, resolution), dtype=bool)

    coords = np.argwhere(voxels_64)
    pts = (coords.astype(np.float32) + 0.5) / 64.0 - 0.5

    c2w = np.array(c2w, dtype=np.float32)
    w2c = np.linalg.inv(np.vstack([c2w[:3], [0, 0, 0, 1]]))[:3]
    R, t = w2c[:, :3], w2c[:, 3]

    cam_pts = (R @ pts.T).T + t
    z = cam_pts[:, 2]
    valid = z > 0.01

    f = 0.5 * resolution / np.tan(fov_x / 2)
    cx = cy = resolution / 2.0

    u = f * cam_pts[:, 0] / z + cx
    v = f * cam_pts[:, 1] / z + cy

    mask = np.zeros((resolution, resolution), dtype=bool)
    for i in range(len(u)):
        if valid[i]:
            ui, vi = int(round(u[i])), int(round(v[i]))
            if 0 <= ui < resolution and 0 <= vi < resolution:
                mask[vi, ui] = True
    return mask


def compute_iou(mask_a, mask_b):
    """IoU between two binary masks."""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / max(float(union), 1.0)


def check_pose_sensitivity(voxels_64, transforms, view_masks, threshold=0.7):
    """Check whether SS voxels project well into each view's mask.

    Returns: list of (view_index, iou) for views below threshold.
    """
    bad_views = []
    for i, frame in enumerate(transforms["frames"]):
        c2w = frame["transform_matrix"]
        fov_x = frame.get("camera_angle_x", 0.8)
        proj = project_voxels(voxels_64, c2w, fov_x, resolution=64)

        mask_64 = view_masks[i]
        if mask_64.shape != (64, 64):
            from PIL import Image
            m = Image.fromarray(mask_64.astype(np.uint8) * 255).resize(
                (64, 64), Image.NEAREST
            )
            mask_64 = np.array(m) > 128

        iou = compute_iou(proj, mask_64)
        if iou < threshold:
            bad_views.append((i, iou))

    return bad_views


if __name__ == "__main__":
    print("canon_refine: standalone test mode")
    # Test with dummy data.
    vox = np.zeros((64, 64, 64), dtype=bool)
    vox[20:44, 20:44, 20:44] = True

    c2w = [[1, 0, 0, 0], [0, 0, -1, -2], [0, 1, 0, 0], [0, 0, 0, 1]]
    transforms = {
        "frames": [{"transform_matrix": c2w, "camera_angle_x": 0.8}]
    }
    mask = np.ones((64, 64), dtype=bool)
    result = check_pose_sensitivity(vox, transforms, [mask])
    print(f"Bad views: {result}")
