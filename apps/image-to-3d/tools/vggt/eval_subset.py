#!/usr/bin/env python3
"""Evaluate VGGT-Ω on small dataset subsets: AUC@3/@30 (pose) and depth AbsRel/delta<1.25.

Usage:
    python eval_subset.py --dataset eth3d \
        --data-root ~/spl-work/datasets/vggt-eval/eth3d \
        --checkpoint ~/spl-work/models/vggt-omega/vggt_omega_1b_512.pt \
        --output results_eth3d.json

    python eval_subset.py --dataset sintel \
        --data-root ~/spl-work/datasets/vggt-eval/sintel \
        --checkpoint ~/spl-work/models/vggt-omega/vggt_omega_1b_512.pt \
        --output results_sintel.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


# ---------- metrics (self-contained, matching the official eval/metrics.py) ----------

def pad_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    values = np.asarray(extrinsics, dtype=np.float64)
    if values.shape[-2] == 4:
        return values
    output = np.zeros((values.shape[0], 4, 4), dtype=np.float64)
    output[:, :3] = values
    output[:, 3, 3] = 1.0
    return output


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    shape = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(shape + (9,)), dim=-1
    )
    magnitude = torch.sqrt(torch.clamp(torch.stack([
        1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22,
    ], dim=-1), min=0.0))
    candidates = torch.stack([
        torch.stack([magnitude[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, magnitude[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, magnitude[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, magnitude[..., 3] ** 2], dim=-1),
    ], dim=-2)
    floor = torch.tensor(0.1, dtype=magnitude.dtype, device=magnitude.device)
    candidates = candidates / (2.0 * magnitude[..., None].max(floor))
    quaternion = candidates[
        F.one_hot(magnitude.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(shape + (4,))
    quaternion = quaternion[..., [1, 2, 3, 0]]
    return torch.where(quaternion[..., 3:4] < 0, -quaternion, quaternion)


def _inverse_se3(matrix: torch.Tensor) -> torch.Tensor:
    rotation = matrix[:, :3, :3]
    translation = matrix[:, :3, 3:]
    rotation_t = rotation.transpose(1, 2)
    output = torch.eye(4, dtype=matrix.dtype, device=matrix.device)[None].repeat(len(matrix), 1, 1)
    output[:, :3, :3] = rotation_t
    output[:, :3, 3:] = -torch.bmm(rotation_t, translation)
    return output


def pose_errors(predicted: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted_tensor = torch.from_numpy(pad_extrinsics(predicted)).double()
    target_tensor = torch.from_numpy(pad_extrinsics(target)).double()
    first, second = torch.combinations(torch.arange(len(target_tensor)), 2, with_replacement=False).unbind(-1)
    relative_target = target_tensor[second].bmm(_inverse_se3(target_tensor[first]))
    relative_predicted = predicted_tensor[second].bmm(_inverse_se3(predicted_tensor[first]))
    target_q = _matrix_to_quaternion(relative_target[:, :3, :3])
    predicted_q = _matrix_to_quaternion(relative_predicted[:, :3, :3])
    rot_loss = (1.0 - (predicted_q * target_q).sum(dim=1) ** 2).clamp(min=1e-15)
    rotation = torch.arccos(1.0 - 2.0 * rot_loss) * 180.0 / np.pi
    t_target = relative_target[:, :3, 3]
    t_pred = relative_predicted[:, :3, 3]
    t_target = t_target / (t_target.norm(dim=1, keepdim=True) + 1e-15)
    t_pred = t_pred / (t_pred.norm(dim=1, keepdim=True) + 1e-15)
    t_loss = torch.clamp_min(1.0 - torch.sum(t_pred * t_target, dim=1) ** 2, 1e-15)
    translation = torch.acos(torch.sqrt(1.0 - t_loss)) * 180.0 / np.pi
    translation = torch.minimum(translation, (180.0 - translation).abs())
    return rotation.numpy(), translation.numpy()


def _auc(rotation: np.ndarray, translation: np.ndarray, threshold: int) -> float:
    errors = np.maximum(rotation, translation)
    histogram, _ = np.histogram(errors, bins=np.arange(threshold + 1))
    return float(np.mean(np.cumsum(histogram.astype(float) / len(errors))) * 100.0)


def pose_metrics(rotation: np.ndarray, translation: np.ndarray) -> dict[str, float]:
    return {"AUC@3": _auc(rotation, translation, 3), "AUC@30": _auc(rotation, translation, 30)}


def depth_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    delta, abs_rel = [], []
    for pf, tf in zip(predicted, target):
        tf = tf.astype(np.float64, copy=True)
        pf = pf.astype(np.float64, copy=True)
        mask = np.isfinite(tf) & (tf > 0.1) & (tf < 100.0)
        if mask.sum() < 10:
            continue
        tf = np.clip(tf, 0.1, 100.0)
        pf = np.clip(pf, 0.1, 100.0)
        tc = tf[mask] - np.median(tf[mask]) + 1e-8
        pc = pf[mask] - np.median(pf[mask]) + 1e-8
        scale = np.median(tc / pc)
        shift = np.median(tf[mask] - scale * pf[mask])
        aligned = np.clip((pf * scale + shift)[mask], 1e-6, None)
        tv = tf[mask]
        abs_rel.append(float(np.mean(np.abs(aligned - tv) / tv)))
        ratio = np.maximum(aligned / tv, tv / aligned)
        delta.append(float(np.mean(ratio < 1.25)))
    if not delta:
        return {"delta125": float("nan"), "AbsRel": float("nan")}
    return {"delta125": float(np.mean(delta)), "AbsRel": float(np.mean(abs_rel))}


def crop_depth(depth: np.ndarray, crop: tuple[float, float, float, float]) -> np.ndarray:
    left, top, right, bottom = crop
    height, width = depth.shape[-2:]
    x0, x1 = round(left * width), round(right * width)
    y0, y1 = round(top * height), round(bottom * height)
    return np.ascontiguousarray(depth[:, y0:y1, x0:x1])


def resize_depth(predicted: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if predicted.shape[1:3] == shape:
        return predicted
    tensor = torch.from_numpy(np.ascontiguousarray(predicted)).float().unsqueeze(1)
    resized = F.interpolate(tensor, shape, mode="bilinear", align_corners=False, antialias=True)
    return resized.squeeze(1).numpy()


# ---------- dataset loading (matches official eval/datasets.py) ----------

def _read_depth(path: Path) -> np.ndarray:
    compressed = path.with_suffix(".npz")
    if compressed.is_file():
        with np.load(compressed) as archive:
            return archive["depth"].astype(np.float32)
    array = path.with_suffix(".npy")
    if array.is_file():
        return np.load(array).astype(np.float32)
    raise FileNotFoundError(compressed)


class Dataset:
    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def load(self, scene: str, indices: list[int]):
        root = self.root / scene
        names = [line.strip() for line in (root / "frames.txt").read_text().splitlines()]
        with np.load(root / "cameras.npz") as cameras:
            extrinsics = cameras["extrinsics"]
        selected_names = [names[i] for i in indices]
        image_paths = [str(root / "images" / name) for name in selected_names]
        depths = []
        for name in selected_names:
            stem = Path(name).stem
            depths.append(_read_depth(root / "depths" / stem))
        return image_paths, np.stack(depths), extrinsics[indices], selected_names


# ---------- predictor ----------

def _crop_box(width: int, height: int):
    aspect = height / width
    if aspect < 0.5:
        crop_width = min(width, max(1, round(height / 0.5)))
        left = max((width - crop_width) // 2, 0)
        return left / width, 0.0, (left + crop_width) / width, 1.0
    if aspect > 2.0:
        crop_height = min(height, max(1, round(width * 2.0)))
        top = max((height - crop_height) // 2, 0)
        return 0.0, top / height, 1.0, (top + crop_height) / height
    return 0.0, 0.0, 1.0, 1.0


def _source_crop(image_paths: list[str]):
    from PIL import Image
    sizes = []
    for path in image_paths:
        with Image.open(path) as image:
            sizes.append(image.size)
    crops = [_crop_box(w, h) for w, h in sizes]
    return crops[0]


class Predictor:
    def __init__(self, checkpoint: str, device: str = "cuda"):
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.pose_enc import encoding_to_camera
        self._encoding_to_camera = encoding_to_camera
        self.device = device
        state = torch.load(Path(checkpoint).expanduser().resolve(), map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            state = state.get("model", state.get("state_dict", state))
        self.model = VGGTOmega()
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.eval().to(device)

    def predict(self, image_paths: list[str]):
        import warnings
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        crop = _source_crop(image_paths)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            images = load_and_preprocess_images(image_paths, mode="max_size", image_resolution=512, patch_size=16)
        images = images.to(self.device)
        with torch.inference_mode():
            prediction = self.model(images)
        extrinsics, _ = self._encoding_to_camera(prediction["pose_enc"], prediction["images"].shape[-2:])
        depth = prediction["depth"].detach().float().cpu().numpy()[0]
        if depth.ndim == 4:
            depth = depth[..., 0]
        poses = extrinsics.detach().float().cpu().numpy()[0]
        return depth.astype(np.float32), pad_extrinsics(poses), crop


# ---------- main ----------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("eth3d", "sintel"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--frame-map", default=None, help="Path to frame map JSON (default: bundled)")
    args = parser.parse_args()

    started = time.time()

    if args.frame_map:
        frame_map = json.loads(Path(args.frame_map).read_text())
    else:
        frame_map = json.loads((Path(__file__).parent / "frames" / f"{args.dataset}.json").read_text())
    if args.max_scenes is not None:
        frame_map = dict(list(frame_map.items())[:args.max_scenes])

    dataset = Dataset(args.data_root)
    predictor = Predictor(args.checkpoint, args.device)
    rotations, translations = [], []
    per_scene = {}

    for scene, indices in frame_map.items():
        image_paths, target_depth, gt_extrinsics, frame_names = dataset.load(scene, indices)
        predicted_depth, predicted_extrinsics, crop = predictor.predict(image_paths)
        target_depth_cropped = crop_depth(target_depth, crop)
        predicted_depth = resize_depth(predicted_depth, target_depth_cropped.shape[-2:])
        depth = depth_metrics(predicted_depth, target_depth_cropped)
        rotation, translation = pose_errors(predicted_extrinsics, gt_extrinsics)
        rotations.append(rotation)
        translations.append(translation)
        per_scene[scene] = {
            **depth,
            "rotation_mean": float(rotation.mean()),
            "translation_mean": float(translation.mean()),
        }
        print(f"{scene}: delta125={depth['delta125'] * 100:.2f}% AbsRel={depth['AbsRel']:.4f} "
              f"rot={rotation.mean():.2f} trans={translation.mean():.2f}")

    rotation = np.concatenate(rotations)
    translation = np.concatenate(translations)
    camera = pose_metrics(rotation, translation)

    summary = {
        "AUC@3": camera["AUC@3"],
        "AUC@30": camera["AUC@30"],
        "delta125": float(np.mean([r["delta125"] for r in per_scene.values()])) * 100.0,
        "AbsRel": float(np.mean([r["AbsRel"] for r in per_scene.values()])),
    }

    output = {
        "checkpoint": Path(args.checkpoint).name,
        "dataset": args.dataset,
        "elapsed_seconds": round(time.time() - started, 1),
        "environment": {
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "metrics": summary,
        "per_scene": per_scene,
        "pose_pairs": int(rotation.size),
        "scenes": len(frame_map),
    }

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
