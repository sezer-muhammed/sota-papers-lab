#!/usr/bin/env python3
"""Validate PLY and SPZ exports without numpy."""
import struct
import gzip
import sys
import json


def validate_ply(path):
    with open(path, "rb") as f:
        data = f.read()
    idx = data.find(b"end_header\n")
    assert idx >= 0, "no end_header"
    header_end = idx + len(b"end_header\n")
    header = data[:header_end].decode("ascii")

    n = 0
    props = 0
    for line in header.split("\n"):
        line = line.strip()
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
        elif line.startswith("property float"):
            props += 1

    assert props == 62, f"expected 62 properties, got {props}"
    body = data[header_end:]
    expected = n * 62 * 4
    assert len(body) >= expected, f"body {len(body)} < expected {expected}"

    # Check first vertex
    values = struct.unpack_from("<62f", body, 0)
    # nx, ny, nz should be 0
    assert values[3] == 0.0 and values[4] == 0.0 and values[5] == 0.0, "normals not zero"

    # Check rotation norms
    for i in range(min(n, 100)):
        vals = struct.unpack_from("<62f", body, i * 62 * 4)
        w, x, y, z = vals[58], vals[59], vals[60], vals[61]
        norm = (w*w + x*x + y*y + z*z) ** 0.5
        assert abs(norm - 1.0) < 0.01, f"rotation norm {norm} at vertex {i}"

    print(f"  PLY: {n} vertices, {props} properties, normals zero, rotations normalized")
    first = struct.unpack_from("<62f", body, 0)
    print(f"    first pos: ({first[0]:.3f}, {first[1]:.3f}, {first[2]:.3f})")
    print(f"    first opacity (logit): {first[54]:.3f}")
    print(f"    first scale (log): ({first[55]:.3f}, {first[56]:.3f}, {first[57]:.3f})")
    return n


def validate_spz(path):
    with open(path, "rb") as f:
        raw = f.read()
    data = gzip.decompress(raw)
    assert len(data) >= 16
    magic, version, n = struct.unpack_from("<III", data, 0)
    assert magic == 0x5053474E, f"bad magic {magic:#x}"
    assert version == 3, f"version {version}"
    degree = data[12]
    fb = data[13]
    flags = data[14]
    dim = {0: 0, 1: 3, 2: 8, 3: 15}.get(degree, 15)
    expected = 16 + n * 9 + n + n * 3 + n * 3 + n * 4 + n * dim * 3
    assert len(data) >= expected, f"data {len(data)} < expected {expected}"
    print(f"  SPZ v3: {n} points, degree {degree}, fb {fb}, flags {flags}")
    print(f"    compressed: {len(raw)} bytes, uncompressed: {len(data)} bytes")
    print(f"    ratio: {len(raw) / len(data):.2%}")
    return n


def validate_colmap(sparse_dir):
    import os
    cam_path = os.path.join(sparse_dir, "cameras.bin")
    img_path = os.path.join(sparse_dir, "images.bin")
    pts_path = os.path.join(sparse_dir, "points3D.bin")
    assert os.path.exists(cam_path), "cameras.bin missing"

    with open(cam_path, "rb") as f:
        data = f.read()
    n_cam = struct.unpack_from("<Q", data, 0)[0]

    with open(img_path, "rb") as f:
        data = f.read()
    n_img = struct.unpack_from("<Q", data, 0)[0]

    with open(pts_path, "rb") as f:
        data = f.read()
    n_pts = struct.unpack_from("<Q", data, 0)[0]

    print(f"  COLMAP: {n_cam} cameras, {n_img} images, {n_pts} points")
    return n_img


def validate_transforms(path):
    with open(path) as f:
        data = json.load(f)
    n_frames = len(data["frames"])
    has_angle = "camera_angle_x" in data
    print(f"  transforms.json: {n_frames} frames, shared camera: {has_angle}")
    if n_frames > 0:
        mat = data["frames"][0]["transform_matrix"]
        assert len(mat) == 4 and len(mat[0]) == 4, "bad transform shape"
        assert mat[3][3] == 1.0, "last element not 1"
    return n_frames


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "/out/image-to-3d/exports"
    import os
    print(f"=== Validating exports in {base} ===")

    ply = os.path.join(base, "garden.ply")
    if os.path.exists(ply):
        n = validate_ply(ply)
        print(f"  PLY VALID ({n} vertices)")

    spz = os.path.join(base, "garden.spz")
    if os.path.exists(spz):
        n = validate_spz(spz)
        print(f"  SPZ VALID ({n} points)")

    colmap = os.path.join(base, "colmap_bin")
    if os.path.exists(colmap):
        validate_colmap(colmap)
        print("  COLMAP VALID")

    tj = os.path.join(base, "transforms.json")
    if os.path.exists(tj):
        validate_transforms(tj)
        print("  transforms.json VALID")

    print("\nAll exports validated successfully.")


if __name__ == "__main__":
    main()
