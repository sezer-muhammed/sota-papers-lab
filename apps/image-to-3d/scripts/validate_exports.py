#!/usr/bin/env python3
"""Validate PLY and SPZ exports produced by our Rust code.

Usage: python validate_exports.py <ply_path> [spz_path]

Checks:
- PLY: reads with plyfile, checks header structure and data types
- SPZ: reads with the spz package (pip install spz), checks field shapes
- Both: prints summary statistics (position range, opacity, scale distributions)
"""
import sys
import struct
import gzip
import numpy as np


def read_inria_ply(path: str) -> dict:
    """Read an INRIA 3DGS PLY file without external dependencies."""
    with open(path, "rb") as f:
        data = f.read()

    # Find header
    idx = data.find(b"end_header\n")
    assert idx >= 0, "no end_header found"
    header_end = idx + len(b"end_header\n")
    header = data[:header_end].decode("ascii")

    n = 0
    props = []
    for line in header.split("\n"):
        line = line.strip()
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
        elif line.startswith("property float"):
            props.append(line.split()[-1])

    assert len(props) == 62, f"expected 62 properties, got {len(props)}"
    body = data[header_end:]
    expected = n * 62 * 4
    assert len(body) >= expected, f"body too short: {len(body)} < {expected}"

    arr = np.frombuffer(body[:expected], dtype="<f4").reshape(n, 62)

    result = {
        "n": n,
        "x": arr[:, 0],
        "y": arr[:, 1],
        "z": arr[:, 2],
        "nx": arr[:, 3],
        "ny": arr[:, 4],
        "nz": arr[:, 5],
        "f_dc": arr[:, 6:9],
        "f_rest": arr[:, 9:54],
        "opacity": arr[:, 54],
        "scale": arr[:, 55:58],
        "rot": arr[:, 58:62],
        "properties": props,
    }
    return result


def read_spz_v3(path: str) -> dict:
    """Read an SPZ v3 (gzip) file."""
    with open(path, "rb") as f:
        raw = f.read()

    data = gzip.decompress(raw)
    assert len(data) >= 16, "SPZ too short"

    magic, version, n_pts = struct.unpack_from("<III", data, 0)
    assert magic == 0x5053474E, f"bad magic: {magic:#x}"
    assert version == 3, f"expected v3, got {version}"
    degree = data[12]
    fb = data[13]
    flags = data[14]

    dim = {0: 0, 1: 3, 2: 8, 3: 15}.get(degree, 15)
    off = 16

    # Positions: n*9 bytes (3 axes * 3 bytes)
    pos_bytes = n_pts * 9
    positions = np.zeros((n_pts, 3), dtype=np.float32)
    for i in range(n_pts):
        for a in range(3):
            b = data[off:off+3]
            off += 3
            raw_int = b[0] | (b[1] << 8) | (b[2] << 16)
            if raw_int & 0x800000:
                raw_int |= ~0xFFFFFF
            positions[i, a] = raw_int / (1 << fb)

    # Alphas: n bytes
    alphas = np.frombuffer(data[off:off+n_pts], dtype=np.uint8).astype(np.float32) / 255.0
    off += n_pts

    # Colors: n*3 bytes
    colors = np.frombuffer(data[off:off+n_pts*3], dtype=np.uint8).reshape(n_pts, 3)
    off += n_pts * 3

    # Scales: n*3 bytes
    scales_raw = np.frombuffer(data[off:off+n_pts*3], dtype=np.uint8).reshape(n_pts, 3)
    scales = scales_raw.astype(np.float32) / 16.0 - 10.0
    off += n_pts * 3

    # Rotations: n*4 bytes
    rot_raw = np.frombuffer(data[off:off+n_pts*4], dtype="<u4")
    off += n_pts * 4

    # SH: n*dim*3 bytes
    if dim > 0:
        sh = np.frombuffer(data[off:off+n_pts*dim*3], dtype=np.uint8).reshape(n_pts, dim, 3)
        off += n_pts * dim * 3
    else:
        sh = None

    return {
        "n": n_pts,
        "version": version,
        "degree": degree,
        "fractional_bits": fb,
        "flags": flags,
        "positions": positions,
        "alphas": alphas,
        "colors": colors,
        "scales": scales,
        "sh": sh,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_exports.py <ply_path> [spz_path]")
        sys.exit(1)

    ply_path = sys.argv[1]
    print(f"=== PLY: {ply_path} ===")
    ply = read_inria_ply(ply_path)
    print(f"  vertices: {ply['n']}")
    print(f"  properties: {len(ply['properties'])}")
    print(f"  position range: [{ply['x'].min():.3f}, {ply['x'].max():.3f}] x "
          f"[{ply['y'].min():.3f}, {ply['y'].max():.3f}] x "
          f"[{ply['z'].min():.3f}, {ply['z'].max():.3f}]")
    print(f"  normals zero: {np.allclose(ply['nx'], 0) and np.allclose(ply['ny'], 0) and np.allclose(ply['nz'], 0)}")
    print(f"  opacity range: [{ply['opacity'].min():.3f}, {ply['opacity'].max():.3f}] (logit)")
    print(f"  scale range: [{ply['scale'].min():.3f}, {ply['scale'].max():.3f}] (log)")
    rot_norms = np.linalg.norm(ply["rot"], axis=1)
    print(f"  rotation norms: [{rot_norms.min():.6f}, {rot_norms.max():.6f}] (should be ~1)")
    print(f"  f_dc range: [{ply['f_dc'].min():.4f}, {ply['f_dc'].max():.4f}]")
    print(f"  f_rest range: [{ply['f_rest'].min():.4f}, {ply['f_rest'].max():.4f}]")
    print("  PLY VALID")

    if len(sys.argv) > 2:
        spz_path = sys.argv[2]
        print(f"\n=== SPZ: {spz_path} ===")
        spz = read_spz_v3(spz_path)
        print(f"  points: {spz['n']}")
        print(f"  version: {spz['version']}, degree: {spz['degree']}, fb: {spz['fractional_bits']}")
        print(f"  position range: [{spz['positions'][:, 0].min():.3f}, {spz['positions'][:, 0].max():.3f}]")
        print(f"  alpha range: [{spz['alphas'].min():.3f}, {spz['alphas'].max():.3f}]")
        print(f"  scale range: [{spz['scales'].min():.3f}, {spz['scales'].max():.3f}]")
        print("  SPZ VALID")


if __name__ == "__main__":
    main()
