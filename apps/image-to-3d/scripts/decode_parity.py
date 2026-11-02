"""JPEG decode parity with PIL (spec-train 8.4).

Compares the decodes written by the Rust test
`io::dataset_tests::garden_decode_dump_for_pil_parity` with PIL's, for the
same files. Run on the GPU host in any image with numpy and PIL:

    docker run --rm --user "$(id -u):$(id -g)" -v ~/spl-work/out:/out \
      -v ~/spl-work/datasets:/datasets:ro -v "$PWD/apps/image-to-3d/scripts:/s:ro" \
      --entrypoint "" pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime \
      python -I /s/decode_parity.py

Rule: if more than 0.1% of samples differ by more than 1 LSB, the decoder is
not acceptable for the benchmark path.
"""

import sys

import numpy as np
from PIL import Image

OUT = "/out/image-to-3d/decode_parity"
IMAGES = "/datasets/mipnerf360/garden/images_4"


def compare(npz_name: str) -> bool:
    z = np.load(f"{OUT}/{npz_name}")
    total = differ = over1 = 0
    worst = 0
    for name in sorted(z.files):
        ours = z[name].astype(np.int16)
        pil = np.asarray(Image.open(f"{IMAGES}/{name}").convert("RGB")).astype(np.int16)
        assert ours.shape == pil.shape, (name, ours.shape, pil.shape)
        d = np.abs(ours - pil)
        total += d.size
        differ += int((d > 0).sum())
        over1 += int((d > 1).sum())
        worst = max(worst, int(d.max()))
    ok = over1 <= 0.001 * total
    print(
        f"{npz_name}: {len(z.files)} images, {total} samples; differ {100 * differ / total:.3f}%, "
        f">1 LSB {100 * over1 / total:.4f}%, max {worst} -> {'OK' if ok else 'NOT OK'} (PIL {Image.__version__})"
    )
    return ok


if __name__ == "__main__":
    results = [compare(n) for n in sys.argv[1:] or ["garden.npz", "garden_zune.npz"]]
    sys.exit(0 if results[0] else 1)
