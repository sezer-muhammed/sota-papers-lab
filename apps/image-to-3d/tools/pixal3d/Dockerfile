# spl-pixal3d: Pixal3D-mv sidecar for object generation.
# Derived from pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime (already on host).
#
# Build:
#   docker build -t spl-pixal3d -f apps/image-to-3d/tools/pixal3d/Dockerfile \
#     apps/image-to-3d/tools/pixal3d/
#
# Weights are mounted read-only at /weights, not baked in.

FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ATTN_BACKEND=sdpa \
    SPARSE_ATTN_BACKEND=sdpa \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    TRANSFORMERS_CACHE=/tmp/hf_cache

# Extension wheels built on the host (FlexGEMM, CuMesh, o-voxel, nvdiffrast).
COPY wheels/ /tmp/wheels/

RUN pip install --no-deps /tmp/wheels/*.whl \
 && pip install \
      https://github.com/SHI-Labs/NATTEN/releases/download/v0.21.5/natten-0.21.5+torch290cu128-cp311-cp311-linux_x86_64.whl \
      https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl \
      "transformers>=4.56" easydict trimesh opencv-python-headless imageio zstandard kornia timm safetensors \
      diffusers accelerate plyfile tqdm huggingface_hub scipy \
 && rm -rf /tmp/wheels

# Pixal3D code (sparse checkout: code only, no hdri/assets).
COPY pixal3d/ /opt/pixal3d/

# Our sidecar scripts.
COPY sidecar/ /opt/spl-p3d/

WORKDIR /opt/pixal3d
