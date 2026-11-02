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

# Triton autotuner needs gcc at runtime for JIT compilation.
RUN apt-get update -qq && apt-get install -y -qq gcc g++ >/dev/null 2>&1 && rm -rf /var/lib/apt/lists/*

# Extension wheels built on the host (FlexGEMM, CuMesh, o-voxel, nvdiffrast).
COPY wheels/ /tmp/wheels/

RUN pip install --no-deps /tmp/wheels/*.whl \
 && pip install \
      https://github.com/SHI-Labs/NATTEN/releases/download/v0.21.5/natten-0.21.5+torch290cu128-cp311-cp311-linux_x86_64.whl \
      https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl \
      "transformers>=4.56" easydict trimesh opencv-python-headless imageio zstandard kornia timm safetensors \
      diffusers accelerate plyfile tqdm huggingface_hub scipy einops \
 && rm -rf /tmp/wheels

# Pixal3D code (sparse checkout: code only, no hdri/assets).
COPY pixal3d/ /opt/pixal3d/

# Patch Pixal3D source for transformers 5.x and our rembg-free setup.
RUN sed -i 's/for layer_module in self\.model\.layer:/for layer_module in self.model.model.layer:/' \
      /opt/pixal3d/pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py \
 && python3 -c "\
import re, pathlib; \
p = pathlib.Path('/opt/pixal3d/pixal3d/pipelines/pixal3d_image_to_3d.py'); \
t = p.read_text(); \
t = t.replace( \
    \"pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])\", \
    \"pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args']) if 'rembg_model' in args else None\"); \
p.write_text(t)"

# Our sidecar scripts.
COPY sidecar/ /opt/spl-p3d/

WORKDIR /opt/pixal3d
