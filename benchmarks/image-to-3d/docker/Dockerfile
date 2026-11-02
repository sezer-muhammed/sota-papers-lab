# spl-gs-ref: the Python reference stack for the image-to-3d harness.
#   - metrics (PSNR, SSIM, LPIPS exactly as 3DGS / gsplat evaluate)
#   - the gsplat oracle (gs-oracle) for comparing our CUDA kernels
#   - the gsplat baseline trainer (examples/simple_trainer.py)
#
# Build with ../install.sh --image (context: benchmarks/image-to-3d).
#
# Disk: the runtime is the official PyTorch 2.9.1 / CUDA 12.8 image (Python
# 3.11), shareable with anything else built on it; this image adds about
# 2.4 GB. The CUDA extensions are compiled in a throwaway stage that borrows
# only nvcc and the headers (about 0.4 GB) from the shared CUDA 12.8 devel
# image, against the runtime's own torch and Python, for sm_89 only.
ARG RUNTIME=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
ARG DEVEL=spl-cuda-devel

FROM ${DEVEL} AS cuda

FROM ${RUNTIME} AS build
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ARG CU=/usr/local/cuda-12.8
COPY --from=cuda ${CU}/bin ${CU}/bin
COPY --from=cuda ${CU}/nvvm ${CU}/nvvm
COPY --from=cuda ${CU}/targets/x86_64-linux/include ${CU}/targets/x86_64-linux/include
COPY --from=cuda ${CU}/targets/x86_64-linux/lib/libcudart* ${CU}/targets/x86_64-linux/lib/
RUN ln -s targets/x86_64-linux/include ${CU}/include && ln -s targets/x86_64-linux/lib ${CU}/lib64 \
 && pip install ninja
ENV CUDA_HOME=${CU} PATH=${CU}/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=16

ARG GSPLAT_REF=937e29912570c372bed6747a5c9bf85fed877bae
RUN git init -q /src/gsplat && cd /src/gsplat \
 && git remote add origin https://github.com/nerfstudio-project/gsplat.git \
 && git fetch -q --depth 1 origin ${GSPLAT_REF} && git checkout -q FETCH_HEAD \
 && git submodule update -q --init --recursive --depth 1 \
 && pip wheel --no-build-isolation --no-deps -w /wheels . \
 && mkdir -p /opt/gsplat && cp -r examples /opt/gsplat/examples
ARG FUSED_SSIM_REF=a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38
RUN git init -q /src/fused-ssim && cd /src/fused-ssim \
 && git remote add origin https://github.com/rahul-goel/fused-ssim.git \
 && git fetch -q --depth 1 origin ${FUSED_SSIM_REF} && git checkout -q FETCH_HEAD \
 && CUDA_ARCHITECTURES=89 pip wheel --no-build-isolation --no-deps -w /wheels .

FROM ${RUNTIME}
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# Metrics and gsplat example dependencies (examples/requirements.txt at the
# pinned commit, minus video and the optional bilateral-grid package).
# constraints.txt pins the rest (install.sh --freeze from the last good build).
COPY docker/constraints.txt /opt/spl-bench/constraints.txt
COPY --from=build /wheels /tmp/wheels
RUN pip install -c /opt/spl-bench/constraints.txt /tmp/wheels/*.whl \
      "numpy<2" torchmetrics lpips imageio scipy scikit-learn tqdm "tyro>=0.8.8" tensorboard tensorly \
      pyyaml matplotlib splines viser opencv-python-headless \
      "pycolmap @ https://github.com/rmbrualla/pycolmap/archive/cc7ea4b7301720ac29287dbe450952511b32125e.zip" \
      "nerfview @ https://github.com/nerfstudio-project/nerfview/archive/4538024fe0d15fd1a0e4d760f3695fc44ca72787.zip" \
 && rm -rf /tmp/wheels \
 && cd / && python -c "import gsplat, gsplat.csrc, importlib.util as u; assert u.find_spec('fused_ssim_cuda'); print('gsplat', gsplat.__version__)"
COPY --from=build /opt/gsplat/examples /opt/gsplat/examples

# LPIPS backbones (VGG16 and AlexNet ImageNet weights) baked in, so metrics
# run offline and every run uses the same files.
ENV TORCH_HOME=/opt/torch-cache
RUN python -c "import torchvision as tv; tv.models.vgg16(weights='IMAGENET1K_V1'); tv.models.alexnet(weights='IMAGENET1K_V1')" \
 && python -c "from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as L; L(net_type='vgg'); L(net_type='alex')" \
 && python -c "import lpips; lpips.LPIPS(net='vgg', verbose=False)" \
 && chmod -R a+rX /opt/torch-cache

# Our scripts last, so editing them only rebuilds this layer.
COPY py/ /opt/spl-bench/
RUN pip freeze > /opt/spl-bench/pip-freeze.txt \
 && python -m compileall -q /opt/spl-bench \
 && chmod -R a+rX /opt/spl-bench /opt/gsplat
ENV PYTHONPATH=/opt/spl-bench HOME=/tmp MPLCONFIGDIR=/tmp
WORKDIR /tmp
