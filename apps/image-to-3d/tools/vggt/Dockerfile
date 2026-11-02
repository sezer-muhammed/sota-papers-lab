# spl-vggt-ref: reference VGGT-Ω inference image.
# Build:  docker build -t spl-vggt-ref -f Dockerfile .
# Run:    docker run --rm --gpus all -v ~/spl-work/models:/models:ro spl-vggt-ref ...
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN pip install --no-cache-dir \
        safetensors==0.5.3 \
        einops==0.8.1 \
        opencv-python-headless==4.11.0.86 \
        huggingface_hub==0.30.2

# Pin the official repo at a known commit (download tarball, no git needed).
# Install the package AND keep the eval/ directory for dataset preparation.
ARG VGGT_COMMIT=48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6
ADD https://github.com/facebookresearch/vggt-omega/archive/${VGGT_COMMIT}.tar.gz /tmp/vggt.tar.gz
RUN tar xzf /tmp/vggt.tar.gz -C /opt && \
    mv /opt/vggt-omega-${VGGT_COMMIT} /opt/vggt-omega && \
    pip install --no-cache-dir /opt/vggt-omega && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless==4.11.0.86 && \
    rm /tmp/vggt.tar.gz

WORKDIR /workspace
