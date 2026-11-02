# syntax=docker/dockerfile:1
# One image for every GPU host: local Linux box or cloud VM.
#   docker build -f deploy/docker/Dockerfile -t sota-papers-lab .
# Needs the NVIDIA Container Toolkit on the host to see GPUs.
#
# Building needs no CUDA: cudarc loads the driver at runtime and kernels are
# compiled by NVRTC when an op first runs. Only the final stage carries CUDA
# libraries. Switch the build stage to nvidia/cuda:*-devel once an app
# compiles kernels ahead of time with nvcc.
#
# The cuda:*-runtime base provides: NVRTC (runtime kernel compilation),
# cuBLAS / cuBLASLt (VGGT-Ω engine GEMMs), and the CUDA runtime.
# Kernels use no #include (types and PTX wrappers are inlined in common.cuh),
# so the devel image's headers are not needed.
ARG CUDA_VERSION=12.8.1
ARG UBUNTU=ubuntu24.04

FROM node:24-bookworm-slim AS web
WORKDIR /src
RUN corepack enable
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY web ./web
RUN --mount=type=cache,id=spl-pnpm,target=/pnpm-store \
    pnpm config set store-dir /pnpm-store \
 && pnpm install --frozen-lockfile \
 && pnpm build

FROM rust:1.99-bookworm AS rust
WORKDIR /src
COPY . .
ARG SPL_GIT_SHA=unknown
ENV SPL_GIT_SHA=${SPL_GIT_SHA}
# Cache the registry and target dir across builds; copy the binary out of the
# cache mount so the next stage can see it.
RUN --mount=type=cache,id=spl-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=spl-cargo-target,target=/src/target \
    rustup toolchain install \
 && cargo build --release --locked -p spl-cli \
 && cp target/release/spl /usr/local/bin/spl

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU}
# ffmpeg/ffprobe: video frame extraction for image-to-3d (Phase 4).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
COPY --from=rust /usr/local/bin/spl /usr/local/bin/spl
COPY --from=web /src/web/shell/dist /opt/spl/web
COPY deploy/config /etc/spl
ENV SPL_CONFIG=/etc/spl/local.toml \
    SPL_WEB_DIR=/opt/spl/web \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
VOLUME /var/lib/spl
EXPOSE 7070
CMD ["spl", "serve"]
