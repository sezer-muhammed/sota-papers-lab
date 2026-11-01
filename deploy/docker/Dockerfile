# One image for every GPU host: local Linux box or cloud VM.
#   docker build -f deploy/docker/Dockerfile -t sota-papers-lab .
# Needs the NVIDIA Container Toolkit on the host to see GPUs.
ARG CUDA_VERSION=12.8.1
ARG UBUNTU=ubuntu24.04

FROM node:24-bookworm-slim AS web
WORKDIR /src
RUN corepack enable
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY web ./web
RUN pnpm install --frozen-lockfile && pnpm build

FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS rust
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates build-essential pkg-config \
 && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
WORKDIR /src
COPY . .
ARG SPL_GIT_SHA=unknown
ENV SPL_GIT_SHA=${SPL_GIT_SHA}
# Installs the toolchain pinned in rust-toolchain.toml, then builds.
RUN rustup toolchain install && cargo build --release --locked -p spl-cli

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU}
COPY --from=rust /src/target/release/spl /usr/local/bin/spl
COPY --from=web /src/web/shell/dist /opt/spl/web
COPY deploy/config /etc/spl
ENV SPL_CONFIG=/etc/spl/local.toml \
    SPL_WEB_DIR=/opt/spl/web \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
VOLUME /var/lib/spl
EXPOSE 7070
CMD ["spl", "serve"]
