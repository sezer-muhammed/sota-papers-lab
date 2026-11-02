# Deploy

The same image runs on a local Linux GPU machine and on a cloud GPU VM.
Nothing GPU-related runs on macOS; there you edit, type-check and run the UI.

## Requirements on a GPU host

- Linux with an NVIDIA driver for CUDA 12.8 or newer (`nvidia-smi` works).
- Docker + NVIDIA Container Toolkit, **or** Rust (via `rustup`) for bare-metal runs.
- GPU compute capability 7.0 (Volta) or newer; image-to-3d ops need 8.0+ (Ampere).

## Local Linux machine

```sh
git clone https://github.com/sezer-muhammed/sota-papers-lab && cd sota-papers-lab
SPL_GIT_SHA=$(git rev-parse --short HEAD) docker compose -f deploy/compose.yaml up -d --build
curl localhost:7070/v1/health
```

Bare metal: `pnpm install && pnpm build && SPL_WEB_DIR=web/shell/dist cargo run --release -p spl-cli -- serve`.

### Tailnet-only (no token)

Reachable only from devices on your Tailscale network; nothing is published on the LAN:

```sh
cd deploy
printf 'COMPOSE_FILE=compose.yaml:compose.tailnet.yaml\nSPL_TAILNET_IP=%s\n' "$(tailscale ip -4)" > .env
SPL_GIT_SHA=$(git rev-parse --short HEAD) docker compose up -d --build
```

Open `http://<machine>.<tailnet>.ts.net:7070`. Update with `git pull && docker compose up -d --build`
(Rust and pnpm caches persist between builds).

Data (SQLite file + artifacts) lives in `./data` (bind-mounted to `/var/lib/spl`).

When `spl serve` starts, it marks runs that the previous process left
`queued` or `running` as `failed` ("host restarted"): nothing would ever
finish them. It stops promptly on SIGTERM (`docker stop`).

### Claude inside the container (opt-in)

Ops that ask Claude for judgments (`raw_retouch.auto`) run the `claude` CLI on
the GPU host user's subscription login. `deploy/compose.claude.yaml` runs the
container as that user, with the same `HOME`, and mounts only what the native
CLI needs at the same absolute paths: `~/.claude` read-write (credentials are
refreshed there in place, shared with the host's CLI), `~/.claude.json`
read-only, `~/.local/share/claude` and `~/.local/bin` read-only. The file
comments explain each mount.

Once, because the root-run container created `data/` as root: build the new
image, stop the service so nothing writes during the change, and hand the
data to your user.

```sh
cd deploy
SPL_GIT_SHA=$(git rev-parse --short HEAD) docker compose build
docker compose stop spl
docker run --rm -v "$PWD/../data:/d" busybox chown -R "$(id -u):$(id -g)" /d
```

Then add the override to `COMPOSE_FILE` in `deploy/.env` and start it:

```sh
# deploy/.env (tailnet setup; without it: compose.yaml:compose.claude.yaml)
COMPOSE_FILE=compose.yaml:compose.tailnet.yaml:compose.claude.yaml
# Only when `id -u` / `id -g` print something other than 1000:
# SPL_UID=1001
# SPL_GID=1001
```

```sh
docker compose up -d
docker compose exec -w /tmp spl claude -p "Reply OK" --model sonnet --output-format json --tools ""
```

Run `docker compose` as that user, never with `sudo`: the override takes
`HOME` from your shell. It sets `SPL_CLAUDE_BIN=$HOME/.local/bin/claude`,
puts `$HOME/.local/bin` on `PATH` and sets `DISABLE_AUTOUPDATER=1` (the host's
CLI updates the mounted binaries). Do not set `ANTHROPIC_API_KEY`: it would
switch the CLI to billed API use. To go back to running as root, drop
`compose.claude.yaml` from `COMPOSE_FILE`; root can still use the data.

## image-to-3d deployment

The `image_to_3d.train` and `image_to_3d.render` ops are GPU-exclusive: each
run holds the GPU lock as a writer, so no other GPU op runs alongside it.

### Runtime dependencies in the image

The `cuda:12.8.1-runtime` base provides everything the splat trainer and the
VGGT-Ω engine need at runtime:

| Library | Used by | Already in runtime image? |
|---|---|---|
| NVRTC (`libnvrtc.so`) | Kernel compilation (cudarc) | Yes |
| cuBLAS / cuBLASLt | VGGT-Ω GEMMs | Yes |
| CUDA runtime (`libcudart.so`) | cudarc driver API | Yes |
| ffmpeg / ffprobe | Video frame extraction (Phase 4) | Added by Dockerfile |

Kernels use no `#include` (types and PTX wrappers are inlined in
`common.cuh`), so the `devel` image's headers are not needed and the image
stays small.

### VGGT-Ω model weights

**Do not bake weights into the image.** The VGGT-Ω weights (~11 GB) are
licensed under Meta's FAIR non-commercial research licence and must not be
redistributed in a Docker image.

Mount them read-only from the host:

```sh
# deploy/.env
SPL_MODELS_HOST=/path/to/models    # contains vggt-omega/
SPL_DATASETS_HOST=/path/to/datasets # contains mipnerf360/ etc.
```

The compose file mounts `$SPL_MODELS_HOST` at `/models` (read-only) and sets
`SPL_MODELS_DIR=/models` in the container. The engine loads weights from
`$SPL_MODELS_DIR/vggt-omega/`.

### Datasets

Datasets can be provided two ways:

1. **Host directory** (bind-mounted read-only): set `SPL_DATASETS_HOST` in
   `deploy/.env` to the host path; compose mounts it at `/data/datasets` and
   sets `SPL_DATASETS=/data/datasets`. Ops reference datasets by relative path
   (e.g. `mipnerf360/garden`).
2. **Uploaded zip**: `POST /v1/artifacts` a zip with `sparse/0/` and images,
   then pass its artifact id as `zip` in the train input.

### Quick deploy (local GPU machine)

```sh
cd deploy
cat > .env <<'EOF'
SPL_DATASETS_HOST=$HOME/spl-work/datasets
SPL_MODELS_HOST=$HOME/spl-work/models
EOF

# Build (with tailnet, models, and claude as needed):
COMPOSE_FILE=compose.yaml:compose.tailnet.yaml
SPL_GIT_SHA=$(git rev-parse --short HEAD) docker compose up -d --build

# Verify:
curl -s localhost:7070/v1/health | jq .
curl -s localhost:7070/v1/ops | jq '.[].id'
```

### End-to-end test

```sh
# Upload a dataset zip:
ARTIFACT=$(curl -s -X POST http://localhost:7070/v1/artifacts \
  -H 'content-type: application/zip' \
  -H 'x-spl-kind: blob' \
  --data-binary @garden-tiny.zip | jq -r .id)

# Run training:
curl -s -X POST http://localhost:7070/v1/runs \
  -H 'content-type: application/json' \
  -d "{\"op\": \"image_to_3d.train\", \"input\": {\"zip\": \"$ARTIFACT\", \"train\": {\"preset\": \"fast\"}}}" \
  | jq .

# Download the .spz:
SPZ_ID=$(curl ... | jq -r .output.spz.id)
curl -o scene.spz http://localhost:7070/v1/artifacts/$SPZ_ID
```

## Cloud GPU VM

1. Build and push the image (`just docker`, then tag for your registry).
2. Run it with `--gpus all`, `SPL_CONFIG=/etc/spl/cloud.toml`, a persistent disk at `/var/lib/spl`, and **`SPL_AUTH_TOKEN` set**.
3. Mount model weights read-only; set `SPL_MODELS_DIR`.
4. Put TLS in front (Caddy, or the provider's load balancer).

Later: artifacts on S3-compatible storage and the database on PostgreSQL or Oracle ADB, so cloud hosts can be stateless.

## From the Mac

```sh
ssh -L 7070:localhost:7070 gpu-box      # or use Tailscale and the box's name
open http://localhost:7070              # the built UI, served by the host
SPL_API=http://localhost:7070 pnpm dev  # or the UI with hot reload
```

## Configuration

`SPL_CONFIG` points at a TOML file (`deploy/config/*.toml`); environment variables override it:

| Variable | Meaning | Default |
|---|---|---|
| `SPL_BIND` | listen address | `127.0.0.1:7070` |
| `SPL_DATA_DIR` | data root (artifacts under `artifacts/`) | `./data` |
| `SPL_DATABASE_URL` | `sqlite://…` (later `postgres://…`, `oracle://…`) | `sqlite://./data/spl.db` |
| `SPL_WEB_DIR` | built UI to serve | unset |
| `SPL_AUTH_TOKEN` | bearer token required on `/v1/*` (not `/v1/health`) | unset |
| `SPL_MAX_CONCURRENT_RUNS` | ops executing at once; further runs wait as `queued`, in order | `3` |
| `SPL_MAX_UPLOAD_MB` | largest `POST /v1/artifacts` body and largest chunked upload (`/v1/uploads`) file, in MiB. Uploads stream to `<data_dir>/staging` and are renamed into the artifact store, so they cost disk, not memory; keep both on one filesystem | `2048` |
| `SPL_CLAUDE_BIN` | `claude` CLI that ops call; else `PATH`, then `~/.local/bin/claude` | unset (`compose.claude.yaml` sets it) |
| `SPL_DATASETS` | root for dataset folders (ops use relative paths) | `<data_dir>/datasets` |
| `SPL_MODELS_DIR` | root for model weights (e.g. VGGT-Ω) | unset |
