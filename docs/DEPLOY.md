# Deploy

The same image runs on a local Linux GPU machine and on a cloud GPU VM.
Nothing GPU-related runs on macOS; there you edit, type-check and run the UI.

## Requirements on a GPU host

- Linux with an NVIDIA driver for CUDA 12.8 or newer (`nvidia-smi` works).
- Docker + NVIDIA Container Toolkit, **or** Rust (via `rustup`) for bare-metal runs.
- GPU compute capability 7.0 (Volta) or newer.

## Local Linux machine

```sh
git clone https://github.com/sezer-muhammed/sota-papers-lab && cd sota-papers-lab
SPL_GIT_SHA=$(git rev-parse --short HEAD) docker compose -f deploy/compose.yaml up -d --build
curl localhost:7070/v1/health
```

Bare metal: `pnpm install && pnpm build && SPL_WEB_DIR=web/shell/dist cargo run --release -p spl-cli -- serve`.

Data (SQLite file + artifacts) lives in `./data` (bind-mounted to `/var/lib/spl`).

## Cloud GPU VM

1. Build and push the image (`just docker`, then tag for your registry).
2. Run it with `--gpus all`, `SPL_CONFIG=/etc/spl/cloud.toml`, a persistent disk at `/var/lib/spl`, and **`SPL_AUTH_TOKEN` set**.
3. Put TLS in front (Caddy, or the provider's load balancer).

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
