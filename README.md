# SOTA Papers Lab

A workspace for re-implementing state-of-the-art papers as GPU apps in
**Rust + CUDA**, testing each one alone, chaining them together, and
benchmarking them against the numbers the papers claim.

- **One contract:** every app exposes typed *ops* (`imgcodec.encode`, `metrics.compare`, …).
- **One host:** the `spl` binary serves every op through one HTTP API, records every run in a database, and serves the web UI.
- **Runs anywhere with an NVIDIA GPU:** a local Linux machine or a cloud VM, from the same Docker image. Development and type-checking work on macOS too.

```
 Web UI ─┐   ┌──────────────── spl host ────────────────┐
 CLI ────┼──▶│ registry · runs · pipelines · benchmarks │──▶ apps/* (Rust + CUDA)
 HTTP ───┘   │ SQLite (→ Postgres / Oracle ADB)         │
             │ artifact store (disk → S3)               │
             └──────────────────────────────────────────┘
```

## Layout

| Path | What |
|---|---|
| `platform/` | Shared foundation. `spl-core` (the op contract), `spl-gpu` (CUDA via cudarc), `spl-data` (artifacts), `spl-db` (records), `spl-host` (runtime + API) |
| `apps/` | One crate per app. Apps never depend on each other |
| `catalog/` | The single list of apps a host links in |
| `bins/spl/` | The `spl` binary: `serve`, `ops`, `run` |
| `web/` | `shell` (UI), `ui` (design system), `sdk` (API client), `bindings` (generated types) |
| `templates/app/` | Starting point for a new app (`scripts/new-app.sh`) |
| `pipelines/`, `benchmarks/` | Op chains and benchmark suites |
| `deploy/` | Dockerfile, compose file, configs for local and cloud |
| `docs/` | Architecture, app contract, database, design, deployment |

## Quick start

On the GPU machine (Linux, NVIDIA driver):

```sh
docker compose -f deploy/compose.yaml up -d --build   # or: just serve
curl localhost:7070/v1/health
```

From anywhere:

```sh
cargo run -p spl-cli -- ops                       # list ops
cargo run -p spl-cli -- run device_info.list      # run one, recorded in ./data
SPL_API=http://gpu-box:7070 pnpm dev              # UI with hot reload against a GPU host
```

New app: `scripts/new-app.sh imgcodec "Learned image compression from <paper>"`.

## Docs

- [Architecture](docs/ARCHITECTURE.md): layers, how apps connect, roadmap
- [App contract](docs/APP_CONTRACT.md): writing an app and its ops
- [Database](docs/DATABASE.md): schema rules that keep the PostgreSQL and Oracle ADB moves cheap
- [Deploy](docs/DEPLOY.md): local Linux, cloud, the Mac workflow
- [Design](docs/DESIGN.md): the UI design language (shared with ATSGrade)
