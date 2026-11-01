# Architecture

## Goals

1. Many independent apps, each a paper re-implementation or a helper (metrics, baselines).
2. Every app testable **alone**, **connected** to other apps, and **under benchmark**, without app-specific glue.
3. Same code on a local Linux GPU box and on a cloud GPU VM; macOS for editing and type-checking only.
4. Records that survive a move from SQLite to PostgreSQL or Oracle Autonomous Database.

## Layers

```
front doors     web/shell (UI) · spl CLI · HTTP clients · (later) remote spl hosts
                        │ one API: /v1/ops, /v1/runs, /v1/artifacts
runtime         platform/spl-host   registry · run lifecycle · metrics · auth · static UI
                        │ Op trait (spl-core)
apps            apps/*              typed ops + CUDA kernels; no knowledge of each other
                        │
services        spl-gpu (cudarc) · spl-data (artifacts) · spl-db (Store trait)
storage         SQLite → Postgres / Oracle ADB · local disk → S3-compatible
```

| Crate | Owns |
|---|---|
| `spl-core` | `Op` trait, `RunCx`, `Registry`, `ArtifactRef`/`ArtifactKind`, run events. No CUDA, no SQL. |
| `spl-gpu` | Device discovery, contexts, runtime kernel compilation (NVRTC). Loads CUDA at runtime, so it compiles everywhere. |
| `spl-data` | `ArtifactStore` implementations: content-addressed local disk (blake3). S3 via `object_store` next. |
| `spl-db` | `Store` trait + SQLite backend; schema migrations for SQLite, PostgreSQL and Oracle. |
| `spl-host` | Running ops (queued, async, cancellable), recording runs, metrics, artifact lineage and environment, the HTTP API, config. |
| `spl-catalog` | Which apps a binary links (one Cargo feature per app). |
| `spl-cli` | The `spl` binary. |

## How apps connect

- **Contract:** an app implements `Op` for each operation: a stable ID, typed input and output (JSON Schema derived automatically), and `run(&RunCx, input)`. See [APP_CONTRACT.md](APP_CONTRACT.md).
- **Data:** large data moves as `ArtifactRef`s (content hash + kind + media type), never inline. `ArtifactKind` (`image`, `tensor`, `bitstream`, `table`, `report`, `blob`) is the shared vocabulary; add kinds in `spl-core`, not in apps.
- **Composition:** apps never call each other's Rust code. A pipeline (or a benchmark) calls ops by ID through the registry, wiring one op's output artifacts into the next op's input.
- **Universal interface:** because every op has a schema, every host exposes every op the same way (HTTP, CLI, UI form) with zero per-app code. An app adds a custom UI page only when it wants richer views.

Example: a new image codec.

| Mode | How |
|---|---|
| Alone | `spl run imgcodec.encode --input '{"image": …, "quality": 0.4}'` |
| Connected | pipeline: load → `imgcodec.encode` → `imgcodec.decode` → `metrics.compare` |
| Benchmark | suite: that pipeline × Kodak × quality sweep → RD curves next to `paper.toml` claims and baseline codecs |

## Runs and the host API

Every front door goes through `spl-host`, which executes each run in a task
of its own: a client that hangs up never strands a run, and a panicking op
becomes a failed run. At most `max_concurrent_runs` ops execute at once (one
GPU is shared); later runs wait as `queued` and start in submission order.

| Route | Does |
|---|---|
| `POST /v1/runs` `{op, input}` | Runs to completion: 200 with the finished `RunRecord` (a failed op is still 200) |
| `POST /v1/runs` `{op, input, wait: false}` | 202 at once with the run `running` or `queued` |
| `GET /v1/runs/{id}` | `RunDetail`. While the run goes: latest progress, last 50 log lines, metrics so far. Afterwards: saved metrics |
| `POST /v1/runs/{id}/cancel` | 202; the op sees `cx.is_cancelled()` and the run ends `cancelled` (a queued run never starts). 404 unknown, 409 not running |
| `POST /v1/artifacts` | Raw body; `content-type` is the media type, `x-spl-filename` (percent-encoded) is recorded as `meta.filename`. 201 with the `ArtifactRef`; 413 over `max_upload_mb` |
| `GET /v1/artifacts/{id}` | The bytes, cacheable forever (ETag = id). `?download=<name>` answers as an attachment |

Runs a previous process left unfinished are marked `failed` when
`spl serve` starts. `spl put <file>` stores a file the same way an upload does.

## Reproducibility

Every run stores its input, output, status, timings, metrics, and an
environment snapshot (OS, architecture, GPU model, compute capability,
driver version, `spl` version, git SHA when built with `SPL_GIT_SHA`).
Every artifact an op stores is recorded and linked to its run as an output
(`out-0`, `out-1`, ...), so lineage needs nothing from the op.

## Roadmap

Done: op contract, registry, SQLite store with portable schema, local artifact store, HTTP API, CLI, web shell, `device_info` app, app template, async runs (queue, live progress and logs, cancellation), uploads, output lineage.

Next, roughly in order:

1. **Run events:** WebSocket `/v1/runs/{id}/events` streaming `RunEvent`s instead of polling; input lineage.
2. **First real apps:** `metrics` (PSNR, SSIM, MS-SSIM on GPU), `baseline-codecs`, then the first paper app.
3. **Pipelines:** TOML definitions in `pipelines/`, GPU-resident artifact hand-off for in-process steps.
4. **Benchmarks:** `spl-bench` (warm-up, repeats, CUDA-event timing, sweeps), `paper.toml` claim comparison, RD-curve views in the shell.
5. **Storage:** S3-compatible `ArtifactStore`; PostgreSQL `Store`; Oracle ADB `Store` ([DATABASE.md](DATABASE.md)).
6. **Remote ops:** a `RemoteOp` that forwards to another `spl` host, so one pipeline can span the local box and a cloud GPU.
7. **AOT kernels:** nvcc fat binaries per target architecture in `build.rs`, alongside NVRTC.
8. **Desktop shell (optional):** Tauri 2 wrapping `web/shell` for a GPU workstation.
