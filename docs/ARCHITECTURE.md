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
| `spl-core` | `Op` trait (with its `GpuUse`), `RunCx`, `Registry`, `ArtifactRef`/`ArtifactKind`, media types, run events. No CUDA, no SQL. |
| `spl-gpu` | Device discovery, contexts, runtime kernel compilation (NVRTC). Loads CUDA at runtime, so it compiles everywhere. |
| `spl-data` | `ArtifactStore` implementations: content-addressed local disk (blake3; big files are hashed in place on every core and renamed in, never loaded). S3 via `object_store` next. |
| `spl-db` | `Store` trait + SQLite backend; schema migrations for SQLite, PostgreSQL and Oracle. |
| `spl-host` | Running ops (queued, async, cancellable, GPU-aware scheduling), recording runs, metrics, announcements, artifact lineage and environment, the HTTP API, config. |
| `spl-catalog` | Which apps a binary links (one Cargo feature per app). |
| `spl-cli` | The `spl` binary. |

## How apps connect

- **Contract:** an app implements `Op` for each operation: a stable ID, typed input and output (JSON Schema derived automatically), and `run(&RunCx, input)`. See [APP_CONTRACT.md](APP_CONTRACT.md).
- **Data:** large data moves as `ArtifactRef`s (content hash + kind + media type), never inline. `ArtifactKind` (`image`, `tensor`, `bitstream`, `table`, `report`, `blob`, and for 3D `splat`, `mesh`, `cameras`) is the shared vocabulary, and `spl_core::media` names the media types that go with it (`application/x-spz`, `application/ply`, `model/gltf-binary`, ...); add kinds and media types in `spl-core`, not in apps.
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
becomes a failed run.

**Scheduling.** One GPU is shared, so the host admits runs through two fair
(first-in, first-out) locks (`spl-host/src/queue.rs`):

- **Run slots:** at most `max_concurrent_runs` ops execute at once.
- **The GPU lock:** every op declares `Op::GPU`. `shared` ops (the default) hold it as readers, so up to the slot limit run together. An `exclusive` op (one that needs most of the 12 GB, or clean benchmark timings) holds it as the writer: it starts once the GPU ops ahead of it end, and no GPU op starts until it ends. `cpu_only` ops skip the GPU lock and run even beside an exclusive op.

Waiting runs are `queued`, start in submission order (a queued exclusive run
holds back the shared runs submitted after it, so it is never starved), and
`GET /v1/runs/{id}` says whether one waits for the `gpu` or a `slot`.
Exclusivity holds inside one host process: a separate `spl run` does not
coordinate with a running `spl serve`.

| Route | Does |
|---|---|
| `GET /v1/health` | Version, op count, environment, and `max_upload_bytes` so a page can refuse a too-large file before sending it. No token needed |
| `GET /v1/ops` | Every op's descriptor: summary, `gpu` use, input and output JSON Schema |
| `POST /v1/runs` `{op, input}` | Runs to completion: 200 with the finished `RunRecord` (a failed op is still 200) |
| `POST /v1/runs` `{op, input, wait: false}` | 202 at once with the run `running` or `queued` |
| `GET /v1/runs/{id}` | `RunDetail`. While the run goes: latest progress (with the op's structured `data`), last 50 log lines, metrics so far, and while queued `waiting_for`. Afterwards: saved metrics. Always: `announced`, the latest artifact per slot the op announced |
| `POST /v1/runs/{id}/cancel` | 202; the op sees `cx.is_cancelled()` and the run ends `cancelled` (a queued run never starts). 404 unknown, 409 not running |
| `DELETE /v1/runs/{id}` | 204; deletes a finished run, its links, announcements and metrics, and the artifacts nothing else references. 404 unknown, 409 queued or running |
| `POST /v1/artifacts` | Raw body, **streamed to disk** (`<data_dir>/staging`, then renamed into the store), so memory does not grow with the upload. `content-type` is the media type, `x-spl-filename` (percent-encoded) is recorded as `meta.filename`, `x-spl-kind` (`splat`, `mesh`, ...) the kind (`blob` when absent; 400 when unknown). 201 with the `ArtifactRef`; 413 over `max_upload_mb` (default 2 GiB), before reading the body when `content-length` already says so |
| `POST /v1/uploads` | Opens a resumable chunked upload, for files a proxy would refuse in one request (Cloudflare stops at 100 MB): JSON `{size, filename?, media_type?, kind?, part_bytes?}` (parts of 256 KiB to 64 MiB, 16 MiB by default). Preallocates a staging file; 201 with the `UploadSession` (`id`, `parts`, `received`); 413 over `max_upload_mb`; 429 past 32 open sessions. A session idle for an hour is dropped with its file |
| `PUT /v1/uploads/{id}/parts/{n}` | Raw body, exactly part `n`'s bytes, streamed to its offset; any order, in parallel. A part sent again replaces itself; 400 on a wrong length, 409 while the same part is being written |
| `GET /v1/uploads/{id}` | The session with the parts the host holds, so a client resumes after a reload or a dropped connection |
| `POST /v1/uploads/{id}/complete` | Moves the file into the store like `POST /v1/artifacts`: 201 with the `ArtifactRef`; 409 listing missing parts. `DELETE /v1/uploads/{id}` drops a session (204) |
| `GET /v1/artifacts/{id}` | The bytes, streamed from disk, cacheable forever (ETag = id). One byte range (`Range: bytes=a-b`, `a-`, `-n`) answers 206 with `Content-Range`, so video players seek; past the end 416; `If-Range` with another tag gets the whole body; several ranges get the whole body. `?download=<name>` answers as an attachment; with no name, the recorded file name, else `<kind>-<id12>.<ext>` from the media type (`splat-1a2b3c4d5e6f.spz`) |

### Intermediate results

An op shows results before it ends by announcing artifacts it stored
(`cx.announce(slot, &artifact, info)`): `points` within seconds of posing,
then a `splat` snapshot every few seconds of training. The host keeps the
latest artifact per slot (`spl_run_announcements`) with a revision that grows
on each announcement, and returns them in `RunDetail.announced` while the run
goes and after it ends (a cancelled run keeps its last snapshot). Clients
fetch an artifact again only when its revision grew; `SplClient.watchRun`
does that polling and calls `onAnnounce` once per new artifact.

Snapshots (`cx.announce_snapshot`) replace each other: when the run ends the
host deletes the ones a later snapshot of the same slot superseded, but only
bytes this run created, that its output does not name, that no slot shows,
that were never announced to be kept, and that no other run links or
announces. The latest snapshot of each slot stays.

Runs a previous process left unfinished are marked `failed` when
`spl serve` starts. `spl put <file> [--kind splat]` stores a file the same
way an upload does (copied through the staging directory, never read into
memory).

## Reproducibility

Every run stores its input, output, status, timings, metrics, and an
environment snapshot (OS, architecture, GPU model, compute capability,
driver version, `spl` version, git SHA when built with `SPL_GIT_SHA`).
Every artifact an op stores is recorded and linked to its run as an output
(`out-0`, `out-1`, ...), so lineage needs nothing from the op.

## Roadmap

Done: op contract, registry, SQLite store with portable schema, local artifact store, HTTP API, CLI, web shell, `device_info` app, app template, async runs (queue, live progress and logs, cancellation), uploads, output lineage, GPU-exclusive scheduling, announced intermediate artifacts with snapshot pruning, streamed uploads (2 GiB) and ranged downloads, 3D artifact kinds.

Next, roughly in order:

1. **Run events:** WebSocket `/v1/runs/{id}/events` streaming `RunEvent`s (announcements included) instead of polling; input lineage. VRAM leases, finer than `exclusive`, if two big ops ever need to share the card.
2. **First real apps:** `metrics` (PSNR, SSIM, MS-SSIM on GPU), `baseline-codecs`, then the first paper app.
3. **Pipelines:** TOML definitions in `pipelines/`, GPU-resident artifact hand-off for in-process steps.
4. **Benchmarks:** `spl-bench` (warm-up, repeats, CUDA-event timing, sweeps), `paper.toml` claim comparison, RD-curve views in the shell.
5. **Storage:** S3-compatible `ArtifactStore`; PostgreSQL `Store`; Oracle ADB `Store` ([DATABASE.md](DATABASE.md)).
6. **Remote ops:** a `RemoteOp` that forwards to another `spl` host, so one pipeline can span the local box and a cloud GPU.
7. **AOT kernels:** nvcc fat binaries per target architecture in `build.rs`, alongside NVRTC.
8. **Desktop shell (optional):** Tauri 2 wrapping `web/shell` for a GPU workstation.
