# App contract

An app is a crate in `apps/<name>/` that exports `register(&mut Registry)`
and one or more ops. Create one with:

```sh
scripts/new-app.sh imgcodec "Learned image compression from <paper>"
```

This copies `templates/app/`, adds the crate to the workspace and to
`catalog/` behind a feature of the same name. The template contains a
working GPU op (NVRTC-compiled kernel, upload, launch, verify, metrics).

## Layout

```
apps/<name>/
├── Cargo.toml
├── README.md        ops, inputs, how to run alone
├── paper.toml       paper metadata + claimed results (paper apps only)
├── cuda/*.cu        kernels, extern "C"
├── src/lib.rs       register() + ops
├── tests/           CPU-only tests run everywhere; GPU tests behind #[ignore]
└── ui/              optional custom page for web/shell (later)
```

## An op

```rust
pub struct Encode;

impl Op for Encode {
    const ID: &'static str = "imgcodec.encode";
    const SUMMARY: &'static str = "Encode an image to a bitstream";
    // Optional; `GpuUse::Shared` when omitted. See "Sharing the GPU".
    const GPU: GpuUse = GpuUse::Shared;

    type Input = EncodeInput;    // Deserialize + JsonSchema
    type Output = EncodeOutput;  // Serialize + JsonSchema

    async fn run(&self, cx: &RunCx, input: EncodeInput) -> Result<EncodeOutput> {
        let image = cx.artifacts.get(&input.image.id).await?;
        let bitstream = tokio::task::spawn_blocking(move || encode_on_gpu(&image)).await??;
        let out = cx.artifacts.put(bitstream, ArtifactKind::Bitstream, "application/octet-stream").await?;
        cx.metric("bpp", bpp, "bit/px");
        Ok(EncodeOutput { bitstream: out })
    }
}
```

## Rules

- **IDs:** `<app_snake>.<verb>`, lowercase. Never reuse or repurpose an ID once runs reference it; add `encode_v2` instead.
- **Inputs:** parameters have serde defaults where sensible so `{}` is a valid smoke-test input. Doc comments on fields become schema descriptions in the UI.
- **Large data:** take and return `ArtifactRef`, never byte arrays or base64 in JSON. Inputs arrive as artifact ids (uploaded with `POST /v1/artifacts` or `spl put <file>`). Everything stored with `cx.artifacts.put` is recorded and linked to the run as an output; the op does nothing extra.
- **Big files:** an output written to disk (a video's frames, a 500 MB `.ply`) goes in with `cx.artifacts.put_file(&path, kind, media_type)`, which moves the file into the store (hashing it in place) instead of loading it. A big input (a 2 GB phone video) is read where it lies: `cx.artifacts.local_path(&id)` gives the file to stream, memory-map or hand to `ffmpeg`; fall back to `get` when it is `None` (a store without local files). Never write to that path.
- **Kinds and media types:** pick the closest `ArtifactKind` and a media type from `spl_core::media`. 3D results are `splat` (`media::SPZ` `application/x-spz`, or `media::PLY` `application/ply` in the INRIA 3DGS layout), `mesh` (`media::GLB` `model/gltf-binary`, or PLY) and `cameras` (`media::JSON`, or `media::ZIP` for a COLMAP model). Downloads get the matching extension. Never serve SPZ with `Content-Encoding`: it is gzip inside, and the viewer unpacks it.
- **Blocking work:** CUDA calls and heavy CPU work go in `spawn_blocking`. Long loops check `cx.is_cancelled()` (blocking closures take `cx.cancel.clone()`) and report `cx.progress(…)`.
- **Cancellation:** `POST /v1/runs/{id}/cancel` fires `cx.cancel`. Return soon after, `Ok` or `Err`: once cancelled, the run ends as `cancelled` whatever the op returns. Child processes must die with it: `select!` on `cx.cancel.cancelled()` and use `kill_on_drop(true)`.
- **Progress and logs:** `cx.progress` and `cx.log` show live in `GET /v1/runs/{id}` (the latest progress, the last 50 log lines) and in the server log. `cx.progress_with(fraction, message, json!({"stage": "refine", "iteration": 7000, "gaussians": 1.2e6}))` adds structured state a custom page renders (a stage bar, counters). None of it is stored; put what must last in metrics, announcements or the output.
- **Metrics:** report every number a benchmark might plot with `cx.metric(key, value, unit)`. Use `""` for unitless. They are saved when the run ends, failed or not; NaN and infinite values are dropped.
- **Intermediate results:** to show a result before the run ends, store it and announce it under a slot: `cx.announce("cameras", &cameras, json!({"count": 200}))`, `cx.announce("points", &cloud, Value::Null)`. `GET /v1/runs/{id}` lists the latest artifact per slot in `announced`, with a `revision` that grows on each announcement, while the run goes and after it ends; the web SDK's `watchRun(id, { onAnnounce })` calls back once per new artifact. Previews that replace each other (a splat every ~5 s of training) use `cx.announce_snapshot("splat", &snapshot, json!({"iteration": i}))`: when the run ends, the host deletes the snapshots a later one superseded (only bytes this run created, not named in its output, not used elsewhere) and keeps the latest. Keep snapshots small (SPZ, low SH degree), since each one is a stored artifact until the run ends. `info` is small JSON (16 KiB at most); slots are short names (200 characters at most).
- **Sharing the GPU:** up to `max_concurrent_runs` runs (default 3) execute at once, possibly the same op, so size GPU memory and caches for that. An op that needs the card to itself (most of its memory, or clean timings for a benchmark) declares `const GPU: GpuUse = GpuUse::Exclusive;`: the host starts it once the GPU ops ahead of it end and starts no GPU op until it returns, so it may size itself to the whole card. An op that never touches the GPU declares `GpuUse::CpuOnly` and keeps running beside an exclusive one. The default is `GpuUse::Shared`. Free big GPU allocations before returning: the next run starts as soon as the op returns. Per-host caches go under the `DataDir` resource (`cx.resources.get::<DataDir>()`).
- **Errors:** return `Err` with context (`.context("…")`); the host records it as a failed run. Don't panic: the host survives it and records `op panicked: …`, but the message is worse.
- **No cross-app imports.** Shared helpers go to `platform/` (or a new platform crate) once two apps need them.
- **Kernels:** CUDA C++ in `cuda/`, loaded with `spl_gpu::load_cuda_source(&ctx, include_str!(…))`. Target compute capability 7.0+.
