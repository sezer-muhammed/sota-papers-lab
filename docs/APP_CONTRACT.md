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
- **Large data:** take and return `ArtifactRef`, never byte arrays or base64 in JSON.
- **Blocking work:** CUDA calls and heavy CPU work go in `spawn_blocking`. Long loops check `cx.is_cancelled()` and report `cx.progress(…)`.
- **Metrics:** report every number a benchmark might plot with `cx.metric(key, value, unit)`. Use `""` for unitless.
- **Errors:** return `Err` with context (`.context("…")`); the host records it as a failed run. Don't panic.
- **No cross-app imports.** Shared helpers go to `platform/` (or a new platform crate) once two apps need them.
- **Kernels:** CUDA C++ in `cuda/`, loaded with `spl_gpu::load_cuda_source(&ctx, include_str!(…))`. Target compute capability 7.0+.
