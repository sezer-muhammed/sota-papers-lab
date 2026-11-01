# Pipelines

Declarative chains of ops, e.g. `codec-rd.toml`:
load image → `imgcodec.encode` → `imgcodec.decode` → `metrics.compare`.
Steps pass artifacts by reference; in-process steps keep data on the GPU.

The pipeline runner in `spl-host` is on the roadmap (docs/ARCHITECTURE.md).
