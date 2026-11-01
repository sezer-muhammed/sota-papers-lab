# raw-retouch

Automatic RAW photo retouching: develop, search Lightroom-style sliders, judge with Claude, export JPEG and XMP

| | |
|---|---|
| Ops | `raw_retouch.scale` |
| Kernels | `cuda/kernels.cu` (NVRTC at runtime) |
| Paper | see `paper.toml` |

Run alone: `spl run raw_retouch.scale --input '{"n": 1048576}'`
