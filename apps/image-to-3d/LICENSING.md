# image-to-3d: licences and the commercial profile

Every non-commercial (or restricted) dependency of the image-to-3d pipeline,
where it appears, and how the **commercial profile** avoids it. Licences were
read from the model cards, repositories and papers (`reports/image-to-3d/spec-commercial.md`
§1, §7, §9; 1-2 Nov 2026). This is an engineering reading of licences, not
legal advice: a lawyer signs off before anything ships commercially.

## The switch

`image_to_3d.pose` takes `"profile": "research" | "commercial"` (default
`research`; `src/engine/mod.rs`, `LicenceProfile`).

| | Research | Commercial |
|---|---|---|
| Stage A engine | native VGGT-Ω 1B (`src/vggt/`) | DA3-BASE 0.12B on ONNX Runtime (`src/engine/da3.rs`) |
| Weights licence | FAIR Noncommercial Research License v1 (covers outputs) | Apache-2.0 (`depth-anything/DA3-BASE`) |
| Loader check | none | the model directory's `manifest.json` licence must be one of `apache-2.0`, `bsd-2-clause`, `bsd-3-clause`, `mit`, else the run fails before any weights load |
| Recorded in the run output | `engine`, `licence` | `engine`, `licence`, `weights_sha256` (SHA-256 of the source `model.safetensors`) |

Switching profiles unloads the other profile's engine, so a commercial run
never has VGGT-Ω weights resident.

## Non-commercial and restricted dependencies

| Dependency | Licence | Where it would appear | Commercial profile |
|---|---|---|---|
| VGGT-Ω weights (`vggt_omega_1b_512.pt`) | FAIR Noncommercial Research License v1; covers outputs | research Stage A engine (`src/vggt/`, `image_to_3d.pose` default) | Not loaded (`profile: commercial` runs DA3-BASE). Its outputs (poses, depth, COLMAP models, point clouds from research runs) must not feed a commercial job either. |
| DINOv3 (VGGT-Ω initialisation; Pixal3D, spec-pixal3d) | custom DINOv3 License (commercial use with conditions / AUP) | inside VGGT-Ω; Stage E if built | DA3-BASE uses DINOv2 ViT-B (Apache-2.0). Stage E stays off in the commercial profile until legal clears DINOv3. |
| VGGT-1B, DA3-LARGE / GIANT / NESTED (incl. 1.1), Pi3X, MapAnything (non-apache), DVLT | CC-BY-NC-4.0 / FAIR NC | could be exported as engines or teachers | Blocked by the manifest licence allow-list. Never export them with `tools/onnx/`. |
| `facebook/VGGT-1B-Commercial` | gated custom `vggt-aup-license` | not used | Not used (gated approval, AUP needs legal review). |
| Inria 3D Gaussian Splatting and derived code (diff-gaussian-rasterization, Taming-3DGS, FastGS, 3DGS-MCMC, Speedy-Splat, StopThePop, PGSR) | Inria / MPII non-commercial | trainer and rasterizer **ideas** | Read only, never copied. Kernels are written from the specs; derived code may come only from Apache/MIT sources (Faster-GS, gsplat, fused-ssim) with a `// Derived from …` note (`DESIGN.md`). Files that name Inria-licensed methods (`cuda/raster_bwd.cu` "Taming's per-Gaussian backward as refined by Faster-GS", `cuda/densify.cu` FastGS VCD/VCP, `cuda/mcmc.cu` 3DGS-MCMC formulas, presets `fastgs`, `fastgs_big`, `mcmc`) cite them as algorithms; legal review should confirm the algorithm-versus-code line before shipping. |
| Baseline containers (`spl-bl-fastgs`, `spl-bl-fastergs`, `spl-bl-brush-lichtfeld`, `spl-bl-colmapfree`) | mixed, some Inria | benchmarks only | Measurement only; never part of a build or image we ship. |
| nvdiffrast | NVIDIA Source Code License (non-commercial) | texture bake (design doc Stage D) | Not used in any profile; our own CUDA bake. |
| Datasets: Mip-NeRF 360, Deep Blending, Tanks and Temples, DTU, ETH3D, Sintel, DL3DV, CO3D | research / academic / CC BY-NC | benchmarks, evaluation subsets, tests | Internal evaluation only. Never shipped, never used to train or fine-tune anything shipped; demo assets must be our own captures. |
| DA3-BASE training data | "public academic datasets" (model card) | inside the DA3-BASE weights | Weights are published Apache-2.0. Open legal question shared with Pi3: whether academic / NC training sets affect permissively licensed weights. Cleanest-provenance fallback: MapAnything-apache (six owner-approved datasets), not yet integrated. |

## Permissive dependencies of the commercial path

| Dependency | Licence | Notes |
|---|---|---|
| DA3 code (`ByteDance-Seed/Depth-Anything-3`) | Apache-2.0 | used only by `tools/onnx/export_da3.py` at export time; the graphs ship, the Python does not |
| DA3-BASE weights | Apache-2.0 | `manifest.json` records the source and its SHA-256 |
| DINOv2 (inside DA3-BASE) | Apache-2.0 | |
| ONNX Runtime 1.23 (`libonnxruntime.so`, CUDA EP) | MIT | loaded at runtime from `$ORT_DYLIB_PATH` |
| `ort` crate 2.0.0-rc.15 | MIT OR Apache-2.0 | `platform/spl-ort` |
| CUDA 12.8 runtime libraries, cuBLAS, cuDNN 9 | NVIDIA EULA / cuDNN SLA (redistributable parts) | ship only the redistributable libraries, per their terms |
| Faster-GS, gsplat | Apache-2.0 | trainer kernels may derive from them with provenance notes |
| fused-ssim | MIT | loss design |
| INRIA `.ply` layout | file format | export only |

## Rules for contributors

- A new model front end gets a `manifest.json` with an SPDX `licence`; the
  commercial loader enforces it. Never relabel a licence in a manifest.
- No file in `apps/image-to-3d` may cite Inria-licensed code as the source of
  copied code; cite the paper and write the kernel from the spec.
- Do not cache or reuse research-profile outputs in commercial runs.
