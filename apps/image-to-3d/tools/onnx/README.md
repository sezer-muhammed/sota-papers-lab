# ONNX front ends (commercial profile)

Exporters and checks for the commercial-profile engines of `image_to_3d.pose`
(`src/engine/`, `LICENSING.md`). Only Apache / BSD / MIT models go through
here; the engine refuses any other `manifest.json` licence.

| File | Does |
|---|---|
| `export_da3.py` | DA3-BASE → `backbone_<W>x<H>.onnx` + `head_<W>x<H>.onnx` per resolution bucket, one shared fp16/fp32 weights file, `manifest.json` |
| `check_parity.py` | the graphs on ONNX Runtime vs the PyTorch model (fp32), with the official bf16 path as the noise floor |
| `da3_official.py` | the official DA3 PyTorch forward: `eval` dumps predictions on the vggt-ref subsets, `time` measures speed and memory at N frames |
| `PARITY.md` | results |

## Setup on the GPU host

Everything runs in existing images (`spl-gs-ref` for PyTorch, the CUDA 12.8
runtime image for the Rust engine); nothing is installed into them.

```sh
C=~/spl-work/commercial
git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3 $C/src/Depth-Anything-3   # Apache-2.0
# Python extras next to the image's PyTorch (no new image):
docker run --rm --user $(id -u):$(id -g) -v $C:$C --entrypoint "" spl-gs-ref \
  pip install --target $C/py onnxruntime-gpu==1.23.2 onnx einops omegaconf addict imageio huggingface_hub==0.36.0
# cuDNN 9 for the ONNX Runtime CUDA EP (the runtime image has cuBLAS but no cuDNN):
docker run --rm --user $(id -u):$(id -g) -v $C/cudnn:/o --entrypoint bash spl-gs-ref -c \
  'cd /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib && cp libcudnn.so.9 libcudnn_{cnn,engines_precompiled,engines_runtime_compiled,graph,heuristic,ops}.so.9 /o/'
# Weights (Apache-2.0, 542 MB):
curl -L -o ~/spl-work/models/da3-base/model.safetensors https://huggingface.co/depth-anything/DA3-BASE/resolve/main/model.safetensors
```

ONNX Runtime 1.23.2 is the newest release whose default GPU build targets
CUDA 12 (1.24+ default to CUDA 13); the Rust side loads
`$C/py/onnxruntime/capi/libonnxruntime.so.1.23.2` through `ORT_DYLIB_PATH`
and needs `$C/cudnn` on `LD_LIBRARY_PATH` (`spl-ort`, api-23).

## Export

```sh
PY="docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp -e PYTHONPATH=$C/py:$C/src/Depth-Anything-3/src:$PWD ..."
$PY export_da3.py --weights model.safetensors --out-dir ~/spl-work/models/da3-base
```

Export **all** buckets in one call: the shared weights file is rewritten on
each run. Default buckets (W×H): 504×336, 504×378, 504×280, 504×252,
504×504 and the four portrait ones; frames are centre-cropped to the bucket
with the nearest aspect ratio. Lessons (spec-commercial §3) that the
exporter handles:

- `torch.cartesian_prod` has no ONNX symbolic: replaced by `meshgrid`.
- `F.scaled_dot_product_attention` exports as a materialised T×T softmax
  (36 GB per global layer at N = 32): exported as ONNX Runtime's fused
  `com.microsoft::MultiHeadAttention` instead.
- H and W are baked into the position embeddings, the RoPE grid and the
  head's upsampling (the tracer turns `int()` of shapes into constants), so
  each bucket is its own graph; N stays dynamic and is verified at N ≠ the
  export N.
- The checkpoint has no weights for the ray head's unused multi-level aux
  outputs; they are dropped from the graph.
- The camera-decoder and DPT-head fp32 islands of the DA3 code clash with a
  fully half model; the backbone is traced in fp16, the heads in fp32, like
  the official autocast path.
- The reference-view selection (`saddle_balanced`, an argmin over feature
  statistics) is in the graph. On near ties, bf16, fp16 and fp32 can pick
  different reference views, which changes the world frame and the depth
  scale; compare relative poses and scale-aligned depth (`check_parity.py`).

## Rust GPU tests

`~/spl-work/commercial/gtest.sh <filter> [lock]` builds the image-to-3d test
binary on the host and runs it in the CUDA 12.8 runtime image with
`/models`, `/datasets`, ONNX Runtime and cuDNN mounted:
`gpu_da3_dump_eval_predictions`, `gpu_da3_bench_scaling`, and the pose op
end to end with `SPL_POSE_PROFILE=commercial gtest.sh gpu_pose_op_vs_colmap`.
