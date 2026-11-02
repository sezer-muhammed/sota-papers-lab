"""Export DA3-BASE (ByteDance-Seed/Depth-Anything-3, Apache-2.0 code and
weights) to ONNX for the commercial profile's `da3` engine
(`src/engine/da3.rs`).

Two graphs per resolution bucket (H x W, multiples of 14; the DA3 graph
bakes H and W into its position embeddings, RoPE grid and head upsampling,
so N is the only dynamic axis):

- `backbone_<W>x<H>.onnx`: `images` [N, 3, H, W] (ImageNet-normalised) ->
  `f0..f3` [N, H/14, W/14, 1536] (the four DPT feature levels) and
  `pose_enc` [N, 9] (t_xyz, quat_xyzw, fov_h, fov_w of the camera-to-world
  pose; the camera-decoder path, `use_ray_pose=False`; reference-view
  selection `saddle_balanced` is inside the graph and needs N >= 3).
- `head_<W>x<H>.onnx`: `f0..f3` for a chunk of frames -> `depth`, `depth_conf`
  [n, H, W]. The engine runs it 8 frames at a time, as the official code does.

Precision follows the official path: the backbone is traced in fp16 (the
official forward runs it under bf16 autocast), the camera decoder and the DPT
head in fp32 (autocast is off for them there); graph inputs and outputs are
fp16. The large
initializers are moved into one shared external-data file `da3_base.weights`,
so every bucket after the first costs a few MB. Attention is exported as ONNX
Runtime's fused `com.microsoft::MultiHeadAttention`.

Run in spl-gs-ref (with a GPU) with the DA3 repo and onnx on PYTHONPATH:

    python export_da3.py --weights model.safetensors --out-dir da3-base \
        --buckets 504x336,504x252
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

HEADS = 12  # ViT-B
WEIGHTS = "da3_base.weights"
EXTERNAL_MIN_BYTES = 64 * 1024


def _sdpa_symbolic(g, q, k, v, attn_mask=None, dropout_p=None, is_causal=None, scale=None, enable_gqa=None):
    """aten::scaled_dot_product_attention -> `com.microsoft::MultiHeadAttention`
    (flash / memory-efficient kernels). The default export materialises the
    T x T score matrix: 36 GB per global layer at N = 32. DA3 calls SDPA
    without mask, dropout or scale."""
    i64 = lambda xs: g.op("Constant", value_t=torch.tensor(xs, dtype=torch.int64))
    merge = lambda t: g.op("Reshape", g.op("Transpose", t, perm_i=[0, 2, 1, 3]), i64([0, 0, -1]))
    out = g.op("com.microsoft::MultiHeadAttention", merge(q), merge(k), merge(v), num_heads_i=HEADS)
    return g.op("Transpose", g.op("Reshape", out, i64([0, 0, HEADS, -1])), perm_i=[0, 2, 1, 3])


def _cartesian_prod(a, b):
    # ONNX has no cartesian_prod (used by the RoPE position grid).
    return torch.stack(torch.meshgrid(a, b, indexing="ij"), dim=-1).reshape(-1, 2)


class Backbone(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, images):
        x = images[None]
        n, h, w = images.shape[0], images.shape[2], images.shape[3]
        feats, _ = self.net.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy="saddle_balanced")
        # The camera decoder runs in fp32, as under the official autocast.
        pose_enc = self.net.cam_dec(feats[-1][1].float())[0]
        levels = [f[0][0].reshape(n, h // 14, w // 14, -1).to(images.dtype) for f in feats]
        return (*levels, pose_enc.to(images.dtype))


class Head(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, f0, f1, f2, f3):
        n, ph, pw, c = f0.shape
        # The DPT head runs in fp32, as under the official autocast.
        feats = [(f.float().reshape(1, n, ph * pw, c), None) for f in (f0, f1, f2, f3)]
        out = self.net.head(feats, ph * 14, pw * 14, patch_start_idx=0, chunk_size=None)
        return out["depth"][0].to(f0.dtype), out["depth_conf"][0].to(f0.dtype)


def build(weights: str):
    torch.cartesian_prod = _cartesian_prod
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY

    net = create_object(load_config(MODEL_REGISTRY["da3-base"]))
    state = load_file(weights)
    state = {k.removeprefix("model."): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    # The checkpoint has no weights for the ray head's unused multi-level aux
    # outputs (the official loader is non-strict too); they never reach our
    # outputs, so the tracer drops them.
    missing = [k for k in missing if not k.startswith("head.scratch.output_conv2_aux.")]
    if missing or unexpected:
        raise SystemExit(f"weights mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
    return net.eval()


def _export(module, args, path, inputs, outputs, dynamic, opset):
    with torch.no_grad():
        torch.onnx.export(
            module, args, str(path), dynamo=False, opset_version=opset,
            input_names=inputs, output_names=outputs, dynamic_axes=dynamic,
            do_constant_folding=True, custom_opsets={"com.microsoft": 1},
        )


class SharedWeights:
    """Large initializers of every graph in one file, deduplicated by content."""

    def __init__(self, path: Path):
        self.path = path
        self.index: dict[str, tuple[int, int]] = {}
        self.file = open(path, "wb")
        self.offset = 0

    def externalize(self, model):
        from onnx import TensorProto

        for t in model.graph.initializer:
            raw = t.raw_data
            if len(raw) < EXTERNAL_MIN_BYTES:
                continue
            key = hashlib.sha256(raw).hexdigest()
            if key not in self.index:
                pad = (-self.offset) % 64
                self.file.write(b"\0" * pad)
                self.offset += pad
                self.file.write(raw)
                self.index[key] = (self.offset, len(raw))
                self.offset += len(raw)
            off, length = self.index[key]
            t.ClearField("raw_data")
            t.data_location = TensorProto.EXTERNAL
            del t.external_data[:]
            for k, v in (("location", self.path.name), ("offset", str(off)), ("length", str(length))):
                e = t.external_data.add()
                e.key, e.value = k, v


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--buckets", default="504x336,504x378,504x280,504x252,504x504,336x504,378x504,280x504,252x504",
                   help="W x H list")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--fp32", action="store_true", help="keep fp32 (parity checks)")
    a = p.parse_args()
    import onnx

    torch.onnx.register_custom_op_symbolic("aten::scaled_dot_product_attention", _sdpa_symbolic, a.opset)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # fp16: the DINOv2 backbone in half precision on the GPU (the official
    # path runs it under bf16 autocast; fp16 has 3 more mantissa bits and
    # DA3-BASE activations fit its range); camera decoder and DPT head stay
    # fp32 like the official code, which disables autocast for them.
    dtype, dev = (torch.float32, "cpu") if a.fp32 else (torch.float16, "cuda")
    net = build(a.weights).to(dev)
    net.backbone.to(dtype)
    shared = SharedWeights(out / (WEIGHTS if not a.fp32 else "da3_base_fp32.weights"))
    sha = hashlib.sha256(Path(a.weights).read_bytes()).hexdigest()
    buckets = []
    for b in a.buckets.split(","):
        w, h = map(int, b.split("x"))
        assert w % 14 == 0 and h % 14 == 0, b
        x = torch.randn(4, 3, h, w, device=dev, dtype=dtype)
        with torch.no_grad():
            levels = Backbone(net)(x)[:4]
        tag = f"{w}x{h}" + ("_fp32" if a.fp32 else "")
        names = {"backbone": out / f"backbone_{tag}.onnx", "head": out / f"head_{tag}.onnx"}
        feat_axes = {f"f{i}": {0: "n"} for i in range(4)}
        _export(Backbone(net), (x,), names["backbone"], ["images"], ["f0", "f1", "f2", "f3", "pose_enc"],
                {"images": {0: "n"}, "pose_enc": {0: "n"}, **feat_axes}, a.opset)
        _export(Head(net), tuple(levels), names["head"], ["f0", "f1", "f2", "f3"], ["depth", "depth_conf"],
                {**feat_axes, "depth": {0: "n"}, "depth_conf": {0: "n"}}, a.opset)
        for path in names.values():
            m = onnx.load(str(path))
            shared.externalize(m)
            onnx.save(m, str(path))
        buckets.append({"width": w, "height": h, "backbone": names["backbone"].name, "head": names["head"].name})
        print(f"bucket {w}x{h}: backbone {names['backbone'].stat().st_size / 1e6:.1f} MB, "
              f"head {names['head'].stat().st_size / 1e6:.1f} MB, shared {shared.offset / 1e6:.1f} MB", flush=True)
    shared.file.close()
    if not a.fp32:
        manifest = {
            "model": "depth-anything/DA3-BASE",
            "licence": "apache-2.0",
            "source": "https://huggingface.co/depth-anything/DA3-BASE",
            "source_sha256": sha,
            "precision": "fp16",
            "weights": WEIGHTS,
            "buckets": buckets,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
