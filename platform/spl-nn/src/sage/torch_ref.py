#!/usr/bin/env python3
"""PyTorch side of the sage attention benchmark (nn-sage).

Exchanges bf16 tensors with the Rust benchmark (`gpu_bench` in tests.rs)
through a run directory:

    meta.txt           frames=, heads=, head_dim=, special=, grid_h=, grid_w=, source=
    q.bin k.bin v.bin  bf16 [T][H][D] (batch 1), little endian
    out_<mode>.bin     what the Rust benchmark wrote (dense, merge2, merge4;
                       the FP16-PV runs add _pv16)

Commands:
    gen DIR [--frames 200] [--head-dim 64]    synthetic Q/K/V (clustered tokens,
                                              K/V channel offsets, outlier channels)
    baselines DIR                             time FlashAttention-2 and SDPA, error vs fp64
    compare DIR                               error of out_<mode>.bin vs fp64
    all DIR                                   baselines + compare, one JSON report
    timing - [--head-dim D] [--heads H]       FA2 and SDPA speed on random data, no files

Typical run on the GPU host (the Rust side runs in the CUDA container, which
sees ~/spl-work/out as /out):

    torch_ref.py gen ~/spl-work/out/sage-runs/syn200
    SAGE_RUN=/out/sage-runs/syn200 cargo test --release -p spl-nn gpu_bench -- --ignored
    torch_ref.py all ~/spl-work/out/sage-runs/syn200

The fp64 reference covers a random subset of query rows (all heads) against
every key: exact on the bf16 inputs. Merged outputs are compared both with
the dense reference (what merging costs) and with fp64 attention over the
merged keys (what quantisation costs).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda"
REF_ROWS = int(os.environ.get("SAGE_REF_ROWS", "2048"))


def read_meta(d):
    m = {}
    with open(os.path.join(d, "meta.txt")) as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                m[k.strip()] = v.strip()
    return m


def write_meta(d, **kw):
    with open(os.path.join(d, "meta.txt"), "w") as f:
        for k, v in kw.items():
            f.write(f"{k}={v}\n")


def load_bf16(path, shape):
    raw = np.fromfile(path, dtype=np.uint16)
    t = torch.from_numpy(raw.astype(np.int16)).view(torch.bfloat16)
    return t[: math.prod(shape)].reshape(shape)


def save_bf16(path, t):
    t.detach().to(torch.bfloat16).contiguous().cpu().view(torch.int16).numpy().tofile(path)


def layout(m):
    frames, special = int(m["frames"]), int(m["special"])
    gh, gw = int(m["grid_h"]), int(m["grid_w"])
    return frames, special, gh, gw, special + gh * gw


def gen(args):
    os.makedirs(args.dir, exist_ok=True)
    frames, special, gh, gw = args.frames, 17, 21, 32
    T, H, D = frames * (special + gh * gw), 16, args.head_dim
    g = torch.Generator(device=DEV).manual_seed(args.seed)
    clusters = 37
    centers = torch.randn(clusters, H, D, device=DEV, generator=g)
    k_off = 3.0 * torch.randn(H, D, device=DEV, generator=g)
    v_off = 2.0 * torch.randn(H, D, device=DEV, generator=g)
    c = torch.randint(0, clusters, (T,), device=DEV, generator=g)
    ctr = centers[c]  # [T, H, D]
    outlier = torch.ones(D, device=DEV)
    outlier[5::23] = 6.0
    q = 1.6 * ctr + 0.6 * torch.randn(T, H, D, device=DEV, generator=g)
    k = (1.6 * ctr + 0.6 * torch.randn(T, H, D, device=DEV, generator=g)) * outlier + k_off
    v = torch.randn(T, H, D, device=DEV, generator=g) + 0.5 * ctr + v_off
    for name, t in (("q", q), ("k", k), ("v", v)):
        save_bf16(os.path.join(args.dir, f"{name}.bin"), t)
    write_meta(args.dir, frames=frames, heads=H, head_dim=D, special=special, grid_h=gh, grid_w=gw,
               source="synthetic")
    print(f"wrote {args.dir}: T={T} H={H} D={D}")


def load_run(d):
    m = read_meta(d)
    frames, special, gh, gw, tpf = layout(m)
    H, D = int(m["heads"]), int(m["head_dim"])
    T = frames * tpf
    qkv = [load_bf16(os.path.join(d, f"{n}.bin"), (T, H, D)).to(DEV) for n in "qkv"]
    return m, qkv


def ref_rows(T):
    g = torch.Generator().manual_seed(1234)
    return torch.randperm(T, generator=g)[: min(REF_ROWS, T)].sort().values.to(DEV)


def reference(q, k, v, rows, bias=None):
    """fp64 attention for query `rows` over all keys: [len(rows), H, D]."""
    T, H, D = q.shape
    scale = 1.0 / math.sqrt(D)
    out = torch.empty(len(rows), H, D, dtype=torch.float64, device=DEV)
    for h in range(H):
        kh = k[:, h].double()
        vh = v[:, h].double()
        for s in range(0, len(rows), 512):
            qh = q[rows[s : s + 512], h].double()
            logits = qh @ kh.T * scale
            if bias is not None:
                logits = logits + bias.double()[None, :]
            out[s : s + 512, h] = torch.softmax(logits, dim=-1) @ vh
    return out


def merge_kv(m, k, v, stride):
    """Mean-pool K and V like sage's KV merge; returns (k, v, bias_ln)."""
    frames, special, gh, gw, tpf = layout(m)
    T, H, D = k.shape
    kf = k.float().view(frames, tpf, H, D)
    vf = v.float().view(frames, tpf, H, D)

    def pool(x):
        sp = x[:, :special]
        grid = x[:, special:].reshape(frames, gh, gw, H * D).permute(0, 3, 1, 2)
        summed = F.avg_pool2d(grid, stride, stride, ceil_mode=True, divisor_override=1)
        ones = torch.ones(1, 1, gh, gw, device=DEV)
        cnt = F.avg_pool2d(ones, stride, stride, ceil_mode=True, divisor_override=1)
        pooled = (summed / cnt).permute(0, 2, 3, 1).reshape(frames, -1, H, D)
        return torch.cat([sp, pooled], 1).reshape(-1, H, D), cnt.flatten()

    km, cnt = pool(kf)
    vm, _ = pool(vf)
    bias = torch.cat([torch.zeros(special, device=DEV), cnt.log()]).repeat(frames)
    return km, vm, bias


def errors(ours, ref):
    a = ours.double().flatten()
    r = ref.double().flatten()
    d = a - r
    return {
        "cos": float((a @ r) / (a.norm() * r.norm())),
        "rel_l1": float(d.abs().sum() / r.abs().sum()),
        "rel_l2": float(d.norm() / r.norm()),
        "max_abs": float(d.abs().max()),
        "ref_rms": float(r.pow(2).mean().sqrt()),
    }


def timed(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    return times[len(times) // 2]


def gpu_util():
    try:
        out = os.popen("nvidia-smi --query-gpu=utilization.gpu,clocks.sm,memory.used --format=csv,noheader").read()
        return out.strip()
    except Exception:  # noqa: BLE001
        return "?"


def baselines(d, report):
    m, (q, k, v) = load_run(d)
    T, H, D = q.shape
    rows = ref_rows(T)
    ref = reference(q, k, v, rows)
    flops = 4.0 * H * T * T * D
    res = {}
    torch.backends.cuda.matmul.allow_tf32 = False

    # FlashAttention-2 (official package), bf16 in/out, [B, T, H, D].
    try:
        from flash_attn import flash_attn_func

        qb, kb, vb = (x.unsqueeze(0) for x in (q, k, v))
        out = flash_attn_func(qb, kb, vb)
        ms = timed(lambda: flash_attn_func(qb, kb, vb))
        res["flash_attn2_bf16"] = {"ms": ms, "tflops": flops / ms / 1e9, **errors(out[0][rows], ref)}
        qh, kh, vh = (x.half() for x in (qb, kb, vb))
        out = flash_attn_func(qh, kh, vh)
        ms = timed(lambda: flash_attn_func(qh, kh, vh))
        res["flash_attn2_fp16"] = {"ms": ms, "tflops": flops / ms / 1e9, **errors(out[0][rows], ref)}
        del out
    except ImportError as e:
        res["flash_attn2_bf16"] = {"error": str(e)}

    # PyTorch SDPA, [B, H, T, D], by backend.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    qh, kh, vh = (x.permute(1, 0, 2).unsqueeze(0).contiguous() for x in (q, k, v))
    for name, backend in (("sdpa_flash_bf16", SDPBackend.FLASH_ATTENTION),
                          ("sdpa_efficient_bf16", SDPBackend.EFFICIENT_ATTENTION),
                          ("sdpa_cudnn_bf16", SDPBackend.CUDNN_ATTENTION)):
        try:
            with sdpa_kernel(backend):
                out = F.scaled_dot_product_attention(qh, kh, vh)
                ms = timed(lambda: F.scaled_dot_product_attention(qh, kh, vh))
            o = out[0].permute(1, 0, 2)
            res[name] = {"ms": ms, "tflops": flops / ms / 1e9, **errors(o[rows], ref)}
            del out, o
        except Exception as e:  # noqa: BLE001
            res[name] = {"error": str(e).splitlines()[0][:200]}
        torch.cuda.empty_cache()

    # SDPA on the merged keys (what the merge schedule costs PyTorch).
    for stride in (2, 4):
        km, vm, bias = merge_kv(m, k, v, stride)
        kmh, vmh = (x.to(torch.bfloat16).permute(1, 0, 2).unsqueeze(0).contiguous() for x in (km, vm))
        mask = bias.to(torch.bfloat16)[None, None, None, :]
        try:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                ms = timed(lambda: F.scaled_dot_product_attention(qh, kmh, vmh, attn_mask=mask))
            res[f"sdpa_efficient_bf16_merge{stride}_with_bias"] = {"ms": ms}
        except Exception as e:  # noqa: BLE001
            res[f"sdpa_efficient_bf16_merge{stride}_with_bias"] = {"error": str(e).splitlines()[0][:200]}
        torch.cuda.empty_cache()
    report["baselines"] = res
    report["gpu_after_baselines"] = gpu_util()


def timing(args, report):
    """FA2 and SDPA speed on random bf16 data (no files), any head dim."""
    T = args.frames * 689
    H, D = args.heads, args.head_dim
    g = torch.Generator(device=DEV).manual_seed(0)
    q, k, v = (torch.randn(1, T, H, D, device=DEV, generator=g, dtype=torch.bfloat16) for _ in range(3))
    flops = 4.0 * H * T * T * D
    res = {}
    from flash_attn import flash_attn_func
    from torch.nn.attention import SDPBackend, sdpa_kernel

    ms = timed(lambda: flash_attn_func(q, k, v))
    res["flash_attn2_bf16"] = {"ms": ms, "tflops": flops / ms / 1e9}
    qh, kh, vh = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    for name, backend in (("sdpa_flash_bf16", SDPBackend.FLASH_ATTENTION),
                          ("sdpa_cudnn_bf16", SDPBackend.CUDNN_ATTENTION)):
        try:
            with sdpa_kernel(backend):
                ms = timed(lambda: F.scaled_dot_product_attention(qh, kh, vh))
            res[name] = {"ms": ms, "tflops": flops / ms / 1e9}
        except Exception as e:  # noqa: BLE001
            res[name] = {"error": str(e).splitlines()[0][:200]}
    report["timing"] = {"tokens": T, "heads": H, "head_dim": D, **res}


def compare(d, report):
    m, (q, k, v) = load_run(d)
    T, H, D = q.shape
    rows = ref_rows(T)
    ref = reference(q, k, v, rows)
    res = {}
    merged_ref = {}
    for mode in ("dense", "merge2", "merge4"):
        for suffix in ("", "_pv16"):
            path = os.path.join(d, f"out_{mode}{suffix}.bin")
            if not os.path.exists(path):
                continue
            ours = load_bf16(path, (T, H, D)).to(DEV)[rows]
            r = {"vs_dense_fp64": errors(ours, ref)}
            if mode != "dense":
                stride = int(mode[-1])
                if stride not in merged_ref:
                    km, vm, bias = merge_kv(m, k, v, stride)
                    merged_ref[stride] = reference(q, km, vm, rows, bias)
                r["vs_merged_fp64"] = errors(ours, merged_ref[stride])
                # The merge's own error, without quantisation.
                r["merged_fp64_vs_dense_fp64"] = errors(merged_ref[stride], ref)
            res[mode + suffix] = r
    report["sage"] = res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["gen", "baselines", "compare", "all", "timing"])
    p.add_argument("dir")
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.cmd == "gen":
        gen(args)
        return
    if args.cmd == "timing":
        report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "gpu_before": gpu_util()}
        timing(args, report)
        print(json.dumps(report, indent=1))
        return
    report = {"dir": args.dir, "meta": read_meta(args.dir), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "gpu_before": gpu_util(), "ref_rows": REF_ROWS}
    t0 = time.time()
    if args.cmd in ("baselines", "all"):
        baselines(args.dir, report)
    if args.cmd in ("compare", "all"):
        compare(args.dir, report)
    report["seconds"] = time.time() - t0
    text = json.dumps(report, indent=1)
    print(text)
    with open(os.path.join(args.dir, f"report_{args.cmd}.json"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    sys.exit(main())
