"""PyTorch baselines for the spl-nn `bench_*` tests, on the same shapes.

Attention: F.scaled_dot_product_attention per backend (FlashAttention-2,
memory-efficient, cuDNN), bf16, head dim 64, frame batches of 689 tokens
and global sequences of frames x 689. GEMMs: F.linear bf16 (+ GELU) and
torch._scaled_mm FP8 for the VGGT-1B shapes. LayerNorm + FP8 quantisation:
eager and torch.compile. Timing: CUDA events, best of 5 batches.

Run in a PyTorch CUDA container on the GPU host (see torch_ref.py), under
the GPU lock:

    flock -x ~/spl-work/gpu.lock docker run ... python3 \
      platform/spl-nn/scripts/torch_bench.py /out/nn-dense/torch_bench.json
"""

import json
import math
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

OUT = sys.argv[1] if len(sys.argv) > 1 else "/out/nn-dense/torch_bench.json"
DEV = "cuda"
HD, HEADS, TOK = 64, 16, 689


def gpu_state():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw",
             "--format=csv,noheader"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return "nvidia-smi unavailable"


def time_ms(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = math.inf
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            fn()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) / reps)
    return best


def bench_attention(res):
    cases = [("frame_n2", 2, 1), ("frame_n16", 16, 1), ("frame_n64", 64, 1), ("frame_n200", 200, 1),
             ("global_n8", 1, 8), ("global_n32", 1, 32), ("global_n64", 1, 64)]
    backends = [("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                ("cudnn", SDPBackend.CUDNN_ATTENTION)]
    for name, n_seq, frames_per_seq in cases:
        seq = TOK * frames_per_seq
        q, k, v = (torch.randn(n_seq, HEADS, seq, HD, device=DEV, dtype=torch.bfloat16) for _ in range(3))
        flops = 4.0 * seq * seq * HD * HEADS * n_seq
        reps = max(2, min(200, int(2e11 / flops)))
        for bname, be in backends:
            try:
                with sdpa_kernel(be):
                    ms = time_ms(lambda: F.scaled_dot_product_attention(q, k, v, scale=0.125), reps)
            except RuntimeError as e:
                print(f"attention {name} {bname}: unavailable ({str(e).splitlines()[0][:80]})")
                continue
            t = flops / (ms * 1e-3) / 1e12
            print(f"attention {name:<11} torch {bname:<9}: {ms:8.3f} ms {t:6.2f} TFLOPS")
            res[f"torch_attn_{name}_{bname}_tflops"] = t
        del q, k, v
        torch.cuda.empty_cache()


def bench_gemm(res):
    for frames in [2, 32, 200]:
        m = frames * TOK
        for layer, n, k in [("qkv", 3072, 1024), ("proj", 1024, 1024), ("fc1", 4096, 1024), ("fc2", 1024, 4096)]:
            flops = 2.0 * m * n * k
            reps = max(3, min(200, int(1e12 / flops)))
            x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
            w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16) * 0.03
            b = torch.randn(n, device=DEV, dtype=torch.bfloat16) * 0.1
            gelu = layer == "fc1"
            f = (lambda: F.gelu(F.linear(x, w, b), approximate="tanh")) if gelu else (lambda: F.linear(x, w, b))
            ms = time_ms(f, reps)
            t = flops / (ms * 1e-3) / 1e12
            print(f"gemm bf16 {layer:<4} m={m:<6} torch: {ms:8.3f} ms {t:6.1f} TFLOPS")
            res[f"torch_gemm_bf16_{layer}_n{frames}_tflops"] = t
            xq, wq = x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
            one = torch.ones((), device=DEV)
            for kind, sa, sb in [("tensor", one, one),
                                 ("rowwise", torch.ones(m, 1, device=DEV), torch.ones(1, n, device=DEV))]:
                try:
                    def f8():
                        y = torch._scaled_mm(xq, wq.t(), scale_a=sa, scale_b=sb, bias=b, out_dtype=torch.bfloat16)
                        return F.gelu(y, approximate="tanh") if gelu else y
                    ms = time_ms(f8, reps)
                except RuntimeError as e:
                    print(f"gemm fp8 {layer} {kind}: unavailable ({str(e).splitlines()[0][:90]})")
                    continue
                t = flops / (ms * 1e-3) / 1e12
                print(f"gemm fp8  {layer:<4} m={m:<6} torch _scaled_mm {kind}: {ms:8.3f} ms {t:6.1f} TFLOPS")
                res[f"torch_gemm_fp8_{kind}_{layer}_n{frames}_tflops"] = t
            del x, w, xq, wq
            torch.cuda.empty_cache()


def bench_layer_norm(res):
    rows, c = 64 * TOK, 1024
    x = torch.randn(rows, c, device=DEV)
    g, b = torch.ones(c, device=DEV), torch.zeros(c, device=DEV)

    def ln_fp8():
        y = F.layer_norm(x, (c,), g, b, eps=1e-6)
        s = y.abs().amax(dim=1, keepdim=True) / 448.0
        return (y / s).to(torch.float8_e4m3fn), s

    bytes_moved = rows * c * (4 + 1)
    for name, f in [("eager", ln_fp8)]:
        ms = time_ms(f, 20)
        print(f"layer_norm f32 -> fp8 torch {name}: {ms:.3f} ms, {bytes_moved / ms / 1e6:.0f} GB/s")
        res[f"torch_ln_fp8_{name}_gbs"] = bytes_moved / ms / 1e6
    try:
        fc = torch.compile(ln_fp8)
        ms = time_ms(fc, 20)
        print(f"layer_norm f32 -> fp8 torch compiled: {ms:.3f} ms, {bytes_moved / ms / 1e6:.0f} GB/s")
        res["torch_ln_fp8_compiled_gbs"] = bytes_moved / ms / 1e6
    except Exception as e:  # noqa: BLE001 (inductor may be unavailable)
        print("torch.compile unavailable:", str(e).splitlines()[0][:100])


def main():
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    res = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}
    print(res, "\nGPU before:", gpu_state())
    bench_attention(res)
    bench_gemm(res)
    bench_layer_norm(res)
    print("GPU after:", gpu_state())
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
