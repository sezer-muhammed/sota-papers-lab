"""PyTorch reference fixtures for the spl-nn `torch_*` GPU tests.

Writes .npy files (bf16 as raw uint16 bits, FP8 as raw uint8 bits, f32
otherwise) that `src/tests.rs` reads. References are computed in float64
on the bf16/FP8 values the kernels see; PyTorch's own bf16/FP8 outputs are
saved next to them so the tests can compare our error with PyTorch's.

Run in a PyTorch CUDA container on the GPU host, e.g.

    docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -v "$PWD:/w" -v ~/spl-work/out:/out -w /w \
      pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime \
      python3 platform/spl-nn/scripts/torch_ref.py /out/nn-dense/fixtures
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

OUT = sys.argv[1] if len(sys.argv) > 1 else "/out/nn-dense/fixtures"
DEV = "cuda"
HD = 64
FP8_MAX = 448.0


def save(name, t):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
        if t.dtype == torch.bfloat16:
            a = t.view(torch.int16).numpy().view(np.uint16)
        elif t.dtype == torch.float8_e4m3fn:
            a = t.view(torch.uint8).numpy()
        else:
            a = t.to(torch.float32).numpy()
    else:
        a = np.asarray(t, dtype=np.float32)
    np.save(os.path.join(OUT, name + ".npy"), np.ascontiguousarray(a))


def attention_fixtures(g):
    # Frame attention: 4 frames of 689 tokens; global: the same 2756 tokens
    # as one sequence. Layout [heads][n_seq * seq_len][64], as spl-nn uses.
    heads = 16
    for case, frames, seq_len in [("frame", 4, 689), ("global", 1, 4 * 689)]:
        total = frames * seq_len
        q = (torch.randn(heads, total, HD, generator=g, device=DEV) * 2.0).bfloat16()
        k = torch.randn(heads, total, HD, generator=g, device=DEV).bfloat16()
        v = torch.randn(heads, total, HD, generator=g, device=DEV).bfloat16()

        def batched(t):  # [n_seq, heads, seq_len, 64]
            return t.view(heads, frames, seq_len, HD).transpose(0, 1)

        def token_major(o):  # [n_seq, heads, L, 64] -> [total, heads * 64]
            return o.permute(0, 2, 1, 3).reshape(total, heads * HD)

        ref = F.scaled_dot_product_attention(
            batched(q).double(), batched(k).double(), batched(v).double(), scale=0.125
        )
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            flash = F.scaled_dot_product_attention(batched(q), batched(k), batched(v), scale=0.125)
        for n, t in [("q", q), ("k", k), ("v", v)]:
            save(f"attn_{case}_{n}", t)
        save(f"attn_{case}_seq_len", [float(seq_len)])
        save(f"attn_{case}_ref_fp32", token_major(ref).float())
        save(f"attn_{case}_sdpa_bf16", token_major(flash).contiguous())


def layer_norm_fixtures(g):
    rows, c = 2 * 689 + 3, 1024
    x = torch.randn(rows, c, generator=g, device=DEV)
    x += torch.randn(rows, 1, generator=g, device=DEV) * 3.0
    x[:, 3::97] += 20.0
    gamma = 1.0 + 0.3 * torch.randn(c, generator=g, device=DEV)
    beta = 0.1 * torch.randn(c, generator=g, device=DEV)
    y = F.layer_norm(x.double(), (c,), gamma.double(), beta.double(), eps=1e-6).float()
    amax = y.abs().amax(dim=1, keepdim=True)
    scale = amax / FP8_MAX
    q = (y * (FP8_MAX / amax)).to(torch.float8_e4m3fn)
    save("ln_x", x)
    save("ln_gamma", gamma)
    save("ln_beta", beta)
    save("ln_ref_fp32", y)
    save("ln_scale", scale.squeeze(1))
    save("ln_q", q)


def gemm_fixtures(g):
    m, n, k = 300, 512, 1024
    x = torch.randn(m, k, generator=g, device=DEV).bfloat16()
    w = (torch.randn(n, k, generator=g, device=DEV) / math.sqrt(k)).bfloat16()
    bias = (0.5 * torch.randn(n, generator=g, device=DEV)).bfloat16()
    lin = F.linear(x.double(), w.double(), bias.double())
    save("gemm_x", x)
    save("gemm_w", w)
    save("gemm_bias", bias)
    save("gemm_ref_gelu_tanh_fp32", F.gelu(lin, approximate="tanh").float())
    save("gemm_ref_gelu_erf_fp32", F.gelu(lin).float())
    save("gemm_torch_gelu_tanh_bf16", F.gelu(F.linear(x, w, bias), approximate="tanh"))

    # FP8 per-tensor scales; torch._scaled_mm for PyTorch's own result.
    xf, wf = x.float(), w.float()
    xs = xf.abs().amax() / FP8_MAX
    ws = wf.abs().amax() / FP8_MAX
    xq = (xf / xs).to(torch.float8_e4m3fn)
    wq = (wf / ws).to(torch.float8_e4m3fn)
    deq = F.linear(xq.double() * xs.double(), wq.double() * ws.double(), bias.double())
    torch_fp8 = torch._scaled_mm(
        xq, wq.t(), scale_a=xs.reshape(()), scale_b=ws.reshape(()), bias=bias, out_dtype=torch.bfloat16
    )
    save("gemm8_xq", xq)
    save("gemm8_wq", wq)
    save("gemm8_xs", xs.reshape(1))
    save("gemm8_ws", ws.reshape(1))
    save("gemm8_ref_fp32", deq.float())
    save("gemm8_torch_bf16", torch_fp8)


def vggt_rope(t, pos, base=100.0):
    """VGGT-style 2D RoPE, written from the formula: channels 0-31 rotate
    by the row position, 32-63 by the column, rotate-half in each half."""
    out = torch.empty_like(t)
    for half, p in [(0, pos[:, 0]), (1, pos[:, 1])]:
        x = t[..., half * 32:(half + 1) * 32]
        inv = base ** (-torch.arange(0, 32, 2, device=t.device, dtype=torch.float64) / 32)
        ang = p.double()[:, None] * inv[None, :]
        ang = torch.cat([ang, ang], dim=-1)
        rot = torch.cat([-x[..., 16:], x[..., :16]], dim=-1)
        out[..., half * 32:(half + 1) * 32] = x * ang.cos() + rot * ang.sin()
    return out


def qkv_fixtures(g):
    grid_h, grid_w, special, frames, heads = 21, 32, 17, 2, 16
    frame_len = special + grid_h * grid_w
    tokens = frames * frame_len
    width = 3 * heads * HD
    y = torch.randn(tokens, width, generator=g, device=DEV).bfloat16()
    bias = (0.2 * torch.randn(width, generator=g, device=DEV)).bfloat16()
    qn_w = 1.0 + 0.2 * torch.randn(HD, generator=g, device=DEV)
    qn_b = 0.1 * torch.randn(HD, generator=g, device=DEV)
    kn_w = 1.0 + 0.2 * torch.randn(HD, generator=g, device=DEV)
    kn_b = 0.1 * torch.randn(HD, generator=g, device=DEV)
    t = (y.double() + bias.double()).view(tokens, 3, heads, HD)
    q = F.layer_norm(t[:, 0], (HD,), qn_w.double(), qn_b.double(), eps=1e-6)
    k = F.layer_norm(t[:, 1], (HD,), kn_w.double(), kn_b.double(), eps=1e-6)
    v = t[:, 2]
    yy, xx = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    patch = torch.stack([yy.flatten(), xx.flatten()], dim=-1) + 1
    pos = torch.cat([torch.zeros(special, 2, dtype=torch.long), patch]).repeat(frames, 1).to(DEV)
    q = torch.stack([vggt_rope(q[:, h], pos) for h in range(heads)], dim=1)
    k = torch.stack([vggt_rope(k[:, h], pos) for h in range(heads)], dim=1)
    save("qkv_y", y)
    save("qkv_bias", bias)
    for n, p in [("qn_w", qn_w), ("qn_b", qn_b), ("kn_w", kn_w), ("kn_b", kn_b)]:
        save(f"qkv_{n}", p)
    save("qkv_meta", [grid_h, grid_w, special])
    for n, o in [("q", q), ("k", k), ("v", v)]:
        save(f"qkv_ref_{n}", o.transpose(0, 1).contiguous().float())  # [heads][tokens][64]


def main():
    os.makedirs(OUT, exist_ok=True)
    g = torch.Generator(device=DEV).manual_seed(1234)
    print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
    attention_fixtures(g)
    layer_norm_fixtures(g)
    gemm_fixtures(g)
    qkv_fixtures(g)
    print("fixtures in", OUT, sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
