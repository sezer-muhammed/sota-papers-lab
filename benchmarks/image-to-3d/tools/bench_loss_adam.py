"""Reference timings for the image-to-3d loss and optimizer kernels (gs-ref image).

Usage (GPU host): gs-ref --entry benchmarks/image-to-3d/tools/bench_loss_adam.py [--json out.json]

Times, with CUDA events (median of 200 after 20 warm-ups, like the Rust benches):
- fused-ssim (the kernel 3DGS, gsplat's trainer, Faster-GS and FastGS use) forward +
  backward of 0.8 L1 + 0.2 (1 - SSIM) at 1297x840 (garden 1/4), 1237x822, 1600x1063, 4K;
  also the SSIM call alone.
- torch.optim.Adam (fused=True and foreach) over the 6 Gaussian parameter groups
  (59 floats per Gaussian) at 1M / 3M / 6M Gaussians, all groups and main groups only.
- gsplat's selective Adam (Taming sparse Adam kernel) with 50% visible.
"""

import argparse
import json
import statistics

import torch


def timed(fn, warmup=20, reps=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(reps):
        a.record()
        fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b))
    out.sort()
    return {"median_ms": statistics.median(out), "p10_ms": out[len(out) // 10], "p90_ms": out[9 * len(out) // 10]}


def bench_loss(res):
    from fused_ssim import fused_ssim

    for w, h in [(1297, 840), (1237, 822), (1600, 1063), (3840, 2160)]:
        g = torch.Generator(device="cuda").manual_seed(0)
        y = torch.rand(1, 3, h, w, device="cuda", generator=g)
        x = (y + 0.02 * torch.randn(1, 3, h, w, device="cuda", generator=g)).clamp(0, 1).requires_grad_(True)

        def loss_step():
            l1 = (x - y).abs().mean()
            loss = 0.8 * l1 + 0.2 * (1.0 - fused_ssim(x, y, padding="same", train=True))
            (gx,) = torch.autograd.grad(loss, x)
            return gx

        def ssim_only():
            s = fused_ssim(x, y, padding="same", train=True)
            (gx,) = torch.autograd.grad(s, x)
            return gx

        res[f"loss {w}x{h} fused-ssim + torch L1"] = timed(loss_step)
        res[f"ssim {w}x{h} fused-ssim only"] = timed(ssim_only)
        print(w, h, res[f"loss {w}x{h} fused-ssim + torch L1"], res[f"ssim {w}x{h} fused-ssim only"], flush=True)


def bench_adam(res):
    from gsplat.cuda._wrapper import adam as selective_adam

    floats = {"means": 3, "scales": 3, "quats": 4, "opacities": 1, "sh0": 3, "shN": 45}
    for n in [1_000_000, 3_000_000, 6_000_000]:
        try:
            params = {k: torch.zeros(n, f, device="cuda", requires_grad=True) for k, f in floats.items()}
            for p in params.values():
                p.grad = torch.randn_like(p) * 1e-3
        except torch.cuda.OutOfMemoryError:
            print("n", n, "out of memory", flush=True)
            continue
        for name, keys in [("all 59", list(floats)), ("main 14", [k for k in floats if k != "shN"])]:
            for mode in ["fused", "foreach"]:
                try:
                    opt = torch.optim.Adam(
                        [{"params": [params[k]], "lr": 1e-3} for k in keys],
                        eps=1e-15,
                        fused=mode == "fused",
                        foreach=mode == "foreach",
                    )
                    r = timed(opt.step, 5, 50)
                except torch.cuda.OutOfMemoryError:
                    r = "out of memory"
                res[f"adam n {n} {name} torch {mode}"] = r
                print(n, name, mode, r, flush=True)
                del opt
                torch.cuda.empty_cache()
        # gsplat selective Adam, 50% visible, every group.
        vis = (torch.arange(n, device="cuda") // 64) % 2 == 0
        state = {k: (torch.zeros_like(p), torch.zeros_like(p)) for k, p in params.items()}

        def sel():
            for k, p in params.items():
                m, v = state[k]
                selective_adam(p.data, p.grad, m, v, vis, 1e-3, 0.9, 0.999, 1e-15)

        r = timed(sel, 5, 50)
        res[f"adam n {n} all 59 gsplat selective 50%"] = r
        print(n, "selective", r, flush=True)
        del params, state
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {"device": torch.cuda.get_device_name(), "torch": torch.__version__}
    bench_loss(res)
    bench_adam(res)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
