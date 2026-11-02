"""Times gsplat's densification ops on the same GPU as our E6 kernels.

Run in the spl-gs-ref image:
    gs-ref --entry benchmarks/image-to-3d/py/bench_densify_gsplat.py [--n 1000000,3000000] [--json out.json]

Per size: a 3DGS parameter dict (means, scales, quats, opacities, sh0, shN with
45 SH-rest floats) with torch.optim.Adam states (every moment populated), then
the same workloads as `gpu_bench_densify`:
  - ADC step: duplicate 5%, split 5%, remove 10% (gsplat.strategy.ops), the
    equivalent of one emission of ours over parameters and moments;
  - MCMC: relocate 2% dead + sample_add 5%;
  - MCMC position noise (inject_noise_to_position).
Median of `--reps` CUDA-event timings after one warm-up; the state is rebuilt
before each repetition (not timed).
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

DEV = "cuda"


def make(n: int, seed: int = 0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    params = torch.nn.ParameterDict(
        {
            "means": torch.randn(n, 3, device=DEV, generator=g),
            "scales": torch.rand(n, 3, device=DEV, generator=g) * 3 - 6,
            "quats": torch.randn(n, 4, device=DEV, generator=g),
            "opacities": torch.randn(n, device=DEV, generator=g),
            "sh0": torch.randn(n, 1, 3, device=DEV, generator=g),
            "shN": torch.randn(n, 15, 3, device=DEV, generator=g),
        }
    )
    opts = {}
    for k, p in params.items():
        opt = torch.optim.Adam([{"params": p, "lr": 1e-3, "name": k}], eps=1e-15)
        p.grad = torch.randn_like(p) * 1e-3
        opt.step()
        p.grad = None
        opts[k] = opt
    state = {
        "grad2d": torch.rand(n, device=DEV),
        "count": torch.rand(n, device=DEV) * 10,
        "radii": torch.rand(n, device=DEV),
    }
    return params, opts, state


def timed(fn, reps: int, setup) -> dict:
    out = []
    for r in range(reps + 1):
        args = setup()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn(*args)
        b.record()
        torch.cuda.synchronize()
        if r > 0:
            out.append(a.elapsed_time(b))
        del args
        torch.cuda.empty_cache()
    return {"median_ms": statistics.median(out), "min_ms": min(out), "max_ms": max(out), "n": len(out)}


def main() -> int:
    from gsplat.strategy import ops

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="1000000,3000000,6000000")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--json")
    a = ap.parse_args()
    binoms = torch.zeros((51, 51), device=DEV)
    import math

    for i in range(51):
        for k in range(i + 1):
            binoms[i, k] = math.comb(i, k)
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rows": []}
    for n in [int(x) for x in a.n.split(",")]:
        row = {"n": n}
        try:

            def setup():
                p, o, s = make(n)
                g = torch.Generator(device=DEV).manual_seed(1)
                u = torch.rand(n, device=DEV, generator=g)
                return p, o, s, u

            def adc(p, o, s, u):
                ops.duplicate(p, o, s, u < 0.05)
                ops.split(p, o, s, torch.cat([(u >= 0.05) & (u < 0.10), torch.zeros(len(p["means"]) - n, dtype=torch.bool, device=DEV)]))
                m = len(p["means"])
                ops.remove(p, o, s, torch.rand(m, device=DEV) < 0.10)

            row["adc_step"] = timed(adc, a.reps, setup)

            def dup(p, o, s, u):
                ops.duplicate(p, o, s, u < 0.05)

            row["duplicate_5pct"] = timed(dup, a.reps, setup)

            def rem(p, o, s, u):
                ops.remove(p, o, s, u < 0.10)

            row["remove_10pct"] = timed(rem, a.reps, setup)

            def mcmc(p, o, s, u):
                dead = u < 0.02
                ops.relocate(p, o, s, dead, binoms)
                ops.sample_add(p, o, s, int(0.05 * n), binoms)

            row["mcmc_round"] = timed(mcmc, a.reps, setup)

            def noise(p, o, s, u):
                ops.inject_noise_to_position(p, o, s, 1e-5 * 5e5)

            row["mcmc_noise"] = timed(noise, a.reps, setup)
        except torch.cuda.OutOfMemoryError as e:
            row["error"] = f"OOM: {e}".splitlines()[0]
        torch.cuda.empty_cache()
        print(json.dumps(row))
        report["rows"].append(row)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
