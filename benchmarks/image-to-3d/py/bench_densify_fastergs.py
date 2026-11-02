"""Times Faster-GS's densification paths on the same GPU as our E6 kernels.

Run in the spl-bl-fastergs image (Faster-GS / NeRFICG installed under /opt/nerficg):
    docker run --rm --gpus all -v <py dir>:/bench:ro -v ~/spl-work/out:/out --entrypoint "" spl-bl-fastergs \
        python /bench/bench_densify_fastergs.py --n 1000000,3000000 --json /out/.../fastergs.json

Reproduces the tensor operations of `Methods/FasterGS/Model.py` (Apache-2.0;
read, not copied into the product) on a parameter set with FusedAdam state:
  - ADC step: duplicate 5%, split 5%, prune 10% (extend_param_groups +
    prune_param_groups, the same masks as gpu_bench_densify);
  - Morton re-sort: morton_encode + argsort + sort_param_groups;
  - MCMC densification: relocate 2% dead + add 5% (relocation_adjustment);
  - MCMC noise (add_noise).
Median of `--reps` CUDA-event timings; the state is rebuilt before each repetition.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys

import torch

from FasterGSCudaBackend.torch_bindings.densification import add_noise, relocation_adjustment  # noqa: E402
from MortonEncoding import morton_encode  # noqa: E402

sys.path.append("/opt/nerficg/src")
from Optim.adam_utils import extend_param_groups, prune_param_groups, reset_state, sort_param_groups  # noqa: E402

DEV = "cuda"
NAMES = ["means", "sh_coefficients_0", "sh_coefficients_rest", "opacities", "scales", "rotations"]


def make(n: int):
    g = torch.Generator(device=DEV).manual_seed(0)
    shapes = {
        "means": (n, 3),
        "sh_coefficients_0": (n, 1, 3),
        "sh_coefficients_rest": (n, 15, 3),
        "opacities": (n, 1),
        "scales": (n, 3),
        "rotations": (n, 4),
    }
    params = {k: torch.nn.Parameter(torch.randn(s, device=DEV, generator=g)) for k, s in shapes.items()}
    params["scales"].data.mul_(0.5).sub_(5)
    opt = torch.optim.Adam([{"params": [params[k]], "lr": 1e-3, "name": k} for k in NAMES], eps=1e-15)
    for p in params.values():
        p.grad = torch.randn_like(p) * 1e-3
    opt.step()
    for p in params.values():
        p.grad = None
    return params, opt


def cur(opt):
    return {gr["name"]: gr["params"][0] for gr in opt.param_groups}


def quat_to_rot(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
         2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def adc(params, opt, u):
    p = cur(opt)
    dup, split = u < 0.05, (u >= 0.05) & (u < 0.10)
    s = p["scales"][split].exp().expand(2, -1, -1).flatten(end_dim=1)
    r = p["rotations"][split].expand(2, -1, -1).flatten(end_dim=1)
    off = (quat_to_rot(r) @ (s * torch.randn_like(s))[..., None])[..., 0]
    new = {
        "means": torch.cat([p["means"][dup], p["means"][split].expand(2, -1, -1).flatten(end_dim=1) + off]),
        "scales": torch.cat([p["scales"][dup], s.mul(0.625).log()]),
        "rotations": torch.cat([p["rotations"][dup], r]),
    }
    for k in ("sh_coefficients_0", "sh_coefficients_rest", "opacities"):
        x = p[k][split]
        new[k] = torch.cat([p[k][dup], x.expand(2, *x.shape).flatten(end_dim=1)])
    extend_param_groups(opt, new)
    p = cur(opt)
    n_new = p["means"].shape[0] - u.shape[0]
    prune = torch.cat([split | ((u >= 0.10) & (u < 0.20)), torch.zeros(n_new, dtype=torch.bool, device=DEV)])
    prune |= p["opacities"].flatten() < math.log(0.005 / 0.995)
    prune |= p["rotations"].mul(p["rotations"]).sum(1) < 1e-8
    prune_param_groups(opt, ~prune)


def morton(params, opt, u):
    p = cur(opt)
    order = torch.argsort(morton_encode(p["means"].data))
    sort_param_groups(opt, order)


def mcmc(params, opt, u):
    p = cur(opt)
    min_o = 0.005
    with torch.no_grad():
        p["opacities"][u < 0.02] = -10.0
        dead = p["opacities"].flatten() <= math.log(min_o / (1 - min_o))
        dead |= p["rotations"].mul(p["rotations"]).sum(1) < 1e-8
        nd = int(dead.sum().item())
        if nd > 0:
            di, ai = torch.where(dead)[0], torch.where(~dead)[0]
            o = torch.sigmoid(p["opacities"]).flatten()
            si = ai[torch.multinomial(o[ai], nd, replacement=True)]
            _, inv, cnt = si.unique(sorted=False, return_inverse=True, return_counts=True)
            ao, as_ = relocation_adjustment(o[si], p["scales"][si].exp(), cnt[inv] + 1)
            ao = ao.clamp(min_o, 1 - torch.finfo(torch.float32).eps).logit()
            as_ = as_.log()
            p["opacities"][si] = ao.reshape(-1, 1)
            p["scales"][si] = as_
            for k in ("means", "sh_coefficients_0", "sh_coefficients_rest", "rotations"):
                p[k][di] = p[k][si]
            p["opacities"][di] = ao.reshape(-1, 1)
            p["scales"][di] = as_
            reset_state(opt, indices=si)
        n = p["means"].shape[0]
        na = int(1.05 * n) - n
        o = torch.sigmoid(p["opacities"]).flatten()
        si = torch.multinomial(o, na, replacement=True)
        _, inv, cnt = si.unique(sorted=False, return_inverse=True, return_counts=True)
        ao, as_ = relocation_adjustment(o[si], p["scales"][si].exp(), cnt[inv] + 1)
        ao = ao.clamp(min_o, 1 - torch.finfo(torch.float32).eps).logit().reshape(-1, 1)
        as_ = as_.log()
        p["opacities"][si] = ao
        p["scales"][si] = as_
        extend_param_groups(opt, {
            "means": p["means"][si], "sh_coefficients_0": p["sh_coefficients_0"][si],
            "sh_coefficients_rest": p["sh_coefficients_rest"][si], "opacities": ao, "scales": as_,
            "rotations": p["rotations"][si],
        })
        reset_state(opt, indices=si)


def noise(params, opt, u):
    p = cur(opt)
    add_noise(p["scales"].data, p["rotations"].data, p["opacities"].data, p["means"].data, 5e5 * 1e-5)


def timed(fn, n: int, reps: int) -> dict:
    out = []
    for r in range(reps + 1):
        params, opt = make(n)
        u = torch.rand(n, device=DEV, generator=torch.Generator(device=DEV).manual_seed(1))
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn(params, opt, u)
        b.record()
        torch.cuda.synchronize()
        if r > 0:
            out.append(a.elapsed_time(b))
        del params, opt, u
        torch.cuda.empty_cache()
    return {"median_ms": statistics.median(out), "min_ms": min(out), "max_ms": max(out), "n": len(out)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="1000000,3000000,6000000")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--json")
    a = ap.parse_args()
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rows": []}
    for n in [int(x) for x in a.n.split(",")]:
        row = {"n": n}
        try:
            for name, fn in [("adc_step", adc), ("morton_sort", morton), ("mcmc_round", mcmc), ("mcmc_noise", noise)]:
                row[name] = timed(fn, n, a.reps)
        except torch.cuda.OutOfMemoryError as e:
            row["error"] = f"OOM: {e}".splitlines()[0]
        torch.cuda.empty_cache()
        print(json.dumps(row), flush=True)
        report["rows"].append(row)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
