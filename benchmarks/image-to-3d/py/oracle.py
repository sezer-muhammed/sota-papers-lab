#!/usr/bin/env python3
"""gs-oracle: run gsplat's reference CUDA ops on npz inputs written by our tests.

  gs-oracle list
  gs-oracle <op> in.npz out.npz           one op (ops: sh project isect rasterize ssim adam render)
  gs-oracle batch DIR                     every DIR/<name>.<op>.in.npz -> DIR/<name>.<op>.out.npz
  gs-oracle example <op> out.npz [...]    write a random input (oracle_example.py)
  gs-oracle check <op> in.npz out.npz     compare any out.npz (ours or gsplat's) to the float64 reference
  gs-oracle compare a.npz b.npz           per-key error table between two npz files
  gs-oracle selftest [--json report.json] gsplat vs float64 reference on random scenes, through files

The npz schemas are documented in benchmarks/image-to-3d/ORACLE.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

DEV = torch.device("cuda")
OPS: dict[str, Callable[..., dict]] = {}


def op(name: str):
    def reg(fn):
        OPS[name] = fn
        return fn

    return reg


class Inputs:
    """Typed access to an npz dict; records which keys were used."""

    def __init__(self, data: dict[str, np.ndarray]):
        self.d = data
        self.used: set[str] = set()

    def has(self, k: str) -> bool:
        return k in self.d

    def raw(self, k: str) -> np.ndarray:
        if k not in self.d:
            raise KeyError(f"missing input {k!r}")
        self.used.add(k)
        return self.d[k]

    def f32(self, k: str, grad: bool = False) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(self.raw(k), dtype=np.float32)).to(DEV)
        return t.requires_grad_(True) if grad else t

    def i32(self, k: str) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(self.raw(k), dtype=np.int32)).to(DEV)

    def mask(self, k: str) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(self.raw(k)) != 0).to(DEV)

    def int(self, k: str, default: int | None = None) -> int:
        if k not in self.d:
            if default is None:
                raise KeyError(f"missing input {k!r}")
            return default
        return int(np.asarray(self.raw(k)).reshape(()).item())

    def float(self, k: str, default: float) -> float:
        return float(np.asarray(self.raw(k)).reshape(()).item()) if k in self.d else default

    def flag(self, k: str, default: bool = False) -> bool:
        return bool(self.int(k, int(default)))

    def unused(self) -> list[str]:
        return sorted(set(self.d) - self.used)


CAMERA_MODELS = {0: "pinhole", 1: "ortho", 2: "fisheye"}


def zeros_or(x: Inputs, k: str, like: torch.Tensor) -> torch.Tensor:
    return x.f32(k) if x.has(k) else torch.zeros_like(like)


def zero_culled(out: dict, keys=("means2d", "depths", "conics", "compensations")) -> dict:
    """gsplat leaves culled entries (radii == 0) uninitialised; report them as 0."""
    valid = (torch.as_tensor(out["radii"]) > 0).all(-1)
    for k in keys:
        if k in out and out[k] is not None:
            v = out[k].detach()
            out[k] = torch.where(valid.reshape(valid.shape + (1,) * (v.dim() - valid.dim())), v, torch.zeros_like(v))
    return out


def grads(outputs, upstream, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """v_<name> for every input, given upstream gradients for the outputs."""
    names = list(inputs)
    g = torch.autograd.grad(outputs, [inputs[n] for n in names], upstream, allow_unused=True)
    return {f"v_{n}": (gi if gi is not None else torch.zeros_like(inputs[n])) for n, gi in zip(names, g, strict=True)}


@op("sh")
def op_sh(x: Inputs) -> dict:
    from gsplat.cuda._wrapper import spherical_harmonics

    dirs, coeffs = x.f32("dirs", True), x.f32("coeffs", True)
    masks = x.mask("masks") if x.has("masks") else None
    colors = spherical_harmonics(x.int("degree"), dirs, coeffs, masks)
    out = {"colors": colors}
    if x.has("v_colors"):
        out |= grads([colors], [x.f32("v_colors")], {"dirs": dirs, "coeffs": coeffs})
    return out


@op("project")
def op_project(x: Inputs) -> dict:
    from gsplat.cuda._wrapper import fully_fused_projection

    means, viewmats, ks = x.f32("means", True), x.f32("viewmats", True), x.f32("Ks")
    params: dict[str, torch.Tensor] = {"means": means}
    if x.has("covars"):
        params["covars"] = x.f32("covars", True)
        quats = scales = None
    else:
        params["quats"] = quats = x.f32("quats", True)
        params["scales"] = scales = x.f32("scales", True)
    params["viewmats"] = viewmats
    comp = x.flag("calc_compensations")
    radii, means2d, depths, conics, comps = fully_fused_projection(
        means,
        params.get("covars"),
        quats,
        scales,
        viewmats,
        ks,
        x.int("width"),
        x.int("height"),
        eps2d=x.float("eps2d", 0.3),
        near_plane=x.float("near_plane", 0.01),
        far_plane=x.float("far_plane", 1e10),
        radius_clip=x.float("radius_clip", 0.0),
        packed=False,
        calc_compensations=comp,
        camera_model=CAMERA_MODELS[x.int("camera_model", 0)],
        opacities=x.f32("opacities") if x.has("opacities") else None,
    )
    out = {"radii": radii, "means2d": means2d, "depths": depths, "conics": conics}
    outs = [means2d, depths, conics]
    keys = ["v_means2d", "v_depths", "v_conics"]
    if comp:
        out["compensations"] = comps
        outs.append(comps)
        keys.append("v_compensations")
    if any(x.has(k) for k in keys):
        out |= grads(outs, [zeros_or(x, k, o) for k, o in zip(keys, outs, strict=True)], params)
    return zero_culled(out)


def tile_grid(x: Inputs) -> tuple[int, int, int, int, int]:
    w, h, ts = x.int("width"), x.int("height"), x.int("tile_size", 16)
    tw = x.int("tile_width", math.ceil(w / ts))
    th = x.int("tile_height", math.ceil(h / ts))
    return w, h, ts, tw, th


@op("isect")
def op_isect(x: Inputs) -> dict:
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    _, _, ts, tw, th = tile_grid(x)
    means2d, radii, depths = x.f32("means2d"), x.i32("radii"), x.f32("depths")
    tpg, ids, flat = isect_tiles(means2d, radii, depths, ts, tw, th, sort=True, packed=False)
    offsets = isect_offset_encode(ids, means2d.shape[0], tw, th)
    return {
        "tiles_per_gauss": tpg,
        "isect_ids": ids,
        "flatten_ids": flat,
        "isect_offsets": offsets,
        "tile_width": np.array(tw, np.int32),
        "tile_height": np.array(th, np.int32),
    }


@op("rasterize")
def op_rasterize(x: Inputs) -> dict:
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles, rasterize_to_pixels

    w, h, ts, tw, th = tile_grid(x)
    means2d, conics, colors = x.f32("means2d", True), x.f32("conics", True), x.f32("colors", True)
    C, N = means2d.shape[:2]
    opac_in = x.f32("opacities", True)
    opacities = opac_in.expand(C, N) if opac_in.dim() == 1 else opac_in
    bg = x.f32("backgrounds", True) if x.has("backgrounds") else None
    out: dict = {}
    if x.has("isect_offsets"):
        offsets, flat = x.i32("isect_offsets"), x.i32("flatten_ids")
    else:
        _, ids, flat = isect_tiles(means2d.detach(), x.i32("radii"), x.f32("depths"), ts, tw, th, sort=True)
        offsets = isect_offset_encode(ids, C, tw, th)
        out |= {"isect_ids": ids, "flatten_ids": flat, "isect_offsets": offsets}
    absgrad = x.flag("absgrad")
    rc, ra = rasterize_to_pixels(
        means2d, conics, colors, opacities, w, h, ts, offsets, flat, backgrounds=bg, packed=False, absgrad=absgrad
    )
    out |= {"render_colors": rc, "render_alphas": ra}
    if x.has("v_render_colors") or x.has("v_render_alphas"):
        loss = (rc * zeros_or(x, "v_render_colors", rc)).sum() + (ra * zeros_or(x, "v_render_alphas", ra)).sum()
        loss.backward()
        out |= {"v_means2d": means2d.grad, "v_conics": conics.grad, "v_colors": colors.grad}
        out["v_opacities"] = opac_in.grad
        if bg is not None:
            out["v_backgrounds"] = bg.grad
        if absgrad:
            out["v_means2d_abs"] = means2d.absgrad
    return out


@op("ssim")
def op_ssim(x: Inputs) -> dict:
    from fused_ssim import fused_ssim

    import oracle_ref as ref

    img1, img2 = x.f32("img1"), x.f32("img2")
    if img1.dim() == 3:
        img1, img2 = img1[None], img2[None]
    valid = x.flag("padding_valid")
    padding = "valid" if valid else "same"
    lam = x.float("lambda_dssim", 0.2)
    v_ssim = x.float("v_ssim", 1.0)

    a = img1.clone().requires_grad_(True)
    s = fused_ssim(a, img2, padding=padding, train=True)
    (v_img1,) = torch.autograd.grad(s, [a], torch.tensor(v_ssim, device=DEV))

    b = img1.clone().requires_grad_(True)
    l1 = (b - img2).abs().mean()
    loss = (1.0 - lam) * l1 + lam * (1.0 - fused_ssim(b, img2, padding=padding, train=True))
    (v_loss,) = torch.autograd.grad(loss, [b])

    m = ref.ssim_map(img1.double(), img2.double(), valid)
    return {
        "ssim": s,
        "v_img1": v_img1,
        "l1": l1,
        "loss": loss,
        "v_img1_loss": v_loss,
        "ssim_map_ref": m,
        "ssim_ref": m.mean(),
    }


@op("adam")
def op_adam(x: Inputs) -> dict:
    from gsplat.cuda._wrapper import adam

    param, grad, m, v = x.f32("param"), x.f32("grad"), x.f32("exp_avg"), x.f32("exp_avg_sq")
    step = x.int("step", 0)
    lr, b1, b2, eps = x.float("lr", 1e-3), x.float("beta1", 0.9), x.float("beta2", 0.999), x.float("eps", 1e-15)

    p = param.clone()
    opt = torch.optim.Adam([p], lr=lr, betas=(b1, b2), eps=eps, foreach=False, fused=False)
    p.grad = grad.clone()
    opt.state[p] = {"step": torch.tensor(float(step)), "exp_avg": m.clone(), "exp_avg_sq": v.clone()}
    opt.step()
    st = opt.state[p]
    out = {"param_out": p.detach(), "exp_avg_out": st["exp_avg"], "exp_avg_sq_out": st["exp_avg_sq"]}
    if x.has("valid"):
        p2, m2, v2 = param.clone(), m.clone(), v.clone()
        adam(p2, grad, m2, v2, x.mask("valid"), lr, b1, b2, eps)
        out |= {"sel_param_out": p2, "sel_exp_avg_out": m2, "sel_exp_avg_sq_out": v2}
    return out


@op("render")
def op_render(x: Inputs) -> dict:
    from gsplat.rendering import rasterization

    params = {k: x.f32(k, True) for k in ("means", "quats", "scales", "opacities", "colors")}
    viewmats = x.f32("viewmats", x.flag("grad_viewmats", True))
    if viewmats.requires_grad:
        params["viewmats"] = viewmats
    colors = params["colors"]
    sh_degree = x.int("sh_degree", math.isqrt(colors.shape[-2]) - 1) if colors.dim() == 3 else -1
    absgrad = x.flag("absgrad")
    rc, ra, meta = rasterization(
        params["means"],
        params["quats"],
        params["scales"],
        params["opacities"],
        params["colors"],
        viewmats,
        x.f32("Ks"),
        x.int("width"),
        x.int("height"),
        near_plane=x.float("near_plane", 0.01),
        far_plane=x.float("far_plane", 1e10),
        radius_clip=x.float("radius_clip", 0.0),
        eps2d=x.float("eps2d", 0.3),
        sh_degree=None if sh_degree < 0 else sh_degree,
        packed=False,
        tile_size=x.int("tile_size", 16),
        backgrounds=x.f32("backgrounds") if x.has("backgrounds") else None,
        absgrad=absgrad,
        rasterize_mode="antialiased" if x.int("rasterize_mode", 0) == 1 else "classic",
        camera_model=CAMERA_MODELS[x.int("camera_model", 0)],
    )
    means2d = meta["means2d"]
    means2d.retain_grad()
    out = {
        "render_colors": rc,
        "render_alphas": ra,
        "radii": meta["radii"],
        "means2d": means2d,
        "depths": meta["depths"],
        "conics": meta["conics"],
        "opacities_view": meta["opacities"],
        "tiles_per_gauss": meta["tiles_per_gauss"],
        "isect_ids": meta["isect_ids"],
        "flatten_ids": meta["flatten_ids"],
        "isect_offsets": meta["isect_offsets"],
    }
    if x.has("v_render_colors") or x.has("v_render_alphas"):
        loss = (rc * zeros_or(x, "v_render_colors", rc)).sum() + (ra * zeros_or(x, "v_render_alphas", ra)).sum()
        loss.backward()
        out |= {f"v_{k}": t.grad for k, t in params.items()}
        out["v_means2d"] = means2d.grad
        if absgrad:
            out["v_means2d_abs"] = means2d.absgrad
    return zero_culled(out)


# ---------------------------------------------------------------- file I/O


def to_numpy(v) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        v = v.detach()
        if v.dtype == torch.float64 or v.is_floating_point():
            v = v.to(torch.float64 if v.dtype == torch.float64 else torch.float32)
        return v.cpu().numpy()
    return np.asarray(v)


def versions() -> dict:
    import gsplat

    info = {"gsplat": gsplat.__version__, "torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        from importlib.metadata import version

        info["fused_ssim"] = version("fused_ssim")
    except Exception:  # noqa: BLE001 - metadata is best effort
        pass
    info["device"] = torch.cuda.get_device_name(0)
    return info


def run(op_name: str, data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    if op_name not in OPS:
        raise SystemExit(f"unknown op {op_name!r}; ops: {', '.join(OPS)}")
    x = Inputs(data)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = OPS[op_name](x)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    arrays = {k: to_numpy(v) for k, v in out.items() if v is not None}
    meta = {"op": op_name, "ms": round(ms, 3), "unused_inputs": x.unused(), **versions()}
    return arrays, meta


def run_file(op_name: str, src: Path, dst: Path) -> dict:
    with np.load(src, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    arrays, meta = run(op_name, data)
    if meta["unused_inputs"]:
        print(f"warning: {src.name}: unused inputs {meta['unused_inputs']}", file=sys.stderr)
    arrays["_meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(dst)
    return meta


# ------------------------------------------------------------- comparison


def err_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Errors of a against b. `rel` is max|a-b| / max|b| (scale-relative)."""
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "ok": False}
    if a.dtype.kind in "iub" and b.dtype.kind in "iub":
        bad = int(np.count_nonzero(a.astype(np.int64) != b.astype(np.int64)))
        return {"n": int(a.size), "mismatch": bad, "frac": bad / max(a.size, 1)}
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    d = np.abs(a64 - b64)
    scale = float(np.max(np.abs(b64))) if b64.size else 0.0
    max_abs = float(d.max()) if d.size else 0.0
    rel = max_abs / scale if scale > 0 else max_abs
    p999 = float(np.quantile(d, 0.999)) / scale if (d.size and scale > 0) else 0.0
    return {
        "n": int(a.size),
        "max_abs": max_abs,
        "scale": scale,
        "rel": rel,
        "p999_rel": p999,
        "nonfinite": int(np.count_nonzero(~np.isfinite(a64))),
    }


def compare(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict:
    keys = sorted((set(a) & set(b)) - {"_meta"})
    return {k: err_stats(a[k], b[k]) for k in keys}


def print_table(title: str, stats: dict, tol: dict | None = None) -> bool:
    print(f"\n{title}")
    ok_all = True
    for k, s in stats.items():
        if "mismatch" in s:
            ok = s["mismatch"] == 0 if tol is None else s["frac"] <= tol.get(k, tol.get("*int", 0.0))
            line = f"  {k:<22} int    mismatches {s['mismatch']}/{s['n']}"
        elif "rel" in s:
            limit = None if tol is None else tol.get(k, tol.get("*", 1e-4))
            ok = (
                s["nonfinite"] == 0
                and (limit is None or s["p999_rel"] <= limit)
                and (limit is None or s["rel"] <= 20 * limit)
            )
            line = (
                f"  {k:<22} float  max_abs {s['max_abs']:.3e}  scale {s['scale']:.3e}  "
                f"rel {s['rel']:.3e}  p99.9 {s['p999_rel']:.3e}" + (f"  (tol {limit:.0e})" if limit else "")
            )
        else:
            ok = False
            line = f"  {k:<22} shape mismatch {s}"
        ok_all &= ok
        print(("  ok " if ok else "  BAD") + line[1:])
    return ok_all


# ------------------------------------------------------- float64 references

TOL = {
    # p99.9 of |ours - ref| / max|ref|; the max itself may be up to 20x that.
    "sh": {"*": 1e-5},
    "project": {"*": 1e-4, "radii": 0.0, "*int": 0.0},
    "isect": {"*int": 0.0},
    "rasterize": {"*": 1e-4, "v_means2d_abs": 1e-3},
    "ssim": {"*": 1e-4},
    "adam": {"*": 1e-5},
    "render": {"*": 2e-4, "*int": 1e-3},
}


def reference(op_name: str, data: dict[str, np.ndarray], cand: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Float64 reference outputs for the keys an op produces.

    Discrete inputs to later stages (tile lists) are taken from `cand` when the
    op computes them itself, so a float boundary case does not cascade.
    """
    import oracle_ref as ref

    x = Inputs(data)
    t = lambda k: torch.from_numpy(np.asarray(x.raw(k))).to(DEV)  # noqa: E731
    f = lambda k, g=False: t(k).double().requires_grad_(g)  # noqa: E731
    out: dict[str, torch.Tensor] = {}
    if op_name == "sh":
        dirs, coeffs = f("dirs", True), f("coeffs", True)
        masks = t("masks") != 0 if x.has("masks") else None
        colors = ref.sh(x.int("degree"), dirs, coeffs, masks)
        out["colors"] = colors
        if x.has("v_colors"):
            out |= grads([colors], [f("v_colors")], {"dirs": dirs, "coeffs": coeffs})
    elif op_name == "project":
        means, viewmats = f("means", True), f("viewmats", True)
        params = {"means": means}
        if x.has("covars"):
            params["covars"] = f("covars", True)
            cov = ref.covar_triu_to_full(params["covars"])
        else:
            params["quats"], params["scales"] = f("quats", True), f("scales", True)
            cov = ref.quat_scale_to_covar(params["quats"], params["scales"])
        params["viewmats"] = viewmats
        comp = x.flag("calc_compensations")
        radii, means2d, depths, conics, comps = ref.project(
            means,
            cov,
            viewmats,
            f("Ks"),
            x.int("width"),
            x.int("height"),
            x.float("eps2d", 0.3),
            x.float("near_plane", 0.01),
            x.float("far_plane", 1e10),
            comp,
            camera_model=CAMERA_MODELS[x.int("camera_model", 0)],
            opacities=f("opacities") if x.has("opacities") else None,
            radius_clip=x.float("radius_clip", 0.0),
        )
        out |= {"radii": radii, "means2d": means2d, "depths": depths, "conics": conics}
        outs, keys = [means2d, depths, conics], ["v_means2d", "v_depths", "v_conics"]
        if comp:
            out["compensations"] = comps
            outs.append(comps)
            keys.append("v_compensations")
        if any(x.has(k) for k in keys):
            # gsplat zeroes the gradient of culled Gaussians (radii == 0).
            valid = (radii > 0).all(-1)
            ups = []
            for k, o in zip(keys, outs, strict=True):
                g = f(k) if x.has(k) else torch.zeros_like(o)
                ups.append(g * valid.reshape(valid.shape + (1,) * (g.dim() - valid.dim())))
            out |= grads(outs, ups, params)
    elif op_name == "isect":
        _, _, ts, tw, th = tile_grid(x)
        tpg, ids, flat, offsets = ref.isect(t("means2d"), t("radii"), t("depths"), ts, tw, th)
        out |= {"tiles_per_gauss": tpg, "isect_ids": ids, "flatten_ids": flat, "isect_offsets": offsets}
    elif op_name == "rasterize":
        w, h, ts, tw, th = tile_grid(x)
        means2d, conics, colors = f("means2d", True), f("conics", True), f("colors", True)
        C, N = means2d.shape[:2]
        opac_in = f("opacities", True)
        opac = opac_in.expand(C, N) if opac_in.dim() == 1 else opac_in
        params = {"means2d": means2d, "conics": conics, "colors": colors, "opacities": opac_in}
        bg = None
        if x.has("backgrounds"):
            params["backgrounds"] = bg = f("backgrounds", True)
        if x.has("isect_offsets"):
            offsets, flat = t("isect_offsets"), t("flatten_ids")
        else:
            offsets = torch.from_numpy(cand["isect_offsets"]).to(DEV)
            flat = torch.from_numpy(cand["flatten_ids"]).to(DEV)
            _, ids_ref, flat_ref, off_ref = ref.isect(t("means2d"), t("radii"), t("depths"), ts, tw, th)
            out |= {"isect_ids": ids_ref, "flatten_ids": flat_ref, "isect_offsets": off_ref}
        rc, ra = ref.rasterize(means2d, conics, colors, opac, w, h, ts, offsets, flat, bg)
        out |= {"render_colors": rc, "render_alphas": ra}
        if x.has("v_render_colors") or x.has("v_render_alphas"):
            ups = [
                f("v_render_colors") if x.has("v_render_colors") else torch.zeros_like(rc),
                f("v_render_alphas") if x.has("v_render_alphas") else torch.zeros_like(ra),
            ]
            out |= grads([rc, ra], ups, params)
    elif op_name == "ssim":
        img1, img2 = f("img1", True), f("img2")
        if img1.dim() == 3:
            img1, img2 = img1[None].detach().requires_grad_(True), img2[None]
        valid = x.flag("padding_valid")
        lam = x.float("lambda_dssim", 0.2)
        m = ref.ssim_map(img1, img2, valid)
        s = m.mean()
        l1 = (img1 - img2).abs().mean()
        loss = (1 - lam) * l1 + lam * (1 - s)
        v_ssim = torch.tensor(x.float("v_ssim", 1.0), device=DEV, dtype=torch.float64)
        (v_img1,) = torch.autograd.grad(s, [img1], v_ssim, retain_graph=True)
        (v_loss,) = torch.autograd.grad(loss, [img1])
        out |= {
            "ssim": s,
            "v_img1": v_img1,
            "l1": l1,
            "loss": loss,
            "v_img1_loss": v_loss,
            "ssim_map_ref": m,
            "ssim_ref": s,
        }
    elif op_name == "adam":
        args = [f("param"), f("grad"), f("exp_avg"), f("exp_avg_sq")]
        hp = (x.float("lr", 1e-3), x.float("beta1", 0.9), x.float("beta2", 0.999), x.float("eps", 1e-15))
        p, m, v = ref.adam_torch(*args, x.int("step", 0), *hp)
        out |= {"param_out": p, "exp_avg_out": m, "exp_avg_sq_out": v}
        if x.has("valid"):
            p, m, v = ref.adam_selective(*args, t("valid") != 0, *hp)
            out |= {"sel_param_out": p, "sel_exp_avg_out": m, "sel_exp_avg_sq_out": v}
    elif op_name == "render":
        out |= reference_render(x, cand)
    else:
        raise SystemExit(f"no reference for {op_name!r}")
    if op_name in ("project", "render"):
        zero_culled(out)
    return {k: to_numpy(v) for k, v in out.items()}


def reference_render(x: Inputs, cand: dict[str, np.ndarray]) -> dict:
    """gsplat.rasterization (classic, dense) composed from the float64 references."""
    import oracle_ref as ref

    def f(k, g=False):
        return torch.from_numpy(np.asarray(x.raw(k))).to(DEV).double().requires_grad_(g)

    if x.int("rasterize_mode", 0) != 0:
        raise SystemExit("render reference covers rasterize_mode=0 (classic) only")
    params = {k: f(k, True) for k in ("means", "quats", "scales", "opacities", "colors")}
    viewmats = f("viewmats", x.flag("grad_viewmats", True))
    if viewmats.requires_grad:
        params["viewmats"] = viewmats
    w, h, ts, tw, th = tile_grid(x)
    means, colors = params["means"], params["colors"]
    cov = ref.quat_scale_to_covar(params["quats"], params["scales"])
    radii, means2d, depths, conics, _ = ref.project(
        means,
        cov,
        viewmats,
        f("Ks"),
        w,
        h,
        x.float("eps2d", 0.3),
        x.float("near_plane", 0.01),
        x.float("far_plane", 1e10),
        False,
        camera_model=CAMERA_MODELS[x.int("camera_model", 0)],
        opacities=params["opacities"].detach(),
        radius_clip=x.float("radius_clip", 0.0),
    )
    C, N = means2d.shape[:2]
    if colors.dim() == 3:
        campos = torch.inverse(viewmats)[:, :3, 3]
        dirs = means[None] - campos[:, None]
        rgb = ref.sh(x.int("sh_degree"), dirs, colors[None].expand(C, *colors.shape), (radii > 0).all(-1))
        rgb = torch.clamp_min(rgb + 0.5, 0.0)
    else:
        rgb = colors[None].expand(C, *colors.shape)
    opac = params["opacities"][None].expand(C, N)
    offsets = torch.from_numpy(cand["isect_offsets"]).to(DEV)
    flat = torch.from_numpy(cand["flatten_ids"]).to(DEV)
    bg = f("backgrounds") if x.has("backgrounds") else None
    rc, ra = ref.rasterize(means2d, conics, rgb, opac, w, h, ts, offsets, flat, bg)
    out = {
        "render_colors": rc,
        "render_alphas": ra,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
    }
    if x.has("v_render_colors") or x.has("v_render_alphas"):
        ups = [
            f("v_render_colors") if x.has("v_render_colors") else torch.zeros_like(rc),
            f("v_render_alphas") if x.has("v_render_alphas") else torch.zeros_like(ra),
        ]
        out |= grads([rc, ra], ups, params)
    return out


def check(op_name: str, src: Path, cand_path: Path, tol: dict | None = None) -> tuple[bool, dict]:
    with np.load(src) as z:
        data = {k: z[k] for k in z.files}
    with np.load(cand_path) as z:
        cand = {k: z[k] for k in z.files}
    want = {k: v for k, v in reference(op_name, data, cand).items() if not k.endswith("_ref")}
    stats = compare(cand, want)
    missing = sorted(set(want) - set(cand))
    ok = print_table(f"{op_name}: {cand_path.name} vs float64 reference", stats, tol or TOL.get(op_name))
    if missing:
        print(f"  missing outputs: {missing}")
    return ok and not missing, stats


# ---------------------------------------------------------------- selftest

# (case, op, oracle_example.make kwargs, input overrides)
SELFTEST_CASES = [
    ("sh", "sh", dict(n=512), {}),
    ("project", "project", dict(n=512, c=2), {}),
    ("isect", "isect", dict(n=256, c=2), {}),
    ("rasterize", "rasterize", dict(n=128, w=64, h=48, c=2), {}),
    ("ssim", "ssim", dict(w=64, h=48, c=2), {}),
    ("ssim_valid", "ssim", dict(w=64, h=48, c=1), {"padding_valid": np.array(1, np.int32)}),
    ("adam", "adam", dict(n=1000), {}),
    ("render", "render", dict(n=200, w=64, h=48, c=1), {}),
    ("render_2cam", "render", dict(n=300, w=80, h=60, c=2, seed=2), {}),
]


def selftest(report_path: str | None, keep: str | None) -> int:
    import oracle_example

    torch.manual_seed(0)
    tmp = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="gs-oracle-"))
    tmp.mkdir(parents=True, exist_ok=True)
    report = {"versions": versions(), "cases": {}}
    all_ok = True
    for name, op_name, kw, overrides in SELFTEST_CASES:
        args = dict(seed=1, n=256, w=64, h=48, c=1) | kw
        src, dst = tmp / f"{name}.{op_name}.in.npz", tmp / f"{name}.{op_name}.out.npz"
        np.savez(src, **(oracle_example.make(op_name, **args) | overrides))
        meta = run_file(op_name, src, dst)  # the same path a test's file takes
        ok, stats = check(op_name, src, dst)
        # Determinism: a second run through the file path (atomics may reorder sums).
        dst2 = tmp / f"{name}.{op_name}.rerun.npz"
        run_file(op_name, src, dst2)
        with np.load(dst) as z1, np.load(dst2) as z2:
            rerun = compare({k: z1[k] for k in z1.files}, {k: z2[k] for k in z2.files})
        nondet = {
            k: s.get("max_abs", s.get("mismatch")) for k, s in rerun.items() if s.get("max_abs", s.get("mismatch", 0))
        }
        print(f"  rerun differences: {nondet or 'none (bitwise identical)'}; op time {meta['ms']:.2f} ms")
        report["cases"][name] = {
            "ok": ok,
            "op": op_name,
            "args": args,
            "ms": meta["ms"],
            "vs_fp64": stats,
            "rerun": nondet,
        }
        all_ok &= ok
    report["ok"] = all_ok
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nselftest {'PASSED' if all_ok else 'FAILED'}; files in {tmp}")
    return 0 if all_ok else 1


# --------------------------------------------------------------------- CLI


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "list":
        for name, fn in OPS.items():
            print(f"{name:<10} {(fn.__doc__ or '').strip()}")
        return 0
    if cmd == "example":
        import oracle_example

        sys.argv = ["oracle_example.py", *argv[1:]]
        oracle_example.main()
        return 0
    if cmd == "batch":
        d = Path(argv[1])
        files = sorted(d.glob("*.in.npz"))
        if not files:
            print(f"no *.in.npz files in {d}", file=sys.stderr)
            return 1
        for src in files:
            stem = src.name[: -len(".in.npz")]
            op_name = stem.rsplit(".", 1)[-1]
            dst = src.with_name(stem + ".out.npz")
            meta = run_file(op_name, src, dst)
            print(f"{src.name} -> {dst.name} ({meta['ms']:.2f} ms)")
        return 0
    if cmd == "check":
        ap = argparse.ArgumentParser(prog="gs-oracle check")
        ap.add_argument("op")
        ap.add_argument("inp")
        ap.add_argument("out")
        ap.add_argument("--json")
        a = ap.parse_args(argv[1:])
        ok, stats = check(a.op, Path(a.inp), Path(a.out))
        if a.json:
            Path(a.json).write_text(json.dumps({"ok": ok, "stats": stats}, indent=2) + "\n")
        return 0 if ok else 1
    if cmd == "compare":
        with np.load(argv[1]) as za, np.load(argv[2]) as zb:
            stats = compare({k: za[k] for k in za.files}, {k: zb[k] for k in zb.files})
        print_table(f"{argv[1]} vs {argv[2]}", stats)
        return 0
    if cmd == "selftest":
        ap = argparse.ArgumentParser(prog="gs-oracle selftest")
        ap.add_argument("--json")
        ap.add_argument("--keep", help="write the npz files here instead of a temp dir")
        a = ap.parse_args(argv[1:])
        return selftest(a.json, a.keep)
    if cmd in OPS:
        if len(argv) != 3:
            print(f"usage: gs-oracle {cmd} in.npz out.npz", file=sys.stderr)
            return 2
        meta = run_file(cmd, Path(argv[1]), Path(argv[2]))
        print(json.dumps(meta))
        return 0
    print(f"unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
