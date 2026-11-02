"""PPISP oracle cases for `ppisp::tests` (owner: ppisp).

`torch_reference.py` is upstream's PyTorch reference, copied verbatim from
nv-tlabs/ppisp a247368 (Apache-2.0, notice kept in the file). This script
builds random cases, evaluates that reference in float64 on the CPU, and
writes inputs, outputs and autograd gradients to one .npz per case:

    python ref.py OUT_DIR

Run in any image with torch (spl-gs-ref or pytorch/pytorch).
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_reference as ref  # noqa: E402

ref._COLOR_PINV_BLOCK_DIAG = ref._COLOR_PINV_BLOCK_DIAG.double()


def softplus_inv(x):
    return float(np.log(np.expm1(x)))


def reg_loss(e, vig, col, crf, w):
    """Upstream tests/test_regularization_loss.py formula."""
    total = 0.0
    total = total + w[0] * F.smooth_l1_loss(e.mean(), torch.zeros(()), beta=0.1)
    total = total + w[1] * (vig[:, :, :2] ** 2).sum(-1).mean()
    total = total + w[3] * F.relu(vig[:, :, 2:]).mean()
    total = total + w[2] * vig.var(dim=1, unbiased=False).mean()
    r = (col @ ref._COLOR_PINV_BLOCK_DIAG).mean(0)
    total = total + w[4] * F.smooth_l1_loss(r, torch.zeros_like(r), beta=0.005)
    total = total + w[5] * crf.var(dim=1, unbiased=False).mean()
    return total


def case(name, out, seed, w, h, cams, frames, camera, frame):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    e = 0.3 * rn(frames)
    vig = torch.zeros(cams, 3, 5)
    vig[:, :, :2] = 0.05 * rn(cams, 3, 2)
    vig[:, :, 2:] = -0.4 * rn(cams, 3, 3).abs() + 0.05
    col = rn(frames, 8)
    crf = torch.tensor([softplus_inv(0.7), softplus_inv(0.7), softplus_inv(0.9), 0.0])
    crf = crf.repeat(cams, 3, 1) + 0.3 * rn(cams, 3, 4)
    rgb = torch.rand(h * w, 3, generator=g) * 1.3 - 0.1
    wgt = rn(h * w, 3)
    params = [t.clone().requires_grad_(True) for t in (e, vig, col, crf, rgb)]
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    coords = torch.stack([xs.reshape(-1) + 0.5, ys.reshape(-1) + 0.5], -1).double()
    o = ref.ppisp_apply_torch(*params[:4], params[4], coords, w, h, camera, frame)
    (o * wgt).sum().backward()
    for t in params:  # unused parameters (effects disabled) get zero gradients
        if t.grad is None:
            t.grad = torch.zeros_like(t)
    weights = [0.7, 0.03, 0.2, 0.05, 1.3, 0.15]
    rp = [t.clone().requires_grad_(True) for t in (e, vig, col, crf)]
    R = reg_loss(*rp, weights)
    R.backward()
    planar = lambda t: t.detach().T.reshape(3, h, w).numpy().astype(np.float32)  # noqa: E731
    f32 = lambda t: t.detach().numpy().astype(np.float32)  # noqa: E731
    np.savez(
        os.path.join(out, f"{name}.npz"),
        dims=np.array([w, h, cams, frames, camera, frame], dtype=np.int32),
        exposure=f32(e), vig=f32(vig), color=f32(col), crf=f32(crf),
        rgb=planar(rgb), wgt=planar(wgt), out=planar(o),
        g_exposure=f32(params[0].grad), g_vig=f32(params[1].grad),
        g_color=f32(params[2].grad), g_crf=f32(params[3].grad), g_rgb=planar(params[4].grad),
        reg_w=np.array(weights, dtype=np.float32), reg=np.array([R.item()], dtype=np.float32),
        r_exposure=f32(rp[0].grad), r_vig=f32(rp[1].grad),
        r_color=f32(rp[2].grad), r_crf=f32(rp[3].grad),
    )
    print(name, "out range", float(o.min()), float(o.max()))


def lr_case(out):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=0.002)
    gamma = 0.01 ** (1.0 / 30000)
    sch = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=500),
         torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)],
        milestones=[500],
    )
    lrs = []
    for _ in range(31000):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    np.savez(os.path.join(out, "lr.npz"), lr=np.array(lrs, dtype=np.float64))


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    case("full", out, 1, 67, 45, 2, 5, 1, 3)
    case("cam_only", out, 2, 40, 31, 2, 5, 0, -1)
    case("frame_only", out, 3, 40, 31, 2, 5, -1, 2)
    lr_case(out)
