"""Float64 PyTorch references for the oracle ops (used by `check` and `selftest`).

Each function mirrors the semantics of the gsplat CUDA op it checks, written
plainly so it can be read as a spec. Where gsplat ships a PyTorch reference
(gsplat.cuda._torch_impl, Apache-2.0) we call it in float64; the rasterizer,
SSIM and Adam references are our own. Discrete decisions (radii, tile lists,
alpha skips, early termination) are computed in float64 too, so a handful of
boundary elements may legitimately differ from a float32 kernel.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from gsplat.cuda import _torch_impl as ti

F64 = torch.float64
ALPHA_THRESHOLD = 1.0 / 255.0
T_THRESHOLD = 1e-4
MAX_ALPHA = 0.999


def sh(degree: int, dirs: torch.Tensor, coeffs: torch.Tensor, masks: torch.Tensor | None) -> torch.Tensor:
    out = ti._spherical_harmonics(degree, dirs.to(F64), coeffs.to(F64))
    if masks is not None:
        out = out * masks[..., None].to(F64)
    return out


def quat_scale_to_covar(quats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    covars, _ = ti._quat_scale_to_covar_preci(quats.to(F64), scales.to(F64), compute_preci=False, triu=False)
    return covars


def covar_triu_to_full(c6: torch.Tensor) -> torch.Tensor:
    a, b, c, d, e, f = c6.unbind(-1)
    return torch.stack([a, b, c, b, d, e, c, e, f], -1).reshape(*c6.shape[:-1], 3, 3)


def project(
    means,
    covars33,
    viewmats,
    Ks,
    width,
    height,
    eps2d,
    near,
    far,
    calc_comp,
    camera_model="pinhole",
    opacities=None,
    radius_clip=0.0,
):
    """gsplat fully_fused_projection (dense) in float64, including opacity-aware radii.

    Built from gsplat's PyTorch pieces (_world_to_cam, _persp_proj, ...); the
    culling and radius rules follow ProjectionEWA3DGSFused.cu:
      cull if z < near or z > far, or det(S + eps2d I) <= 0
      opacity-aware: o' = o * compensation (when computed); cull if o' < 1/255;
                     extend = min(3.33, sqrt(2 ln(o' * 255))), else extend = 3.33
      radius = ceil(extend * sqrt(S_ii)); cull if both <= radius_clip or off-image
    """
    means_c, covars_c = ti._world_to_cam(means.to(F64), covars33.to(F64), viewmats.to(F64))
    proj = {"pinhole": ti._persp_proj, "ortho": ti._ortho_proj, "fisheye": ti._fisheye_proj}[camera_model]
    means2d, cov = proj(means_c, covars_c, Ks.to(F64), width, height)
    det_orig = cov[..., 0, 0] * cov[..., 1, 1] - cov[..., 0, 1] * cov[..., 1, 0]
    cov = cov + torch.eye(2, dtype=F64, device=cov.device) * eps2d
    det = cov[..., 0, 0] * cov[..., 1, 1] - cov[..., 0, 1] * cov[..., 1, 0]
    safe_det = det.clamp(min=1e-10)
    comp = torch.sqrt(torch.clamp(det_orig / safe_det, min=0.0))
    conics = torch.stack(
        [cov[..., 1, 1] / safe_det, -(cov[..., 0, 1] + cov[..., 1, 0]) / 2.0 / safe_det, cov[..., 0, 0] / safe_det], -1
    )
    depths = means_c[..., 2]
    with torch.no_grad():
        valid = (det > 0) & (depths >= near) & (depths <= far)
        extend = torch.full_like(depths, 3.33)
        if opacities is not None:
            o = opacities.to(F64)[None].expand_as(depths) * (comp if calc_comp else 1.0)
            valid &= o >= ALPHA_THRESHOLD
            extend = torch.minimum(extend, torch.sqrt(2.0 * torch.log(o.clamp(min=ALPHA_THRESHOLD) / ALPHA_THRESHOLD)))
        rx = torch.ceil(extend * torch.sqrt(cov[..., 0, 0].clamp(min=0)))
        ry = torch.ceil(extend * torch.sqrt(cov[..., 1, 1].clamp(min=0)))
        valid &= ~((rx <= radius_clip) & (ry <= radius_clip))
        valid &= (means2d[..., 0] + rx > 0) & (means2d[..., 0] - rx < width)
        valid &= (means2d[..., 1] + ry > 0) & (means2d[..., 1] - ry < height)
        radii = (torch.stack([rx, ry], -1) * valid[..., None]).int()
    return radii, means2d, depths, conics, (comp if calc_comp else None)


@torch.no_grad()
def isect(means2d, radii, depths, tile_size, tile_width, tile_height):
    """gsplat's own (slow, loop-based) reference, on the float32 inputs as the kernel sees them."""
    tpg, ids, flat = ti._isect_tiles(
        means2d.float(), radii.int(), depths.float(), tile_size, tile_width, tile_height, sort=True
    )
    offsets = ti._isect_offset_encode(ids, means2d.shape[0], tile_width, tile_height)
    return tpg, ids, flat, offsets


def rasterize(means2d, conics, colors, opacities, width, height, tile_size, offsets, flatten_ids, backgrounds=None):
    """Front-to-back alpha compositing over the given per-tile lists, float64.

    Per pixel (center at +0.5), for each Gaussian in its tile's list in order:
      sigma = 0.5 * (a dx^2 + c dy^2) + b dx dy   (conic = [a, b, c], d = mean - pixel)
      alpha = min(0.999, opacity * exp(-sigma)); skipped if sigma < 0 or alpha < 1/255
      stop *before* a Gaussian that would bring T to <= 1e-4
      color += alpha * T * c; T *= 1 - alpha
    render_alpha = 1 - T; color += T * background.
    """
    C, N = means2d.shape[:2]
    D = colors.shape[-1]
    th, tw = offsets.shape[-2:]
    dev = means2d.device
    m2 = means2d.to(F64).reshape(C * N, 2)
    cn = conics.to(F64).reshape(C * N, 3)
    co = colors.to(F64).reshape(C * N, D)
    op = opacities.to(F64).reshape(C * N)
    bounds = offsets.reshape(-1).long().tolist() + [int(flatten_ids.numel())]
    fid = flatten_ids.long()
    images, alphas = [], []
    for c in range(C):
        color_rows, t_rows = [], []
        for ty in range(th):
            y0, y1 = ty * tile_size, min((ty + 1) * tile_size, height)
            if y0 >= height:
                break
            color_row, t_row = [], []
            for tx in range(tw):
                x0, x1 = tx * tile_size, min((tx + 1) * tile_size, width)
                if x0 >= width:
                    break
                t = (c * th + ty) * tw + tx
                ids = fid[bounds[t] : bounds[t + 1]]
                ys, xs = torch.meshgrid(
                    torch.arange(y0, y1, device=dev, dtype=F64),
                    torch.arange(x0, x1, device=dev, dtype=F64),
                    indexing="ij",
                )
                px, py = (xs + 0.5).reshape(-1, 1), (ys + 0.5).reshape(-1, 1)
                if ids.numel() == 0:
                    col = torch.zeros(px.shape[0], D, dtype=F64, device=dev)
                    T = torch.ones(px.shape[0], 1, dtype=F64, device=dev)
                else:
                    dx = m2[ids, 0][None] - px
                    dy = m2[ids, 1][None] - py
                    a, b, cc = cn[ids, 0][None], cn[ids, 1][None], cn[ids, 2][None]
                    sigma = 0.5 * (a * dx * dx + cc * dy * dy) + b * dx * dy
                    alpha = torch.clamp(op[ids][None] * torch.exp(-sigma), max=MAX_ALPHA)
                    use = (sigma >= 0) & (alpha >= ALPHA_THRESHOLD)
                    alpha = torch.where(use, alpha, torch.zeros_like(alpha))
                    with torch.no_grad():
                        stop = torch.cumprod(1.0 - alpha, dim=1) <= T_THRESHOLD
                        keep = torch.cumsum(stop.to(torch.int32), dim=1) == 0
                    alpha = alpha * keep
                    t_excl = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha[:, :-1]], 1), dim=1)
                    col = (alpha * t_excl) @ co[ids]
                    T = t_excl[:, -1:] * (1.0 - alpha[:, -1:])
                color_row.append(col.reshape(y1 - y0, x1 - x0, D))
                t_row.append(T.reshape(y1 - y0, x1 - x0, 1))
            color_rows.append(torch.cat(color_row, 1))
            t_rows.append(torch.cat(t_row, 1))
        img, T = torch.cat(color_rows, 0), torch.cat(t_rows, 0)
        if backgrounds is not None:
            img = img + T * backgrounds.to(F64)[c][None, None, :]
        images.append(img)
        alphas.append(1.0 - T)
    return torch.stack(images), torch.stack(alphas)


def gaussian_window(size: int = 11, sigma: float = 1.5, dtype=F64, device="cuda") -> torch.Tensor:
    x = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(x * x) / (2 * sigma * sigma))
    g = g / g.sum()
    return g[:, None] @ g[None, :]


def ssim_map(img1: torch.Tensor, img2: torch.Tensor, valid: bool = False) -> torch.Tensor:
    """3DGS utils/loss_utils.ssim: 11x11 Gaussian (sigma 1.5), zero padding, C1=0.01^2, C2=0.03^2.

    valid=True crops 5 px per side, as fused-ssim's padding="valid" does.
    """
    ch = img1.shape[1]
    w = gaussian_window(dtype=img1.dtype, device=img1.device)[None, None].expand(ch, 1, 11, 11).contiguous()

    def conv(x):
        return F.conv2d(x, w, padding=5, groups=ch)

    mu1, mu2 = conv(img1), conv(img2)
    s11 = conv(img1 * img1) - mu1 * mu1
    s22 = conv(img2 * img2) - mu2 * mu2
    s12 = conv(img1 * img2) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 * mu1 + mu2 * mu2 + c1) * (s11 + s22 + c2))
    return m[:, :, 5:-5, 5:-5] if valid else m


def adam_torch(param, grad, m, v, step, lr, b1, b2, eps):
    """torch.optim.Adam (single-tensor path) after `step` previous steps, float64."""
    p, g, m, v = (t.to(F64) for t in (param, grad, m, v))
    t = step + 1
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    bc1 = 1 - b1**t
    bc2 = 1 - b2**t
    denom = v.sqrt() / math.sqrt(bc2) + eps
    p = p - (lr / bc1) * m / denom
    return p, m, v


def adam_selective(param, grad, m, v, valid, lr, b1, b2, eps):
    """gsplat's fused Adam kernel (Taming-3DGS style): no bias correction, rows with valid=False untouched."""
    p, g, m0, v0 = (t.to(F64) for t in (param, grad, m, v))
    m1 = b1 * m0 + (1 - b1) * g
    v1 = b2 * v0 + (1 - b2) * g * g
    p1 = p - lr * m1 / (v1.sqrt() + eps)
    rows = valid.reshape(valid.shape[0], *([1] * (param.dim() - 1))).bool()
    return torch.where(rows, p1, p), torch.where(rows, m1, m0), torch.where(rows, v1, v0)
