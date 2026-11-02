"""VRAM fixes for Pixal3D on 12 GB cards: head-chunked NAF fused with the
projection, CuMesh cache release.

Pixal3D's texture stage upsamples 1024-channel DINOv3 features to 1024x1024
with NAF (valeoai/NAF, Apache-2.0). Its CrossAttention.forward resizes the
values to full resolution, runs na2d and rearranges, holding three 4 GiB fp32
tensors at once; on a 12 GB card that OOMs even with the GPU to itself.
Attention heads are independent, so running them in groups and writing each
group's result into one preallocated [B, C, H, W] output is exact and caps
the extra memory at one group.
"""
import torch
import torch.nn.functional as F

GROUP_BYTES = 512 << 20


def _chunked_forward(self, q, k, v, image=None, return_weights=False, **kwargs):
    if return_weights:
        return self._spl_orig_forward(q, k, v, image, return_weights=True, **kwargs)
    from natten.functional import na2d

    b, cq, hq, wq = q.shape
    hk, wk = k.shape[-2:]
    dilation = (hq // hk, wq // wk)
    self.dilation = dilation
    n = self.num_heads
    dq, dk, dv = cq // n, k.shape[1] // n, v.shape[1] // n
    out = torch.empty(b, v.shape[1], hq, wq, device=q.device, dtype=q.dtype)
    per_head = b * hq * wq * (dq + dk + 2 * dv) * q.element_size()
    g = max(1, min(n, GROUP_BYTES // max(per_head, 1)))
    for h0 in range(0, n, g):
        h1 = min(n, h0 + g)

        def heads(x, d, resize):
            x = x[:, h0 * d:h1 * d]
            if resize:
                x = F.interpolate(x, size=(hq, wq), mode="nearest-exact")
            x = x.to(q.dtype)
            return x.reshape(b, h1 - h0, d, hq, wq).permute(0, 3, 4, 1, 2).contiguous()

        o = na2d(heads(q, dq, False), heads(k, dk, True), heads(v, dv, True),
                 kernel_size=self.kernel_size, dilation=dilation, stride=1,
                 backend="cutlass-fna")
        out[:, h0 * dv:h1 * dv] = o.permute(0, 3, 4, 1, 2).reshape(b, (h1 - h0) * dv, hq, wq)
        del o
    return out


def _project_single_view(self, z_patchtokens_spatial, image_for_naf,
                         camera_angle_x_v, distance_v, mesh_scale_v, calc_mat_v):
    """Fused NAF + projection: never materialises the [b, C, H, W] HR map.

    Same result as DinoV3ProjMultiViewFeatureExtractor._project_single_view:
    the projection samples each channel independently, so each head group's
    NAF output is projected and written into its slice of the result.
    """
    from natten.functional import na2d

    args = (camera_angle_x_v, distance_v, mesh_scale_v, calc_mat_v)
    z_lr = self.proj_grid(z_patchtokens_spatial, *args)  # [b, K, D]
    if not self.use_naf_upsample:
        return z_lr
    self._load_naf()
    naf = self.naf_model
    feats = z_patchtokens_spatial.permute(0, 3, 1, 2)
    x = naf.image_encoder(image_for_naf, output_size=self.naf_target_size)
    q = naf.query_encoder(x)
    k = naf.key_encoder(x, feats)
    del x
    up = naf.upsampler
    b, cq, hq, wq = q.shape
    dilation = (hq // k.shape[-2], wq // k.shape[-1])
    n = up.num_heads
    d_lr = z_lr.shape[-1]
    dq, dk, dv = cq // n, k.shape[1] // n, feats.shape[1] // n
    out = torch.empty(b, z_lr.shape[1], d_lr + feats.shape[1],
                      device=z_lr.device, dtype=z_lr.dtype)
    out[..., :d_lr] = z_lr
    del z_lr
    per_head = b * hq * wq * (dq + dk + 2 * dv) * q.element_size()
    g = max(1, min(n, GROUP_BYTES // max(per_head, 1)))
    for h0 in range(0, n, g):
        h1 = min(n, h0 + g)

        def heads(t, d, resize):
            t = t[:, h0 * d:h1 * d]
            if resize:
                t = F.interpolate(t, size=(hq, wq), mode="nearest-exact")
            t = t.to(q.dtype)
            return t.reshape(b, h1 - h0, d, hq, wq).permute(0, 3, 4, 1, 2).contiguous()

        o = na2d(heads(q, dq, False), heads(k, dk, True), heads(feats, dv, True),
                 kernel_size=up.kernel_size, dilation=dilation, stride=1,
                 backend="cutlass-fna")
        o = o.permute(0, 3, 4, 1, 2).reshape(b, (h1 - h0) * dv, hq, wq)
        out[..., d_lr + h0 * dv:d_lr + h1 * dv] = self.proj_grid(o, *args, BHWC=False)
        del o
    return out


def patch(pipeline):
    """Patch the NAF upsampler class used by the pipeline's extractors."""
    for attr in ("image_cond_model_ss", "image_cond_model_shape_512",
                 "image_cond_model_shape_1024", "image_cond_model_tex_1024"):
        m = getattr(pipeline, attr, None)
        naf = getattr(m, "naf_model", None) if m is not None else None
        if naf is None:
            continue
        cls = type(naf.upsampler)
        if not hasattr(cls, "_spl_orig_forward"):
            cls._spl_orig_forward = cls.forward
            cls.forward = _chunked_forward
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjMultiViewFeatureExtractor as mv,
    )
    mv._project_single_view = _project_single_view


def patch_cumesh():
    """Return PyTorch's cached blocks before every CuMesh call.

    CuMesh allocates with raw cudaMalloc, so memory the PyTorch caching
    allocator still reserves after the denoisers (several GB) is invisible to
    it and decode/remesh OOM on a 12 GB card.
    """
    import functools

    import cumesh

    cls = cumesh.CuMesh
    if getattr(cls, "_spl_patched", False):
        return
    for name, fn in list(vars(cls).items()):
        if name.startswith("__") or not callable(fn):
            continue

        def wrap(fn):
            @functools.wraps(fn)
            def inner(*a, **k):
                torch.cuda.empty_cache()
                return fn(*a, **k)
            return inner

        setattr(cls, name, wrap(fn))
    cls._spl_patched = True
