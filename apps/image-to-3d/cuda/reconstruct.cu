// image_to_3d.reconstruct (Phase 4): joint camera refinement and the
// depth-prior resampling between the render and the VGGT-Omega depth maps.
// Owner: mvp-pipeline. Compiled alone (src/reconstruct/joint.rs, SOURCE).
//
// Pose: per-image twist (rho, phi) in CameraTable::pose_delta, applied by
// frame_begin as exp(delta) * T_base; gradients from cam_reduce (left
// perturbation in the camera frame, spec-train 10.1). Adam with per-image
// bias correction (host counts the steps of each image); the gradient row is
// zeroed after use. Focal: one log-scale kappa per intrinsics entry,
// fx = fx0 * e^kappa, fy = fy0 * e^kappa, dL/dkappa = fx dL/dfx + fy dL/dfy
// at the render scale (scale invariant).

extern "C" __global__ void recon_pose_adam(float* __restrict__ pose_grad, float* __restrict__ pose_delta,
                                           float* __restrict__ m, float* __restrict__ v, unsigned int image,
                                           float lr_rot, float lr_trans, float b1, float b2, float bc1,
                                           float bc2, float eps, int frozen) {
    const unsigned int i = threadIdx.x;
    if (i >= 6) return;
    const size_t o = (size_t)image * 6 + i;
    const float g = pose_grad[o];
    pose_grad[o] = 0.f;
    if (frozen || !(g == g)) return;
    const float mi = b1 * m[o] + (1.f - b1) * g;
    const float vi = b2 * v[o] + (1.f - b2) * g * g;
    m[o] = mi;
    v[o] = vi;
    const float lr = i < 3 ? lr_trans : lr_rot;
    pose_delta[o] -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
}

// intr / intr0 / intr_grad: [n][4] = fx, fy, cx, cy. state: [n][3] = kappa, m, v.
extern "C" __global__ void recon_focal_adam(float* __restrict__ intr_grad, float* __restrict__ intr,
                                            const float* __restrict__ intr0, float* __restrict__ state,
                                            unsigned int id, float sx, float sy, float lr, float b1, float b2,
                                            float bc1, float bc2, float eps, int frozen) {
    if (threadIdx.x != 0) return;
    float* g = intr_grad + 4 * (size_t)id;
    const float gfx = g[0], gfy = g[1];
    g[0] = g[1] = g[2] = g[3] = 0.f;
    if (frozen || !(gfx == gfx) || !(gfy == gfy)) return;
    float* st = state + 3 * (size_t)id;
    float k = st[0];
    const float fx0 = intr0[4 * id], fy0 = intr0[4 * id + 1];
    const float e = expf(k);
    const float gk = fx0 * e * sx * gfx + fy0 * e * sy * gfy;
    const float mi = b1 * st[1] + (1.f - b1) * gk;
    const float vi = b2 * st[2] + (1.f - b2) * gk * gk;
    st[1] = mi;
    st[2] = vi;
    k -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
    st[0] = k;
    intr[4 * id] = fx0 * expf(k);
    intr[4 * id + 1] = fy0 * expf(k);
}

// Box of render pixels covered by prior pixel (u, v): [x0, x1) x [y0, y1).
__device__ __forceinline__ void recon_box(int u, int v, int w, int h, int pw, int ph, int& x0, int& x1, int& y0,
                                          int& y1) {
    x0 = (int)((long long)u * w / pw);
    x1 = (int)((long long)(u + 1) * w / pw);
    y0 = (int)((long long)v * h / ph);
    y1 = (int)((long long)(v + 1) * h / ph);
    if (x1 <= x0) x1 = min(x0 + 1, w);
    if (y1 <= y0) y1 = min(y0 + 1, h);
    x0 = min(x0, w - 1);
    y0 = min(y0, h - 1);
}

// Box-average of the rendered expected depth ([h][w]) at the prior's size.
extern "C" __global__ void recon_depth_gather(const float* __restrict__ depth, int w, int h,
                                              float* __restrict__ out, int pw, int ph) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pw * ph) return;
    const int u = i % pw, v = i / pw;
    int x0, x1, y0, y1;
    recon_box(u, v, w, h, pw, ph, x0, x1, y0, y1);
    float s = 0.f;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) s += depth[y * w + x];
    out[i] = s / (float)((x1 - x0) * (y1 - y0));
}

// Transpose of recon_depth_gather: grad[box] += g / |box|.
extern "C" __global__ void recon_depth_scatter(const float* __restrict__ g, int pw, int ph,
                                               float* __restrict__ grad, int w, int h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pw * ph) return;
    const float gi = g[i];
    if (gi == 0.f) return;
    const int u = i % pw, v = i / pw;
    int x0, x1, y0, y1;
    recon_box(u, v, w, h, pw, ph, x0, x1, y0, y1);
    const float a = gi / (float)((x1 - x0) * (y1 - y0));
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) atomicAdd(&grad[y * w + x], a);
}
