// PPISP: physically plausible ISP applied to the rendered image before the
// loss (arXiv 2601.18336). Owner: ppisp. Rust side: src/ppisp/mod.rs.
//
// Our own implementation of the model of nv-tlabs/ppisp (Apache-2.0): per-frame
// exposure (2^e) and colour homography on (r, g, r+g+b), per-camera radial
// vignetting and a toe/shoulder/gamma response curve. The math follows the
// upstream PyTorch reference (tests/torch_reference.py) so its autograd is the
// test oracle; the kernel structure (a per-frame constant block, per-block
// partial gradients, forward-mode homography backward) is ours.
//
// Images are f32 planar [3][H][W] (the raster/loss layout). The parameter
// buffer is one f32 array, offsets in PpispLayout (mirrored in Rust):
//   exposure [F] | vignetting [C][3][5] (cx, cy, a0, a1, a2) | color [F][8] | crf [C][3][4]
// No #include: compiled alone by NVRTC.

#define PPISP_BLOCK 256
#define PPISP_NG 37           // exposure scale 1 | vig 15 | H 9 | crf decoded 12
#define PPISP_CONSTS 48
#define PPISP_EPS_POW 1e-6f

struct PpispLayout {
    int n_frames, n_cameras;
    int off_exposure, off_vig, off_color, off_crf;
};

// Frame constants (written by ppisp_prepare, read by every pixel):
//   [0] exposure scale, [1..10) H row-major, [10..25) vig 3x5,
//   [25..43) crf 3x(toe, shoulder, gamma, center, a, b), [43] use_frame, [44] use_cam
#define PC_S 0
#define PC_H 1
#define PC_VIG 10
#define PC_CRF 25
#define PC_USE_FRAME 43
#define PC_USE_CAM 44

__constant__ float PPISP_PINV[8][8] = {
    {0.0480542f, -0.0043631f, 0, 0, 0, 0, 0, 0},
    {-0.0043631f, 0.0481283f, 0, 0, 0, 0, 0, 0},
    {0, 0, 0.0580570f, -0.0179872f, 0, 0, 0, 0},
    {0, 0, -0.0179872f, 0.0431061f, 0, 0, 0, 0},
    {0, 0, 0, 0, 0.0433336f, -0.0180537f, 0, 0},
    {0, 0, 0, 0, -0.0180537f, 0.0580500f, 0, 0},
    {0, 0, 0, 0, 0, 0, 0.0128369f, -0.0034654f},
    {0, 0, 0, 0, 0, 0, -0.0034654f, 0.0128158f},
};

// ---------------------------------------------------------------------------
// Homography from the 8 latent colour parameters, generic over float and a
// forward-mode dual so the backward is exact without a hand-derived adjoint.

struct Dual {
    float v, d;
};
__device__ inline Dual mkd(float v, float d = 0.f) { return Dual{v, d}; }
__device__ inline Dual operator+(Dual a, Dual b) { return {a.v + b.v, a.d + b.d}; }
__device__ inline Dual operator-(Dual a, Dual b) { return {a.v - b.v, a.d - b.d}; }
__device__ inline Dual operator-(Dual a) { return {-a.v, -a.d}; }
__device__ inline Dual operator*(Dual a, Dual b) { return {a.v * b.v, a.d * b.v + a.v * b.d}; }
__device__ inline Dual operator/(Dual a, Dual b) {
    return {a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v)};
}
__device__ inline float val(float x) { return x; }
__device__ inline float val(Dual x) { return x.v; }
template <typename T> __device__ inline T cst(float x);
template <> __device__ inline float cst<float>(float x) { return x; }
template <> __device__ inline Dual cst<Dual>(float x) { return mkd(x); }

template <typename T> __device__ inline void cross3(const T a[3], const T b[3], T o[3]) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}

template <typename T> __device__ void ppisp_homography(const T cp[8], T H[9]) {
    T off[8];
    for (int j = 0; j < 8; ++j) {
        T acc = cst<T>(0.f);
        for (int i = 0; i < 8; ++i) acc = acc + cp[i] * cst<T>(PPISP_PINV[i][j]);
        off[j] = acc;
    }
    const T one = cst<T>(1.f), zero = cst<T>(0.f);
    // Targets (columns of T): blue (0,0), red (1,0), green (0,1), plus offsets.
    T tb[3] = {off[0], off[1], one};
    T tr[3] = {one + off[2], off[3], one};
    T tg[3] = {off[4], one + off[5], one};
    T tn[3] = {cst<T>(1.f / 3.f) + off[6], cst<T>(1.f / 3.f) + off[7], one};
    T Tm[3][3] = {{tb[0], tr[0], tg[0]}, {tb[1], tr[1], tg[1]}, {tb[2], tr[2], tg[2]}};
    T sk[3][3] = {{zero, -tn[2], tn[1]}, {tn[2], zero, -tn[0]}, {-tn[1], tn[0], zero}};
    T M[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            M[i][j] = sk[i][0] * Tm[0][j] + sk[i][1] * Tm[1][j] + sk[i][2] * Tm[2][j];
    T l01[3], l02[3], l12[3];
    cross3(M[0], M[1], l01);
    cross3(M[0], M[2], l02);
    cross3(M[1], M[2], l12);
    float n01 = 0.f, n02 = 0.f, n12 = 0.f;
    for (int k = 0; k < 3; ++k) {
        n01 += val(l01[k]) * val(l01[k]);
        n02 += val(l02[k]) * val(l02[k]);
        n12 += val(l12[k]) * val(l12[k]);
    }
    const T *lam = n01 >= n02 ? (n01 >= n12 ? l01 : l12) : (n02 >= n12 ? l02 : l12);
    // S^-1 for S = [s_b s_r s_g]: rows (-1,-1,1), (1,0,0), (0,1,0).
    // H = T diag(lam) S^-1, so column j of H = sum_k T[:,k] lam_k Sinv[k][j].
    const float Sinv[3][3] = {{-1.f, -1.f, 1.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            T acc = zero;
            for (int k = 0; k < 3; ++k)
                if (Sinv[k][j] != 0.f) acc = acc + Tm[i][k] * lam[k] * cst<T>(Sinv[k][j]);
            H[i * 3 + j] = acc;
        }
    T denom = H[8] + cst<T>(1e-10f);
    for (int k = 0; k < 9; ++k) H[k] = H[k] / denom;
}

__device__ inline float softplusf(float x) { return x > 20.f ? x : log1pf(expf(x)); }
__device__ inline float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }

// Per-frame constants. frame < 0 disables per-frame effects (or, with
// `use_override` set, uses those 9 values: exposure then 8 colour latents,
// the novel-view path); camera < 0 disables per-camera effects.
extern "C" __global__ void ppisp_prepare(const float *__restrict__ params, PpispLayout L,
                                         int camera, int frame,
                                         int use_override, const float *__restrict__ override_ec,
                                         float *__restrict__ consts) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const bool use_frame = frame >= 0 || use_override;
    float e = 0.f, cp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (frame >= 0) {
        e = params[L.off_exposure + frame];
        for (int k = 0; k < 8; ++k) cp[k] = params[L.off_color + frame * 8 + k];
    } else if (use_override) {
        e = override_ec[0];
        for (int k = 0; k < 8; ++k) cp[k] = override_ec[1 + k];
    }
    consts[PC_S] = exp2f(e);
    float H[9];
    ppisp_homography<float>(cp, H);
    for (int k = 0; k < 9; ++k) consts[PC_H + k] = H[k];
    const bool use_cam = camera >= 0;
    for (int k = 0; k < 15; ++k)
        consts[PC_VIG + k] = use_cam ? params[L.off_vig + camera * 15 + k] : 0.f;
    for (int c = 0; c < 3; ++c) {
        float toe = 1.f, sh = 1.f, g = 1.f, ctr = 0.5f;
        if (use_cam) {
            const float *r = params + L.off_crf + camera * 12 + c * 4;
            toe = 0.3f + softplusf(r[0]);
            sh = 0.3f + softplusf(r[1]);
            g = 0.1f + softplusf(r[2]);
            ctr = sigmoidf_(r[3]);
        }
        float a = sh * ctr / (toe + ctr * (sh - toe));
        float *o = consts + PC_CRF + c * 6;
        o[0] = toe; o[1] = sh; o[2] = g; o[3] = ctr; o[4] = a; o[5] = 1.f - a;
    }
    consts[PC_USE_FRAME] = use_frame ? 1.f : 0.f;
    consts[PC_USE_CAM] = use_cam ? 1.f : 0.f;
}

__device__ inline float2 ppisp_uv(int px, int W, int H) {
    float inv = 1.f / (float)max(W, H);
    return make_float2(((float)(px % W) + 0.5f - 0.5f * W) * inv,
                       ((float)(px / W) + 0.5f - 0.5f * H) * inv);
}

__device__ inline float ppisp_falloff_raw(const float *v, float2 uv, float &r2) {
    float dx = uv.x - v[0], dy = uv.y - v[1];
    r2 = dx * dx + dy * dy;
    return 1.f + r2 * (v[2] + r2 * (v[3] + r2 * v[4]));
}

__device__ inline float ppisp_crf(const float *c, float x) {
    x = fminf(fmaxf(x, 0.f), 1.f);
    float y = x <= c[3] ? c[4] * powf(x / c[3], c[0])
                        : 1.f - c[5] * powf((1.f - x) / (1.f - c[3]), c[1]);
    return y > 0.f ? powf(y, c[2]) : 0.f;
}

extern "C" __global__ void ppisp_fwd(const float *__restrict__ consts,
                                     const float *__restrict__ rgb_in, float *__restrict__ rgb_out,
                                     int W, int H) {
    __shared__ float k[PPISP_CONSTS];
    if (threadIdx.x < PPISP_CONSTS) k[threadIdx.x] = consts[threadIdx.x];
    __syncthreads();
    const int n = W * H;
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= n) return;
    float c[3] = {rgb_in[px], rgb_in[n + px], rgb_in[2 * n + px]};
    const bool uf = k[PC_USE_FRAME] != 0.f, ucam = k[PC_USE_CAM] != 0.f;
    if (uf)
        for (int i = 0; i < 3; ++i) c[i] *= k[PC_S];
    if (ucam) {
        float2 uv = ppisp_uv(px, W, H);
        for (int i = 0; i < 3; ++i) {
            float r2;
            c[i] *= fminf(fmaxf(ppisp_falloff_raw(k + PC_VIG + 5 * i, uv, r2), 0.f), 1.f);
        }
    }
    if (uf) {
        const float *Hm = k + PC_H;
        float I = c[0] + c[1] + c[2];
        float q[3];
        for (int i = 0; i < 3; ++i) q[i] = Hm[3 * i] * c[0] + Hm[3 * i + 1] * c[1] + Hm[3 * i + 2] * I;
        float s = I / (q[2] + 1e-5f);
        c[0] = q[0] * s;
        c[1] = q[1] * s;
        c[2] = q[2] * s - c[0] - c[1];
    }
    if (ucam)
        for (int i = 0; i < 3; ++i) c[i] = ppisp_crf(k + PC_CRF + 6 * i, c[i]);
    rgb_out[px] = c[0];
    rgb_out[n + px] = c[1];
    rgb_out[2 * n + px] = c[2];
}

__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Recomputes the forward per pixel, writes dL/d(rgb_in) (may alias v_out) and
// one row of PPISP_NG partial gradients per block (deterministic reduction in
// ppisp_bwd_finalize). Partial columns: 0 = dL/d(exposure scale),
// 1..16 = vig raw, 16..25 = H, 25..37 = crf (toe, shoulder, gamma, center).
extern "C" __global__ void __launch_bounds__(PPISP_BLOCK)
    ppisp_bwd(const float *__restrict__ consts, const float *__restrict__ rgb_in,
              const float *v_out, float *v_in, float *__restrict__ partial, int W, int H) {
    __shared__ float k[PPISP_CONSTS];
    __shared__ float red[PPISP_BLOCK / 32][PPISP_NG];
    if (threadIdx.x < PPISP_CONSTS) k[threadIdx.x] = consts[threadIdx.x];
    __syncthreads();
    const int n = W * H;
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    float g[PPISP_NG];
    for (int j = 0; j < PPISP_NG; ++j) g[j] = 0.f;
    if (px < n) {
        const bool uf = k[PC_USE_FRAME] != 0.f, ucam = k[PC_USE_CAM] != 0.f;
        float x0[3] = {rgb_in[px], rgb_in[n + px], rgb_in[2 * n + px]};
        // Forward with saved intermediates.
        float x1[3], x2[3], x3[3], fall[3], fraw[3], r2[3];
        for (int i = 0; i < 3; ++i) x1[i] = uf ? x0[i] * k[PC_S] : x0[i];
        float2 uv = ppisp_uv(px, W, H);
        for (int i = 0; i < 3; ++i) {
            if (ucam) {
                fraw[i] = ppisp_falloff_raw(k + PC_VIG + 5 * i, uv, r2[i]);
                fall[i] = fminf(fmaxf(fraw[i], 0.f), 1.f);
            } else {
                fraw[i] = fall[i] = 1.f;
                r2[i] = 0.f;
            }
            x2[i] = x1[i] * fall[i];
        }
        const float *Hm = k + PC_H;
        float I = x2[0] + x2[1] + x2[2], q[3], s = 1.f;
        if (uf) {
            for (int i = 0; i < 3; ++i)
                q[i] = Hm[3 * i] * x2[0] + Hm[3 * i + 1] * x2[1] + Hm[3 * i + 2] * I;
            s = I / (q[2] + 1e-5f);
            x3[0] = q[0] * s;
            x3[1] = q[1] * s;
            x3[2] = q[2] * s - x3[0] - x3[1];
        } else {
            for (int i = 0; i < 3; ++i) x3[i] = x2[i];
        }
        float d[3] = {v_out[px], v_out[n + px], v_out[2 * n + px]};
        // CRF backward.
        if (ucam) {
            for (int i = 0; i < 3; ++i) {
                const float *c = k + PC_CRF + 6 * i;
                const float toe = c[0], sh = c[1], gm = c[2], ct = c[3], a = c[4], b = c[5];
                float xv = x3[i];
                bool pass = xv >= 0.f && xv <= 1.f;
                float x = fminf(fmaxf(xv, 0.f), 1.f);
                float y, dydx = 0.f, dyda = 0.f, dyt = 0.f, dys = 0.f, dyc = 0.f;
                if (x <= ct) {
                    float u = x / ct, p = powf(u, toe);
                    y = a * p;
                    dyda = p;
                    if (u > PPISP_EPS_POW) {
                        float dpu = toe * powf(u, toe - 1.f);
                        dydx = a * dpu / ct;
                        dyt = a * p * logf(u);
                        dyc = -a * dpu * x / (ct * ct);
                    }
                } else {
                    float w = (1.f - x) / (1.f - ct), qq = powf(w, sh);
                    y = 1.f - b * qq;
                    dyda = qq;  // b = 1 - a
                    if (w > PPISP_EPS_POW) {
                        float dqw = sh * powf(w, sh - 1.f);
                        dydx = b * dqw / (1.f - ct);
                        dys = -b * qq * logf(w);
                        dyc = -b * dqw * (1.f - x) / ((1.f - ct) * (1.f - ct));
                    }
                }
                float dz = d[i], dy = 0.f;
                if (y > 0.f) {
                    float z = powf(y, gm);
                    dy = dz * gm * powf(y, gm - 1.f);
                    g[25 + 4 * i + 2] += dz * z * logf(y);
                }
                // a = sh*ct / Lr, Lr = toe + ct*(sh - toe)
                float Lr = toe + ct * (sh - toe), L2 = Lr * Lr;
                float da_dt = -sh * ct * (1.f - ct) / L2;
                float da_ds = ct * toe * (1.f - ct) / L2;
                float da_dc = sh * toe / L2;
                float ga = dy * dyda;
                g[25 + 4 * i + 0] += dy * dyt + ga * da_dt;
                g[25 + 4 * i + 1] += dy * dys + ga * da_ds;
                g[25 + 4 * i + 3] += dy * dyc + ga * da_dc;
                d[i] = pass ? dy * dydx : 0.f;
            }
        }
        // Colour backward.
        if (uf) {
            float dO[3] = {d[0] - d[2], d[1] - d[2], d[2]};
            float dk = dO[0] * q[0] + dO[1] * q[1] + dO[2] * q[2];
            float den = q[2] + 1e-5f;
            float dq[3] = {dO[0] * s, dO[1] * s, dO[2] * s - dk * I / (den * den)};
            float dI = dk / den;
            float rgi[3] = {x2[0], x2[1], I};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) g[16 + 3 * i + j] += dq[i] * rgi[j];
            float dr[3];
            for (int j = 0; j < 3; ++j) dr[j] = Hm[j] * dq[0] + Hm[3 + j] * dq[1] + Hm[6 + j] * dq[2];
            float dIt = dr[2] + dI;
            d[0] = dr[0] + dIt;
            d[1] = dr[1] + dIt;
            d[2] = dIt;
        }
        // Vignetting backward.
        if (ucam) {
            for (int i = 0; i < 3; ++i) {
                const float *v = k + PC_VIG + 5 * i;
                float df = d[i] * x1[i];
                d[i] *= fall[i];
                if (fraw[i] >= 0.f && fraw[i] <= 1.f) {
                    float r = r2[i];
                    g[1 + 5 * i + 2] += df * r;
                    g[1 + 5 * i + 3] += df * r * r;
                    g[1 + 5 * i + 4] += df * r * r * r;
                    float dr2 = df * (v[2] + r * (2.f * v[3] + 3.f * r * v[4]));
                    g[1 + 5 * i + 0] += -2.f * dr2 * (uv.x - v[0]);
                    g[1 + 5 * i + 1] += -2.f * dr2 * (uv.y - v[1]);
                }
            }
        }
        // Exposure backward.
        if (uf) {
            g[0] += d[0] * x0[0] + d[1] * x0[1] + d[2] * x0[2];
            for (int i = 0; i < 3; ++i) d[i] *= k[PC_S];
        }
        v_in[px] = d[0];
        v_in[n + px] = d[1];
        v_in[2 * n + px] = d[2];
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int j = 0; j < PPISP_NG; ++j) {
        float s = warp_sum(g[j]);
        if (lane == 0) red[wid][j] = s;
    }
    __syncthreads();
    if (threadIdx.x < PPISP_NG) {
        float s = 0.f;
        for (int w = 0; w < PPISP_BLOCK / 32; ++w) s += red[w][threadIdx.x];
        partial[blockIdx.x * PPISP_NG + threadIdx.x] = s;
    }
}

// One block: sums the partial rows and adds the parameter gradients into
// `grads` (same layout as params). With `use_override` (novel view /
// controller path) the per-frame gradients go to `grad_override[9]` instead.
extern "C" __global__ void ppisp_bwd_finalize(const float *__restrict__ partial, int n_blocks,
                                              const float *__restrict__ params, PpispLayout L,
                                              int camera, int frame,
                                              int use_override, const float *__restrict__ override_ec,
                                              const float *__restrict__ consts,
                                              float *__restrict__ grads,
                                              float *__restrict__ grad_override) {
    __shared__ float tot[PPISP_NG];
    if (threadIdx.x < PPISP_NG) {
        float s = 0.f;
        for (int b = 0; b < n_blocks; ++b) s += partial[b * PPISP_NG + threadIdx.x];
        tot[threadIdx.x] = s;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    if (camera >= 0) {
        for (int k = 0; k < 15; ++k) grads[L.off_vig + camera * 15 + k] += tot[1 + k];
        for (int c = 0; c < 3; ++c) {
            const float *r = params + L.off_crf + camera * 12 + c * 4;
            float *gr = grads + L.off_crf + camera * 12 + c * 4;
            gr[0] += tot[25 + 4 * c + 0] * sigmoidf_(r[0]);  // softplus' = sigmoid
            gr[1] += tot[25 + 4 * c + 1] * sigmoidf_(r[1]);
            gr[2] += tot[25 + 4 * c + 2] * sigmoidf_(r[2]);
            float sg = sigmoidf_(r[3]);
            gr[3] += tot[25 + 4 * c + 3] * sg * (1.f - sg);
        }
    }
    if (frame < 0 && !use_override) return;
    float cp[8];
    float *ge, *gc;
    if (frame >= 0) {
        for (int k = 0; k < 8; ++k) cp[k] = params[L.off_color + frame * 8 + k];
        ge = grads + L.off_exposure + frame;
        gc = grads + L.off_color + frame * 8;
    } else {
        for (int k = 0; k < 8; ++k) cp[k] = override_ec[1 + k];
        ge = grad_override;
        gc = grad_override + 1;
    }
    *ge += tot[0] * consts[PC_S] * 0.69314718056f;
    for (int t = 0; t < 8; ++t) {
        Dual cd[8], Hd[9];
        for (int k = 0; k < 8; ++k) cd[k] = mkd(cp[k], k == t ? 1.f : 0.f);
        ppisp_homography<Dual>(cd, Hd);
        float s = 0.f;
        for (int k = 0; k < 9; ++k) s += tot[16 + k] * Hd[k].d;
        gc[t] += s;
    }
}

__device__ inline float sl1(float x, float beta) {
    float ax = fabsf(x);
    return ax < beta ? 0.5f * x * x / beta : ax - 0.5f * beta;
}
__device__ inline float sl1_d(float x, float beta) {
    return fabsf(x) < beta ? x / beta : (x > 0.f ? 1.f : (x < 0.f ? -1.f : 0.f));
}

// Regulariser of the upstream module (weights: exposure_mean, vig_center,
// vig_channel, vig_non_pos, color_mean, crf_channel). One block of
// PPISP_BLOCK threads. Adds dR/dparams into grads; writes R to loss[0].
extern "C" __global__ void __launch_bounds__(PPISP_BLOCK)
    ppisp_reg(const float *__restrict__ params, PpispLayout L, float w_exp, float w_vc,
              float w_vch, float w_vnp, float w_col, float w_crf, float *__restrict__ grads,
              float *__restrict__ loss) {
    __shared__ float sums[PPISP_BLOCK / 32][9];
    __shared__ float tot[9];
    const int F = L.n_frames, C = L.n_cameras;
    float acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int f = threadIdx.x; f < F; f += blockDim.x) {
        acc[0] += params[L.off_exposure + f];
        const float *cp = params + L.off_color + f * 8;
        for (int j = 0; j < 8; ++j) {
            float o = 0.f;
            for (int i = 0; i < 8; ++i) o += cp[i] * PPISP_PINV[i][j];
            acc[1 + j] += o;
        }
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int j = 0; j < 9; ++j) {
        float s = warp_sum(acc[j]);
        if (lane == 0) sums[wid][j] = s;
    }
    __syncthreads();
    if (threadIdx.x < 9) {
        float s = 0.f;
        for (int w = 0; w < (int)(blockDim.x / 32); ++w) s += sums[w][threadIdx.x];
        tot[threadIdx.x] = s / (float)F;
    }
    __syncthreads();
    // Per-frame gradients.
    float ge = w_exp * sl1_d(tot[0], 0.1f) / (float)F;
    float gr[8];
    for (int j = 0; j < 8; ++j) gr[j] = w_col * sl1_d(tot[1 + j], 0.005f) / (8.f * (float)F);
    for (int f = threadIdx.x; f < F; f += blockDim.x) {
        grads[L.off_exposure + f] += ge;
        for (int i = 0; i < 8; ++i) {
            float s = 0.f;
            for (int j = 0; j < 8; ++j) s += PPISP_PINV[i][j] * gr[j];
            grads[L.off_color + f * 8 + i] += s;
        }
    }
    if (threadIdx.x != 0) return;
    float R = w_exp * sl1(tot[0], 0.1f);
    for (int j = 0; j < 8; ++j) R += w_col * sl1(tot[1 + j], 0.005f) / 8.f;
    for (int cam = 0; cam < C; ++cam) {
        const float *v = params + L.off_vig + cam * 15;
        float *gv = grads + L.off_vig + cam * 15;
        for (int ch = 0; ch < 3; ++ch) {
            float cx = v[5 * ch], cy = v[5 * ch + 1];
            R += w_vc * (cx * cx + cy * cy) / (3.f * C);
            gv[5 * ch] += w_vc * 2.f * cx / (3.f * C);
            gv[5 * ch + 1] += w_vc * 2.f * cy / (3.f * C);
            for (int a = 2; a < 5; ++a) {
                float x = v[5 * ch + a];
                if (x > 0.f) {
                    R += w_vnp * x / (9.f * C);
                    gv[5 * ch + a] += w_vnp / (9.f * C);
                }
            }
        }
        for (int p = 0; p < 5; ++p) {
            float m = (v[p] + v[5 + p] + v[10 + p]) / 3.f, var = 0.f;
            for (int ch = 0; ch < 3; ++ch) var += (v[5 * ch + p] - m) * (v[5 * ch + p] - m);
            R += w_vch * var / 3.f / (5.f * C);
            for (int ch = 0; ch < 3; ++ch)
                gv[5 * ch + p] += w_vch * 2.f * (v[5 * ch + p] - m) / 3.f / (5.f * C);
        }
        const float *r = params + L.off_crf + cam * 12;
        float *grf = grads + L.off_crf + cam * 12;
        for (int p = 0; p < 4; ++p) {
            float m = (r[p] + r[4 + p] + r[8 + p]) / 3.f, var = 0.f;
            for (int ch = 0; ch < 3; ++ch) var += (r[4 * ch + p] - m) * (r[4 * ch + p] - m);
            R += w_crf * var / 3.f / (4.f * C);
            for (int ch = 0; ch < 3; ++ch)
                grf[4 * ch + p] += w_crf * 2.f * (r[4 * ch + p] - m) / 3.f / (4.f * C);
        }
    }
    loss[0] = R;
}

// Adam over [lo, hi) of the parameter buffer (torch.optim.Adam semantics,
// bias-corrected), then zeroes those gradients. bc1 = 1 - b1^t, bc2 = 1 - b2^t.
extern "C" __global__ void ppisp_adam(float *__restrict__ p, float *__restrict__ g,
                                      float *__restrict__ m, float *__restrict__ v, int lo, int hi,
                                      float lr, float b1, float b2, float eps, float bc1,
                                      float bc2) {
    int i = lo + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hi) return;
    float gi = g[i];
    float mi = b1 * m[i] + (1.f - b1) * gi;
    float vi = b2 * v[i] + (1.f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    p[i] -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
    g[i] = 0.f;
}
