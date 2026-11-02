// Geometry rendering and geometry regularisers of the Geometry preset. Owner: mesh-mode.
//
// Compiled with common.cuh + splat_common.cuh + raster_common.cuh and the raster variant's
// defines, so the blend (alpha clamp, 1/255 skip, T_MIN stop) is the raster's exactly.
// Our own code after the papers:
//   - median depth: the depth where the transmittance crosses 0.5 (2DGS, GGGS);
//   - per-Gaussian plane: normal = the shortest axis facing the camera, depth along a pixel
//     ray = ray/plane intersection (PGSR's unbiased depth), clamped to 3 sigma of the centre;
//   - depth distortion sum_{i,j} w_i w_j (m_i - m_j)^2 with m = 1 - near / z (2DGS, its
//     NDC mapping for far >> near), closed form through prefix moments;
//   - depth-normal consistency 1 - N . (alpha N_d), N the blended normal, N_d the normal of
//     the median depth map (2DGS; N_d detached).
//
// Pipeline per training iteration (after raster_fwd, before raster_bwd; geom_post after
// preprocess_bwd):
//   geom_prep  (per visible v)  aux[v] = camera-space normal, centre z, plane offset, clamp
//   geom_fwd   (per tile)       per-pixel A, M1, M2, N, median depth, expected depth
//   geom_pixel (per pixel)      depth normal, per-pixel loss gradients, loss sums
//   geom_bwd   (per tile)       alpha-chain gradients into g2d (mx, my, A, B, C, o), dz, dn
//   geom_post  (per visible v)  dz -> mean, dn -> quat, flattening -> log_scale
// Mesh renders run geom_prep + geom_fwd only (after raster_fwd in Infer mode).

#define GEOM_CH 8      // median z, A, M1, M2, Nx, Ny, Nz, expected z
#define GPIX_CH 5      // gNx, gNy, gNz, distortion weight, sum_k w_k g_k

struct GeomParams {
    float near_m;      // distortion depth mapping m = 1 - near_m / max(z, near_m), in [0, 1)
    float lambda_dist; // both pixel losses are means over the frame
    float lambda_normal;
    float lambda_flat; // per Gaussian
    uint32_t dist_from, normal_from, flat_from;  // first iteration each term is on
    uint32_t v_cap;
};

struct GeomAux {
    float nx, ny, nz, z;   // camera-space unit normal (n . mean_c <= 0), centre depth
    float d, zr;           // plane n . x = d, |z_hit - z| clamp
};

SPL_INLINE float geom_hit_z(const GeomAux& g, float rx, float ry) {
    const float den = g.nx * rx + g.ny * ry + g.nz;
    const float zh = fabsf(den) > 1e-4f ? g.d / den : g.z;
    return fminf(fmaxf(zh, g.z - g.zr), g.z + g.zr);
}

// Shortest axis k and the rotation of the unit quaternion of p.
SPL_INLINE int geom_axis(const float* __restrict__ log_scale, const float* __restrict__ quat, uint32_t p,
                         float q[4], float Rq[9], float s[3]) {
    float w = quat[4 * (size_t)p], x = quat[4 * (size_t)p + 1], y = quat[4 * (size_t)p + 2],
          z = quat[4 * (size_t)p + 3];
    const float inv = rsqrtf(fmaxf(w * w + x * x + y * y + z * z, 1e-24f));
    q[0] = w * inv;
    q[1] = x * inv;
    q[2] = y * inv;
    q[3] = z * inv;
    quat_to_rotmat(q[0], q[1], q[2], q[3], Rq);
    for (int k = 0; k < 3; ++k) s[k] = __expf(log_scale[3 * (size_t)p + k]);
    return s[0] <= s[1] ? (s[0] <= s[2] ? 0 : 2) : (s[1] <= s[2] ? 1 : 2);
}

extern "C" __global__ void geom_prep(const FrameCam* __restrict__ cam, const uint32_t* __restrict__ vis_idx,
                                     const float* __restrict__ mean, const float* __restrict__ log_scale,
                                     const float* __restrict__ quat, GeomAux* __restrict__ aux,
                                     float* __restrict__ gz, float* __restrict__ gn, uint32_t v_cap,
                                     const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    const uint32_t V = min(status[ST_N_VIS], v_cap);
    const float* R = cam->R;
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < V; v += gridDim.x * blockDim.x) {
        const uint32_t p = vis_idx[v];
        float q[4], Rq[9], s[3];
        const int k = geom_axis(log_scale, quat, p, q, Rq, s);
        const float ax = Rq[k], ay = Rq[3 + k], az = Rq[6 + k];  // column k: world axis
        float n[3], m[3];
        const float mw[3] = {mean[3 * (size_t)p], mean[3 * (size_t)p + 1], mean[3 * (size_t)p + 2]};
        for (int r = 0; r < 3; ++r) {
            n[r] = R[3 * r] * ax + R[3 * r + 1] * ay + R[3 * r + 2] * az;
            m[r] = R[3 * r] * mw[0] + R[3 * r + 1] * mw[1] + R[3 * r + 2] * mw[2] + cam->t[r];
        }
        float d = n[0] * m[0] + n[1] * m[1] + n[2] * m[2];
        const float sg = d > 0.f ? -1.f : 1.f;
        GeomAux a;
        a.nx = sg * n[0];
        a.ny = sg * n[1];
        a.nz = sg * n[2];
        a.z = m[2];
        a.d = sg * d;
        a.zr = 3.f * fmaxf(s[0], fmaxf(s[1], s[2]));
        aux[v] = a;
        if (gz) {
            gz[v] = 0.f;
            gn[v] = 0.f;
            gn[(size_t)v_cap + v] = 0.f;
            gn[2 * (size_t)v_cap + v] = 0.f;
        }
    }
}

// Shared batch of one tile: record (mean, conic, opacity) and aux.
struct GeomBatch {
    float4 a[TILE_W * TILE_H];   // mx, my, ca, cb
    float2 b[TILE_W * TILE_H];   // cc, o
    GeomAux g[TILE_W * TILE_H];
    uint32_t v[TILE_W * TILE_H];
};

SPL_INLINE void geom_stage(GeomBatch& S, const SplatRec* __restrict__ rec, const GeomAux* __restrict__ aux,
                           const uint32_t* __restrict__ inst_v, uint32_t s0, uint32_t j, uint32_t end,
                           uint32_t tid) {
    if (j < end) {
        const uint32_t v = inst_v[s0 + j];
        const SplatRec r = rec[v];
        S.a[tid] = make_float4(r.mx, r.my, r.ca, r.cb);
        S.b[tid] = make_float2(r.cc, r.o);
        S.g[tid] = aux[v];
        S.v[tid] = v;
    }
}

// mode 0: mesh render (median + expected depth only need the A..z channels);
// out[GEOM_CH][P]. Pixels where T never reaches 0.5 get median z = 0 (invalid).
extern "C" __global__ void __launch_bounds__(TILE_W* TILE_H)
    geom_fwd(const FrameCam* __restrict__ cam, const SplatRec* __restrict__ rec, const GeomAux* __restrict__ aux,
             const uint32_t* __restrict__ inst_v, const uint32_t* __restrict__ tile_start, GeomParams gp,
             float* __restrict__ out, const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    const uint32_t t = blockIdx.x;
    if (t >= cam->n_tiles) return;
    __shared__ GeomBatch S;
    const uint32_t tid = threadIdx.x, npx = TILE_W * TILE_H;
    const uint32_t gx = (t % cam->grid_w) * TILE_W + tid % TILE_W, gy = (t / cam->grid_w) * TILE_H + tid / TILE_W;
    const bool inside = gx < cam->width && gy < cam->height;
    const float pcx = (float)gx + 0.5f, pcy = (float)gy + 0.5f;
    const float rx = (pcx - cam->cx) / cam->fx, ry = (pcy - cam->cy) / cam->fy;
    const uint32_t s0 = tile_start[t], n = tile_start[t + 1] - s0;
    float T = 1.f, A = 0.f, M1 = 0.f, M2 = 0.f, N0 = 0.f, N1 = 0.f, N2 = 0.f, zmed = 0.f, zexp = 0.f;
    bool done = !inside;
    for (uint32_t b0 = 0; b0 < n; b0 += npx) {
        if (__syncthreads_count(done) == (int)npx) break;
        geom_stage(S, rec, aux, inst_v, s0, b0 + tid, n, tid);
        __syncthreads();
        const uint32_t cnt = min(npx, n - b0);
        for (uint32_t j = 0; j < cnt && !done; ++j) {
            const float4 a = S.a[j];
            const float2 b = S.b[j];
            const float e = rs_power(a.z, a.w, b.x, a.x - pcx, a.y - pcy);
            float G, a_raw, alpha;
            if (!rs_alpha(b.y, e, G, a_raw, alpha)) continue;
            const float Tn = T * (1.f - alpha);
#if BLEND == BLEND_FASTERGS
            const bool stop = Tn < T_MIN;
#else
            if (Tn <= T_MIN) {
                done = true;
                break;
            }
            const bool stop = false;
#endif
            const GeomAux g = S.g[j];
            const float zh = geom_hit_z(g, rx, ry);
            const float m = 1.f - gp.near_m / fmaxf(zh, gp.near_m);
            const float w = alpha * T;
            A += w;
            M1 = fmaf(w, m, M1);
            M2 = fmaf(w * m, m, M2);
            N0 = fmaf(w, g.nx, N0);
            N1 = fmaf(w, g.ny, N1);
            N2 = fmaf(w, g.nz, N2);
            zexp = fmaf(w, zh, zexp);
            if (T > 0.5f && Tn <= 0.5f) zmed = zh;
            T = Tn;
            if (stop) done = true;
        }
        __syncthreads();
    }
    if (inside) {
        const uint32_t P = cam->width * cam->height, pix = gy * cam->width + gx;
        out[0 * P + pix] = zmed;
        out[1 * P + pix] = A;
        out[2 * P + pix] = M1;
        out[3 * P + pix] = M2;
        out[4 * P + pix] = N0;
        out[5 * P + pix] = N1;
        out[6 * P + pix] = N2;
        out[7 * P + pix] = zexp;
    }
}

// Per-pixel loss gradients. sums[0] += distortion, sums[1] += normal loss (unscaled means
// are sums / P on the host).
extern "C" __global__ void geom_pixel(const IterParams* __restrict__ params, const FrameCam* __restrict__ cam,
                                      const float* __restrict__ out, GeomParams gp, float* __restrict__ gpix,
                                      float* __restrict__ sums, const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    const uint32_t W = cam->width, H = cam->height, P = W * H;
    const uint32_t it = params->frame.iteration;
    const float ld = it >= gp.dist_from ? gp.lambda_dist / (float)P : 0.f;
    const float ln = it >= gp.normal_from ? gp.lambda_normal / (float)P : 0.f;
    float l_d = 0.f, l_n = 0.f;
    for (uint32_t pix = blockIdx.x * blockDim.x + threadIdx.x; pix < P; pix += gridDim.x * blockDim.x) {
        const uint32_t x = pix % W, y = pix / W;
        const float A = out[P + pix], M1 = out[2 * P + pix], M2 = out[3 * P + pix];
        const float N0 = out[4 * P + pix], N1 = out[5 * P + pix], N2 = out[6 * P + pix];
        float g0 = 0.f, g1 = 0.f, g2 = 0.f;
        if (ln > 0.f && x > 0 && y > 0 && x + 1 < W && y + 1 < H) {
            const float* zm = out;
            const float zl = zm[pix - 1], zr = zm[pix + 1], zu = zm[pix - W], zd = zm[pix + W];
            if (zl > 0.f && zr > 0.f && zu > 0.f && zd > 0.f) {
                auto unp = [&](float u, float v, float z, float o[3]) {
                    o[0] = (u + 0.5f - cam->cx) / cam->fx * z;
                    o[1] = (v + 0.5f - cam->cy) / cam->fy * z;
                    o[2] = z;
                };
                float pl[3], pr[3], pu[3], pd[3];
                unp((float)x - 1.f, (float)y, zl, pl);
                unp((float)x + 1.f, (float)y, zr, pr);
                unp((float)x, (float)y - 1.f, zu, pu);
                unp((float)x, (float)y + 1.f, zd, pd);
                const float ex[3] = {pr[0] - pl[0], pr[1] - pl[1], pr[2] - pl[2]};
                const float ey[3] = {pd[0] - pu[0], pd[1] - pu[1], pd[2] - pu[2]};
                float c[3] = {ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2],
                              ex[0] * ey[1] - ex[1] * ey[0]};
                const float len = sqrtf(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
                if (len > 0.f) {
                    // Face the camera: the centre pixel's ray is ~ +z, so the normal has z < 0.
                    const float sg = (c[2] > 0.f ? -1.f : 1.f) * A / len;
                    c[0] *= sg;
                    c[1] *= sg;
                    c[2] *= sg;
                    l_n += 1.f - (N0 * c[0] + N1 * c[1] + N2 * c[2]);
                    g0 = -ln * c[0];
                    g1 = -ln * c[1];
                    g2 = -ln * c[2];
                }
            }
        }
        l_d += A * M2 - M1 * M1;  // half the double sum sum_{j,k} = 2 (A M2 - M1^2)
        gpix[pix] = g0;
        gpix[P + pix] = g1;
        gpix[2 * P + pix] = g2;
        gpix[3 * P + pix] = ld;
        gpix[4 * P + pix] = ld * 4.f * (A * M2 - M1 * M1) + g0 * N0 + g1 * N1 + g2 * N2;
    }
    if (sums) {
        l_d = warp_sum(l_d);
        l_n = warp_sum(l_n);
        if ((threadIdx.x & 31) == 0) {
            atomicAdd(&sums[0], 2.f * l_d);
            atomicAdd(&sums[1], l_n);
        }
    }
}

// Same traversal as geom_fwd; the suffix sums come from the totals (sum_k w_k g_k), so no
// reverse pass is needed. Warp-reduced atomics per (Gaussian, batch).
extern "C" __global__ void __launch_bounds__(TILE_W* TILE_H)
    geom_bwd(const IterParams* __restrict__ params, const FrameCam* __restrict__ cam,
             const SplatRec* __restrict__ rec, const GeomAux* __restrict__ aux, const uint32_t* __restrict__ inst_v,
             const uint32_t* __restrict__ tile_start, GeomParams gp, const float* __restrict__ out,
             const float* __restrict__ gpix, float* __restrict__ g2d, float* __restrict__ gz, float* __restrict__ gn,
             const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    const uint32_t it = params->frame.iteration;
    if (it < gp.dist_from && it < gp.normal_from) return;
    const uint32_t t = blockIdx.x;
    if (t >= cam->n_tiles) return;
    __shared__ GeomBatch S;
    const uint32_t tid = threadIdx.x, npx = TILE_W * TILE_H, lane = tid & 31;
    const uint32_t gx = (t % cam->grid_w) * TILE_W + tid % TILE_W, gy = (t / cam->grid_w) * TILE_H + tid / TILE_W;
    const bool inside = gx < cam->width && gy < cam->height;
    const uint32_t P = cam->width * cam->height, pix = inside ? gy * cam->width + gx : 0;
    const float pcx = (float)gx + 0.5f, pcy = (float)gy + 0.5f;
    const float rx = (pcx - cam->cx) / cam->fx, ry = (pcy - cam->cy) / cam->fy;
    float A = 0.f, M1 = 0.f, M2 = 0.f, gN0 = 0.f, gN1 = 0.f, gN2 = 0.f, ld = 0.f, tot = 0.f;
    if (inside) {
        A = out[P + pix];
        M1 = out[2 * P + pix];
        M2 = out[3 * P + pix];
        gN0 = gpix[pix];
        gN1 = gpix[P + pix];
        gN2 = gpix[2 * P + pix];
        ld = gpix[3 * P + pix];
        tot = gpix[4 * P + pix];
    }
    const uint32_t s0 = tile_start[t], n = tile_start[t + 1] - s0;
    float T = 1.f, pref = 0.f;
    bool done = !inside || (ld == 0.f && gN0 == 0.f && gN1 == 0.f && gN2 == 0.f);
    for (uint32_t b0 = 0; b0 < n; b0 += npx) {
        if (__syncthreads_count(done) == (int)npx) break;
        geom_stage(S, rec, aux, inst_v, s0, b0 + tid, n, tid);
        __syncthreads();
        const uint32_t cnt = min(npx, n - b0);
        for (uint32_t j = 0; j < cnt; ++j) {
            const float4 a = S.a[j];
            const float2 b = S.b[j];
            const float dx = a.x - pcx, dy = a.y - pcy;
            const float e = rs_power(a.z, a.w, b.x, dx, dy);
            float G, a_raw, alpha;
            bool ok = !done && rs_alpha(b.y, e, G, a_raw, alpha);
            float Tn = T;
            if (ok) {
                Tn = T * (1.f - alpha);
#if BLEND == BLEND_FASTERGS
                if (Tn < T_MIN) done = true;  // blended, then stop
#else
                if (Tn <= T_MIN) {
                    done = true;
                    ok = false;
                }
#endif
            }
            float v_mx = 0.f, v_my = 0.f, v_a = 0.f, v_b = 0.f, v_c = 0.f, v_o = 0.f, v_z = 0.f, v_n0 = 0.f,
                  v_n1 = 0.f, v_n2 = 0.f;
            if (ok) {
                const GeomAux g = S.g[j];
                const float zh = geom_hit_z(g, rx, ry);
                const float zc = fmaxf(zh, 1e-6f);
                const float m = 1.f - gp.near_m / fmaxf(zc, gp.near_m);
                const float w = alpha * T;
                // d/dw_i of sum_{j,k} w_j w_k (m_j - m_k)^2 = 2 sum_k w_k (m_i - m_k)^2.
                const float gw = 2.f * ld * (m * m * A - 2.f * m * M1 + M2) + gN0 * g.nx + gN1 * g.ny + gN2 * g.nz;
                pref = fmaf(w, gw, pref);
                const float suffix = tot - pref;
#if BLEND == BLEND_FASTERGS
                const float ra_ = 1.f / fmaxf(1.f - alpha, ONE_MINUS_ALPHA_MIN);
                const bool through = true;
#else
                const float ra_ = 1.f / (1.f - alpha);
                const bool through = a_raw <= ALPHA_MAX;
#endif
                const float d_alpha = T * gw - suffix * ra_;
                if (through) {
                    v_o = G * d_alpha;
                    const float q = a_raw * d_alpha;
                    const float qx = q * dx, qy = q * dy;
                    v_a = qx * dx;
                    v_b = qx * dy;
                    v_c = qy * dy;
                    v_mx = fmaf(2.f * a.z, qx, a.w * qy);
                    v_my = fmaf(a.w, qx, 2.f * b.x * qy);
                }
                // dm/dz through the centre depth only (the hit depth's plane terms are dropped).
                v_z = zc > gp.near_m ? ld * 4.f * w * (m * A - M1) * gp.near_m / (zc * zc) : 0.f;
                v_n0 = w * gN0;
                v_n1 = w * gN1;
                v_n2 = w * gN2;
                T = Tn;
            }
            const bool any = v_o != 0.f || v_mx != 0.f || v_my != 0.f || v_z != 0.f || v_n0 != 0.f ||
                             v_n1 != 0.f || v_n2 != 0.f;
            if (__any_sync(FULL_MASK, any)) {
                v_mx = warp_sum(v_mx);
                v_my = warp_sum(v_my);
                v_a = warp_sum(v_a);
                v_b = warp_sum(v_b);
                v_c = warp_sum(v_c);
                v_o = warp_sum(v_o);
                v_z = warp_sum(v_z);
                v_n0 = warp_sum(v_n0);
                v_n1 = warp_sum(v_n1);
                v_n2 = warp_sum(v_n2);
                if (lane == 0) {
                    const uint32_t v = S.v[j];
                    const size_t vc = gp.v_cap;
                    const float ln2 = 0.69314718055994531f;
                    atomicAdd(&g2d[0 * vc + v], ln2 * v_mx);
                    atomicAdd(&g2d[1 * vc + v], ln2 * v_my);
                    atomicAdd(&g2d[2 * vc + v], -0.5f * v_a);
                    atomicAdd(&g2d[3 * vc + v], -v_b);
                    atomicAdd(&g2d[4 * vc + v], -0.5f * v_c);
                    atomicAdd(&g2d[5 * vc + v], v_o);
                    if (v_z != 0.f) atomicAdd(&gz[v], v_z);
                    atomicAdd(&gn[v], v_n0);
                    atomicAdd(&gn[vc + v], v_n1);
                    atomicAdd(&gn[2 * vc + v], v_n2);
                }
            }
        }
        __syncthreads();
    }
}

// dz -> mean (z = R[2,:] . x + t_z), dn -> quaternion (n = sg R_cam Rq[:,k]), flattening
// lambda_flat * min scale -> log_scale. Adds to the gradients preprocess_bwd wrote.
extern "C" __global__ void geom_post(const IterParams* __restrict__ params, const FrameCam* __restrict__ cam,
                                     const uint32_t* __restrict__ vis_idx, const float* __restrict__ mean,
                                     const float* __restrict__ log_scale, const float* __restrict__ quat,
                                     const float* __restrict__ gz, const float* __restrict__ gn, GeomParams gp,
                                     float* __restrict__ grad_mean, float* __restrict__ grad_log_scale,
                                     float* __restrict__ grad_quat, const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    const uint32_t it = params->frame.iteration;
    const bool geo = it >= gp.dist_from || it >= gp.normal_from;
    const bool flat = it >= gp.flat_from && gp.lambda_flat > 0.f;
    if (!geo && !flat) return;
    const uint32_t V = min(status[ST_N_VIS], gp.v_cap);
    const float* R = cam->R;
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < V; v += gridDim.x * blockDim.x) {
        const uint32_t p = vis_idx[v];
        float q[4], Rq[9], s[3];
        const int k = geom_axis(log_scale, quat, p, q, Rq, s);
        if (flat) atomicAdd(&grad_log_scale[3 * (size_t)p + k], gp.lambda_flat * s[k]);
        if (!geo) continue;
        const float dz = gz[v];
        if (dz != 0.f)
            for (int c = 0; c < 3; ++c) atomicAdd(&grad_mean[3 * (size_t)p + c], dz * R[6 + c]);
        const float dn[3] = {gn[v], gn[(size_t)gp.v_cap + v], gn[2 * (size_t)gp.v_cap + v]};
        if (dn[0] == 0.f && dn[1] == 0.f && dn[2] == 0.f) continue;
        // Recompute the facing sign.
        float n[3], m[3];
        const float mw[3] = {mean[3 * (size_t)p], mean[3 * (size_t)p + 1], mean[3 * (size_t)p + 2]};
        for (int r = 0; r < 3; ++r) {
            n[r] = R[3 * r] * Rq[k] + R[3 * r + 1] * Rq[3 + k] + R[3 * r + 2] * Rq[6 + k];
            m[r] = R[3 * r] * mw[0] + R[3 * r + 1] * mw[1] + R[3 * r + 2] * mw[2] + cam->t[r];
        }
        const float sg = n[0] * m[0] + n[1] * m[1] + n[2] * m[2] > 0.f ? -1.f : 1.f;
        float G[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int r = 0; r < 3; ++r)  // dL/dRq[r][k] = sg (R_cam^T dn)_r
            G[3 * r + k] = sg * (R[r] * dn[0] + R[3 + r] * dn[1] + R[6 + r] * dn[2]);
        const float4 gq = quat_to_rotmat_vjp(q[0], q[1], q[2], q[3], G);
        // Through the normalisation q = raw / |raw|.
        const float raw[4] = {quat[4 * (size_t)p], quat[4 * (size_t)p + 1], quat[4 * (size_t)p + 2],
                              quat[4 * (size_t)p + 3]};
        const float inv = rsqrtf(fmaxf(raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] + raw[3] * raw[3], 1e-24f));
        const float g4[4] = {gq.x, gq.y, gq.z, gq.w};
        const float dot = g4[0] * q[0] + g4[1] * q[1] + g4[2] * q[2] + g4[3] * q[3];
        for (int c = 0; c < 4; ++c) atomicAdd(&grad_quat[4 * (size_t)p + c], (g4[c] - q[c] * dot) * inv);
    }
}
