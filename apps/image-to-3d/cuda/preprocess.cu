// Per-iteration camera, preprocess forward and backward, camera-gradient reduction.
// Owner: preprocess (E1). Spec: spec-raster Sections 1.2-1.3, 2, 5.3-5.5.
//
// Our own code, written from the spec. The EWA Jacobian clip and the compensation VJP follow
// gsplat's conventions (gsplat@937e299 gsplat/cuda/include/Utils.cuh, Apache-2.0) so the
// oracle matches; the structure (one thread per primitive, order-preserving compaction,
// tight tile counts, gradient recomputation per visible Gaussian) is spec-raster's.

// ---------------------------------------------------------------- per-iteration camera

// SE(3) exponential of the twist xi = (rho, phi): rotation Rd (row-major) and translation td.
SPL_INLINE void se3_exp(const float* xi, float Rd[9], float td[3]) {
    const float rx = xi[0], ry = xi[1], rz = xi[2];
    const float px = xi[3], py = xi[4], pz = xi[5];
    const float th2 = px * px + py * py + pz * pz;
    float a, b, c;
    if (th2 < 2.5e-3f) {  // series: error below 1e-9 here
        a = 1.f - th2 * (1.f / 6.f) * (1.f - th2 * (1.f / 20.f));
        b = 0.5f - th2 * (1.f / 24.f) * (1.f - th2 * (1.f / 30.f));
        c = (1.f / 6.f) - th2 * (1.f / 120.f) * (1.f - th2 * (1.f / 42.f));
    } else {
        const float th = sqrtf(th2);
        float s, co;
        sincosf(th, &s, &co);
        a = s / th;
        b = (1.f - co) / th2;
        c = (th - s) / (th2 * th);
    }
    // K = phi^, K^2 = phi phi^T - th2 I.
    const float K[9] = {0.f, -pz, py, pz, 0.f, -px, -py, px, 0.f};
    const float pp[3] = {px, py, pz};
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float k2 = pp[i] * pp[j] - (i == j ? th2 : 0.f);
            Rd[3 * i + j] = (i == j ? 1.f : 0.f) + a * K[3 * i + j] + b * k2;
        }
    }
    const float rho[3] = {rx, ry, rz};
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float k2 = pp[i] * pp[j] - (i == j ? th2 : 0.f);
            s += ((i == j ? 1.f : 0.f) + b * K[3 * i + j] + c * k2) * rho[j];
        }
        td[i] = s;
    }
}

// <<<1, 64>>>. With ring_slots > 0, copies ring[*iter_counter % ring_slots] (mapped pinned
// memory, read over PCIe) into *params; with ring_slots == 0 the caller uploaded *params
// already (eager frames). Composes *cam: W = exp(xi^) W0 with xi = pose_delta[cam_index],
// intrinsics scaled per axis to the render size, J clip limits, tile grid. Zeroes the
// status counters; ST_OVERFLOW and ST_OVERFLOW_ITER stay sticky until the host clears them.
extern "C" __global__ void frame_begin(const IterParams* ring, const uint32_t* __restrict__ iter_counter,
                                       uint32_t ring_slots, const CamBase* __restrict__ cams,
                                       const Intr* __restrict__ intr, const float* __restrict__ pose_delta,
                                       IterParams* params, FrameCam* __restrict__ cam,
                                       uint32_t* __restrict__ status) {
    __shared__ IterParams sp;
    const uint32_t t = threadIdx.x;
    const uint32_t* src =
        (const uint32_t*)(ring_slots ? ring + (*iter_counter % ring_slots) : (const IterParams*)params);
    for (uint32_t i = t; i < sizeof(IterParams) / 4; i += blockDim.x) {
        const uint32_t w = src[i];
        ((uint32_t*)&sp)[i] = w;
        if (ring_slots) ((uint32_t*)params)[i] = w;
    }
    if (t < STATUS_LEN && t != ST_OVERFLOW && t != ST_OVERFLOW_ITER && t != ST_ITER) status[t] = 0u;
    __syncthreads();
    if (t != 0) return;

    const FrameParams& fp = sp.frame;
    const uint32_t ci = fp.cam_index;
    const CamBase b = cams[ci];
    const Intr in = intr[b.intr_id];
    float Rd[9], td[3];
    se3_exp(pose_delta + 6 * ci, Rd, td);
    FrameCam c;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j)
            c.R[3 * i + j] = Rd[3 * i] * b.R[j] + Rd[3 * i + 1] * b.R[3 + j] + Rd[3 * i + 2] * b.R[6 + j];
        c.t[i] = Rd[3 * i] * b.t[0] + Rd[3 * i + 1] * b.t[1] + Rd[3 * i + 2] * b.t[2] + td[i];
    }
    const uint32_t W = fp.render_w ? fp.render_w : b.width;
    const uint32_t H = fp.render_h ? fp.render_h : b.height;
    const float sx = (float)W / (float)b.width, sy = (float)H / (float)b.height;
    c.fx = in.fx * sx;
    c.fy = in.fy * sy;
    c.cx = in.cx * sx;
    c.cy = in.cy * sy;
#pragma unroll
    for (int j = 0; j < 3; ++j) c.campos[j] = -(c.R[j] * c.t[0] + c.R[3 + j] * c.t[1] + c.R[6 + j] * c.t[2]);
    c.near_z = NEAR_DEFAULT;
    c.far_z = FAR_DEFAULT;
    c.lim_x0 = -(c.cx + CLIP_MARGIN * (float)W) / c.fx;
    c.lim_x1 = ((1.f + CLIP_MARGIN) * (float)W - c.cx) / c.fx;
    c.lim_y0 = -(c.cy + CLIP_MARGIN * (float)H) / c.fy;
    c.lim_y1 = ((1.f + CLIP_MARGIN) * (float)H - c.cy) / c.fy;
    c.width = W;
    c.height = H;
    c.grid_w = (W + TILE_W - 1) / TILE_W;
    c.grid_h = (H + TILE_H - 1) / TILE_H;
    c.n_tiles = c.grid_w * c.grid_h;
    c.image_index = ci;
    c.intr_id = b.intr_id;
    *cam = c;
}

// <<<1, 32>>>, the last kernel of an iteration: status[ST_ITER] = iteration, copies the
// status words into the mapped pinned mirror (when non-null) and increments *iter_counter.
extern "C" __global__ void frame_end(const IterParams* __restrict__ params, uint32_t* __restrict__ status,
                                     uint32_t* __restrict__ status_mirror, uint32_t* __restrict__ iter_counter) {
    const uint32_t t = threadIdx.x;
    if (t >= STATUS_LEN) return;
    uint32_t w = status[t];
    if (t == ST_ITER) {
        w = params->frame.iteration;
        status[t] = w;
    }
    if (status_mirror) {
        status_mirror[t] = w;
        __threadfence_system();
    }
    if (t == 0) *iter_counter += 1u;
}

// ---------------------------------------------------------------- shared forward state

// Everything the forward computes about one Gaussian, so preprocess_bwd recomputes exactly
// the forward's numbers (the compiler drops what a kernel does not use).
struct Proj {
    float3 mu;           // world mean
    float3 pc;           // camera-space mean (u, v, z)
    float rz;            // 1 / z
    float x, y;          // u / z, v / z
    float xc, yc;        // clipped to the J limits
    bool in_x, in_y;     // x, y inside the limits (gradient flows through x, y)
    float qn[4];         // unit quaternion (w, x, y, z)
    float qinv;          // 1 / |q|
    float Rq[9];         // rotation of the Gaussian
    float s2[3];         // exp(2 s_hat)
    float sf2[3];        // s2 + v3 (3D filter) or s2
    float v3;            // 3D filter variance (0 = off)
    float c3;            // 3D filter opacity compensation
    float M[6];          // J R_cam (2 x 3)
    float T[6];          // M R_q (2 x 3)
    float a0, b, c0;     // T diag(sf2) T^T without the 2D dilation
    float a, c;          // with eps2d
    float det, det0;     // a c - b^2, a0 c0 - b^2
    float c2;            // 2D (AA) compensation
    float o, oeff;       // sigmoid(o_hat), o c2 c3
};

// Depth and opacity culls, then the EWA projection (spec-raster 2 steps 1-5). Returns false
// when the Gaussian is culled before the conic. `flags` is FrameParams.flags.
SPL_INLINE bool project(uint32_t p, const FrameCam& cam, uint32_t flags, float eps2d,
                        const float* __restrict__ mean, const float* __restrict__ log_scale,
                        const float4* __restrict__ quat, const float* __restrict__ opacity,
                        const float* __restrict__ filter3d, Proj& g) {
    g.mu = make_float3(mean[3 * p], mean[3 * p + 1], mean[3 * p + 2]);
    const float* R = cam.R;
    g.pc.x = R[0] * g.mu.x + R[1] * g.mu.y + R[2] * g.mu.z + cam.t[0];
    g.pc.y = R[3] * g.mu.x + R[4] * g.mu.y + R[5] * g.mu.z + cam.t[1];
    g.pc.z = R[6] * g.mu.x + R[7] * g.mu.y + R[8] * g.mu.z + cam.t[2];
    if (!(g.pc.z >= cam.near_z && g.pc.z <= cam.far_z)) return false;
    g.o = 1.f / (1.f + __expf(-opacity[p]));
    if (g.o < ALPHA_MIN) return false;
    const float4 q = quat[p];
    const float n2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    if (!(n2 >= QUAT_MIN_NORM2)) return false;
    g.qinv = rsqrtf(n2);
    g.qn[0] = q.x * g.qinv;
    g.qn[1] = q.y * g.qinv;
    g.qn[2] = q.z * g.qinv;
    g.qn[3] = q.w * g.qinv;
    quat_to_rotmat(g.qn[0], g.qn[1], g.qn[2], g.qn[3], g.Rq);
    g.v3 = (filter3d && (flags & FLAG_FILTER3D)) ? filter3d[p] : 0.f;
    g.c3 = 1.f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        g.s2[k] = __expf(2.f * log_scale[3 * p + k]);
        g.sf2[k] = g.s2[k] + g.v3;
    }
    if (g.v3 > 0.f) g.c3 = sqrtf((g.s2[0] * g.s2[1] * g.s2[2]) / (g.sf2[0] * g.sf2[1] * g.sf2[2]));

    g.rz = 1.f / g.pc.z;
    g.x = g.pc.x * g.rz;
    g.y = g.pc.y * g.rz;
    g.in_x = g.x >= cam.lim_x0 && g.x <= cam.lim_x1;
    g.in_y = g.y >= cam.lim_y0 && g.y <= cam.lim_y1;
    g.xc = fminf(fmaxf(g.x, cam.lim_x0), cam.lim_x1);
    g.yc = fminf(fmaxf(g.y, cam.lim_y0), cam.lim_y1);
    const float j11 = cam.fx * g.rz, j13 = -cam.fx * g.xc * g.rz;
    const float j22 = cam.fy * g.rz, j23 = -cam.fy * g.yc * g.rz;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        g.M[j] = j11 * R[j] + j13 * R[6 + j];
        g.M[3 + j] = j22 * R[3 + j] + j23 * R[6 + j];
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
        for (int k = 0; k < 3; ++k)
            g.T[3 * r + k] = g.M[3 * r] * g.Rq[k] + g.M[3 * r + 1] * g.Rq[3 + k] + g.M[3 * r + 2] * g.Rq[6 + k];
    }
    g.a0 = g.T[0] * g.T[0] * g.sf2[0] + g.T[1] * g.T[1] * g.sf2[1] + g.T[2] * g.T[2] * g.sf2[2];
    g.b = g.T[0] * g.T[3] * g.sf2[0] + g.T[1] * g.T[4] * g.sf2[1] + g.T[2] * g.T[5] * g.sf2[2];
    g.c0 = g.T[3] * g.T[3] * g.sf2[0] + g.T[4] * g.T[4] * g.sf2[1] + g.T[5] * g.T[5] * g.sf2[2];
    g.a = g.a0 + eps2d;
    g.c = g.c0 + eps2d;
    g.det = g.a * g.c - g.b * g.b;
    if (!(g.det > 0.f)) return false;
    g.det0 = g.a0 * g.c0 - g.b * g.b;
    g.c2 = (flags & FLAG_AA) ? sqrtf(fmaxf(g.det0 / g.det, MIN_COMPENSATION * MIN_COMPENSATION)) : 1.f;
    g.oeff = g.o * g.c2 * g.c3;
    return g.oeff >= ALPHA_MIN;
}

// Unit view direction and distance of a Gaussian from the camera centre.
SPL_INLINE float3 view_dir(const float3 mu, const FrameCam& cam, float& dist) {
    const float3 d = make_float3(mu.x - cam.campos[0], mu.y - cam.campos[1], mu.z - cam.campos[2]);
    const float l2 = fmaxf(d.x * d.x + d.y * d.y + d.z * d.z, 1e-30f);
    const float inv = rsqrtf(l2);
    dist = l2 * inv;
    return make_float3(d.x * inv, d.y * inv, d.z * inv);
}

// ---------------------------------------------------------------- preprocess_fwd

#define LB_AGG (1u << 30)
#define LB_INC (2u << 30)
#define LB_VAL ((1u << 30) - 1u)

// Decoupled look-back over 256-primitive chunks (warp-collective): publishes this chunk's
// count, returns the number of visible Gaussians in all earlier chunks and publishes the
// inclusive prefix. Chunk ids come from an atomic counter, so every predecessor is owned by a
// running block (forward progress without relying on block scheduling order).
SPL_INLINE uint32_t chunk_lookback(uint32_t* lb, uint32_t chunk, uint32_t total, uint32_t lane) {
    if (chunk == 0) {
        if (lane == 0) st_release_gpu(&lb[0], LB_INC | total);
        return 0u;
    }
    if (lane == 0) st_release_gpu(&lb[chunk], LB_AGG | total);
    uint32_t excl = 0, ns = 32;
    int j = (int)chunk - 1;
    for (;;) {
        const int idx = j - (int)lane;
        const uint32_t w = idx >= 0 ? ld_acquire_gpu(&lb[idx]) : LB_INC;
        const uint32_t flag = w >> 30;
        const uint32_t not_ready = __ballot_sync(FULL_MASK, flag == 0u);
        const uint32_t inc = __ballot_sync(FULL_MASK, flag == 2u);
        const uint32_t upto = inc ? ((inc & (0u - inc)) << 1) - 1u : FULL_MASK;
        if (not_ready & upto) {
            __nanosleep(ns);
            ns = min(ns * 2u, 256u);
            continue;
        }
        excl += __reduce_add_sync(FULL_MASK, ((upto >> lane) & 1u) ? (w & LB_VAL) : 0u);
        if (inc) break;
        j -= 32;
    }
    if (lane == 0) st_release_gpu(&lb[chunk], LB_INC | (excl + total));
    return excl;
}

// Rows per Gaussian a lane counts on its own; larger spans are counted by the whole warp.
#define SPAN_LANE_ROWS 4

// Persistent grid (SMs x occupancy, PRE_THREADS threads), chunks of PRE_THREADS primitives
// from chunk_ctr. Culls, projects, evaluates SH, counts tiles exactly (span_tile_row), and
// compacts visible primitives in order into v-space (decoupled look-back over `lookback`):
// vis_idx, rec, depth_key = float bits of z (z > 0, so monotone), depth_val = v, n_tiles,
// zeroes g2d[.][v] (+ abs, extra), writes vis_mask words, last_visible[p] = iteration and
// status[ST_N_VIS]. chunk_ctr and lookback are zero on entry (memset by the stage function).
extern "C" __global__ void __launch_bounds__(PRE_THREADS)
    preprocess_fwd(const IterParams* __restrict__ params, const FrameCam* __restrict__ cam_p,
                   const float* __restrict__ mean, const float* __restrict__ log_scale,
                   const float4* __restrict__ quat, const float* __restrict__ opacity,
                   const float* __restrict__ sh_dc, const float* __restrict__ sh_rest,
                   const float* __restrict__ filter3d, uint32_t n_cap, uint32_t v_cap,
                   uint32_t* __restrict__ vis_mask, uint32_t* __restrict__ last_visible,
                   uint32_t* __restrict__ vis_idx, SplatRec* __restrict__ rec, uint32_t* __restrict__ depth_key,
                   uint32_t* __restrict__ depth_val, uint32_t* __restrict__ n_tiles, float* __restrict__ g2d,
                   float* __restrict__ g2d_abs, float* __restrict__ g2d_extra, uint32_t* __restrict__ chunk_ctr,
                   uint32_t* __restrict__ lookback, uint32_t* __restrict__ status) {
    __shared__ uint32_t s_chunk;
    __shared__ uint32_t s_warp[PRE_THREADS / 32];
    const FrameCam cam = *cam_p;
    const FrameParams fp = params->frame;
    const uint32_t n = min(fp.n_gaussians, n_cap);
    const uint32_t deg = min(fp.sh_degree, (uint32_t)MAX_SH_DEGREE);
    const uint32_t n_rest = sh_rest_count(deg);
    const uint32_t n_chunks = (n + PRE_THREADS - 1) / PRE_THREADS;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;

    for (;;) {
        if (threadIdx.x == 0) s_chunk = atomicAdd(chunk_ctr, 1u);
        __syncthreads();
        const uint32_t chunk = s_chunk;
        if (chunk >= n_chunks) break;
        const uint32_t p = chunk * PRE_THREADS + threadIdx.x;

        Proj g;
        bool alive = p < n && project(p, cam, fp.flags, fp.eps2d, mean, log_scale, quat, opacity, filter3d, g);
        SplatRec r;
        SpanEllipse e;
        int px0 = 0, px1 = -1, py0 = 0, py1 = -1, ty0 = 0, ty1 = -1;
        if (alive) {
            const float inv = 1.f / g.det;
            const float A = g.c * inv, B = -g.b * inv, C = g.a * inv;
            r.mx = cam.fx * g.x + cam.cx;
            r.my = cam.fy * g.y + cam.cy;
            r.ca = CONIC_SCALE_AC * A;
            r.cb = CONIC_SCALE_B * B;
            r.cc = CONIC_SCALE_AC * C;
            r.o = g.oeff;
            r.pthr = fmaxf(__log2f(255.f * g.oeff), 0.f);
            r.z = g.pc.z;
            e = span_ellipse(r, fp.box_beta);
            alive = e.det > 0.f && span_pixel_box(e, cam.width, cam.height, px0, px1, py0, py1);
            ty0 = py0 / TILE_H;
            ty1 = py1 / TILE_H;
        }
        // Tile count: small spans per lane, large ones by the whole warp.
        uint32_t ntl = 0;
        const int rows = alive ? ty1 - ty0 + 1 : 0;
        if (alive && rows <= SPAN_LANE_ROWS) {
            for (int ty = ty0; ty <= ty1; ++ty) {
                int a0, a1;
                ntl += span_tile_row(e, ty, px0, px1, py0, py1, a0, a1);
            }
        }
        uint32_t big = __ballot_sync(FULL_MASK, rows > SPAN_LANE_ROWS);
        while (big) {
            const int src = __ffs(big) - 1;
            big &= big - 1u;
            SpanEllipse eb;
            eb.mx = __shfl_sync(FULL_MASK, e.mx, src);
            eb.my = __shfl_sync(FULL_MASK, e.my, src);
            eb.A = __shfl_sync(FULL_MASK, e.A, src);
            eb.B = __shfl_sync(FULL_MASK, e.B, src);
            eb.C = __shfl_sync(FULL_MASK, e.C, src);
            eb.det = __shfl_sync(FULL_MASK, e.det, src);
            eb.k = __shfl_sync(FULL_MASK, e.k, src);
            eb.x_ext = __shfl_sync(FULL_MASK, e.x_ext, src);
            eb.y_ext = __shfl_sync(FULL_MASK, e.y_ext, src);
            const int bx0 = __shfl_sync(FULL_MASK, px0, src), bx1 = __shfl_sync(FULL_MASK, px1, src);
            const int by0 = __shfl_sync(FULL_MASK, py0, src), by1 = __shfl_sync(FULL_MASK, py1, src);
            const int bt0 = __shfl_sync(FULL_MASK, ty0, src), bt1 = __shfl_sync(FULL_MASK, ty1, src);
            uint32_t cnt = 0;
            for (int ty = bt0 + (int)lane; ty <= bt1; ty += 32) {
                int a0, a1;
                cnt += span_tile_row(eb, ty, bx0, bx1, by0, by1, a0, a1);
            }
            cnt = __reduce_add_sync(FULL_MASK, cnt);
            if ((int)lane == src) ntl = cnt;
        }
        const bool vis = alive && ntl > 0;

        // Colour (visible only): 0.5 + SH, clamped at 0 (spec-raster 2.7).
        if (vis) {
            float dist;
            const float3 d = view_dir(g.mu, cam, dist);
            float Y[15];
            sh_basis(deg, d.x, d.y, d.z, Y);
            float col[3];
#pragma unroll
            for (int ch = 0; ch < 3; ++ch) col[ch] = 0.5f + SH_C0 * sh_dc[3 * p + ch];
#pragma unroll
            for (int k = 0; k < 15; ++k) {
                if ((uint32_t)k < n_rest) {
#pragma unroll
                    for (int ch = 0; ch < 3; ++ch) col[ch] += Y[k] * sh_rest[(size_t)(3 * k + ch) * n_cap + p];
                }
            }
            r.flags = (col[0] <= 0.f ? 1u : 0u) | (col[1] <= 0.f ? 2u : 0u) | (col[2] <= 0.f ? 4u : 0u);
            r.r = fmaxf(col[0], 0.f);
            r.g = fmaxf(col[1], 0.f);
            r.b = fmaxf(col[2], 0.f);
        }

        // Order-preserving compaction: block scan, then look-back over chunks.
        const uint32_t ballot = __ballot_sync(FULL_MASK, vis);
        if (lane == 0) {
            s_warp[warp] = __popc(ballot);
            const uint32_t word = chunk * (PRE_THREADS / 32) + warp;
            if (word < (n_cap + 31) / 32) vis_mask[word] = ballot;
        }
        __syncthreads();
        if (warp == 0) {
            const uint32_t cnt = lane < PRE_THREADS / 32 ? s_warp[lane] : 0u;
            uint32_t incl = cnt;
#pragma unroll
            for (uint32_t o = 1; o < PRE_THREADS / 32; o <<= 1) {
                const uint32_t up = __shfl_up_sync(FULL_MASK, incl, o);
                if (lane >= o) incl += up;
            }
            const uint32_t total = __shfl_sync(FULL_MASK, incl, PRE_THREADS / 32 - 1);
            const uint32_t base = chunk_lookback(lookback, chunk, total, lane);
            if (lane < PRE_THREADS / 32) s_warp[lane] = base + incl - cnt;
            if (lane == 0 && chunk == n_chunks - 1) status[ST_N_VIS] = base + total;
        }
        __syncthreads();
        if (vis) {
            const uint32_t v = s_warp[warp] + __popc(ballot & lanemask_lt());
            if (v < v_cap) {
                vis_idx[v] = p;
                float4* dst = (float4*)(rec + v);
                dst[0] = make_float4(r.mx, r.my, r.ca, r.cb);
                dst[1] = make_float4(r.cc, r.o, r.r, r.g);
                dst[2] = make_float4(r.b, r.pthr, r.z, __uint_as_float(r.flags));
                depth_key[v] = __float_as_uint(r.z);
                depth_val[v] = v;
                n_tiles[v] = ntl;
#pragma unroll
                for (int j = 0; j < 9; ++j) g2d[(size_t)j * v_cap + v] = 0.f;
                if (g2d_abs) {
                    g2d_abs[v] = 0.f;
                    g2d_abs[(size_t)v_cap + v] = 0.f;
                }
                if (g2d_extra) {
#pragma unroll
                    for (int j = 0; j < EXTRA_CH; ++j) g2d_extra[(size_t)j * v_cap + v] = 0.f;
                }
                last_visible[p] = fp.iteration;
            } else {
                atomicOr(&status[ST_OVERFLOW], OVF_VISIBLE);
            }
        }
        // No barrier needed here: s_chunk and s_warp are rewritten next round only by
        // threads that passed its first barrier, after every read of this round.
    }
}

// ---------------------------------------------------------------- preprocess_bwd

#define SINK_WRITE 0
#define SINK_FUSED_ADAM 1

// grad[i] = g (overwrite) or += g (accumulate); one thread owns each p, so no atomics.
SPL_INLINE void sink_put(float* __restrict__ grad, size_t i, float g, bool overwrite) {
    grad[i] = overwrite ? g : grad[i] + g;
}

// One thread per visible v (fixed grid, grid-stride). Recomputes the forward from the
// parameters and chains g2d back to mean, log_scale, quat, opacity and SH (spec-raster 5.3).
// sink == SINK_WRITE writes gradients at p (Adam's overwrite_mask bit g: = instead of +=).
// With FLAG_CAM_GRAD and cam_partial, per-block sums of (g_rho, g_phi, g_fx W, g_fy H,
// g_cx W, g_cy H) go to cam_partial[block][CAM_GRAD_N] (fixed grid: deterministic).
// With FLAG_DENSIFY_STATS and stat_vis_count, the densification statistics accumulate at p.
extern "C" __global__ void __launch_bounds__(PRE_THREADS)
    preprocess_bwd(const IterParams* __restrict__ params, const FrameCam* __restrict__ cam_p,
                   const float* __restrict__ mean, const float* __restrict__ log_scale,
                   const float4* __restrict__ quat, const float* __restrict__ opacity,
                   const float* __restrict__ sh_dc, const float* __restrict__ sh_rest,
                   const float* __restrict__ filter3d, uint32_t n_cap, uint32_t v_cap,
                   const uint32_t* __restrict__ vis_idx, const float* __restrict__ g2d,
                   const float* __restrict__ g2d_abs, const float* __restrict__ g2d_extra, uint32_t sink,
                   float* __restrict__ grad_mean, float* __restrict__ grad_log_scale,
                   float* __restrict__ grad_quat, float* __restrict__ grad_opacity,
                   float* __restrict__ grad_sh_dc, float* __restrict__ grad_sh_rest,
                   float* __restrict__ cam_partial, float* __restrict__ stat_vis_count,
                   float* __restrict__ stat_grad_sum, float* __restrict__ stat_grad_abs_sum,
                   const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    __shared__ float s_cam[PRE_THREADS / 32][CAM_GRAD_N];
    const FrameCam cam = *cam_p;
    const FrameParams fp = params->frame;
    const uint32_t ow = params->adam.overwrite_mask;
    const uint32_t V = min(status[ST_N_VIS], v_cap);
    const uint32_t deg = min(fp.sh_degree, (uint32_t)MAX_SH_DEGREE);
    const uint32_t n_rest = sh_rest_count(deg);
    const bool want_cam = cam_partial && (fp.flags & FLAG_CAM_GRAD);
    const bool want_stats = stat_vis_count && (fp.flags & FLAG_DENSIFY_STATS);
    const bool aa_grad = (fp.flags & FLAG_AA) && (fp.flags & FLAG_AA_GRAD);
    float acc[CAM_GRAD_N];
#pragma unroll
    for (int i = 0; i < CAM_GRAD_N; ++i) acc[i] = 0.f;

    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < V; v += gridDim.x * blockDim.x) {
        const uint32_t p = vis_idx[v];
        const float gmx = g2d[v], gmy = g2d[(size_t)v_cap + v];
        const float gA = g2d[2 * (size_t)v_cap + v], gB = g2d[3 * (size_t)v_cap + v];
        const float gC = g2d[4 * (size_t)v_cap + v], go = g2d[5 * (size_t)v_cap + v];
        float gcol[3] = {g2d[6 * (size_t)v_cap + v], g2d[7 * (size_t)v_cap + v], g2d[8 * (size_t)v_cap + v]};
        const float gz_extra = (EXTRA_CH > 0 && g2d_extra) ? g2d_extra[v] : 0.f;

        Proj g;
        // Visible in the forward, so the culls pass again (same inputs, same code).
        project(p, cam, fp.flags, fp.eps2d, mean, log_scale, quat, opacity, filter3d, g);

        // ---- SH colour (spec-raster 5.3 step 8)
        float dist;
        const float3 d = view_dir(g.mu, cam, dist);
        float Y[15];
        sh_basis(deg, d.x, d.y, d.z, Y);
        float f[15][3];
        float raw[3];
#pragma unroll
        for (int ch = 0; ch < 3; ++ch) raw[ch] = 0.5f + SH_C0 * sh_dc[3 * p + ch];
#pragma unroll
        for (int k = 0; k < 15; ++k) {
#pragma unroll
            for (int ch = 0; ch < 3; ++ch) {
                f[k][ch] = (uint32_t)k < n_rest ? sh_rest[(size_t)(3 * k + ch) * n_cap + p] : 0.f;
                raw[ch] += Y[k] * f[k][ch];
            }
        }
#pragma unroll
        for (int ch = 0; ch < 3; ++ch) gcol[ch] = raw[ch] > 0.f ? gcol[ch] : 0.f;
        float w[15];
#pragma unroll
        for (int k = 0; k < 15; ++k) w[k] = f[k][0] * gcol[0] + f[k][1] * gcol[1] + f[k][2] * gcol[2];
        const float3 gd = sh_basis_vjp(deg, d.x, d.y, d.z, w);
        const float gdd = gd.x * d.x + gd.y * d.y + gd.z * d.z;
        const float idist = 1.f / dist;
        const float3 gmu_sh =
            make_float3((gd.x - gdd * d.x) * idist, (gd.y - gdd * d.y) * idist, (gd.z - gdd * d.z) * idist);

        // ---- conic -> 2D covariance (step 1), compensation (step 2)
        const float id2 = 1.f / (g.det * g.det);
        float ga = (-g.c * g.c * gA + g.b * g.c * gB - g.b * g.b * gC) * id2;
        float gb = (2.f * g.b * g.c * gA - (g.a * g.c + g.b * g.b) * gB + 2.f * g.a * g.b * gC) * id2;
        float gc = (-g.b * g.b * gA + g.a * g.b * gB - g.a * g.a * gC) * id2;
        if (aa_grad) {
            const float rr = go * g.o * g.c3 * 0.5f / g.c2;
            ga += rr * (g.c0 * g.det - g.det0 * g.c) * id2;
            gc += rr * (g.a0 * g.det - g.det0 * g.a) * id2;
            gb += rr * (-2.f * g.b * (g.det - g.det0)) * id2;
        }
        const float g_ohat = go * g.c2 * g.c3 * g.o * (1.f - g.o);

        // ---- Sigma2 = T diag(sf2) T^T (steps 3, 6)
        float gsf2[3], dT[6];
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const float t1 = g.T[k], t2 = g.T[3 + k];
            gsf2[k] = ga * t1 * t1 + gb * t1 * t2 + gc * t2 * t2;
            dT[k] = 2.f * g.sf2[k] * (ga * t1 + 0.5f * gb * t2);
            dT[3 + k] = 2.f * g.sf2[k] * (0.5f * gb * t1 + gc * t2);
        }
        // T = M R_q: dR_q = M^T dT, dM = dT R_q^T
        float dRq[9], dM[6];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
#pragma unroll
            for (int k = 0; k < 3; ++k) dRq[3 * j + k] = g.M[j] * dT[k] + g.M[3 + j] * dT[3 + k];
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int j = 0; j < 3; ++j)
                dM[3 * r + j] = dT[3 * r] * g.Rq[3 * j] + dT[3 * r + 1] * g.Rq[3 * j + 1] + dT[3 * r + 2] * g.Rq[3 * j + 2];
        }
        // M = J R_cam: dJ = dM R_cam^T (step 4)
        const float* R = cam.R;
        float dJ[6];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int i = 0; i < 3; ++i)
                dJ[3 * r + i] = dM[3 * r] * R[3 * i] + dM[3 * r + 1] * R[3 * i + 1] + dM[3 * r + 2] * R[3 * i + 2];
        }

        // ---- J and mean2d -> camera-space mean (step 5)
        const float fx = cam.fx, fy = cam.fy, rz = g.rz, rz2 = rz * rz;
        float3 gp;
        gp.x = fx * rz * gmx - (g.in_x ? fx * rz2 * dJ[2] : 0.f);
        gp.y = fy * rz * gmy - (g.in_y ? fy * rz2 * dJ[5] : 0.f);
        gp.z = -fx * g.x * rz * gmx - fy * g.y * rz * gmy - fx * rz2 * dJ[0] - fy * rz2 * dJ[4] +
               (g.in_x ? 2.f : 1.f) * fx * g.xc * rz2 * dJ[2] + (g.in_y ? 2.f : 1.f) * fy * g.yc * rz2 * dJ[5] +
               gz_extra;
        float3 gmu;
        gmu.x = R[0] * gp.x + R[3] * gp.y + R[6] * gp.z + gmu_sh.x;
        gmu.y = R[1] * gp.x + R[4] * gp.y + R[7] * gp.z + gmu_sh.y;
        gmu.z = R[2] * gp.x + R[5] * gp.y + R[8] * gp.z + gmu_sh.z;

        // ---- scales (step 6) and quaternion (step 7)
        float gs[3];
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            gs[k] = 2.f * g.s2[k] * gsf2[k];
            if (g.v3 > 0.f) gs[k] += go * g.oeff * g.v3 / g.sf2[k];
        }
        const float4 gqu = quat_to_rotmat_vjp(g.qn[0], g.qn[1], g.qn[2], g.qn[3], dRq);
        const float gdot = gqu.x * g.qn[0] + gqu.y * g.qn[1] + gqu.z * g.qn[2] + gqu.w * g.qn[3];
        const float4 gq = make_float4((gqu.x - gdot * g.qn[0]) * g.qinv, (gqu.y - gdot * g.qn[1]) * g.qinv,
                                      (gqu.z - gdot * g.qn[2]) * g.qinv, (gqu.w - gdot * g.qn[3]) * g.qinv);

        // ---- sink
        if (sink == SINK_WRITE) {
            // Bits of AdamIterPlan.overwrite_mask: GROUP_* order of adam.cuh.
            const bool o_mean = ow & 1u, o_scale = ow & 2u, o_rot = ow & 4u, o_op = ow & 8u;
            const bool o_dc = ow & 16u, o_rest = ow & 32u;
            sink_put(grad_mean, 3 * (size_t)p, gmu.x, o_mean);
            sink_put(grad_mean, 3 * (size_t)p + 1, gmu.y, o_mean);
            sink_put(grad_mean, 3 * (size_t)p + 2, gmu.z, o_mean);
#pragma unroll
            for (int k = 0; k < 3; ++k) sink_put(grad_log_scale, 3 * (size_t)p + k, gs[k], o_scale);
            if (o_rot) {
                ((float4*)grad_quat)[p] = gq;
            } else {
                float4 old = ((float4*)grad_quat)[p];
                ((float4*)grad_quat)[p] = make_float4(old.x + gq.x, old.y + gq.y, old.z + gq.z, old.w + gq.w);
            }
            sink_put(grad_opacity, p, g_ohat, o_op);
#pragma unroll
            for (int ch = 0; ch < 3; ++ch) sink_put(grad_sh_dc, 3 * (size_t)p + ch, SH_C0 * gcol[ch], o_dc);
#pragma unroll
            for (int k = 0; k < 15; ++k) {
                if ((uint32_t)k < n_rest || o_rest) {
#pragma unroll
                    for (int ch = 0; ch < 3; ++ch)
                        sink_put(grad_sh_rest, (size_t)(3 * k + ch) * n_cap + p, Y[k] * gcol[ch], o_rest);
                }
            }
        }

        // ---- densification statistics (spec-raster 5.5, NDC units)
        if (want_stats) {
            const float hw = 0.5f * (float)cam.width, hh = 0.5f * (float)cam.height;
            stat_vis_count[p] += 1.f;
            stat_grad_sum[p] += sqrtf(gmx * hw * gmx * hw + gmy * hh * gmy * hh);
            if (stat_grad_abs_sum && g2d_abs) {
                const float ax = g2d_abs[v] * hw, ay = g2d_abs[(size_t)v_cap + v] * hh;
                stat_grad_abs_sum[p] += sqrtf(ax * ax + ay * ay);
            }
        }

        // ---- camera pose and intrinsics (spec-raster 5.4)
        if (want_cam) {
            // rho: camera-space mean plus the SH path through the camera centre.
            acc[0] += gp.x + R[0] * gmu_sh.x + R[1] * gmu_sh.y + R[2] * gmu_sh.z;
            acc[1] += gp.y + R[3] * gmu_sh.x + R[4] * gmu_sh.y + R[5] * gmu_sh.z;
            acc[2] += gp.z + R[6] * gmu_sh.x + R[7] * gmu_sh.y + R[8] * gmu_sh.z;
            // phi: p x g_p + vee'(J^T dJ) with J = [[j11, 0, j13], [0, j22, j23]].
            const float j11 = fx * rz, j13 = -fx * g.xc * rz, j22 = fy * rz, j23 = -fy * g.yc * rz;
            const float3 pg = cross3(g.pc, gp);
            acc[3] += pg.x + j13 * dJ[1] + j23 * dJ[4] - j22 * dJ[5];
            acc[4] += pg.y + j11 * dJ[2] - j13 * dJ[0] - j23 * dJ[3];
            acc[5] += pg.z + j22 * dJ[3] - j11 * dJ[1];
            // Intrinsics normalized by the render size (fx/W, fy/H, cx/W, cy/H), so frames at
            // different render scales accumulate consistently. Clip limits held constant.
            acc[6] += (gmx * g.x + (dJ[0] - g.xc * dJ[2]) * rz) * (float)cam.width;
            acc[7] += (gmy * g.y + (dJ[4] - g.yc * dJ[5]) * rz) * (float)cam.height;
            acc[8] += gmx * (float)cam.width;
            acc[9] += gmy * (float)cam.height;
        }
    }

    if (want_cam) {
        const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
#pragma unroll
        for (int i = 0; i < CAM_GRAD_N; ++i) {
            const float s = warp_sum(acc[i]);
            if (lane == 0) s_cam[warp][i] = s;
        }
        __syncthreads();
        if (threadIdx.x < CAM_GRAD_N) {
            float s = 0.f;
#pragma unroll
            for (int w2 = 0; w2 < PRE_THREADS / 32; ++w2) s += s_cam[w2][threadIdx.x];
            cam_partial[(size_t)blockIdx.x * CAM_GRAD_N + threadIdx.x] = s;
        }
    }
}

// <<<1, 32 * CAM_GRAD_N>>>: warp i sums column i of cam_partial[n_blocks][CAM_GRAD_N] in a
// fixed order and adds it to pose_grad[image][i] (i < 6) or intr_grad[intr_id][i - 6].
// The pose optimizer zeroes what it steps.
extern "C" __global__ void cam_reduce(const FrameCam* __restrict__ cam, const float* __restrict__ cam_partial,
                                      uint32_t n_blocks, float* __restrict__ pose_grad,
                                      float* __restrict__ intr_grad, const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    const uint32_t lane = threadIdx.x & 31u, i = threadIdx.x >> 5;
    if (i >= CAM_GRAD_N) return;
    float s = 0.f;
    for (uint32_t b = lane; b < n_blocks; b += 32) s += cam_partial[(size_t)b * CAM_GRAD_N + i];
    s = warp_sum(s);
    if (lane != 0) return;
    if (i < 6)
        pose_grad[(size_t)cam->image_index * 6 + i] += s;
    else
        intr_grad[(size_t)cam->intr_id * 4 + (i - 6)] += s;
}
