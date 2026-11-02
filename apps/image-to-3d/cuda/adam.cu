// Multi-group Adam over SoA parameters, invisible-Gaussian handling, LazyCatchUp.
// Owner: optimizer (E5). Spec: spec-train Section 4. Compiled as
// common.cuh + splat_common.cuh + adam.cuh + adam.cu.
//
// One launch per iteration covers every due group: a persistent grid strides over the
// quads (4 consecutive elements of one group) of all due groups, the group found from a
// 6-entry prefix computed from IterParams.frame.n_gaussians. Interleaved groups
// ([cap][k]) are one contiguous range of n*k floats; SH rest is planar [45][cap], so its
// quads run row by row over ceil(n/4) quads per row. Full quads use 16-byte loads and
// stores. Every kernel starts by checking status[ST_OVERFLOW].
//
// Write modes (spec-train 4.3): Overwrite reads g = (last_visible[p] == t) ? grad : 0, so
// stale entries of invisible Gaussians never count and nothing needs zeroing; the plan's
// zero-after bit makes the kernel clear the group's gradient when the next iteration
// accumulates (Accumulate groups always, Overwrite groups at a plan change).
//
// Sparsity (spec-train 4.5): Dense steps everything; VisibleOnly and LazyCatchUp skip a
// Gaussian unless it was visible after the group's previous step (last_visible[p] >
// prev_due(t)): no load, no moment decay, no store.

// Launched with 256-thread blocks (kernels::adam::THREADS).

struct GroupRanges {
    uint32_t start[N_GROUPS + 1];  // quad prefix over due groups
};

SPL_INLINE uint32_t group_quads(int g, uint32_t n) {
    switch (g) {
        case GROUP_MEANS:
        case GROUP_SCALES:
        case GROUP_SH0: return (3 * n + 3) / 4;
        case GROUP_ROTATIONS: return n;
        case GROUP_OPACITIES: return (n + 3) / 4;
        default: return 45 * ((n + 3) / 4);
    }
}

// Element and Gaussian indices of quad q of a group with K floats per Gaussian (K = 45:
// planar SH rest).
template <int K>
SPL_INLINE void quad_index(uint32_t q, uint32_t n, uint32_t n_cap, uint32_t& base, uint32_t p[4], bool valid[4]) {
    if (K == 45) {
        uint32_t nq = (n + 3) / 4;
        uint32_t j = q / nq;
        uint32_t p0 = (q - j * nq) * 4;
        base = j * n_cap + p0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            p[i] = p0 + i;
            valid[i] = p0 + i < n;
        }
    } else {
        base = 4 * q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            p[i] = (base + i) / K;
            valid[i] = base + i < n * K;
        }
    }
}

// Philox bits for the bf16 stochastic rounding of one quad (unused in f32 mode).
SPL_INLINE uint4 sr_bits(const IterParams& P, uint32_t q, uint32_t t, int g, uint32_t salt) {
#if ADAM_MOMENTS_BF16
    return philox4x32_10(make_uint4(q, t, RNG_BF16_SR, salt + (uint32_t)g), make_uint2(P.seed[0], P.seed[1]));
#else
    (void)P, (void)q, (void)t, (void)g, (void)salt;
    return make_uint4(0, 0, 0, 0);
#endif
}

// 8-byte moment words seen as two floats (f32 and mixed modes; bits preserved).
#if ADAM_MOMENTS_BF16 == 2
SPL_INLINE MomentT moment_bits(float a, float b) { return make_uint2(__float_as_uint(a), __float_as_uint(b)); }
SPL_INLINE float2 moment_floats(MomentT w) { return make_float2(__uint_as_float(w.x), __uint_as_float(w.y)); }
#elif ADAM_MOMENTS_BF16 == 0
SPL_INLINE MomentT moment_bits(float a, float b) { return make_float2(a, b); }
SPL_INLINE float2 moment_floats(MomentT w) { return w; }
#endif

SPL_INLINE uint32_t pick(const uint4& r, int i) { return i == 0 ? r.x : i == 1 ? r.y : i == 2 ? r.z : r.w; }

struct StepCtx {
    uint32_t t;
    float step_size, bc2;
    bool overwrite, zero_after;
    uint32_t prev_due;
    float reg;  // MCMC regularizer weight of this group (opacity / scale), 0 = off
};

// INVIS: adam_invisible (only Gaussians with last_visible != t, g = 0).
template <int K, bool INVIS>
SPL_INLINE void quad_step(const IterParams& P, const GroupPtrs& ptrs, const AdamHyper& h, int g, uint32_t q,
                          uint32_t n, uint32_t n_cap, const uint32_t* __restrict__ last_visible, const StepCtx& c) {
    uint32_t base, p[4];
    bool valid[4];
    quad_index<K>(q, n, n_cap, base, p, valid);
    const bool full = valid[0] && valid[1] && valid[2] && valid[3];

    uint32_t lv[4];
    if ((K == 45 || K == 1) && full) {
        uint4 l = *reinterpret_cast<const uint4*>(last_visible + p[0]);
        lv[0] = l.x, lv[1] = l.y, lv[2] = l.z, lv[3] = l.w;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) lv[i] = valid[i] ? last_visible[p[i]] : 0u;
    }

    bool step[4], vis_now[4];
    bool any_step = false, any_vis = false;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vis_now[i] = lv[i] == c.t;
        bool s;
        if (INVIS)
            s = !vis_now[i];
        else if (h.sparsity == SPARSE_DENSE)
            s = true;
        else
            s = lv[i] > c.prev_due;
        step[i] = valid[i] && s;
        any_step |= step[i];
        any_vis |= valid[i] && vis_now[i];
    }
    // Zero-after: stepped elements always; skipped ones only in Overwrite mode (their
    // entries may be stale; skipped Accumulate entries are already zero).
    const bool zero_any = !INVIS && c.zero_after && (any_step || c.overwrite);
    if (!any_step && !zero_any) return;

    float* __restrict__ par = ptrs.param[g] + base;
    float* __restrict__ grd = ptrs.grad[g] + base;
    MomentT* __restrict__ mom = ptrs.moments[g] + base;

    // Gradients: not read at all by adam_invisible, nor in Overwrite mode when no element
    // of the quad was rendered this iteration.
    float gr[4] = {0.f, 0.f, 0.f, 0.f};
    const bool need_grad = !INVIS && (c.overwrite ? any_vis : any_step);
    if (need_grad) {
        if (full) {
            float4 v = *reinterpret_cast<const float4*>(grd);
            gr[0] = v.x, gr[1] = v.y, gr[2] = v.z, gr[3] = v.w;
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) gr[i] = valid[i] ? grd[i] : 0.f;
        }
    }
    if (any_step) {
        float pv[4], m[4], v[4];
        if (full) {
            float4 x = *reinterpret_cast<const float4*>(par);
            pv[0] = x.x, pv[1] = x.y, pv[2] = x.z, pv[3] = x.w;
#if ADAM_MOMENTS_BF16 == 1
            uint4 w = *reinterpret_cast<const uint4*>(mom);
            moment_load(w.x, m[0], v[0]);
            moment_load(w.y, m[1], v[1]);
            moment_load(w.z, m[2], v[2]);
            moment_load(w.w, m[3], v[3]);
#else
            const float4* mp = reinterpret_cast<const float4*>(mom);
            float4 a = mp[0], b = mp[1];
            moment_load(moment_bits(a.x, a.y), m[0], v[0]);
            moment_load(moment_bits(a.z, a.w), m[1], v[1]);
            moment_load(moment_bits(b.x, b.y), m[2], v[2]);
            moment_load(moment_bits(b.z, b.w), m[3], v[3]);
#endif
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                pv[i] = 0.f, m[i] = 0.f, v[i] = 0.f;
                if (valid[i]) {
                    pv[i] = par[i];
                    moment_load(mom[i], m[i], v[i]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (!step[i]) continue;
            float gi = (c.overwrite && !vis_now[i]) ? 0.f : gr[i];
            if (c.reg != 0.f) {
                if (g == GROUP_OPACITIES) {
                    float s = 1.f / (1.f + __expf(-pv[i]));
                    gi += c.reg * s * (1.f - s);
                } else {
                    gi += c.reg * __expf(pv[i]);
                }
            }
            adam_update(gi, pv[i], m[i], v[i], c.step_size, c.bc2, h.beta1, h.beta2, h.one_minus_beta1, h.one_minus_beta2,
                        h.eps);
        }
        uint4 r = sr_bits(P, q, c.t, g, INVIS ? 0x40u : 0u);
        if (full) {
            *reinterpret_cast<float4*>(par) = make_float4(pv[0], pv[1], pv[2], pv[3]);
#if ADAM_MOMENTS_BF16 == 1
            *reinterpret_cast<uint4*>(mom) = make_uint4(moment_pack(m[0], v[0], r.x), moment_pack(m[1], v[1], r.y),
                                                        moment_pack(m[2], v[2], r.z), moment_pack(m[3], v[3], r.w));
#else
            float2 w0 = moment_floats(moment_pack(m[0], v[0], r.x)), w1 = moment_floats(moment_pack(m[1], v[1], r.y));
            float2 w2 = moment_floats(moment_pack(m[2], v[2], r.z)), w3 = moment_floats(moment_pack(m[3], v[3], r.w));
            float4* mp = reinterpret_cast<float4*>(mom);
            mp[0] = make_float4(w0.x, w0.y, w1.x, w1.y);
            mp[1] = make_float4(w2.x, w2.y, w3.x, w3.y);
#endif
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                if (step[i]) {
                    par[i] = pv[i];
                    mom[i] = moment_pack(m[i], v[i], pick(r, i));
                }
        }
    }
    if (zero_any) {
        if (full && c.overwrite) {
            *reinterpret_cast<float4*>(grd) = make_float4(0.f, 0.f, 0.f, 0.f);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                if (valid[i] && (step[i] || c.overwrite)) grd[i] = 0.f;
        }
    }
}

template <bool INVIS>
SPL_INLINE void adam_body(const IterParams* __restrict__ params, const GroupPtrs& ptrs, const AdamHyper& h,
                          const StepPlansDev& plans, const uint32_t* __restrict__ last_visible, uint32_t n_cap,
                          float2 reg, float2* __restrict__ lr_hist, const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    const IterParams& P = *params;
    const uint32_t t = P.frame.iteration, n = min(P.frame.n_gaussians, n_cap);
    const uint32_t due = P.adam.due_mask, ow = P.adam.overwrite_mask;

    uint32_t start[N_GROUPS + 1];
    start[0] = 0;
#pragma unroll
    for (int g = 0; g < N_GROUPS; ++g) start[g + 1] = start[g] + (((due >> g) & 1u) ? group_quads(g, n) : 0u);

    // LazyCatchUp: remember this step's (step size, bias correction) per group.
    if (!INVIS && lr_hist && blockIdx.x == 0 && threadIdx.x < N_GROUPS && ((due >> threadIdx.x) & 1u)) {
        int g = threadIdx.x;
        uint32_t s = plan_due_count(plans, g, 0, t);
        lr_hist[g * LR_HIST + s % LR_HIST] = make_float2(P.adam.step_size[g], P.adam.bc2_sqrt_rcp[g]);
    }

    const uint32_t total = start[N_GROUPS];
    int g = 0;
    StepCtx c;
    c.t = t;
    bool have = false;
    for (uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < total; q += gridDim.x * blockDim.x) {
        int gq = 0;
        while (q >= start[gq + 1]) ++gq;
        if (!have || gq != g) {
            g = gq;
            have = true;
            c.step_size = P.adam.step_size[g];
            c.bc2 = P.adam.bc2_sqrt_rcp[g];
            c.overwrite = (ow >> g) & 1u;
            c.zero_after = (ow >> (ZERO_AFTER_SHIFT + g)) & 1u;
            c.prev_due = h.sparsity == SPARSE_DENSE ? 0u : plan_prev_due(plans, g, t);
            c.reg = g == GROUP_OPACITIES ? reg.x : g == GROUP_SCALES ? reg.y : 0.f;
        }
        uint32_t lq = q - start[g];
        switch (g) {
            case GROUP_MEANS:
            case GROUP_SCALES:
            case GROUP_SH0: quad_step<3, INVIS>(P, ptrs, h, g, lq, n, n_cap, last_visible, c); break;
            case GROUP_ROTATIONS: quad_step<4, INVIS>(P, ptrs, h, g, lq, n, n_cap, last_visible, c); break;
            case GROUP_OPACITIES: quad_step<1, INVIS>(P, ptrs, h, g, lq, n, n_cap, last_visible, c); break;
            default: quad_step<45, INVIS>(P, ptrs, h, g, lq, n, n_cap, last_visible, c); break;
        }
    }
}

// The optimizer step of every due group. MCMC regularizer gradients (opacity, scale) are
// added here, densely, with weights reg = (lambda_o / n, lambda_s / (3 n)) already folded:
// g += reg.x sigmoid'(o), g += reg.y exp(s). lr_hist: LazyCatchUp ring [6][LR_HIST], or null.
extern "C" __global__ void adam_step(const IterParams* __restrict__ params, GroupPtrs ptrs, AdamHyper h, StepPlansDev plans,
              const uint32_t* __restrict__ last_visible, uint32_t n_cap, float2 reg, float2* __restrict__ lr_hist,
              const uint32_t* __restrict__ status) {
    adam_body<false>(params, ptrs, h, plans, last_visible, n_cap, reg, lr_hist, status);
}

// Dense mode with fused Adam in the backward: steps only Gaussians with
// last_visible[i] != t (g = 0).
extern "C" __global__ void adam_invisible(const IterParams* __restrict__ params, GroupPtrs ptrs, AdamHyper h, StepPlansDev plans,
                   const uint32_t* __restrict__ last_visible, uint32_t n_cap, const uint32_t* __restrict__ status) {
    adam_body<true>(params, ptrs, h, plans, last_visible, n_cap, make_float2(0.f, 0.f), nullptr, status);
}

// Applies j zero-gradient steps (group step indices s_u + 1 .. s_u + j) to one element.
// Stops summing once an update falls below half an ulp of p (a dense f32 step would not
// change p either) and decays the moments of the remaining steps in closed form.
SPL_INLINE void catchup_elem(const IterParams& P, const AdamHyper& h, const float2* __restrict__ hist, float* par,
                             MomentT* mom, uint32_t elem, uint32_t s_u, uint32_t j, uint32_t salt) {
    float p = par[elem], m, v;
    moment_load(mom[elem], m, v);
    uint32_t done = 0;
    const uint32_t lim = min(j, (uint32_t)CATCHUP_MAX);
    for (; done < lim; ++done) {
        float2 e = hist[(s_u + done + 1) % LR_HIST];
        m *= h.beta1;
        v *= h.beta2;
        float upd = e.x * m / (sqrtf(v) * e.y + h.eps);
        if (p != 0.f && fabsf(upd) < 5.9604645e-8f * fabsf(p)) {  // below 2^-24 |p|
            ++done;
            break;
        }
        p -= upd;
    }
    if (j > done) {
        float r = (float)(j - done);
        m *= __powf(h.beta1, r);
        v *= __powf(h.beta2, r);
    }
    par[elem] = p;
    uint4 rb = philox4x32_10(make_uint4(elem, P.frame.iteration, RNG_BF16_SR, salt), make_uint2(P.seed[0], P.seed[1]));
    mom[elem] = moment_pack(m, v, rb.x);
}

// LazyCatchUp (4.6): applies the effect of the zero-gradient steps a Gaussian skipped since
// its stamp last_update[plan][p] (one u32 per Gaussian per distinct step plan), with the
// per-step learning rates and bias corrections adam_step recorded in lr_hist.
// flush = 0: before adam_step, only for Gaussians that step at t (steps in (u, t)), stamp t.
// flush = 1: after adam_step, every Gaussian (steps in (u, t]), stamp t.
extern "C" __global__ void adam_catchup(const IterParams* __restrict__ params, GroupPtrs ptrs, AdamHyper h, StepPlansDev plans,
                 const uint32_t* __restrict__ last_visible, uint32_t* __restrict__ last_update,
                 const float2* __restrict__ lr_hist, uint32_t n_cap, uint32_t flush,
                 const uint32_t* __restrict__ status) {
    if (status[ST_OVERFLOW]) return;
    const IterParams& P = *params;
    const uint32_t t = P.frame.iteration, n = min(P.frame.n_gaussians, n_cap);
    for (uint32_t p = blockIdx.x * blockDim.x + threadIdx.x; p < n; p += gridDim.x * blockDim.x) {
        const uint32_t lv = last_visible[p];
        for (int k = 0; k < N_GROUPS; ++k) {
            int lead = -1;
            for (int g = 0; g < N_GROUPS; ++g)
                if (plans.plan_id[g] == (uint32_t)k) {
                    lead = g;
                    break;
                }
            if (lead < 0) continue;
            uint32_t* stamp = last_update + (size_t)k * n_cap + p;
            const uint32_t u = *stamp;
            uint32_t j;
            if (flush) {
                if (u >= t) continue;
                j = plan_due_count(plans, lead, u, t);
            } else {
                if (!plan_due(plans, lead, t) || !(lv > plan_prev_due(plans, lead, t))) continue;
                j = plan_due_count(plans, lead, u, t - 1);
            }
            if (j > 0) {
                const uint32_t s_u = plan_due_count(plans, lead, 0, u);
                for (int g = 0; g < N_GROUPS; ++g) {
                    if (plans.plan_id[g] != (uint32_t)k) continue;
                    const float2* hist = lr_hist + g * LR_HIST;
                    if (g == GROUP_SH_REST) {
                        for (uint32_t c = 0; c < 45; ++c)
                            catchup_elem(P, h, hist, ptrs.param[g], ptrs.moments[g], c * n_cap + p, s_u, j, 0x80u + g);
                    } else {
                        const uint32_t kf = g == GROUP_ROTATIONS ? 4 : g == GROUP_OPACITIES ? 1 : 3;
                        for (uint32_t c = 0; c < kf; ++c)
                            catchup_elem(P, h, hist, ptrs.param[g], ptrs.moments[g], p * kf + c, s_u, j, 0x80u + g);
                    }
                }
            }
            *stamp = t;
        }
    }
}
