// Densification decisions: VCD masks and view scores, clone/split/prune flags, VCP
// sampling keys, opacity clamp and reset. Owner: densify (E6).
// Spec: spec-train Sections 5.1-5.4, 5.7. FastGS is read-only (Inria-derived): this is a
// reimplementation from the spec.
//
// Thresholds are compared in parameter space (logit opacity, log scale) and averages use
// IEEE division (__fdiv_rn), so the CPU reference (kernels::densify::plan_cpu) makes the
// same decisions bit for bit under --use_fast_math.

#define NONE 0xffffffffu

// densify_plan modes.
#define PLAN_ADC 0       // 3DGS / Faster-GS ADC
#define PLAN_FASTGS 1    // FastGS densify step (prune candidates -> out_count 2, resolved by VCP)
#define PLAN_FINAL 2     // FastGS final VCP prune (no emission)

struct DensifyRules {          // 64 B, mirrors kernels::densify::DensifyRules
    uint32_t mode;
    float grad_thresh;         // average NDC gradient (clone; ADC split too)
    float grad_abs_thresh;     // average AbsGS gradient (FastGS split)
    float clone_log_scale;     // ln(dense * r): clone iff max log scale <= this
    float big_log_scale;       // ln(0.1 r) when the world-size prune is on, else +inf
    float min_opacity_logit;   // logit(min_opacity)
    float views;               // K
    float tau;                 // candidate iff floor(count / K) > tau
    float final_opacity_logit; // logit(0.1)
    float final_score;         // 0.9
    uint32_t pad[6];
};

// Order-preserving float <-> u32.
SPL_INLINE uint32_t f2key(float f) {
    uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
SPL_INLINE float key2f(uint32_t k) { return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k); }

// mask[p] = (err[p] - min) / (max - min) > tau (u8), err and minmax from loss.cu error_map.
extern "C" __global__ void vcd_mask(const float* __restrict__ err, const float* __restrict__ minmax, float tau,
                                    uint32_t n_pixels, uint8_t* __restrict__ mask) {
    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;
    float lo = minmax[0], range = minmax[1] - lo;
    // (e - lo) / range > tau, without a division (range > 0); an empty range masks nothing.
    mask[p] = range > 0.f && __fdiv_rn(err[p] - lo, range) > tau;
}

// After a count-mode raster pass: count_p[vis_idx[v]] += count_v[v],
// score_p[vis_idx[v]] += view_weight * count_v[v]; zeroes count_v. vis_idx is injective,
// so no atomics.
extern "C" __global__ void scores_add(const uint32_t* __restrict__ count_v, const uint32_t* __restrict__ vis_idx,
                                      const uint32_t* __restrict__ status, float view_weight,
                                      uint32_t* __restrict__ count_p, float* __restrict__ score_p) {
    if (status[ST_OVERFLOW]) return;
    uint32_t n_vis = status[ST_N_VIS];
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n_vis; v += gridDim.x * blockDim.x) {
        uint32_t c = count_v[v];
        if (c) {
            uint32_t p = vis_idx[v];
            count_p[p] += c;
            score_p[p] += view_weight * (float)c;
            ((uint32_t*)count_v)[v] = 0;
        }
    }
}

// out[0] = min key, out[1] = max key of x[0..n) (init 0xffffffff, 0).
extern "C" __global__ void minmax_f32(const float* __restrict__ x, uint32_t n, uint32_t* __restrict__ out) {
    uint32_t lo = NONE, hi = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        uint32_t k = f2key(x[i]);
        lo = min(lo, k), hi = max(hi, k);
    }
    for (int o = 16; o > 0; o >>= 1) {
        lo = min(lo, __shfl_xor_sync(FULL_MASK, lo, o));
        hi = max(hi, __shfl_xor_sync(FULL_MASK, hi, o));
    }
    if ((threadIdx.x & 31) == 0) atomicMin(&out[0], lo), atomicMax(&out[1], hi);
}

// (score - min) / (max - min); 0 when the range is empty (PyTorch would give NaN: no
// Gaussian passes > 0.9 and every VCP weight is equal, which 0 reproduces).
SPL_INLINE float prune_score(float s, const uint32_t* mm) {
    float lo = key2f(mm[0]), hi = key2f(mm[1]);
    return hi > lo ? __fdiv_rn(s - lo, hi - lo) : 0.f;
}

SPL_INLINE float avg(float sum, float cnt) { return cnt > 0.f ? __fdiv_rn(sum, cnt) : 0.f; }

// Per Gaussian: clone / split / prune flags (all decisions on the pre-step indexing).
// out_count: 0 removed (pruned or split parent), 1 kept, 2 FastGS prune candidate (kept
// unless VCP draws it). priority: the statistic that ranks tail entries when the capacity
// is full. counters[0] += prune candidates.
extern "C" __global__ void densify_plan(const float* __restrict__ log_scale, const float4* __restrict__ quat,
                                        const float* __restrict__ opacity, const float* __restrict__ vis_count,
                                        const float* __restrict__ grad_sum, const float* __restrict__ grad_abs_sum,
                                        const uint32_t* __restrict__ count_p, const float* __restrict__ score_p,
                                        const uint32_t* __restrict__ score_minmax, DensifyRules r, uint32_t n,
                                        uint32_t* __restrict__ out_count, uint8_t* __restrict__ clone_flag,
                                        uint8_t* __restrict__ split_flag, float* __restrict__ priority,
                                        uint32_t* __restrict__ counters) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t cand = 0;
    if (i < n) {
        float ls = fmaxf(fmaxf(log_scale[3 * i], log_scale[3 * i + 1]), log_scale[3 * i + 2]);
        float o = opacity[i];
        uint32_t oc = 1, cl = 0, sp = 0;
        float pri = 0.f;
        if (r.mode == PLAN_FINAL) {
            oc = !(o < r.final_opacity_logit || prune_score(score_p[i], score_minmax) > r.final_score);
        } else {
            float cnt = vis_count[i];
            float g = avg(grad_sum[i], cnt);
            bool small = ls <= r.clone_log_scale;
            if (r.mode == PLAN_ADC) {
                float4 q = quat[i];
                bool dead = o < r.min_opacity_logit || q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w < 1e-8f;
                bool hot = g >= r.grad_thresh;
                cl = hot && small;
                sp = hot && !small;
                // Prune on the post-step set: a clone has its parent's values, split
                // children have the parent's scale / 1.6.
                bool kill = dead || ls > r.big_log_scale;
                bool kill_child = dead || ls - 0.47000363f > r.big_log_scale;
                oc = !sp && !kill;
                cl = cl && !kill;
                sp = sp && !kill_child;
                pri = g;
            } else {
                float imp = floorf(__fdiv_rn((float)count_p[i], r.views));
                bool c = imp > r.tau;
                float ga = grad_abs_sum ? avg(grad_abs_sum[i], cnt) : 0.f;
                cl = c && small && g >= r.grad_thresh;
                sp = c && !small && ga >= r.grad_abs_thresh;
                bool p = !sp && (o < r.min_opacity_logit || ls > r.big_log_scale);
                oc = sp ? 0 : (p ? 2 : 1);
                cand = p;
                pri = sp ? ga : g;
            }
        }
        out_count[i] = oc;
        clone_flag[i] = cl;
        split_flag[i] = sp;
        priority[i] = pri;
    }
    uint32_t b = __ballot_sync(FULL_MASK, cand);
    if ((threadIdx.x & 31) == 0 && b) atomicAdd(&counters[0], __popc(b));
}

// Gumbel-top-k keys for in-densify VCP: key = ln w - ln(-ln u), u from Philox
// (i, iteration, RNG_VCP_GUMBEL), w = 1 / (1e-6 + 1 - prune_score), stored so that an
// ascending sort yields descending keys (key ties keep index order).
extern "C" __global__ void gumbel_keys(const float* __restrict__ score_p, const uint32_t* __restrict__ minmax,
                                       uint2 seed, uint32_t iteration, uint32_t n, uint32_t* __restrict__ keys,
                                       uint32_t* __restrict__ idx) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float ps = prune_score(score_p[i], minmax);
    float u = u01_open(rng4(seed, i, iteration, RNG_VCP_GUMBEL).x);
    float key = -logf(1e-6f + 1.f - ps) - logf(-logf(u));
    keys[i] = ~f2key(key);
    idx[i] = i;
}

// The k drawn (first k of the sorted order): candidates (2) become pruned (0).
extern "C" __global__ void vcp_mark(const uint32_t* __restrict__ sorted_idx, uint32_t k,
                                    uint32_t* __restrict__ out_count) {
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= k) return;
    uint32_t p = sorted_idx[r];
    if (out_count[p] == 2) out_count[p] = 0;
}

// opacity <- min(opacity, logit_max) for all Gaussians and zero the opacity moments
// (`words` u32 per element: 2 for f32 moments, 1 for bf16; 0 skips).
extern "C" __global__ void opacity_clamp(float* __restrict__ opacity, uint32_t* __restrict__ moments, uint32_t words,
                                         float logit_max, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    opacity[i] = fminf(opacity[i], logit_max);
    for (uint32_t w = 0; w < words; w++) moments[(uint64_t)i * words + w] = 0;
}
