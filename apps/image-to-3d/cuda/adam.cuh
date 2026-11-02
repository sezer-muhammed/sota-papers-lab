// Device Adam helpers, spliced into adam.cu and (Phase 3b, fused Adam) into
// preprocess.cu. Owner: optimizer (E5). Spec: spec-train Sections 4.1, 4.4, 4.8.
// Update form after Faster-GS adam.cu (Apache-2.0, 4f97d14: m and v interleaved as
// float2, fma form of the moment updates); semantics are PyTorch Adam's.
//
// Moments are float2 (m, v) at the same element index as the parameter, so "moment of
// element e of group g" uses the parameter's index math (planar SH included).
// ADAM_MOMENTS_BF16 = 1 stores them as two bf16 in a u32 (m low, v high), = 2 as
// uint2 {m bf16 in the low half of x, v f32 bits in y} (accuracy ablation, no saving).
// bf16 stores use stochastic rounding (spec-train 4.7).

#ifndef ADAM_MOMENTS_BF16
#define ADAM_MOMENTS_BF16 0
#endif

#define N_GROUPS 6
#define GROUP_MEANS 0
#define GROUP_SCALES 1
#define GROUP_ROTATIONS 2
#define GROUP_OPACITIES 3
#define GROUP_SH0 4
#define GROUP_SH_REST 5

#define WRITE_ACCUMULATE 0
#define WRITE_OVERWRITE 1

// AdamIterPlan.overwrite_mask: bit g = Overwrite write mode; bit (ZERO_AFTER_SHIFT + g) =
// zero the group's gradient buffer after stepping (the next iteration accumulates).
#define ZERO_AFTER_SHIFT 8

// Sparsity modes (kernels::adam::Sparsity).
#define SPARSE_DENSE 0
#define SPARSE_VISIBLE 1
#define SPARSE_LAZY 2

// Piecewise step plans: at most this many (from, every) segments per group.
#define PLAN_SEGS 4
// LazyCatchUp: per-step (step_size, bc2) ring entries per group, and the summed terms cap.
#define LR_HIST 8192
#define CATCHUP_MAX 256

#if ADAM_MOMENTS_BF16 == 1
typedef uint32_t MomentT;
#elif ADAM_MOMENTS_BF16 == 2
typedef uint2 MomentT;
#else
typedef float2 MomentT;
#endif

struct AdamGroupStep {
    float step_size;
    float bc2_sqrt_rcp;
    uint32_t due;
    uint32_t write_mode;
};

// Device pointers of every group, passed by value (Rust mirror: kernels::adam::GroupPtrs).
struct GroupPtrs {                   // 144 B
    float* param[N_GROUPS];
    float* grad[N_GROUPS];
    MomentT* moments[N_GROUPS];
};

// Fixed per run (kernel argument; Rust mirror kernels::adam::AdamHyper). 24 B.
// one_minus_beta*: 1 - beta rounded once from f64 on the host (PyTorch's `value=1-beta`).
struct AdamHyper {
    float beta1, beta2, eps;
    uint32_t sparsity;
    float one_minus_beta1, one_minus_beta2;
};

// The step plan of every group (Rust mirror kernels::adam::StepPlansDev). 240 B.
// Segment i of group g covers iterations [from[g][i], from[g][i+1]) and steps when
// t % every == 0. plan_id groups sharing a plan (one LazyCatchUp stamp array each).
struct StepPlansDev {
    uint32_t from[N_GROUPS][PLAN_SEGS];
    uint32_t every[N_GROUPS][PLAN_SEGS];
    uint32_t n_seg[N_GROUPS];
    uint32_t plan_id[N_GROUPS];
};

SPL_INLINE uint32_t plan_every(const StepPlansDev& pl, int g, uint32_t t) {
    uint32_t e = 1;
    for (uint32_t i = 0; i < pl.n_seg[g]; ++i)
        if (pl.from[g][i] <= t) e = pl.every[g][i];
    return e;
}

SPL_INLINE bool plan_due(const StepPlansDev& pl, int g, uint32_t t) { return t % plan_every(pl, g, t) == 0; }

// Number of due iterations in (a, b] (0 when b <= a).
SPL_INLINE uint32_t plan_due_count(const StepPlansDev& pl, int g, uint32_t a, uint32_t b) {
    if (b <= a) return 0;
    uint32_t ns = pl.n_seg[g];
    if (ns == 0) return b - a;
    uint32_t count = 0;
    // Before the first segment the plan is "every 1".
    uint32_t lo = a, first = pl.from[g][0];
    if (first > 0 && lo + 1 < first) {
        uint32_t hi = min(b, first - 1);
        if (hi > lo) count += hi - lo;
    }
    for (uint32_t i = 0; i < ns; ++i) {
        uint32_t s0 = pl.from[g][i];
        uint32_t s1 = i + 1 < ns ? pl.from[g][i + 1] : 0xffffffffu;  // exclusive end
        uint32_t l = max(a, s0 == 0 ? 0u : s0 - 1);
        uint32_t h = min(b, s1 - 1);
        if (h > l) {
            uint32_t e = max(pl.every[g][i], 1u);
            count += h / e - l / e;
        }
    }
    return count;
}

// Largest due iteration t' < t, 0 if none (iterations start at 1).
SPL_INLINE uint32_t plan_prev_due(const StepPlansDev& pl, int g, uint32_t t) {
    uint32_t x = t - 1;  // candidate upper bound (t >= 1)
    while (x > 0) {
        uint32_t e = 1, s0 = 0;
        for (uint32_t i = 0; i < pl.n_seg[g]; ++i)
            if (pl.from[g][i] <= x) {
                e = max(pl.every[g][i], 1u);
                s0 = pl.from[g][i];
            }
        uint32_t c = (x / e) * e;
        if (c >= s0 || s0 == 0) return c;
        x = s0 - 1;
    }
    return 0;
}

// ---------------------------------------------------------------- moments storage

SPL_INLINE void moment_load(const MomentT& w, float& m, float& v) {
#if ADAM_MOMENTS_BF16 == 1
    m = bf16_to_f32((uint16_t)(w & 0xffffu));
    v = bf16_to_f32((uint16_t)(w >> 16));
#elif ADAM_MOMENTS_BF16 == 2
    m = bf16_to_f32((uint16_t)(w.x & 0xffffu));
    v = __uint_as_float(w.y);
#else
    m = w.x;
    v = w.y;
#endif
}

// `rand`: 32 random bits (16 for m, 16 for v); unused in f32 mode.
SPL_INLINE MomentT moment_pack(float m, float v, uint32_t rand) {
#if ADAM_MOMENTS_BF16 == 1
    return (uint32_t)f32_to_bf16_sr(m, rand) | ((uint32_t)f32_to_bf16_sr(v, rand >> 16) << 16);
#elif ADAM_MOMENTS_BF16 == 2
    return make_uint2((uint32_t)f32_to_bf16_sr(m, rand), __float_as_uint(v));
#else
    (void)rand;
    return make_float2(m, v);
#endif
}

// One Adam step of one element in registers, in PyTorch's order (m*b1 + (1-b1) g,
// v*b2 + (1-b2) g^2). Faster-GS's fma(b2, v - g^2, g^2) form saves a constant but
// cancels when v << g^2 (early steps, rarely visible Gaussians): 2.5e-5 relative error
// in v against f64 after 100 steps, versus 3e-7 for this form.
SPL_INLINE void adam_update(float g, float& p, float& m, float& v, float step_size, float bc2, float b1, float b2,
                            float omb1, float omb2, float eps) {
    m = fmaf(omb1, g, b1 * m);
    v = fmaf(omb2, g * g, b2 * v);
    p -= step_size * m / (sqrtf(v) * bc2 + eps);
}

// m <- b1 m + (1 - b1) g; v <- b2 v + (1 - b2) g^2; p <- p - step_size m / (sqrt(v) bc2 + eps).
// Fused-Adam entry point (betas 0.9 / 0.999, eps 1e-15: the only values the presets use;
// stochastic-rounding bits from the element index).
SPL_INLINE void adam_apply(float g, float* p, MomentT* mv, uint32_t elem, const AdamGroupStep& s) {
    if (!s.due) return;
    float m, v, pp = p[elem];
    moment_load(mv[elem], m, v);
    adam_update(g, pp, m, v, s.step_size, s.bc2_sqrt_rcp, 0.9f, 0.999f, 0.1f, 0.001f, 1e-15f);
    p[elem] = pp;
    mv[elem] = moment_pack(m, v, elem * 0x9E3779B9u ^ __float_as_uint(g));
}

// grad[elem] += g (Accumulate) or = g (Overwrite), per the group's write mode.
SPL_INLINE void grad_accumulate(float g, float* grad, uint32_t elem, uint32_t write_mode) {
    if (write_mode == WRITE_OVERWRITE)
        grad[elem] = g;
    else
        grad[elem] += g;
}
