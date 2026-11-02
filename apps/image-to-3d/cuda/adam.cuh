// Device Adam helpers, spliced into adam.cu and (Phase 3b, fused Adam) into
// preprocess.cu. Owner: optimizer (E5). Spec: spec-train Sections 4.1, 4.4, 4.8.
// Update form after Faster-GS adam.cu (Apache-2.0); semantics are PyTorch Adam's.
//
// Moments are float2 (m, v) at the same element index as the parameter, so "moment of
// element e of group g" uses the parameter's index math (planar SH included).
// ADAM_MOMENTS_BF16 stores them as two bf16 in a u32 (stochastic rounding).
//
// STUBS: the helpers below have their final signatures and empty bodies.

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

#if ADAM_MOMENTS_BF16
typedef uint32_t MomentT;
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

// m <- b1 m + (1 - b1) g; v <- b2 v + (1 - b2) g^2; p <- p - step_size m / (sqrt(v) bc2 + eps).
SPL_INLINE void adam_apply(float g, float* p, MomentT* mv, uint32_t elem, const AdamGroupStep& s) {}

// grad[elem] += g (Accumulate) or = g (Overwrite), per the group's write mode.
SPL_INLINE void grad_accumulate(float g, float* grad, uint32_t elem) {}
