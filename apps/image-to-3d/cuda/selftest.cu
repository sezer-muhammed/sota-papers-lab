// Self-test kernels for the shared headers: struct sizes and offsets as the device sees
// them, Philox, f16/bf16 conversions and stochastic rounding. Owner: preprocess (E1).
// Used only by src/kernels/selftest.rs (GPU tests). Implemented, not stubs.

// NVRTC has no offsetof or __builtin_offsetof: offsets are address differences
// inside one local instance per struct (folded to constants by the compiler).
#define OFF(T, f) ((uint32_t)((const char*)&probe_##T.f - (const char*)&probe_##T))

// out[]: sizeof and selected offsets, in the order of layout_fields() in
// src/kernels/selftest.rs.
extern "C" __global__ void layout_probe(uint32_t* __restrict__ out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    CamBase probe_CamBase;
    FrameParams probe_FrameParams;
    FrameCam probe_FrameCam;
    AdamIterPlan probe_AdamIterPlan;
    IterParams probe_IterParams;
    SplatRec probe_SplatRec;
    uint32_t v[] = {
        (uint32_t)sizeof(CamBase),      OFF(CamBase, intr_id),       OFF(CamBase, width),
        (uint32_t)sizeof(Intr),         (uint32_t)sizeof(FrameParams), OFF(FrameParams, bg),
        OFF(FrameParams, eps2d),        OFF(FrameParams, flags),     OFF(FrameParams, adam_lr_scale),
        OFF(FrameParams, render_w),     (uint32_t)sizeof(FrameCam),  OFF(FrameCam, fx),
        OFF(FrameCam, campos),          OFF(FrameCam, near_z),       OFF(FrameCam, lim_x0),
        OFF(FrameCam, width),           OFF(FrameCam, image_index),  OFF(FrameCam, intr_id),
        (uint32_t)sizeof(AdamIterPlan), OFF(AdamIterPlan, bc2_sqrt_rcp), OFF(AdamIterPlan, due_mask),
        OFF(AdamIterPlan, elem_prefix), (uint32_t)sizeof(IterParams), OFF(IterParams, target_offset),
        OFF(IterParams, target_w),      OFF(IterParams, res_scale),  OFF(IterParams, means_lr),
        OFF(IterParams, noise_scale),   OFF(IterParams, seed),       OFF(IterParams, adam),
        (uint32_t)sizeof(SplatRec),     OFF(SplatRec, ca),           OFF(SplatRec, o),
        OFF(SplatRec, r),               OFF(SplatRec, pthr),         OFF(SplatRec, z),
        OFF(SplatRec, flags),
    };
    for (uint32_t i = 0; i < sizeof(v) / sizeof(v[0]); ++i) out[i] = v[i];
}

// out[i] = philox4x32_10(ctr[i], key[i]).
extern "C" __global__ void philox_probe(const uint4* __restrict__ ctr, const uint2* __restrict__ key, uint32_t n,
                                        uint4* __restrict__ out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = philox4x32_10(ctr[i], key[i]);
}

// Conversions of x[i]: f16 bits, bf16 round-to-nearest bits, bf16 stochastic-rounded bits
// with the given random word, and the f16 value converted back.
extern "C" __global__ void half_probe(const float* __restrict__ x, const uint32_t* __restrict__ rand, uint32_t n,
                                      uint16_t* __restrict__ f16, uint16_t* __restrict__ bf16_rn,
                                      uint16_t* __restrict__ bf16_sr, float* __restrict__ f16_back) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    f16[i] = f32_to_f16(x[i]);
    bf16_rn[i] = f32_to_bf16_rn(x[i]);
    bf16_sr[i] = f32_to_bf16_sr(x[i], rand[i]);
    f16_back[i] = f16_to_f32(f16[i]);
}

// Warp helpers and memory-ordering PTX: lane id, lanemask_lt, an acquire/release
// round trip, and a cp.async copy of 16 bytes per thread through shared memory.
extern "C" __global__ void ptx_probe(const float4* __restrict__ in, float4* __restrict__ out,
                                     uint32_t* __restrict__ lanes, uint32_t* __restrict__ flag) {
    __shared__ float4 buf[32];
    uint32_t t = threadIdx.x;
    lanes[t] = lane_id() | (__popc(lanemask_lt()) << 8);
    cp_async16(&buf[t], &in[t]);
    cp_async_commit();
    cp_async_wait<0>();
    __syncwarp();
    out[t] = buf[t];
    if (t == 0) {
        st_release_gpu(flag, 7u);
        flag[1] = ld_acquire_gpu(flag) + ld_volatile(flag);
    }
}
