// Generic device helpers shared by every image-to-3d kernel. Owner: preprocess (E1);
// other owners request additions instead of editing.
//
// NVRTC rule (spec-raster 1.6): no #include anywhere. The runtime container has no
// CUDA headers, so this file defines the integer types and wraps everything that
// would normally come from cuda_fp16.h / cuda_bf16.h / cuda/atomic in inline PTX.
// Sources are concatenated by src/gpu.rs: common.cuh + splat_common.cuh + <module>.cu,
// and compiled with --gpu-architecture=compute_<cc> --use_fast_math -std=c++17.
//
// Implemented and tested here (src/kernels/selftest): fixed-width integer types,
// PTX acquire/release loads and stores, cp.async, f16 and bf16 conversions,
// bf16 stochastic rounding, lane masks, Philox-4x32-10 and its uniform/normal draws.
// Every #define below mirrors a constant in src/kernels/layout.rs (unit test).

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;

#define SPL_INLINE __device__ __forceinline__
#define FULL_MASK 0xffffffffu

// RNG purpose ids: the Philox counter is (element, iteration, purpose, 0) and the key is
// the run seed, so any kernel's draws are reproducible without RNG state (spec-train 1).
#define RNG_SPLIT 1
#define RNG_MCMC_NOISE 2
#define RNG_MCMC_DRAW 3
#define RNG_VCP_GUMBEL 4
#define RNG_BF16_SR 5
#define RNG_RANDOM_BG 6
#define RNG_VIEW_SAMPLE 7

// ---------------------------------------------------------------- lanes and warps

SPL_INLINE uint32_t lane_id() {
    uint32_t l;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(l));
    return l;
}

SPL_INLINE uint32_t lanemask_lt() {
    uint32_t m;
    asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
    return m;
}

SPL_INLINE float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);
    return v;
}

SPL_INLINE float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, o));
    return v;
}

SPL_INLINE float warp_min(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fminf(v, __shfl_xor_sync(FULL_MASK, v, o));
    return v;
}

// ---------------------------------------------------------------- memory ordering (decoupled look-back)

SPL_INLINE uint32_t ld_acquire_gpu(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

SPL_INLINE void st_release_gpu(uint32_t* p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

SPL_INLINE uint32_t ld_volatile(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// ---------------------------------------------------------------- cp.async (sm_80+)

SPL_INLINE uint32_t smem_u32(const void* p) {
    uint64_t a;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(a) : "l"(p));
    return (uint32_t)a;
}

// 16-byte global -> shared copy that bypasses L1 (.cg). Both addresses 16 B aligned.
SPL_INLINE void cp_async16(void* smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(smem)), "l"(gmem) : "memory");
}

SPL_INLINE void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N>
SPL_INLINE void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// ---------------------------------------------------------------- half and bfloat16 as raw bits

SPL_INLINE uint16_t f32_to_f16(float x) {
    uint16_t h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return h;
}

SPL_INLINE float f16_to_f32(uint16_t h) {
    float x;
    asm("cvt.f32.f16 %0, %1;" : "=f"(x) : "h"(h));
    return x;
}

SPL_INLINE uint16_t f32_to_bf16_rn(float x) {
    uint16_t h;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return h;
}

SPL_INLINE float bf16_to_f32(uint16_t h) { return __uint_as_float(((uint32_t)h) << 16); }

// Stochastic rounding to bf16 (spec-train 4.7): add 16 random bits to the low half of
// the f32 pattern, then truncate. Sign-magnitude makes it symmetric. NaN and Inf pass
// through unchanged (a carry would otherwise turn the largest finite value into Inf,
// which is the correctly rounded-up result, so only non-finite inputs are special).
SPL_INLINE uint16_t f32_to_bf16_sr(float x, uint32_t rand) {
    uint32_t b = __float_as_uint(x);
    if ((b & 0x7f800000u) == 0x7f800000u) return (uint16_t)(b >> 16);
    return (uint16_t)((b + (rand & 0xffffu)) >> 16);
}

// ---------------------------------------------------------------- Philox-4x32-10 (Salmon et al. 2011)

SPL_INLINE uint4 philox4x32_10(uint4 c, uint2 k) {
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        if (r) {
            k.x += 0x9E3779B9u;
            k.y += 0xBB67AE85u;
        }
        uint32_t lo0 = 0xD2511F53u * c.x, hi0 = __umulhi(0xD2511F53u, c.x);
        uint32_t lo1 = 0xCD9E8D57u * c.z, hi1 = __umulhi(0xCD9E8D57u, c.z);
        c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    }
    return c;
}

// Four independent u32 draws for (element, iteration, purpose) under the run seed.
SPL_INLINE uint4 rng4(uint2 seed, uint32_t elem, uint32_t iteration, uint32_t purpose) {
    return philox4x32_10(make_uint4(elem, iteration, purpose, 0u), seed);
}

// Uniform in the open interval (0, 1): safe for log() (Gumbel keys, Box-Muller).
// 23 bits, so k + 0.5 is exact in f32 and the largest value stays below 1.
SPL_INLINE float u01_open(uint32_t x) { return ((float)(x >> 9) + 0.5f) * (1.0f / 8388608.0f); }

// Two standard normals from two u32 draws (Box-Muller).
SPL_INLINE float2 normal2(uint32_t a, uint32_t b) {
    float r = sqrtf(-2.0f * __logf(u01_open(a)));
    float s, c;
    __sincosf(6.2831853071795864f * u01_open(b), &s, &c);
    return make_float2(r * c, r * s);
}

// ---------------------------------------------------------------- small float3 helpers

SPL_INLINE float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
SPL_INLINE float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
SPL_INLINE float3 operator*(float s, float3 a) { return make_float3(s * a.x, s * a.y, s * a.z); }
SPL_INLINE float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
SPL_INLINE float3 cross3(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
