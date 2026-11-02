// LayerNorm fused with FP8 quantisation, and the row epilogue for GEMM
// outputs (see src/norm.rs). Prefixed with common.cuh.

// Mirrors `LnArgs` in src/norm.rs (8-byte pointers first, then 4-byte fields).
struct LnArgs {
    unsigned long long x;       // residual stream [rows][C], f32 or bf16 (read, and written when y != 0)
    unsigned long long y;       // bf16 [rows][C] residual branch, or 0
    unsigned long long ytok;    // f32 [rows] per-token scale of y, or 0
    unsigned long long ychan;   // f32 [C] per-channel scale of y, or 0
    unsigned long long ybias;   // bf16 [C] bias of y, or 0
    unsigned long long ls;      // f32 [C] LayerScale, or 0
    unsigned long long gamma;   // f32 [C]
    unsigned long long beta;    // f32 [C]
    unsigned long long out;     // bf16 [rows][C] normalised output, or 0
    unsigned long long q;       // fp8 [rows][C] quantised output, or 0
    unsigned long long qscale;  // f32 [rows] per-token scales (written when q != 0 and static_scale == 0)
    unsigned long long static_scale; // f32 [1] fixed quantisation scale, or 0
    int rows;
    float eps;
};

__device__ __forceinline__ void load8f(const float* p, float* v) {
    float4 a = *(const float4*)p, b = *(const float4*)(p + 4);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

__device__ __forceinline__ void store8f(float* p, const float* v) {
    *(float4*)p = make_float4(v[0], v[1], v[2], v[3]);
    *(float4*)(p + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

// One warp per row; each lane owns NCH chunks of 8 channels (C = 256 * NCH),
// held in registers across the residual update, both LayerNorm passes and
// the quantisation.
template <bool XF32, int NCH>
__device__ __forceinline__ void layer_norm_rows(const LnArgs& a) {
    constexpr int C = NCH * 256;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= a.rows) return;
    const size_t base = (size_t)row * C;

    float v[NCH][8];
#pragma unroll
    for (int i = 0; i < NCH; ++i) {
        const int c0 = (i * 32 + lane) * 8;
        if (XF32) load8f((const float*)a.x + base + c0, v[i]);
        else unpack8(*(const uint4*)((const bf16*)a.x + base + c0), v[i]);
    }

    if (a.y) {
        const float tok = a.ytok ? ((const float*)a.ytok)[row] : 1.0f;
#pragma unroll
        for (int i = 0; i < NCH; ++i) {
            const int c0 = (i * 32 + lane) * 8;
            float y[8], f[8];
            unpack8(*(const uint4*)((const bf16*)a.y + base + c0), y);
#pragma unroll
            for (int j = 0; j < 8; ++j) y[j] *= tok;
            if (a.ychan) {
                load8f((const float*)a.ychan + c0, f);
#pragma unroll
                for (int j = 0; j < 8; ++j) y[j] *= f[j];
            }
            if (a.ybias) {
                unpack8(*(const uint4*)((const bf16*)a.ybias + c0), f);
#pragma unroll
                for (int j = 0; j < 8; ++j) y[j] += f[j];
            }
            if (a.ls) {
                load8f((const float*)a.ls + c0, f);
#pragma unroll
                for (int j = 0; j < 8; ++j) y[j] *= f[j];
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) v[i][j] += y[j];
            if (XF32) {
                store8f((float*)a.x + base + c0, v[i]);
            } else {
                uint4 p = pack8(v[i]);
                *(uint4*)((bf16*)a.x + base + c0) = p;
                unpack8(p, v[i]);  // normalise what the stream now holds
            }
        }
    }

    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < NCH; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) s += v[i][j];
    const float mean = warp_sum(s) * (1.0f / C);
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < NCH; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float d = v[i][j] - mean;
            ss += d * d;
        }
    const float rstd = rsqrtf(warp_sum(ss) * (1.0f / C) + a.eps);

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < NCH; ++i) {
        const int c0 = (i * 32 + lane) * 8;
        float g[8], b[8];
        load8f((const float*)a.gamma + c0, g);
        load8f((const float*)a.beta + c0, b);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[i][j] = (v[i][j] - mean) * rstd * g[j] + b[j];
            amax = fmaxf(amax, fabsf(v[i][j]));
        }
        if (a.out) *(uint4*)((bf16*)a.out + base + c0) = pack8(v[i]);
    }

    if (a.q) {
        float inv;
        if (a.static_scale) {
            inv = 1.0f / *(const float*)a.static_scale;
        } else {
            amax = warp_max(amax);
            inv = amax > 0.0f ? FP8_MAX / amax : 1.0f;
            if (lane == 0) ((float*)a.qscale)[row] = amax > 0.0f ? amax / FP8_MAX : 1.0f;
        }
#pragma unroll
        for (int i = 0; i < NCH; ++i) {
            const int c0 = (i * 32 + lane) * 8;
            float t[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) t[j] = v[i][j] * inv;
            *(uint2*)((fp8*)a.q + base + c0) = pack_fp8x8(t);
        }
    }
}

#define LN_KERNELS(NCH)                                                              \
    extern "C" __global__ void __launch_bounds__(128) ln_f32_c##NCH(LnArgs a) {      \
        layer_norm_rows<true, NCH>(a);                                               \
    }                                                                                \
    extern "C" __global__ void __launch_bounds__(128) ln_bf16_c##NCH(LnArgs a) {     \
        layer_norm_rows<false, NCH>(a);                                              \
    }

LN_KERNELS(1)
LN_KERNELS(2)
LN_KERNELS(3)
LN_KERNELS(4)
LN_KERNELS(6)
LN_KERNELS(8)

// Mirrors `RowArgs` in src/norm.rs.
struct RowArgs {
    unsigned long long y;       // bf16 [rows][ld] GEMM output
    unsigned long long ytok;    // f32 [rows] or 0
    unsigned long long ychan;   // f32 [C] or 0
    unsigned long long bias;    // bf16 [C] or 0
    unsigned long long out;     // bf16 [rows][C] or 0 (may alias y when ld == C)
    unsigned long long q;       // fp8 [rows][C] or 0
    unsigned long long qscale;  // f32 [rows]
    unsigned long long static_scale; // f32 [1] or 0
    int rows;
    int ld;
    int act;
    int pad;
};

// One 128-thread block per row; thread t owns NCH chunks of 8 channels
// (C = 1024 * NCH): y · tok · chan + bias → activation → bf16 and/or FP8.
template <int NCH>
__device__ __forceinline__ void row_epilogue(const RowArgs& a) {
    constexpr int C = NCH * 1024;
    __shared__ float red[4];
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const float tok = a.ytok ? ((const float*)a.ytok)[row] : 1.0f;
    float v[NCH][8];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < NCH; ++i) {
        const int c0 = (i * 128 + t) * 8;
        float f[8];
        unpack8(*(const uint4*)((const bf16*)a.y + (size_t)row * a.ld + c0), v[i]);
#pragma unroll
        for (int j = 0; j < 8; ++j) v[i][j] *= tok;
        if (a.ychan) {
            load8f((const float*)a.ychan + c0, f);
#pragma unroll
            for (int j = 0; j < 8; ++j) v[i][j] *= f[j];
        }
        if (a.bias) {
            unpack8(*(const uint4*)((const bf16*)a.bias + c0), f);
#pragma unroll
            for (int j = 0; j < 8; ++j) v[i][j] += f[j];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[i][j] = activate(v[i][j], a.act);
            amax = fmaxf(amax, fabsf(v[i][j]));
        }
    }
    // All reads of y happen before any write, so out may alias y.
    __syncthreads();
    if (a.out) {
#pragma unroll
        for (int i = 0; i < NCH; ++i)
            *(uint4*)((bf16*)a.out + (size_t)row * C + (i * 128 + t) * 8) = pack8(v[i]);
    }
    if (!a.q) return;
    float inv;
    if (a.static_scale) {
        inv = 1.0f / *(const float*)a.static_scale;
    } else {
        amax = warp_max(amax);
        if ((t & 31) == 0) red[t >> 5] = amax;
        __syncthreads();
        amax = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
        inv = amax > 0.0f ? FP8_MAX / amax : 1.0f;
        if (t == 0) ((float*)a.qscale)[row] = amax > 0.0f ? amax / FP8_MAX : 1.0f;
    }
#pragma unroll
    for (int i = 0; i < NCH; ++i) {
        float s[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j] = v[i][j] * inv;
        *(uint2*)((fp8*)a.q + (size_t)row * C + (i * 128 + t) * 8) = pack_fp8x8(s);
    }
}

extern "C" __global__ void __launch_bounds__(128) row_c1(RowArgs a) { row_epilogue<1>(a); }
extern "C" __global__ void __launch_bounds__(128) row_c2(RowArgs a) { row_epilogue<2>(a); }
extern "C" __global__ void __launch_bounds__(128) row_c3(RowArgs a) { row_epilogue<3>(a); }
extern "C" __global__ void __launch_bounds__(128) row_c4(RowArgs a) { row_epilogue<4>(a); }
