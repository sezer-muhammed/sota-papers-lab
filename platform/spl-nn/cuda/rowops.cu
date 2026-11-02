// Small row ops (see src/rowops.rs). Prefixed with common.cuh.

// In-place activation of a bf16 buffer, 8 elements per thread.
extern "C" __global__ void activate_bf16(unsigned long long ptr, long long chunks, int act) {
    uint4* p = (uint4*)ptr;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < chunks;
         t += (long long)gridDim.x * blockDim.x) {
        float f[8];
        unpack8(p[t], f);
#pragma unroll
        for (int e = 0; e < 8; ++e) f[e] = activate(f[e], act);
        p[t] = pack8(f);
    }
}

// Attention for head dim 128 on short sequences (SIMT, fp32 math), reading
// Q, K, V straight out of a packed QKV GEMM output:
// q of token t, head h = qkv[t * ld + q_off + h * 128 .. + 128] (same for k, v).
// One CTA = 16 queries of one head, 4 warps × 4 queries. Each key tile of 32
// keys sits in shared memory; lane j scores key j for the warp's 4 queries
// (rows padded to 130 elements: conflict-free), online softmax in the exp2
// domain, then lane l accumulates output dims 4l..4l+3 over the 32 keys with
// the probabilities broadcast by shuffles.
#define AD 128
#define AQ 16
#define AK 32
#define KPAD (AD + 2)

struct Attn128Args {
    unsigned long long qkv, out;
    int ld, q_off, k_off, v_off;
    int seq, out_ld;
    float scale_log2;
};

extern "C" __global__ void __launch_bounds__(128) attn_d128(const Attn128Args a) {
    __shared__ float Qs[AQ][AD];
    __shared__ bf16 Ks[AK][KPAD];
    __shared__ __align__(16) bf16 Vs[AK][AD];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head = blockIdx.y, q0 = blockIdx.x * AQ;
    const bf16* qkv = (const bf16*)a.qkv;

    for (int i = tid; i < AQ * AD; i += 128) {
        const int r = i / AD, d = i % AD, t = q0 + r;
        Qs[r][d] = t < a.seq ? bf16_f32(qkv[(size_t)t * a.ld + a.q_off + head * AD + d]) * a.scale_log2 : 0.0f;
    }
    float m[4], l[4], o[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; ++e) o[i][e] = 0.0f;
    }

    for (int k0 = 0; k0 < a.seq; k0 += AK) {
        __syncthreads();
        for (int i = tid; i < AK * AD / 2; i += 128) {
            const int r = i / (AD / 2), d = (i % (AD / 2)) * 2, t = k0 + r;
            unsigned kv = 0, vv = 0;
            if (t < a.seq) {
                kv = *(const unsigned*)(qkv + (size_t)t * a.ld + a.k_off + head * AD + d);
                vv = *(const unsigned*)(qkv + (size_t)t * a.ld + a.v_off + head * AD + d);
            }
            *(unsigned*)&Ks[r][d] = kv;
            *(unsigned*)&Vs[r][d] = vv;
        }
        __syncthreads();
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const unsigned* krow = (const unsigned*)&Ks[lane][0];
#pragma unroll 8
        for (int d2 = 0; d2 < AD / 2; ++d2) {
            const unsigned kk = krow[d2];
            const float k_lo = bf16_lo(kk), k_hi = bf16_hi(kk);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float* q = Qs[warp * 4 + i];
                s[i] += q[2 * d2] * k_lo + q[2 * d2 + 1] * k_hi;
            }
        }
        const bool valid = k0 + lane < a.seq;
        float p[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float si = valid ? s[i] : -INFINITY;
            const float mn = fmaxf(m[i], warp_max(si));
            const float corr = exp2f(m[i] - mn);
            p[i] = valid ? exp2f(si - mn) : 0.0f;
            l[i] = l[i] * corr + warp_sum(p[i]);
            m[i] = mn;
#pragma unroll
            for (int e = 0; e < 4; ++e) o[i][e] *= corr;
        }
#pragma unroll 4
        for (int j = 0; j < AK; ++j) {
            const uint2 vv = *(const uint2*)&Vs[j][lane * 4];
            const float v0 = bf16_lo(vv.x), v1 = bf16_hi(vv.x), v2 = bf16_lo(vv.y), v3 = bf16_hi(vv.y);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float pj = __shfl_sync(0xffffffffu, p[i], j);
                o[i][0] += pj * v0;
                o[i][1] += pj * v1;
                o[i][2] += pj * v2;
                o[i][3] += pj * v3;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int t = q0 + warp * 4 + i;
        if (t >= a.seq) continue;
        const float inv = 1.0f / l[i];
        uint2 r;
        r.x = pack_bf16(o[i][0] * inv, o[i][1] * inv);
        r.y = pack_bf16(o[i][2] * inv, o[i][3] * inv);
        *(uint2*)((bf16*)a.out + (size_t)t * a.out_ld + head * AD + lane * 4) = r;
    }
}
