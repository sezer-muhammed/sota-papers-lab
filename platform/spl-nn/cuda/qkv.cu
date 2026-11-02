// QKV prologue (see src/qkv.rs): split the fused QKV projection, apply
// optional dequantisation scales and bias, QK-norm and 2D RoPE, and write
// Q, K and V head-major for the attention kernels. Prefixed with common.cuh.
//
// Head dim is 64. Eight lanes own one (token, part, head) row, 8 channels
// each, so loads and stores are 16 bytes and a row is 128 contiguous bytes.

// Mirrors `QkvArgs` in src/qkv.rs.
struct QkvArgs {
    unsigned long long y;      // bf16 [tokens][ld]: q | k | v, each heads*64 wide
    unsigned long long ytok;   // f32 [tokens] or 0
    unsigned long long ychan;  // f32 [3*heads*64] or 0
    unsigned long long bias;   // bf16 [3*heads*64] or 0
    unsigned long long qn_w;   // f32 [64] QK-norm weight for q
    unsigned long long qn_b;   // f32 [64] QK-norm bias for q (LayerNorm only)
    unsigned long long kn_w;   // f32 [64]
    unsigned long long kn_b;   // f32 [64]
    unsigned long long rope;   // float2 [frame_len][32] (cos, sin) per rotation pair, or 0
    unsigned long long q;      // bf16 [heads][tokens][64]
    unsigned long long k;
    unsigned long long v;
    long long tokens;
    int ld;
    int heads;
    int frame_len;   // tokens per frame: row of the RoPE table = token % frame_len
    int norm;        // 0 none, 1 LayerNorm, 2 RMSNorm
    int rope_half;   // 16: pairs (i, i+16) in each 32-wide half (VGGT); 32: pairs (i, i+32) (DINOv3)
    float eps;
    int v_f16;       // write V as fp16 (for the PV16 attention kernels) instead of bf16
    int pad;
};

__device__ __forceinline__ float sum8(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    return v;
}

extern "C" __global__ void __launch_bounds__(256) qkv_prologue(QkvArgs a) {
    const int lane8 = threadIdx.x & 7;
    const long long rows = a.tokens * 3 * a.heads;
    // Every lane runs the shuffles: rows past the end compute on zeros.
    for (long long base = (long long)blockIdx.x * (blockDim.x >> 3); base < rows;
         base += (long long)gridDim.x * (blockDim.x >> 3)) {
        const long long item = base + (threadIdx.x >> 3);
        const bool valid = item < rows;
        const long long it = valid ? item : 0;
        const long long tok = it / (3 * a.heads);
        const int rem = (int)(it - tok * 3 * a.heads);
        const int part = rem / a.heads;
        const int h = rem - part * a.heads;
        const int ch = rem * 64 + lane8 * 8;  // channel in the fused projection

        float x[8], f[8];
        unpack8(*(const uint4*)((const bf16*)a.y + tok * a.ld + ch), x);
        if (a.ytok) {
            const float t = ((const float*)a.ytok)[tok];
#pragma unroll
            for (int j = 0; j < 8; ++j) x[j] *= t;
        }
        if (a.ychan) {
            const float4 p = *(const float4*)((const float*)a.ychan + ch);
            const float4 q = *(const float4*)((const float*)a.ychan + ch + 4);
            x[0] *= p.x; x[1] *= p.y; x[2] *= p.z; x[3] *= p.w;
            x[4] *= q.x; x[5] *= q.y; x[6] *= q.z; x[7] *= q.w;
        }
        if (a.bias) {
            unpack8(*(const uint4*)((const bf16*)a.bias + ch), f);
#pragma unroll
            for (int j = 0; j < 8; ++j) x[j] += f[j];
        }

        // `a.norm` is uniform; `part` is not when heads % 4 != 0, so the
        // shuffles run on every lane and only the update is conditional.
        if (a.norm) {
            float mean = 0.0f;
            if (a.norm == 1) {
                float s = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) s += x[j];
                mean = sum8(s) * (1.0f / 64.0f);
            }
            float ss = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float d = x[j] - mean;
                ss += d * d;
            }
            const float r = rsqrtf(sum8(ss) * (1.0f / 64.0f) + a.eps);
            if (part < 2) {
                const float* w = (const float*)(part == 0 ? a.qn_w : a.kn_w);
                const float* b = (const float*)(part == 0 ? a.qn_b : a.kn_b);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = lane8 * 8 + j;
                    x[j] = (x[j] - mean) * r * w[d] + (a.norm == 1 ? b[d] : 0.0f);
                }
            }
        }

        if (a.rope) {
            // The partner of channel d is d ^ half, held by lane8 ^ (half / 8).
            const int half = a.rope_half;
            const int xo = half >> 3;
            float p[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) p[j] = __shfl_xor_sync(0xffffffffu, x[j], xo);
            if (part < 2) {
                const float2* t = (const float2*)a.rope + (tok % a.frame_len) * 32;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = lane8 * 8 + j;
                    const int pair = (d / (2 * half)) * half + (d & (half - 1));
                    const float2 cs = t[pair];
                    x[j] = (d & half) ? x[j] * cs.x + p[j] * cs.y : x[j] * cs.x - p[j] * cs.y;
                }
            }
        }

        if (valid) {
            const unsigned long long dst = part == 0 ? a.q : part == 1 ? a.k : a.v;
            uint4 out = pack8(x);
            if (part == 2 && a.v_f16) {
                unsigned w[4];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(w[j]) : "f"(x[2 * j + 1]), "f"(x[2 * j]));
                out = make_uint4(w[0], w[1], w[2], w[3]);
            }
            *(uint4*)((bf16*)dst + ((long long)h * a.tokens + tok) * 64 + lane8 * 8) = out;
        }
    }
}
