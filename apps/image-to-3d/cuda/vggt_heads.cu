// VGGT-Omega head kernels that are model-specific (owner: heads; see
// src/vggt/heads/*.rs and reports/image-to-3d/spec-vggt.md §4-§5).
// Prefixed with spl_nn::COMMON_CUH.

// x[i] = float(bf16 src[i]) — seeds an f32 residual stream from a bf16
// LayerNorm output.
extern "C" __global__ void widen_bf16(unsigned long long src, unsigned long long dst, long long n) {
    const bf16* s = (const bf16*)src;
    float* d = (float*)dst;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        d[i] = bf16_f32(s[i]);
}

// Camera branch tail + pose encoding → cameras, one CTA (256 threads) per
// frame: raw = W2 · gelu(h) + b2 (h is already activated, bf16 [frames][1024]),
// pose_enc = [raw0..6, relu(raw7..8) + 0.01]; quaternion (x, y, z, w) to R
// with s = 2 / |q|²; extrinsics [R | T] (w2c) row-major 3×4; intrinsics with
// fy = (H/2)/tan(fov_h/2), fx = (W/2)/tan(fov_w/2), principal point at the
// centre (utils/pose_enc.py::encoding_to_camera).
struct CamArgs {
    unsigned long long h;     // bf16 [frames][1024]
    unsigned long long w2;    // f32 [9][1024]
    unsigned long long b2;    // f32 [9]
    unsigned long long pose;  // f32 [frames][9]
    unsigned long long extr;  // f32 [frames][12]
    unsigned long long intr;  // f32 [frames][9]
    int frames, width, height, hidden;
};

extern "C" __global__ void __launch_bounds__(256) camera_finish(const CamArgs a) {
    __shared__ float part[9][8];
    __shared__ float pe[9];
    const int f = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const bf16* h = (const bf16*)a.h + (size_t)f * a.hidden;
    const float* w2 = (const float*)a.w2;
    float acc[9];
#pragma unroll
    for (int o = 0; o < 9; ++o) acc[o] = 0.0f;
    for (int i = tid; i < a.hidden; i += 256) {
        const float x = bf16_f32(h[i]);
#pragma unroll
        for (int o = 0; o < 9; ++o) acc[o] += x * w2[o * a.hidden + i];
    }
#pragma unroll
    for (int o = 0; o < 9; ++o) {
        const float s = warp_sum(acc[o]);
        if (lane == 0) part[o][warp] = s;
    }
    __syncthreads();
    if (tid < 9) {
        float s = ((const float*)a.b2)[tid];
        for (int wi = 0; wi < 8; ++wi) s += part[tid][wi];
        if (tid >= 7) s = fmaxf(s, 0.0f) + 0.01f;
        pe[tid] = s;
        ((float*)a.pose)[f * 9 + tid] = s;
    }
    __syncthreads();
    if (tid == 0) {
        const float x = pe[3], y = pe[4], z = pe[5], w = pe[6];
        const float s = 2.0f / (x * x + y * y + z * z + w * w);
        float* E = (float*)a.extr + f * 12;
        E[0] = 1.0f - s * (y * y + z * z); E[1] = s * (x * y - z * w);        E[2] = s * (x * z + y * w);        E[3] = pe[0];
        E[4] = s * (x * y + z * w);        E[5] = 1.0f - s * (x * x + z * z); E[6] = s * (y * z - x * w);        E[7] = pe[1];
        E[8] = s * (x * z - y * w);        E[9] = s * (y * z + x * w);        E[10] = 1.0f - s * (x * x + y * y); E[11] = pe[2];
        float* K = (float*)a.intr + f * 9;
        const float fy = 0.5f * a.height / tanf(0.5f * pe[7]);
        const float fx = 0.5f * a.width / tanf(0.5f * pe[8]);
        K[0] = fx;   K[1] = 0.0f; K[2] = 0.5f * a.width;
        K[3] = 0.0f; K[4] = fy;   K[5] = 0.5f * a.height;
        K[6] = 0.0f; K[7] = 0.0f; K[8] = 1.0f;
    }
}

// Dense-head output, one thread per quarter-resolution pixel:
// v = fused + 0.1 · PE (fp32), [depth | conf] logits = W · v + b (32 × 256,
// fp32, W in shared memory), pixel shuffle ×4 (logit 4i + j → pixel
// (4y + i, 4x + j)), depth = exp, conf = 1 + exp, both written as fp16
// [frames][H][W]; optionally world points [frames][H][W][3] f32 by
// unprojecting the fp32 depth with the frame's camera:
// cam = ((u − cx)/fx · d, (v − cy)/fy · d, d) at integer pixel (u, v),
// world = Rᵀ (cam − T).
#define DO_C 256
#define DO_OUT 32

struct OutArgs {
    unsigned long long fused;  // bf16 [frames][qh][qw][256]
    unsigned long long pe;     // f32 [qh·qw][256] (already × 0.1) or 0
    unsigned long long w;      // f32 [32][256]: rows 0..15 depth, 16..31 conf
    unsigned long long b;      // f32 [32]
    unsigned long long depth;  // f16 [frames][H][W]
    unsigned long long conf;   // f16 [frames][H][W]
    unsigned long long extr;   // f32 [frames][12] or 0
    unsigned long long intr;   // f32 [frames][9]
    unsigned long long world;  // f32 [frames][H][W][3] or 0
    int frames, qh, qw;
};

__device__ __forceinline__ unsigned short f32_f16(float v) {
    unsigned short r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(v));
    return r;
}

extern "C" __global__ void __launch_bounds__(128) dense_output(const OutArgs a) {
    __shared__ float Ws[DO_C][DO_OUT];  // transposed: Ws[c][o]
    __shared__ float Bs[DO_OUT];
    const float* w = (const float*)a.w;
    for (int i = threadIdx.x; i < DO_C * DO_OUT; i += blockDim.x) {
        const int o = i / DO_C, c = i % DO_C;
        Ws[c][o] = w[i];
    }
    if (threadIdx.x < DO_OUT) Bs[threadIdx.x] = ((const float*)a.b)[threadIdx.x];
    __syncthreads();

    const int qp = a.qh * a.qw;
    const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= (long long)a.frames * qp) return;
    const int f = (int)(t / qp), p = (int)(t % qp);
    const int qy = p / a.qw, qx = p % a.qw;
    const bf16* x = (const bf16*)a.fused + (size_t)t * DO_C;
    const float* pe = a.pe ? (const float*)a.pe + (size_t)p * DO_C : nullptr;
    float acc[DO_OUT];
#pragma unroll
    for (int o = 0; o < DO_OUT; ++o) acc[o] = Bs[o];
    for (int c0 = 0; c0 < DO_C; c0 += 8) {
        float v[8];
        unpack8(*(const uint4*)(x + c0), v);
        if (pe) {
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] += pe[c0 + e];
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float* wr = Ws[c0 + e];
#pragma unroll
            for (int o = 0; o < DO_OUT; ++o) acc[o] += v[e] * wr[o];
        }
    }
    const int H = a.qh * 4, W = a.qw * 4;
    unsigned short* D = (unsigned short*)a.depth + (size_t)f * H * W;
    unsigned short* C = (unsigned short*)a.conf + (size_t)f * H * W;
    float E[12], fx = 1, fy = 1, cx = 0, cy = 0;
    const bool world = a.world && a.extr;
    if (world) {
        const float* e = (const float*)a.extr + f * 12;
#pragma unroll
        for (int i = 0; i < 12; ++i) E[i] = e[i];
        const float* k = (const float*)a.intr + f * 9;
        fx = k[0]; cx = k[2]; fy = k[4]; cy = k[5];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int Y = qy * 4 + i, X = qx * 4 + j;
            const size_t o = (size_t)Y * W + X;
            const float d = expf(acc[i * 4 + j]);
            D[o] = f32_f16(d);
            C[o] = f32_f16(1.0f + expf(acc[16 + i * 4 + j]));
            if (world) {
                const float px = ((float)X - cx) / fx * d - E[3];
                const float py = ((float)Y - cy) / fy * d - E[7];
                const float pz = d - E[11];
                float* wp = (float*)a.world + ((size_t)f * H * W + o) * 3;
                wp[0] = E[0] * px + E[4] * py + E[8] * pz;
                wp[1] = E[1] * px + E[5] * py + E[9] * pz;
                wp[2] = E[2] * px + E[6] * py + E[10] * pz;
            }
        }
    }
}
