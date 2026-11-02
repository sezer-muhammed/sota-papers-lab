// LayerScale + bias + residual add in one pass (see src/residual.rs).
// Prefixed with common.cuh.

// Mirrors `ResArgs` in src/residual.rs.
struct ResArgs {
    unsigned long long x;      // residual stream [rows][C], f32 or bf16, updated in place
    unsigned long long y;      // bf16 [rows][ld] branch output
    unsigned long long ytok;   // f32 [rows] or 0
    unsigned long long ychan;  // f32 [C] or 0
    unsigned long long ybias;  // bf16 [C] or 0
    unsigned long long ls;     // f32 [C] or 0
    long long chunks;          // rows * C / 8
    int c;                     // channels (multiple of 8)
    int ld;                    // row stride of y in elements (multiple of 8)
};

// x[r][c] += ls[c] * (y[r][c] * ytok[r] * ychan[c] + ybias[c]); 8 channels
// per thread, grid-stride.
template <bool XF32>
__device__ __forceinline__ void residual(const ResArgs& a) {
    const int cpr = a.c >> 3;  // chunks per row
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < a.chunks;
         i += (long long)gridDim.x * blockDim.x) {
        const long long r = i / cpr;
        const int c0 = (int)(i - r * cpr) * 8;
        float y[8], f[8];
        unpack8(*(const uint4*)((const bf16*)a.y + r * a.ld + c0), y);
        if (a.ytok) {
            const float t = ((const float*)a.ytok)[r];
#pragma unroll
            for (int j = 0; j < 8; ++j) y[j] *= t;
        }
        if (a.ychan) {
            const float4 p = *(const float4*)((const float*)a.ychan + c0);
            const float4 q = *(const float4*)((const float*)a.ychan + c0 + 4);
            y[0] *= p.x; y[1] *= p.y; y[2] *= p.z; y[3] *= p.w;
            y[4] *= q.x; y[5] *= q.y; y[6] *= q.z; y[7] *= q.w;
        }
        if (a.ybias) {
            unpack8(*(const uint4*)((const bf16*)a.ybias + c0), f);
#pragma unroll
            for (int j = 0; j < 8; ++j) y[j] += f[j];
        }
        if (a.ls) {
            const float4 p = *(const float4*)((const float*)a.ls + c0);
            const float4 q = *(const float4*)((const float*)a.ls + c0 + 4);
            y[0] *= p.x; y[1] *= p.y; y[2] *= p.z; y[3] *= p.w;
            y[4] *= q.x; y[5] *= q.y; y[6] *= q.z; y[7] *= q.w;
        }
        const long long off = r * a.c + c0;
        if (XF32) {
            float4* px = (float4*)((float*)a.x + off);
            float4 p = px[0], q = px[1];
            p.x += y[0]; p.y += y[1]; p.z += y[2]; p.w += y[3];
            q.x += y[4]; q.y += y[5]; q.z += y[6]; q.w += y[7];
            px[0] = p;
            px[1] = q;
        } else {
            uint4* px = (uint4*)((bf16*)a.x + off);
            unpack8(*px, f);
#pragma unroll
            for (int j = 0; j < 8; ++j) f[j] += y[j];
            *px = pack8(f);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) residual_f32(ResArgs a) { residual<true>(a); }
extern "C" __global__ void __launch_bounds__(256) residual_bf16(ResArgs a) { residual<false>(a); }
