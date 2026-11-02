// VGGT-Omega runtime kernels (owner: runtime). Compiled by NVRTC through
// spl_nn::Nn::module; no includes.

// RGB8 [N][H][W][3] -> f32 [N][3][H][W], (x / 255 - mean) / std in fp32 with
// round-to-nearest division, as torchvision ToTensor + Aggregator.forward.
extern "C" __global__ void vggt_normalize(const unsigned char* __restrict__ src,
                                          float* __restrict__ dst, long long pixels,
                                          int hw) {
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixels) return;
    long long frame = i / hw;
    int p = (int)(i - frame * hw);
    const unsigned char* s = src + i * 3;
    float* d = dst + frame * 3LL * hw + p;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        float x = __fdiv_rn((float)s[c], 255.0f);
        d[(long long)c * hw] = __fdiv_rn(__fsub_rn(x, mean[c]), stdv[c]);
    }
}

// f16 bits -> f32 (depth / confidence readback helpers are on the host; this
// converts a device buffer in place of a host loop for large N).
extern "C" __global__ void vggt_f16_to_f32(const unsigned short* __restrict__ src,
                                          float* __restrict__ dst, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned short h = src[i];
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    dst[i] = f;
}
