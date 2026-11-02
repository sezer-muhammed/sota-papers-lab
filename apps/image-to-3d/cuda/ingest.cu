// Stage A0 ingest kernels. Owner: ingest (A0). Rust side: src/ingest/gpu.rs.
//
// Frames arrive as packed 8-bit RGB (row stride = 3·w bytes). Two kernels:
//   sharpness_partials  Laplacian variance of BT.601 luma in [0, 1]: each block
//                       writes (Σ L, Σ L²) of its pixels to partials[2·block] in
//                       f32; the host sums the partials in f64 (variance =
//                       E[L²] − E[L]²). The 4-neighbour Laplacian is evaluated on
//                       interior pixels only (n = (w − 2)(h − 2)).
//   resize_area         antialiased resize: each output pixel is the area-weighted
//                       mean of the source pixels its footprint covers (exact box
//                       filter for downscaling; for upscaling it degrades to
//                       nearest-ish, which ingest never asks for). Output packed RGB u8.
// common.cuh is prepended by gpu.rs (integer typedefs).

#define SHARP_BLOCK_X 32
#define SHARP_BLOCK_Y 8

SPL_INLINE float luma_at(const uint8_t* rgb, int w, int x, int y) {
    const uint8_t* p = rgb + (size_t(y) * w + x) * 3;
    return (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) * (1.0f / 255.0f);
}

extern "C" __global__ void sharpness_partials(const uint8_t* __restrict__ rgb, int w, int h,
                                              float* __restrict__ partials) {
    const int x = blockIdx.x * SHARP_BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * SHARP_BLOCK_Y + threadIdx.y;
    float s = 0.f, s2 = 0.f;
    if (x >= 1 && y >= 1 && x < w - 1 && y < h - 1) {
        const float l = luma_at(rgb, w, x - 1, y) + luma_at(rgb, w, x + 1, y) +
                        luma_at(rgb, w, x, y - 1) + luma_at(rgb, w, x, y + 1) -
                        4.f * luma_at(rgb, w, x, y);
        s = l;
        s2 = l * l;
    }
    s = warp_sum(s);
    s2 = warp_sum(s2);
    __shared__ float sh[2][SHARP_BLOCK_X * SHARP_BLOCK_Y / 32];
    const int tid = threadIdx.y * SHARP_BLOCK_X + threadIdx.x;
    if ((tid & 31) == 0) {
        sh[0][tid >> 5] = s;
        sh[1][tid >> 5] = s2;
    }
    __syncthreads();
    if (tid == 0) {
        float a = 0.f, b = 0.f;
        for (int i = 0; i < SHARP_BLOCK_X * SHARP_BLOCK_Y / 32; ++i) {
            a += sh[0][i];
            b += sh[1][i];
        }
        const int block = blockIdx.y * gridDim.x + blockIdx.x;
        partials[2 * block] = a;
        partials[2 * block + 1] = b;
    }
}

extern "C" __global__ void resize_area(const uint8_t* __restrict__ src, int sw, int sh,
                                       uint8_t* __restrict__ dst, int dw, int dh) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || oy >= dh) return;
    const float fx = float(sw) / float(dw);
    const float fy = float(sh) / float(dh);
    const float x0 = ox * fx, x1 = x0 + fx;
    const float y0 = oy * fy, y1 = y0 + fy;
    const int ix0 = int(floorf(x0)), ix1 = min(int(ceilf(x1)), sw);
    const int iy0 = int(floorf(y0)), iy1 = min(int(ceilf(y1)), sh);
    float acc[3] = {0.f, 0.f, 0.f};
    float wsum = 0.f;
    for (int y = iy0; y < iy1; ++y) {
        const float wy = fminf(y + 1.f, y1) - fmaxf(float(y), y0);
        if (wy <= 0.f) continue;
        const uint8_t* row = src + size_t(y) * sw * 3;
        for (int x = ix0; x < ix1; ++x) {
            const float wx = fminf(x + 1.f, x1) - fmaxf(float(x), x0);
            if (wx <= 0.f) continue;
            const float wgt = wx * wy;
            acc[0] += wgt * row[3 * x];
            acc[1] += wgt * row[3 * x + 1];
            acc[2] += wgt * row[3 * x + 2];
            wsum += wgt;
        }
    }
    const float inv = wsum > 0.f ? 1.f / wsum : 0.f;
    uint8_t* o = dst + (size_t(oy) * dw + ox) * 3;
    for (int c = 0; c < 3; ++c) o[c] = (uint8_t)fminf(fmaxf(acc[c] * inv + 0.5f, 0.f), 255.f);
}
