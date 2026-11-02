// VGGT-Ω trunk helpers (src/vggt/trunk): patchify, token assembly, row
// gather/scatter between frame layouts, the fp32-output LayerNorm that ends
// the DINO embedder, and the bf16 cache writes the heads read. Compiled by
// NVRTC after spl_nn::COMMON_CUH (bf16 helpers, warp_sum).
//
// Rows are 1024 channels (C) everywhere. A "row map" addresses row p of
// frame f as f * frame_len + offset + p, so one kernel moves rows between
// the DINO layout ([cls, 4 storage, P patches] per frame), the aggregator
// layout ([camera, 16 registers, P patches]) and the packed register
// sequence ([17] per frame).

#define C 1024

struct RowMap {
    int frame_len;
    int offset;
};

__device__ __forceinline__ long long map_row(RowMap m, int f, int p) {
    return (long long)f * m.frame_len + m.offset + p;
}

// im2col of a 16x16 / stride 16 patch conv: img f32 [N][3][H][W] (already
// normalised) -> out bf16 [N * gh * gw][768] in (c, ky, kx) order, which
// matches proj.weight.reshape(1024, 768). One thread writes 8 values (half a
// kernel row).
extern "C" __global__ void patchify(const float* __restrict__ img, bf16* __restrict__ out,
                                    int n, int h, int w) {
    int gh = h / 16, gw = w / 16;
    long long total = (long long)n * gh * gw * 96;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        int part = (int)(i % 96);
        long long row = i / 96;
        int c = part / 32, ky = (part / 2) % 16, kx0 = (part % 2) * 8;
        int px = (int)(row % gw);
        long long t = row / gw;
        int py = (int)(t % gh);
        int f = (int)(t / gh);
        const float* src = img + (((long long)f * 3 + c) * h + py * 16 + ky) * w + px * 16 + kx0;
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = src[j];
        *(uint4*)(out + row * 768 + part * 8) = pack8(v);
    }
}

// Writes `count` prefix rows per frame from `prefix` (f32 [variants][count][C]):
// frame 0 takes variant 0, later frames variant `variants - 1`.
extern "C" __global__ void put_prefix(const float* __restrict__ prefix, float* __restrict__ x,
                                      int n, int count, int variants, RowMap dst) {
    long long total = (long long)n * count * (C / 4);
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        int c4 = (int)(i % (C / 4));
        long long r = i / (C / 4);
        int p = (int)(r % count), f = (int)(r / count);
        int var = f == 0 ? 0 : variants - 1;
        float4 v = ((const float4*)(prefix + ((long long)var * count + p) * C))[c4];
        ((float4*)(x + map_row(dst, f, p) * C))[c4] = v;
    }
}

// bf16 rows [N * count][C] (packed) -> f32 rows at dst.
extern "C" __global__ void put_bf16_rows(const bf16* __restrict__ src, float* __restrict__ x,
                                         int n, int count, RowMap dst) {
    long long total = (long long)n * count * (C / 8);
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        int c8 = (int)(i % (C / 8));
        long long r = i / (C / 8);
        int p = (int)(r % count), f = (int)(r / count);
        float v[8];
        unpack8(((const uint4*)(src + r * C))[c8], v);
        float4* d = (float4*)(x + map_row(dst, f, p) * C + c8 * 8);
        d[0] = make_float4(v[0], v[1], v[2], v[3]);
        d[1] = make_float4(v[4], v[5], v[6], v[7]);
    }
}

// f32 rows src -> f32 rows dst, `count` per frame.
extern "C" __global__ void copy_rows(const float* __restrict__ src, float* __restrict__ dst,
                                     int n, int count, RowMap sm, RowMap dm) {
    long long total = (long long)n * count * (C / 4);
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        int c4 = (int)(i % (C / 4));
        long long r = i / (C / 4);
        int p = (int)(r % count), f = (int)(r / count);
        ((float4*)(dst + map_row(dm, f, p) * C))[c4] =
            ((const float4*)(src + map_row(sm, f, p) * C))[c4];
    }
}

// LayerNorm (eps given) with f32 output, src rows -> dst rows: the DINO
// final norm writing straight into the aggregator stream. One warp per row.
extern "C" __global__ void ln_rows_f32(const float* __restrict__ src, float* __restrict__ dst,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta, int n, int count,
                                       RowMap sm, RowMap dm, float eps) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (warp >= n * count) return;
    int p = warp % count, f = warp / count;
    const float4* s = (const float4*)(src + map_row(sm, f, p) * C);
    float4 v[C / 128];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < C / 128; j++) {
        v[j] = s[lane + 32 * j];
        sum += v[j].x + v[j].y + v[j].z + v[j].w;
    }
    float mean = warp_sum(sum) * (1.0f / C);
    float var = 0.f;
#pragma unroll
    for (int j = 0; j < C / 128; j++) {
        float a = v[j].x - mean, b = v[j].y - mean, c = v[j].z - mean, d = v[j].w - mean;
        var += a * a + b * b + c * c + d * d;
    }
    float rstd = rsqrtf(warp_sum(var) * (1.0f / C) + eps);
    float4* o = (float4*)(dst + map_row(dm, f, p) * C);
#pragma unroll
    for (int j = 0; j < C / 128; j++) {
        int c4 = lane + 32 * j;
        float4 g = ((const float4*)gamma)[c4], b = ((const float4*)beta)[c4];
        o[c4] = make_float4((v[j].x - mean) * rstd * g.x + b.x, (v[j].y - mean) * rstd * g.y + b.y,
                            (v[j].z - mean) * rstd * g.z + b.z, (v[j].w - mean) * rstd * g.w + b.w);
    }
}

// f32 rows [rows][C] -> bf16 into `cache` [rows][ld] at column `col`.
extern "C" __global__ void to_cache(const float* __restrict__ x, bf16* __restrict__ cache,
                                    long long rows, int ld, int col) {
    long long total = rows * (C / 8);
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        int c8 = (int)(i % (C / 8));
        long long r = i / (C / 8);
        const float4* s = (const float4*)(x + r * C + c8 * 8);
        float4 a = s[0], b = s[1];
        float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        *(uint4*)(cache + r * ld + col + c8 * 8) = pack8(v);
    }
}
