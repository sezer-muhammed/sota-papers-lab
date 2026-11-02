// Tensor-core peak probes for the sage benchmark (tests only). Owner: nn-sage.
//
// Each warp runs 8 independent accumulator chains of one mma.sync shape so the
// pipe stays full; ops per mma = 2 * m * n * k. Compiled after
// cuda/sage_common.cu (types). The inputs are arbitrary bit patterns that stay
// finite: fp8 0x38 = 1.0, bf16 0x3f80 = 1.0.

#define SG_PEAK_CHAINS 8

extern "C" __global__ void sg_peak_s8(int* out, int iters) {
    const sg_u32 a[4] = {threadIdx.x, threadIdx.x * 3u, threadIdx.x * 5u, threadIdx.x * 7u};
    const sg_u32 bm[2] = {threadIdx.x * 11u, threadIdx.x * 13u};
    int c[SG_PEAK_CHAINS][4] = {};
    for (int it = 0; it < iters; it++)
#pragma unroll
        for (int j = 0; j < SG_PEAK_CHAINS; j++)
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                "{%0, %1, %2, %3};\n"
                : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bm[0]), "r"(bm[1]));
    int s = 0;
#pragma unroll
    for (int j = 0; j < SG_PEAK_CHAINS; j++) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

extern "C" __global__ void sg_peak_e4m3_f32(int* out, int iters) {
    const sg_u32 one = 0x38383838u;
    float c[SG_PEAK_CHAINS][4] = {};
    for (int it = 0; it < iters; it++)
#pragma unroll
        for (int j = 0; j < SG_PEAK_CHAINS; j++)
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                "{%0, %1, %2, %3};\n"
                : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                : "r"(one), "r"(one), "r"(one), "r"(one), "r"(one), "r"(one));
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < SG_PEAK_CHAINS; j++) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = (int)s;
}

extern "C" __global__ void sg_peak_e4m3_f16(int* out, int iters) {
    const sg_u32 tiny = 0x08080808u;  // 2^-6 each, keeps f16 sums small
    sg_u32 c[SG_PEAK_CHAINS][2] = {};
    for (int it = 0; it < iters; it++)
#pragma unroll
        for (int j = 0; j < SG_PEAK_CHAINS; j++)
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, "
                "{%0, %1};\n"
                : "+r"(c[j][0]), "+r"(c[j][1])
                : "r"(tiny), "r"(tiny), "r"(tiny), "r"(tiny), "r"(tiny), "r"(tiny));
    sg_u32 s = 0;
#pragma unroll
    for (int j = 0; j < SG_PEAK_CHAINS; j++) s ^= c[j][0] ^ c[j][1];
    out[blockIdx.x * blockDim.x + threadIdx.x] = (int)s;
}

extern "C" __global__ void sg_peak_bf16_f32(int* out, int iters) {
    const sg_u32 one = 0x3f803f80u;
    float c[SG_PEAK_CHAINS][4] = {};
    for (int it = 0; it < iters; it++)
#pragma unroll
        for (int j = 0; j < SG_PEAK_CHAINS; j++)
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                "{%0, %1, %2, %3};\n"
                : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                : "r"(one), "r"(one), "r"(one), "r"(one), "r"(one), "r"(one));
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < SG_PEAK_CHAINS; j++) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = (int)s;
}

extern "C" __global__ void sg_peak_f16_f16(int* out, int iters) {
    const sg_u32 tiny = 0x10001000u;  // 2^-11 each (f16), keeps sums small
    sg_u32 c[SG_PEAK_CHAINS][2] = {};
    for (int it = 0; it < iters; it++)
#pragma unroll
        for (int j = 0; j < SG_PEAK_CHAINS; j++)
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, "
                "{%0, %1};\n"
                : "+r"(c[j][0]), "+r"(c[j][1])
                : "r"(tiny), "r"(tiny), "r"(tiny), "r"(tiny), "r"(tiny), "r"(tiny));
    sg_u32 s = 0;
#pragma unroll
    for (int j = 0; j < SG_PEAK_CHAINS; j++) s ^= c[j][0] ^ c[j][1];
    out[blockIdx.x * blockDim.x + threadIdx.x] = (int)s;
}
