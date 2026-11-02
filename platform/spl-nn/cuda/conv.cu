// Image-space ops on NHWC bf16 tensors (see src/conv.rs). Prefixed with
// common.cuh.
//
// conv2d: implicit-GEMM convolution. GEMM view: M = n·Ho·Wo output pixels,
// N = Cout, K = KH·KW·Cin in (ky, kx, ci) order, weights [Cout][KH][KW][Cin].
// One CTA computes a 128 × 128 tile with 8 warps (4 along M, 2 along N,
// 32 × 64 each) on mma.sync m16n8k16 (bf16 in, fp32 accumulate). K advances
// 32 at a time; Cin % 32 == 0, so a K slice is one filter tap and the A tile
// is 128 contiguous 64-byte pixel runs (zero-filled outside the image),
// loaded with cp.async through a two-stage pipeline into swizzled shared
// memory and read with ldmatrix. Optional ReLU on the input (applied to the A
// fragments), bias and up to two residual adds in the epilogue.

#define CONV_BM 128
#define CONV_BN 128
#define CONV_BK 32

struct ConvArgs {
    unsigned long long x;     // bf16 [n][h][w][cin]
    unsigned long long w;     // bf16 [cout][kh][kw][cin]
    unsigned long long bias;  // f32 [cout] or 0
    unsigned long long res0;  // bf16 [m][cout] or 0
    unsigned long long res1;  // bf16 [m][cout] or 0
    unsigned long long y;     // bf16 [m][cout]
    int n, h, w_, cin;
    int ho, wo, cout;
    int kh, kw, stride, pad;
    int pre_relu;
};

__device__ __forceinline__ unsigned c_smem(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void c_cp16(unsigned dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void c_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void c_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

__device__ __forceinline__ void c_ldsm4(unsigned a, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

__device__ __forceinline__ void c_mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Element offset of 16-byte chunk `ch` (0..3) of row `r` in a [rows][32]
// tile; the XOR by (r >> 1) & 3 makes 8 consecutive rows hit 8 distinct
// bank groups for ldmatrix.
__device__ __forceinline__ int c_swz(int r, int ch) { return r * CONV_BK + ((ch ^ ((r >> 1) & 3)) << 3); }

// ReLU on packed bf16x2: clear each half whose sign bit is set.
__device__ __forceinline__ unsigned relu2(unsigned u) {
    unsigned m = (u & 0x80008000u) >> 15;
    return u & ~(m * 0xffffu);
}

extern "C" __global__ void __launch_bounds__(256) conv2d_bf16(const ConvArgs a) {
    __shared__ __align__(16) bf16 As[2][CONV_BM * CONV_BK];
    __shared__ __align__(16) bf16 Bs[2][CONV_BN * CONV_BK];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 3, wn = warp >> 2;
    const int m_total = a.n * a.ho * a.wo;
    const int bm = blockIdx.x * CONV_BM, bn = blockIdx.y * CONV_BN;
    const int K = a.kh * a.kw * a.cin;
    const int kt_count = K / CONV_BK;
    const bf16* X = (const bf16*)a.x;
    const bf16* Wt = (const bf16*)a.w;

    // Each thread loads chunk `ch` of rows r0 and r0 + 64 of both tiles.
    const int ch = tid & 3, r0 = tid >> 2;
    int pn[2], piy[2], pix[2];
    bool pok[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int m = bm + r0 + i * 64;
        pok[i] = m < m_total;
        const int mm = pok[i] ? m : 0;
        const int ox = mm % a.wo, t = mm / a.wo;
        const int oy = t % a.ho;
        pn[i] = t / a.ho;
        piy[i] = oy * a.stride - a.pad;
        pix[i] = ox * a.stride - a.pad;
    }

    auto load = [&](int stage, int kt) {
        const int k0 = kt * CONV_BK;
        const int tap = k0 / a.cin, ci0 = k0 - tap * a.cin;
        const int ky = tap / a.kw, kx = tap - ky * a.kw;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int r = r0 + i * 64;
            const int iy = piy[i] + ky, ix = pix[i] + kx;
            const bool ok = pok[i] && iy >= 0 && iy < a.h && ix >= 0 && ix < a.w_;
            const bf16* src = ok ? X + (((size_t)pn[i] * a.h + iy) * a.w_ + ix) * a.cin + ci0 + ch * 8 : X;
            c_cp16(c_smem(&As[stage][c_swz(r, ch)]), src, ok ? 16 : 0);
            const int co = bn + r;
            const bool okb = co < a.cout;
            const bf16* srcb = okb ? Wt + (size_t)co * K + k0 + ch * 8 : Wt;
            c_cp16(c_smem(&Bs[stage][c_swz(r, ch)]), srcb, okb ? 16 : 0);
        }
    };

    float acc[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    load(0, 0);
    c_commit();
    for (int kt = 0; kt < kt_count; ++kt) {
        const int st = kt & 1;
        if (kt + 1 < kt_count) load(st ^ 1, kt + 1);
        c_commit();
        c_wait<1>();
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            unsigned af[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r = wm * 32 + i * 16 + (lane & 15);
                const int c = kk * 2 + (lane >> 4);
                c_ldsm4(c_smem(&As[st][c_swz(r, c)]), af[i][0], af[i][1], af[i][2], af[i][3]);
                if (a.pre_relu) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) af[i][e] = relu2(af[i][e]);
                }
            }
#pragma unroll
            for (int j2 = 0; j2 < 4; ++j2) {
                const int r = wn * 64 + j2 * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int c = kk * 2 + ((lane >> 3) & 1);
                unsigned b0, b1, b2, b3;
                c_ldsm4(c_smem(&Bs[st][c_swz(r, c)]), b0, b1, b2, b3);
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    c_mma(acc[i][2 * j2], af[i], b0, b1);
                    c_mma(acc[i][2 * j2 + 1], af[i], b2, b3);
                }
            }
        }
        __syncthreads();
    }

    const float* bias = (const float*)a.bias;
    const unsigned* R0 = (const unsigned*)a.res0;
    const unsigned* R1 = (const unsigned*)a.res1;
    unsigned* Y = (unsigned*)a.y;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int co = bn + wn * 64 + j * 8 + (lane & 3) * 2;
            if (co >= a.cout) continue;
            float b0 = 0.0f, b1 = 0.0f;
            if (bias) { b0 = bias[co]; b1 = bias[co + 1]; }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = bm + wm * 32 + i * 16 + (lane >> 2) + h * 8;
                if (m >= m_total) continue;
                float v0 = acc[i][j][2 * h] + b0, v1 = acc[i][j][2 * h + 1] + b1;
                const size_t o = ((size_t)m * a.cout + co) >> 1;
                if (R0) { const unsigned r = R0[o]; v0 += bf16_lo(r); v1 += bf16_hi(r); }
                if (R1) { const unsigned r = R1[o]; v0 += bf16_lo(r); v1 += bf16_hi(r); }
                Y[o] = pack_bf16(v0, v1);
            }
        }
    }
}

// depth_to_space: in [n][h][w][s·s·c] with channel (i·s + j)·c + o →
// out [n][h·s][w·s][c] at (y·s + i, x·s + j), plus an optional f32 bias [c].
// ConvTranspose2d with kernel = stride is a per-pixel GEMM followed by this;
// pixel shuffle is the same op with c-major channel order (see `cmajor`).
struct D2sArgs {
    unsigned long long in, bias, out;
    int n, h, w, c, s;
    int cmajor;  // 1: channel o·s·s + i·s + j (torch pixel_shuffle)
};

extern "C" __global__ void depth_to_space_bf16(const D2sArgs a) {
    const int c8 = a.c / 8;
    const long long total = (long long)a.n * a.h * a.s * a.w * a.s * c8;
    const bf16* in = (const bf16*)a.in;
    const float* bias = (const float*)a.bias;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < total;
         t += (long long)gridDim.x * blockDim.x) {
        const int cv = (int)(t % c8);
        long long p = t / c8;
        const int X = (int)(p % (a.w * a.s));
        p /= a.w * a.s;
        const int Y = (int)(p % (a.h * a.s));
        const int n = (int)(p / (a.h * a.s));
        const int y = Y / a.s, i = Y - y * a.s, x = X / a.s, j = X - x * a.s;
        const size_t base = (((size_t)n * a.h + y) * a.w + x) * (size_t)(a.s * a.s * a.c);
        float f[8];
        if (!a.cmajor) {
            unpack8(*(const uint4*)(in + base + (size_t)(i * a.s + j) * a.c + cv * 8), f);
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) f[e] = bf16_f32(in[base + (size_t)(cv * 8 + e) * a.s * a.s + i * a.s + j]);
        }
        if (bias) {
#pragma unroll
            for (int e = 0; e < 8; ++e) f[e] += bias[cv * 8 + e];
        }
        *(uint4*)((bf16*)a.out + (size_t)t * 8) = pack8(f);
    }
}

// Bilinear resize with align_corners=True, NHWC bf16, 8 channels per thread.
struct UpArgs {
    unsigned long long in, out;
    int n, h, w, c, ho, wo;
};

extern "C" __global__ void upsample_bilinear_ac_bf16(const UpArgs a) {
    const int c8 = a.c / 8;
    const long long total = (long long)a.n * a.ho * a.wo * c8;
    const float sy = a.ho > 1 ? (float)(a.h - 1) / (float)(a.ho - 1) : 0.0f;
    const float sx = a.wo > 1 ? (float)(a.w - 1) / (float)(a.wo - 1) : 0.0f;
    const bf16* in = (const bf16*)a.in;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < total;
         t += (long long)gridDim.x * blockDim.x) {
        const int cv = (int)(t % c8);
        long long p = t / c8;
        const int ox = (int)(p % a.wo);
        p /= a.wo;
        const int oy = (int)(p % a.ho);
        const int n = (int)(p / a.ho);
        const float fy = oy * sy, fx = ox * sx;
        const int y0 = min((int)fy, a.h - 1), x0 = min((int)fx, a.w - 1);
        const int y1 = min(y0 + 1, a.h - 1), x1 = min(x0 + 1, a.w - 1);
        const float ly = fy - y0, lx = fx - x0;
        const float w00 = (1 - ly) * (1 - lx), w01 = (1 - ly) * lx, w10 = ly * (1 - lx), w11 = ly * lx;
        const size_t row = (size_t)n * a.h;
        float f00[8], f01[8], f10[8], f11[8], o[8];
        unpack8(*(const uint4*)(in + ((row + y0) * a.w + x0) * a.c + cv * 8), f00);
        unpack8(*(const uint4*)(in + ((row + y0) * a.w + x1) * a.c + cv * 8), f01);
        unpack8(*(const uint4*)(in + ((row + y1) * a.w + x0) * a.c + cv * 8), f10);
        unpack8(*(const uint4*)(in + ((row + y1) * a.w + x1) * a.c + cv * 8), f11);
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] = w00 * f00[e] + w01 * f01[e] + w10 * f10[e] + w11 * f11[e];
        *(uint4*)((bf16*)a.out + (size_t)t * 8) = pack8(o);
    }
}

// out[f][p][c] = in[f·rows + start + p][c] + scale · table[p][c]: gathers
// `p_count` rows per frame (dropping prefix rows) and adds a per-position
// table shared by every frame. In place when rows == p_count and start == 0.
struct TableArgs {
    unsigned long long in, table, out;
    int frames, rows, start, p_count, c;
    float scale;
};

extern "C" __global__ void add_table_bf16(const TableArgs a) {
    const int c8 = a.c / 8;
    const long long total = (long long)a.frames * a.p_count * c8;
    const bf16* in = (const bf16*)a.in;
    const float* tab = (const float*)a.table;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < total;
         t += (long long)gridDim.x * blockDim.x) {
        const int cv = (int)(t % c8);
        const long long fp = t / c8;
        const int p = (int)(fp % a.p_count), f = (int)(fp / a.p_count);
        float v[8];
        unpack8(*(const uint4*)(in + ((size_t)f * a.rows + a.start + p) * a.c + cv * 8), v);
        if (tab) {
            const float4 t0 = *(const float4*)(tab + (size_t)p * a.c + cv * 8);
            const float4 t1 = *(const float4*)(tab + (size_t)p * a.c + cv * 8 + 4);
            v[0] += a.scale * t0.x; v[1] += a.scale * t0.y; v[2] += a.scale * t0.z; v[3] += a.scale * t0.w;
            v[4] += a.scale * t1.x; v[5] += a.scale * t1.y; v[6] += a.scale * t1.z; v[7] += a.scale * t1.w;
        }
        *(uint4*)((bf16*)a.out + (size_t)t * 8) = pack8(v);
    }
}
