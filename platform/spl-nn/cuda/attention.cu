// FlashAttention-2-style bf16 attention, head dim 64 (see src/attention.rs).
// Prefixed with common.cuh. Written from the FlashAttention-2 paper (Dao,
// 2023) and the PTX ISA; no code is taken from the FlashAttention sources.
//
// Q, K, V: bf16 [heads][n_seq * seq_len][64]; sequence s of head h is the
// contiguous run of rows [s * seq_len, (s + 1) * seq_len). One CTA owns BM
// query rows of one (sequence, head); each warp owns MT m16 tiles of them.
// The CTA streams K and V tiles of BN keys through a two-stage cp.async
// pipeline; S = Q Kᵀ and O += P V run on mma.sync m16n8k16 (bf16 in, fp32
// accumulate) with operands fetched by ldmatrix from XOR-swizzled shared
// memory (conflict-free for cp.async writes and ldmatrix reads). Softmax is
// online, in the exp2 domain with log2(e) folded into the scale; P never
// leaves registers (the S accumulator layout is the A-operand layout).
//
// PV16 keeps V in fp16 and runs P·V on fp16-accumulating tensor cores,
// which GeForce Ada executes at twice the fp32-accumulate rate: each key
// tile's partial P·V (64 keys) is accumulated in fp16 and then added to the
// fp32 output accumulator, so rounding never compounds across tiles. P is
// in [0, 1] and fp16 carries 3 more mantissa bits than bf16. Q·Kᵀ stays
// bf16 with fp32 accumulation. Needs |V| well below 1000 (64 · |V| < 65504).
//
// Shared memory holds two pipeline stages of [K tile | V tile]. With QA
// ("Q aliased") the Q tile is staged in stage 1, moved to registers, and
// stage 1 is then refilled: 32 KB instead of 40 KB at BM = BN = 64, so three
// CTAs fit per Ada SM instead of two.

#define HD 64

// Mirrors `AttnArgs` in src/attention.rs.
struct AttnArgs {
    unsigned long long q, k, v;  // bf16 [heads][n_seq * seq_len][64]
    unsigned long long o;        // bf16 [n_seq * seq_len][o_ld], head h at columns h*64..
    int seq_len;
    int n_seq;
    int o_ld;
    float scale_log2;  // softmax scale * log2(e)
};

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

// Element offset of 8-element chunk `ch` of row `row` in a [rows][64] bf16
// tile whose 16-byte chunks are XOR-swizzled by the row's low 3 bits.
__device__ __forceinline__ int swz(int row, int ch) { return row * HD + ((ch ^ (row & 7)) << 3); }

__device__ __forceinline__ void cp_async16(unsigned dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void ldsm_x4(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma16816_f16(unsigned* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two floats -> packed f16x2 (round to nearest even); `lo` lands in the low half.
__device__ __forceinline__ unsigned pack_f16(float lo, float hi) {
    unsigned r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ void unpack_f16(unsigned v, float& lo, float& hi) {
    asm("{\n.reg .f16 l, h;\nmov.b32 {l, h}, %2;\ncvt.f32.f16 %0, l;\ncvt.f32.f16 %1, h;\n}"
        : "=f"(lo), "=f"(hi)
        : "r"(v));
}

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Copies rows [row0, row0 + ROWS) of a [len][64] bf16 matrix into a swizzled
// tile; rows at or past `len` are zero-filled (cp.async with 0 source bytes).
template <int ROWS, int THREADS>
__device__ __forceinline__ void load_tile(bf16* tile, const bf16* g, int row0, int len, int tid) {
    static_assert((ROWS * 8) % THREADS == 0, "tile chunks must divide evenly");
#pragma unroll
    for (int i = 0; i < ROWS * 8 / THREADS; ++i) {
        const int c = tid + i * THREADS;
        const int r = c >> 3, ch = c & 7;
        const int gr = row0 + r;
        const bool ok = gr < len;
        cp_async16(smem_addr(tile + swz(r, ch)), g + (size_t)(ok ? gr : 0) * HD + ch * 8, ok ? 16 : 0);
    }
}

template <int WARPS, int MT, int BN, bool QA, bool PV16>
__device__ __forceinline__ void attn_fwd(const AttnArgs& a) {
    constexpr int BM = WARPS * MT * 16;
    constexpr int THREADS = WARPS * 32;
    constexpr int NT = BN / 8;  // n8 tiles of S per key tile
    constexpr int STAGE = 2 * BN * HD;  // elements per [K | V] stage
    static_assert(!QA || BM <= 2 * BN, "an aliased Q tile must fit in one stage");
    extern __shared__ __align__(128) unsigned char smem[];
    bf16* stages = (bf16*)smem + (QA ? 0 : BM * HD);
    bf16* sQ = QA ? stages + STAGE : (bf16*)smem;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int L = a.seq_len;
    const int q0 = blockIdx.x * BM;
    const int seq = blockIdx.y, h = blockIdx.z;
    const size_t rows = (size_t)a.n_seq * L;
    const size_t off = ((size_t)h * rows + (size_t)seq * L) * HD;
    const bf16* Q = (const bf16*)a.q + off;
    const bf16* K = (const bf16*)a.k + off;
    const bf16* V = (const bf16*)a.v + off;
    const int tiles = (L + BN - 1) / BN;

    load_tile<BM, THREADS>(sQ, Q, q0, L, tid);
    load_tile<BN, THREADS>(stages, K, 0, L, tid);
    load_tile<BN, THREADS>(stages + BN * HD, V, 0, L, tid);
    cp_async_commit();
    if (!QA) {
        if (tiles > 1) {
            load_tile<BN, THREADS>(stages + STAGE, K, BN, L, tid);
            load_tile<BN, THREADS>(stages + STAGE + BN * HD, V, BN, L, tid);
        }
        cp_async_commit();
        cp_async_wait<1>();
    } else {
        cp_async_wait<0>();
    }
    __syncthreads();

    // Q fragments stay in registers for the whole key loop.
    unsigned qf[MT][4][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const int r = (warp * MT + mt) * 16 + (lane & 15);
            ldsm_x4(smem_addr(sQ + swz(r, kk * 2 + (lane >> 4))), qf[mt][kk][0], qf[mt][kk][1],
                    qf[mt][kk][2], qf[mt][kk][3]);
        }
    if (QA) {
        __syncthreads();  // every warp holds its Q fragments; stage 1 is free
        if (tiles > 1) {
            load_tile<BN, THREADS>(stages + STAGE, K, BN, L, tid);
            load_tile<BN, THREADS>(stages + STAGE + BN * HD, V, BN, L, tid);
        }
        cp_async_commit();
    }

    float o[MT][8][4];
    float m[MT][2], l[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int j = 0; j < 8; ++j) o[mt][j][0] = o[mt][j][1] = o[mt][j][2] = o[mt][j][3] = 0.0f;
        m[mt][0] = m[mt][1] = -INFINITY;
        l[mt][0] = l[mt][1] = 0.0f;
    }

    // ldmatrix lane roles: lane >> 3 picks one of four 8x8 matrices.
    const int mi = lane >> 3, mr = lane & 7;

    for (int j = 0; j < tiles; ++j) {
        if (j > 0) {
            cp_async_wait<1>();
            __syncthreads();
        }
        const bf16* kt = stages + (j & 1) * STAGE;
        const bf16* vt = kt + BN * HD;

        float s[MT][NT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int n = 0; n < NT; ++n) s[mt][n][0] = s[mt][n][1] = s[mt][n][2] = s[mt][n][3] = 0.0f;

        // S = Q Kᵀ: K rows are keys with contiguous head dims, which is the
        // col-major B operand ldmatrix delivers without transposing.
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
#pragma unroll
            for (int n = 0; n < NT; n += 2) {
                unsigned b0, b1, b2, b3;
                ldsm_x4(smem_addr(kt + swz((n + (mi >> 1)) * 8 + mr, kk * 2 + (mi & 1))), b0, b1, b2, b3);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    mma16816(s[mt][n], qf[mt][kk], b0, b1);
                    mma16816(s[mt][n + 1], qf[mt][kk], b2, b3);
                }
            }

        if ((j + 1) * BN > L) {  // keys past the end of the sequence
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int col = j * BN + n * 8 + (lane & 3) * 2;
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    if (col >= L) s[mt][n][0] = s[mt][n][2] = -INFINITY;
                    if (col + 1 >= L) s[mt][n][1] = s[mt][n][3] = -INFINITY;
                }
            }
        }

        // Online softmax; each thread holds rows lane/4 and lane/4 + 8 of
        // every m16 tile, and the 4 lanes of a quad share a row.
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                float mx = -INFINITY;
#pragma unroll
                for (int n = 0; n < NT; ++n) mx = fmaxf(mx, fmaxf(s[mt][n][rr * 2], s[mt][n][rr * 2 + 1]));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                const float mnew = fmaxf(m[mt][rr], mx * a.scale_log2);
                const float corr = ex2(m[mt][rr] - mnew);
                m[mt][rr] = mnew;
                float sum = 0.0f;
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    const float p0 = ex2(fmaf(s[mt][n][rr * 2], a.scale_log2, -mnew));
                    const float p1 = ex2(fmaf(s[mt][n][rr * 2 + 1], a.scale_log2, -mnew));
                    s[mt][n][rr * 2] = p0;
                    s[mt][n][rr * 2 + 1] = p1;
                    sum += p0 + p1;
                }
                l[mt][rr] = l[mt][rr] * corr + sum;
#pragma unroll
                for (int d = 0; d < 8; ++d) {
                    o[mt][d][rr * 2] *= corr;
                    o[mt][d][rr * 2 + 1] *= corr;
                }
            }

        // O += P V: P's accumulator layout is the m16n8k16 A layout; V rows
        // are keys, so ldmatrix.trans delivers the col-major B operand.
        if constexpr (PV16) {
            unsigned ot[MT][8][2];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int d = 0; d < 8; ++d) ot[mt][d][0] = ot[mt][d][1] = 0u;
#pragma unroll
            for (int kk = 0; kk < BN / 16; ++kk) {
                unsigned pa[MT][4];
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    pa[mt][0] = pack_f16(s[mt][2 * kk][0], s[mt][2 * kk][1]);
                    pa[mt][1] = pack_f16(s[mt][2 * kk][2], s[mt][2 * kk][3]);
                    pa[mt][2] = pack_f16(s[mt][2 * kk + 1][0], s[mt][2 * kk + 1][1]);
                    pa[mt][3] = pack_f16(s[mt][2 * kk + 1][2], s[mt][2 * kk + 1][3]);
                }
#pragma unroll
                for (int d = 0; d < 8; d += 2) {
                    unsigned b0, b1, b2, b3;
                    ldsm_x4_t(smem_addr(vt + swz(kk * 16 + (mi & 1) * 8 + mr, d + (mi >> 1))), b0, b1, b2, b3);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        mma16816_f16(ot[mt][d], pa[mt], b0, b1);
                        mma16816_f16(ot[mt][d + 1], pa[mt], b2, b3);
                    }
                }
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int d = 0; d < 8; ++d) {
                    float x0, x1, x2, x3;
                    unpack_f16(ot[mt][d][0], x0, x1);
                    unpack_f16(ot[mt][d][1], x2, x3);
                    o[mt][d][0] += x0;
                    o[mt][d][1] += x1;
                    o[mt][d][2] += x2;
                    o[mt][d][3] += x3;
                }
        } else {
#pragma unroll
        for (int kk = 0; kk < BN / 16; ++kk) {
            unsigned pa[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                pa[mt][0] = pack_bf16(s[mt][2 * kk][0], s[mt][2 * kk][1]);
                pa[mt][1] = pack_bf16(s[mt][2 * kk][2], s[mt][2 * kk][3]);
                pa[mt][2] = pack_bf16(s[mt][2 * kk + 1][0], s[mt][2 * kk + 1][1]);
                pa[mt][3] = pack_bf16(s[mt][2 * kk + 1][2], s[mt][2 * kk + 1][3]);
            }
#pragma unroll
            for (int d = 0; d < 8; d += 2) {
                unsigned b0, b1, b2, b3;
                ldsm_x4_t(smem_addr(vt + swz(kk * 16 + (mi & 1) * 8 + mr, d + (mi >> 1))), b0, b1, b2, b3);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    mma16816(o[mt][d], pa[mt], b0, b1);
                    mma16816(o[mt][d + 1], pa[mt], b2, b3);
                }
            }
        }
        }

        __syncthreads();  // every warp is done with this stage
        if (j + 2 < tiles) {
            load_tile<BN, THREADS>(stages + (j & 1) * STAGE, K, (j + 2) * BN, L, tid);
            load_tile<BN, THREADS>(stages + (j & 1) * STAGE + BN * HD, V, (j + 2) * BN, L, tid);
        }
        cp_async_commit();
    }

    // Normalise and stage the warp's rows in its own part of sQ, then write
    // 16-byte chunks to global memory. Without QA only this warp read those
    // rows; with QA the loop's last barrier retired every read of stage 1
    // and no load into it is pending.
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            float sum = l[mt][rr];
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            const float inv = 1.0f / sum;
            const int r = (warp * MT + mt) * 16 + (lane >> 2) + rr * 8;
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                const unsigned v = pack_bf16(o[mt][d][rr * 2] * inv, o[mt][d][rr * 2 + 1] * inv);
                *(unsigned*)(sQ + swz(r, d) + (lane & 3) * 2) = v;
            }
        }
    __syncthreads();
    bf16* O = (bf16*)a.o + ((size_t)seq * L) * a.o_ld + h * HD;
#pragma unroll
    for (int i = 0; i < BM * 8 / THREADS; ++i) {
        const int c = tid + i * THREADS;
        const int r = c >> 3, ch = c & 7;
        if (q0 + r < L) *(uint4*)(O + (size_t)(q0 + r) * a.o_ld + ch * 8) = *(const uint4*)(sQ + swz(r, ch));
    }
}

// One extern "C" kernel per compiled tile; names match `Tile::kernel_name`.
#define ATTN(NAME, W, MT, BN, QA, PV16)                                               \
    extern "C" __global__ void __launch_bounds__(W * 32) NAME(AttnArgs a) {          \
        attn_fwd<W, MT, BN, QA, PV16>(a);                                             \
    }

ATTN(attn_w4_m1_n64, 4, 1, 64, false, false)
ATTN(attn_w4_m1_n128, 4, 1, 128, false, false)
ATTN(attn_w4_m2_n32, 4, 2, 32, false, false)
ATTN(attn_w4_m2_n64, 4, 2, 64, false, false)
ATTN(attn_w8_m1_n64, 8, 1, 64, false, false)
ATTN(attn_w8_m1_n128, 8, 1, 128, false, false)
ATTN(attn_w2_m2_n64, 2, 2, 64, false, false)
ATTN(attn_w4_m1_n64_qa, 4, 1, 64, true, false)
ATTN(attn_w4_m2_n64_qa, 4, 2, 64, true, false)
ATTN(attn_w8_m1_n64_qa, 8, 1, 64, true, false)
ATTN(attn_w2_m2_n64_qa, 2, 2, 64, true, false)
ATTN(attn_w4_m1_n64_qa_pv16, 4, 1, 64, true, true)
ATTN(attn_w4_m2_n64_pv16, 4, 2, 64, false, true)
ATTN(attn_w8_m1_n64_pv16, 8, 1, 64, false, true)
ATTN(attn_w8_m1_n64_qa_pv16, 8, 1, 64, true, true)
