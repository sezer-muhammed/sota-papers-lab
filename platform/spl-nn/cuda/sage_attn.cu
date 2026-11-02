// SageAttention2-style attention over the tensors cuda/sage_quant.cu wrote.
// Owner: nn-sage.
//
// One CTA owns SG_BM = 16 * SG_MT * SG_NW query rows of one (batch, head);
// each warp owns SG_MT mma m-tiles (16 rows each, one Q scale per m-tile).
// The CTA walks the keys in 64-key blocks through an SG_STAGES-deep cp.async
// ring of (INT8 K, FP8 V^T) tiles in XOR-swizzled shared memory:
//
//   S  = Q8 K8^T            mma.sync m16n8k32 s8 -> s32, Q fragments stay in
//                           registers for the whole sweep
//   x  = S * sq * sk (+ log2 count when merged); sq carries softmax scale and
//        log2(e), so the softmax runs in exp2 (ex2.approx)
//   online softmax per row: running max m, running sum l
//   P  = 448 * exp2(x - m)  packed to FP8 E4M3 straight from the accumulator
//                           layout (V keys are permuted to match)
//   O  = O * exp2(m_old - m) + P V8    mma.sync m16n8k32 e4m3; each block's
//        PV starts from zero (FP32, or FP16 with SG_PV16) and is folded into
//        the FP32 O with one FFMA: two-level accumulation, since Ada's FP8
//        mma accumulates with fewer than 32 bits and never sums > 64 keys here
//   out = O / l * v_scale[d] + v_mean[d]   (448 cancels between O and l)
//
// Instruction budget matters as much as tensor rate on this card, so:
// - the INT8 accumulators start at 0x4B400000 = bits of 1.5 * 2^23, which
//   makes every S register already the float 1.5 * 2^23 + S (exact while
//   |S| < 2^22, true for D <= 128 and |q8|, |k8| <= 127): no conversions;
// - the first mma of every accumulator reads C from one shared constant
//   register quad instead of a zeroed copy;
// - SG_FAST_EXP folds the 1.5 * 2^23 offset into the exp2 FFMA (dense) or
//   into the per-key bias (merged); that offset carries an fp32 rounding of
//   ulp(1.5 * 2^23 * sc) / 2 into each (row, block) logit, a few 1e-4 in
//   log2 units at VGGT scales (the measured error does not move);
// - the last key block is peeled, so only it pays for the padding mask.
//
// Measured on an RTX 4070 SUPER (200 frames x 689 tokens, 16 heads of 64):
// 176 TOPS with FP32 PV (84% of the 209 TOPS INT8/FP8-f32 mixed peak) and
// 209 TOPS with FP16 PV; FlashAttention-2 bf16 reaches 72 TFLOPS.

#ifndef SG_MT
#define SG_MT 2
#endif
#ifndef SG_NW
#define SG_NW 4
#endif
#ifndef SG_STAGES
#define SG_STAGES 3
#endif
// 1: each block's PV accumulates in FP16 (SageAttention2++, twice the FP32
// rate on GeForce Ada); the prologue then scales V to +-2.25 so 64 keys of
// P (<= 448) times V stay below the FP16 maximum.
#ifndef SG_PV16
#define SG_PV16 0
#endif
// 1: S for every m-tile, then softmax + PV per m-tile (each warp's softmax
// overlaps its own mma); 0: S, softmax and PV each over all m-tiles. 0 is
// faster on Ada (163 vs 154 TOPS: 1 reloads K and V fragments per m-tile).
#ifndef SG_SPLIT
#define SG_SPLIT 0
#endif
#ifndef SG_FAST_EXP
#define SG_FAST_EXP 1
#endif
#define SG_BM (16 * SG_MT * SG_NW)
#define SG_BN SG_KBLOCK
#define SG_MAGIC_I 0x4B400000
// Head dim 128: ask for two CTAs per SM (up to 255 registers a thread);
// otherwise ptxas aims for three and spills. At 64 the hint costs 6%.
#define SG_MINB ((SG_D > 64 && SG_NW <= 4) ? 2 : 1)

template <bool B>
struct sg_bool {
    static constexpr bool value = B;
};

__device__ __forceinline__ void sg_mma_s8(int* c, const sg_u32* a, const sg_u32* b) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// d = a b + {c, c, c, c}: C from one register shared by every tile.
__device__ __forceinline__ void sg_mma_s8_c(int* d, const sg_u32* a, const sg_u32* b, int c) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c));
}

__device__ __forceinline__ void sg_mma_e4m3(float* c, const sg_u32* a, const sg_u32* b) {
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void sg_mma_e4m3_c(float* d, const sg_u32* a, const sg_u32* b, float c) {
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c));
}

// FP16 accumulate: C is two half2, row g and row g + 8, columns 2t and 2t + 1.
__device__ __forceinline__ void sg_mma_e4m3_f16(sg_u32* c, const sg_u32* a, const sg_u32* b) {
    asm("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void sg_mma_e4m3_f16_c(sg_u32* d, const sg_u32* a, const sg_u32* b, sg_u32 c) {
    asm("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %8};\n"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c));
}

__device__ __forceinline__ void sg_h2f(sg_u32 h, float& lo, float& hi) {
    asm("{ .reg .f16 l, u; mov.b32 {l, u}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, u; }"
        : "=f"(lo), "=f"(hi)
        : "r"(h));
}

// Four floats -> four E4M3 in one register, a lowest; the halves are
// assembled by register allocation instead of shifts.
__device__ __forceinline__ sg_u32 sg_p8x4(float a, float b, float c, float d) {
    sg_u32 r;
    asm("{ .reg .b16 lo, hi;\n"
        "cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n"
        "cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
        "mov.b32 %0, {lo, hi}; }"
        : "=r"(r)
        : "f"(a), "f"(b), "f"(c), "f"(d));
    return r;
}

template <bool MERGE>
__device__ __forceinline__ void sg_attn(const sg_i8* __restrict__ q8, const float* __restrict__ q_scale,
                                        const sg_i8* __restrict__ k8, const float* __restrict__ k_scale,
                                        const sg_u8* __restrict__ v8, const float* __restrict__ stats,
                                        const float* __restrict__ kbias, sg_u16* __restrict__ out,
                                        const SgGeom& g) {
    constexpr int D = SG_D, MT = SG_MT, NW = SG_NW, ST = SG_STAGES, BM = SG_BM, BN = SG_BN;
    constexpr int CPR = D / 16;   // 16-byte chunks per Q or K row
    constexpr int VPR = BN / 16;  // 16-byte chunks per V^T row
    constexpr int KT = D / 32;    // k-steps of Q K^T
    constexpr int NT = BN / 8;    // n-tiles of S
    constexpr int DT = D / 8;     // n-tiles of O
    constexpr int KC = BN / 32;   // k-steps of P V
    constexpr int TILE = BN * D;  // bytes of one K tile and of one V tile
    constexpr int NTH = NW * 32;
    static_assert(KT % 2 == 0 && KC % 2 == 0, "ldmatrix.x4 covers two k-steps");

    extern __shared__ __align__(128) sg_u8 smem[];
    const sg_u32 sq = sg_smem(smem);
    const sg_u32 skv = sq + BM * D;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, t4 = lane & 3;
    const int qb = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const long long bh = (long long)b * g.heads + h;
    const int nblk = g.tk_pad / BN;
    const sg_i8* qg = q8 + (bh * g.tq_pad + (long long)qb * BM) * D;
    const sg_i8* kg = k8 + bh * g.tk_pad * (long long)D;
    const sg_u8* vg = v8 + bh * g.tk_pad * (long long)D;
    const float* ksc = k_scale + bh * nblk;

    // cp.async slots: thread tid copies 16-byte chunks c = tid + i * NTH of a
    // tile. The swizzled offset of chunk c depends on c alone (whatever the
    // row length), and adding i * NTH (a multiple of 128 chunks) moves it by
    // whole 2 KB, so one per-thread offset serves every tile and slot.
    static_assert((BM * CPR) % NTH == 0 && (BN * CPR) % NTH == 0 && (D * VPR) % NTH == 0, "whole slots");
    const sg_u32 soff = sg_swz<8>(tid >> 3, tid & 7);
    constexpr int SLOT = NTH * 16;
#pragma unroll
    for (int i = 0; i < BM * CPR / NTH; i++) sg_cp16(sq + soff + i * SLOT, qg + tid * 16 + i * SLOT);
    sg_cp_commit();

    auto load_kv = [&](int blk, int stage) {
        const sg_u32 sk = skv + stage * 2 * TILE + soff;
        const sg_i8* kgb = kg + (long long)blk * TILE + tid * 16;
        const sg_u8* vgb = vg + (long long)blk * TILE + tid * 16;
#pragma unroll
        for (int i = 0; i < BN * CPR / NTH; i++) sg_cp16(sk + i * SLOT, kgb + i * SLOT);
#pragma unroll
        for (int i = 0; i < D * VPR / NTH; i++) sg_cp16(sk + TILE + i * SLOT, vgb + i * SLOT);
    };

#pragma unroll
    for (int s = 0; s < ST - 1; s++) {
        if (s < nblk) load_kv(s, s);
        sg_cp_commit();
    }
    sg_cp_wait<ST - 1>();
    __syncthreads();

    sg_u32 qa[MT][KT][4];
    float qs[MT];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        const int row = (warp * MT + mt) * 16 + (lane & 15);
#pragma unroll
        for (int kk = 0; kk < KT; kk++)
            sg_ldsm4(sq + sg_swz<CPR>(row, kk * 2 + (lane >> 4)), qa[mt][kk][0], qa[mt][kk][1], qa[mt][kk][2],
                     qa[mt][kk][3]);
        qs[mt] = q_scale[bh * (g.tq_pad / SG_QGROUP) + (qb * BM) / SG_QGROUP + warp * MT + mt];
    }

    float o[MT][DT][4];
    float mrow[MT][2], lrow[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
        for (int nt = 0; nt < DT; nt++) o[mt][nt][0] = o[mt][nt][1] = o[mt][nt][2] = o[mt][nt][3] = 0.f;
        mrow[mt][0] = mrow[mt][1] = -1.0e30f;
        lrow[mt][0] = lrow[mt][1] = 0.f;
    }
    const float kMagic = 12582912.0f;          // 1.5 * 2^23
    const float kLog448 = 8.807354922057604f;  // log2(448): P fills the E4M3 range
    const float kNegInf = __int_as_float(0xff800000);
    // Shared C operands for the first mma of each accumulator.
    const int cMagic = SG_MAGIC_I;
    const float cZero = 0.f;
    const sg_u32 cZero16 = 0u;
    (void)cZero;
    (void)cZero16;

    // One 64-key block; TAIL masks keys past g.tk (only the dense kernel's
    // last block can hold padding: the merge bias is -inf there).
    auto step = [&](int blk, int stage, auto tail_tag) {
        constexpr bool TAIL = decltype(tail_tag)::value;
        sg_cp_wait<ST - 2>();
        __syncthreads();
        {
            const int nb = blk + ST - 1;
            if (nb < nblk) load_kv(nb, stage == 0 ? ST - 1 : stage - 1);
            sg_cp_commit();
        }
        const sg_u32 sk = skv + stage * 2 * TILE, sv = sk + TILE;
        const float kscale = __ldg(ksc + blk);
        float bias[NT][2];
        if (MERGE) {
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const float2 bb = __ldg(reinterpret_cast<const float2*>(kbias + blk * BN + nt * 8 + 2 * t4));
                bias[nt][0] = bb.x;
                bias[nt][1] = bb.y;
            }
        }
        // S, biased by 1.5 * 2^23: s[][][i] holds the float bits of kMagic + S.
        int s[MT][NT][4];
        auto load_k = [&](int nt, sg_u32 (&kb)[KT][2]) {
#pragma unroll
            for (int x = 0; x < KT / 2; x++)
                sg_ldsm4(sk + sg_swz<CPR>(nt * 8 + (lane & 7), x * 4 + (lane >> 3)), kb[2 * x][0], kb[2 * x][1],
                         kb[2 * x + 1][0], kb[2 * x + 1][1]);
        };
        auto qk = [&](int mt, int nt, const sg_u32 (&kb)[KT][2]) {
            sg_mma_s8_c(s[mt][nt], qa[mt][0], kb[0], cMagic);
#pragma unroll
            for (int kk = 1; kk < KT; kk++) sg_mma_s8(s[mt][nt], qa[mt][kk], kb[kk]);
        };

        // Online softmax of m-tile mt; P as FP8 A fragments.
        auto softmax = [&](int mt, sg_u32 (&pa)[KC][4], float (&alpha)[2]) {
            const float sc = qs[mt] * kscale;
            float x[NT][4];
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    x[nt][i] = __int_as_float(s[mt][nt][i]);
                    if (!SG_FAST_EXP) x[nt][i] -= kMagic;  // exact
                    // Merged: the log2 logit, with the offset folded into the bias.
                    if (MERGE)
                        x[nt][i] = fmaf(x[nt][i], sc, SG_FAST_EXP ? bias[nt][i & 1] - kMagic * sc : bias[nt][i & 1]);
                }
            if (TAIL) {
#pragma unroll
                for (int nt = 0; nt < NT; nt++)
#pragma unroll
                    for (int i = 0; i < 4; i++)
                        if (blk * BN + nt * 8 + 2 * t4 + (i & 1) >= g.tk) x[nt][i] = kNegInf;
            }
            // Dense: x is S (+ kMagic with SG_FAST_EXP) and the scale is
            // applied inside the exp2 FFMA; merged: x is the log2 logit.
            const float xoff = MERGE ? 0.f : (SG_FAST_EXP ? -kMagic * sc : 0.f);
#pragma unroll
            for (int r = 0; r < 2; r++) {
                float mx = fmaxf(x[0][2 * r], x[0][2 * r + 1]);
#pragma unroll
                for (int nt = 1; nt < NT; nt++) mx = fmaxf(mx, fmaxf(x[nt][2 * r], x[nt][2 * r + 1]));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                const float mnew = fmaxf(mrow[mt][r], MERGE ? mx : fmaf(mx, sc, xoff));
                alpha[r] = sg_ex2(mrow[mt][r] - mnew);
                mrow[mt][r] = mnew;
                const float off = kLog448 - mnew + xoff;
                float sum = 0.f;
#pragma unroll
                for (int nt = 0; nt < NT; nt++)
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        const float y = x[nt][2 * r + e];
                        const float p = sg_ex2(MERGE ? y + off : fmaf(y, sc, off));
                        x[nt][2 * r + e] = p;
                        sum += p;
                    }
                lrow[mt][r] = fmaf(lrow[mt][r], alpha[r], sum);
            }
#pragma unroll
            for (int kc = 0; kc < KC; kc++) {
                const int n0 = 4 * kc;
                pa[kc][0] = sg_p8x4(x[n0][0], x[n0][1], x[n0 + 1][0], x[n0 + 1][1]);
                pa[kc][1] = sg_p8x4(x[n0][2], x[n0][3], x[n0 + 1][2], x[n0 + 1][3]);
                pa[kc][2] = sg_p8x4(x[n0 + 2][0], x[n0 + 2][1], x[n0 + 3][0], x[n0 + 3][1]);
                pa[kc][3] = sg_p8x4(x[n0 + 2][2], x[n0 + 2][3], x[n0 + 3][2], x[n0 + 3][3]);
            }
        };

        auto load_v = [&](int nt, sg_u32 (&vb)[KC][2]) {
#pragma unroll
            for (int x = 0; x < KC / 2; x++)
                sg_ldsm4(sv + sg_swz<VPR>(nt * 8 + (lane & 7), x * 4 + (lane >> 3)), vb[2 * x][0], vb[2 * x][1],
                         vb[2 * x + 1][0], vb[2 * x + 1][1]);
        };
        // O = O * alpha + P V for one (m-tile, channel tile): this block's PV
        // starts from zero and folds into the FP32 O with one FFMA.
        auto pv = [&](int mt, int nt, const sg_u32 (&pa)[KC][4], const sg_u32 (&vb)[KC][2], const float (&alpha)[2]) {
            float c[4];
#if SG_PV16
            sg_u32 ch[2];
            sg_mma_e4m3_f16_c(ch, pa[0], vb[0], cZero16);
#pragma unroll
            for (int kc = 1; kc < KC; kc++) sg_mma_e4m3_f16(ch, pa[kc], vb[kc]);
            sg_h2f(ch[0], c[0], c[1]);
            sg_h2f(ch[1], c[2], c[3]);
#else
            sg_mma_e4m3_c(c, pa[0], vb[0], cZero);
#pragma unroll
            for (int kc = 1; kc < KC; kc++) sg_mma_e4m3(c, pa[kc], vb[kc]);
#endif
            o[mt][nt][0] = fmaf(o[mt][nt][0], alpha[0], c[0]);
            o[mt][nt][1] = fmaf(o[mt][nt][1], alpha[0], c[1]);
            o[mt][nt][2] = fmaf(o[mt][nt][2], alpha[1], c[2]);
            o[mt][nt][3] = fmaf(o[mt][nt][3], alpha[1], c[3]);
        };

#if SG_SPLIT
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                sg_u32 kb[KT][2];
                load_k(nt, kb);
                qk(mt, nt, kb);
            }
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            sg_u32 pa[KC][4];
            float alpha[2];
            softmax(mt, pa, alpha);
#pragma unroll
            for (int nt = 0; nt < DT; nt++) {
                sg_u32 vb[KC][2];
                load_v(nt, vb);
                pv(mt, nt, pa, vb, alpha);
            }
        }
#else
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            sg_u32 kb[KT][2];
            load_k(nt, kb);
#pragma unroll
            for (int mt = 0; mt < MT; mt++) qk(mt, nt, kb);
        }
        sg_u32 pa[MT][KC][4];
        float alpha[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; mt++) softmax(mt, pa[mt], alpha[mt]);
#pragma unroll
        for (int nt = 0; nt < DT; nt++) {
            sg_u32 vb[KC][2];
            load_v(nt, vb);
#pragma unroll
            for (int mt = 0; mt < MT; mt++) pv(mt, nt, pa[mt], vb, alpha[mt]);
        }
#endif
    };

    // Every block but the last runs unmasked; the last is masked only when
    // the dense key count is not a multiple of 64.
    int stage = 0;
    for (int blk = 0; blk < nblk - 1; blk++) {
        step(blk, stage, sg_bool<false>{});
        stage = stage == ST - 1 ? 0 : stage + 1;
    }
    if (!MERGE && g.tk % BN != 0)
        step(nblk - 1, stage, sg_bool<true>{});
    else
        step(nblk - 1, stage, sg_bool<false>{});
    sg_cp_wait<0>();

    const float* vmean = stats + bh * 4 * D + D;
    const float* vscale = vmean + D;
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            float l = lrow[mt][r];
            l += __shfl_xor_sync(0xffffffffu, l, 1);
            l += __shfl_xor_sync(0xffffffffu, l, 2);
            const float inv = 1.0f / l;
            const int row = qb * BM + (warp * MT + mt) * 16 + (lane >> 2) + 8 * r;
            if (row >= g.tokens) continue;
            sg_u16* orow = out + b * g.o_sb + (long long)row * g.o_st + h * g.o_sh;
#pragma unroll
            for (int nt = 0; nt < DT; nt++) {
                const int d = nt * 8 + 2 * t4;
                const float y0 = fmaf(o[mt][nt][2 * r] * inv, vscale[d], vmean[d]);
                const float y1 = fmaf(o[mt][nt][2 * r + 1] * inv, vscale[d + 1], vmean[d + 1]);
                *reinterpret_cast<sg_u32*>(orow + d) = sg_pack_bf16(y0, y1);
            }
        }
    }
}

// grid (tq_pad / SG_BM, heads, batch), SG_NW * 32 threads,
// SG_BM * D + SG_STAGES * 2 * 64 * D bytes of dynamic shared memory.
extern "C" __global__ void __launch_bounds__(SG_NW * 32, SG_MINB)
    sg_attn_dense(const sg_i8* __restrict__ q8, const float* __restrict__ q_scale, const sg_i8* __restrict__ k8,
                  const float* __restrict__ k_scale, const sg_u8* __restrict__ v8, const float* __restrict__ stats,
                  const float* __restrict__ kbias, sg_u16* __restrict__ out, const SgGeom g) {
    sg_attn<false>(q8, q_scale, k8, k_scale, v8, stats, kbias, out, g);
}

extern "C" __global__ void __launch_bounds__(SG_NW * 32, SG_MINB)
    sg_attn_merge(const sg_i8* __restrict__ q8, const float* __restrict__ q_scale, const sg_i8* __restrict__ k8,
                  const float* __restrict__ k_scale, const sg_u8* __restrict__ v8, const float* __restrict__ stats,
                  const float* __restrict__ kbias, sg_u16* __restrict__ out, const SgGeom g) {
    sg_attn<true>(q8, q_scale, k8, k_scale, v8, stats, kbias, out, g);
}
