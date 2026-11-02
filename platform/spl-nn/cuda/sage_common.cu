// SageAttention2-style quantised attention: shared definitions. Owner: nn-sage.
//
// src/sage/mod.rs concatenates this file, cuda/sage_quant.cu and
// cuda/sage_attn.cu into one NVRTC source and compiles it for compute_89 or
// newer (FP8 conversions and FP8 mma.sync need sm_89). SG_D (head dim, 64 or
// 128) and the attention tile shape arrive as -D options. Header-free: bf16
// travels as unsigned short, FP8 E4M3 as unsigned char, and every helper is
// prefixed sg_ so the file can sit next to other spl-nn sources.
//
// Algorithm: SageAttention2 (Zhang et al., arXiv 2411.10958; reference code
// github.com/thu-ml/SageAttention, Apache-2.0). This is an independent
// implementation; the FP8 P-fragment key permutation and the two-level PV
// accumulation follow the paper and that code.

typedef unsigned char sg_u8;
typedef signed char sg_i8;
typedef unsigned short sg_u16;
typedef unsigned int sg_u32;
typedef unsigned long long sg_u64;

#ifndef SG_D
#define SG_D 64
#endif

// Flags in SgGeom::flags; mirror src/sage/mod.rs.
#define SG_LN_Q 1
#define SG_LN_K 2
#define SG_SMOOTH_K 4
#define SG_SMOOTH_V 8

// Rows per Q quantisation group (one mma m-tile, the rows one warp owns),
// queries per prologue tile and keys per K/V block. Mirror src/sage/mod.rs.
#define SG_QGROUP 16
#define SG_QTILE 128
#define SG_KBLOCK 64

// Geometry and options shared by every sage kernel; mirrors `Geom` in
// src/sage/mod.rs field for field (#[repr(C)]).
struct SgGeom {
    // Element strides of the bf16 inputs and of the output; channels are
    // contiguous. Q, K and V may alias one packed [B][T][3][H][D] buffer.
    long long q_sb, q_st, q_sh;
    long long k_sb, k_st, k_sh;
    long long v_sb, v_st, v_sh;
    long long o_sb, o_st, o_sh;
    int batch, heads;
    int tokens;        // T: queries and (unmerged) keys per batch item
    int tq_pad;        // T rounded up to SG_QTILE
    int tk;            // keys after merging (T when dense)
    int tk_pad;        // tk rounded up to SG_KBLOCK
    int merge;         // pooling stride s: 1 dense, 2 or 4
    int frames;        // frames per batch item (merge only)
    int frame_tokens;  // special + grid_h * grid_w
    int special;       // camera and register tokens at the start of each frame
    int grid_h, grid_w;
    int merged_frame;  // special + ceil(grid_h / s) * ceil(grid_w / s)
    int merged_w;      // ceil(grid_w / s)
    int q_tiles;       // tq_pad / SG_QTILE: stat partials per (batch, head)
    int flags;
    float q_mul;       // softmax scale * log2(e), folded into the Q scales
    float ln_eps;
    float v_range;     // V is scaled so max |V - mean| maps to this
    int dense_frames;  // leading frames kept unmerged
};

__device__ __forceinline__ float sg_bf16(sg_u32 bits16) { return __uint_as_float(bits16 << 16); }

__device__ __forceinline__ void sg_unpack8(uint4 v, float* f) {
    f[0] = __uint_as_float(v.x << 16); f[1] = __uint_as_float(v.x & 0xffff0000u);
    f[2] = __uint_as_float(v.y << 16); f[3] = __uint_as_float(v.y & 0xffff0000u);
    f[4] = __uint_as_float(v.z << 16); f[5] = __uint_as_float(v.z & 0xffff0000u);
    f[6] = __uint_as_float(v.w << 16); f[7] = __uint_as_float(v.w & 0xffff0000u);
}

// Two floats -> bf16x2, round to nearest even; lo in the low half.
__device__ __forceinline__ sg_u32 sg_pack_bf16(float lo, float hi) {
    sg_u32 r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// Two floats -> two FP8 E4M3 (RNE, saturating at +-448); lo in the low byte.
__device__ __forceinline__ sg_u32 sg_e4m3x2(float lo, float hi) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return (sg_u32)r;
}

// Four floats -> four E4M3 bytes, a in the lowest byte.
__device__ __forceinline__ sg_u32 sg_e4m3x4(float a, float b, float c, float d) {
    return sg_e4m3x2(a, b) | (sg_e4m3x2(c, d) << 16);
}

// Round to nearest even and clamp to the symmetric INT8 range.
__device__ __forceinline__ sg_u32 sg_s8(float x) {
    int q = __float2int_rn(x);
    q = max(-127, min(127, q));
    return (sg_u32)(q & 0xff);
}

__device__ __forceinline__ sg_u32 sg_s8x4(float a, float b, float c, float d) {
    return sg_s8(a) | (sg_s8(b) << 8) | (sg_s8(c) << 16) | (sg_s8(d) << 24);
}

__device__ __forceinline__ float sg_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float sg_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Shared-memory address (32-bit) of a generic pointer into shared memory.
__device__ __forceinline__ sg_u32 sg_smem(const void* p) {
    sg_u32 a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
    return a;
}

__device__ __forceinline__ void sg_cp16(sg_u32 dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void sg_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void sg_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void sg_ldsm4(sg_u32 addr, sg_u32& r0, sg_u32& r1, sg_u32& r2, sg_u32& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// Byte offset of 16-byte chunk `chunk` of row `row` in a tile whose rows are
// CPR chunks long, XOR-swizzled within each 128-byte line so that ldmatrix
// (8 rows, one chunk column) and the cp.async fills are bank-conflict free.
template <int CPR>
__device__ __forceinline__ sg_u32 sg_swz(int row, int chunk) {
    const int u = row * CPR + chunk;
    const int line = u >> 3;
    return (sg_u32)((line << 7) | (((u & 7) ^ (line & 7)) << 4));
}

// Position of key `k` (0..15) inside its 16-key group of the transposed FP8 V
// tile. The P fragment of mma m16n8k32 built from the S accumulators puts
// keys {2t, 2t+1, 8+2t, 9+2t} at k = 4t..4t+3 (t = lane % 4), so V is stored
// with key 8h + 2t + e at position 4t + 2h + e.
__device__ __forceinline__ int sg_vpos(int k) { return ((k >> 1) & 3) * 4 + ((k >> 3) << 1) + (k & 1); }
