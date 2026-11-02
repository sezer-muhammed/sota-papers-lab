// Stage D: sparse-voxel-hash TSDF fusion and marching cubes. Owner: mesh-tsdf.
//
// Voxel blocks of 8^3 samples; block coordinates hash (open addressing,
// linear probing, 64-bit keys) to a pool index. Sample g sits at
// origin + voxel * g. Conventions (2DGS / Open3D ScalableTSDFVolume):
// projective sdf = depth(pixel) - z_cam, tsdf = min(1, sdf / trunc), voxels
// with sdf < -trunc untouched, weight 1 per observation, capped running mean.
// Pixel (u, v) covers [u, u+1) x [v, v+1) (COLMAP), so x = fx X/Z + cx.

#define BLK 8
#define BLK3 512
#define EMPTY_KEY 0xFFFFFFFFFFFFFFFFull

struct TsdfCam {
    float r[9];  // world to camera, row-major
    float t[3];
    float fx, fy, cx, cy;
    int width, height;
    float depth_min, depth_max;
};

__device__ __forceinline__ unsigned long long pack_key(int x, int y, int z) {
    return ((unsigned long long)(x + (1 << 20)) << 42) |
           ((unsigned long long)(y + (1 << 20)) << 21) |
           (unsigned long long)(z + (1 << 20));
}

__device__ __forceinline__ unsigned int hash_key(unsigned long long k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ull;
    k ^= k >> 33;
    return (unsigned int)k;
}

__device__ int hash_find(const unsigned long long* keys, const int* vals, unsigned int mask,
                         int x, int y, int z) {
    unsigned long long k = pack_key(x, y, z);
    unsigned int h = hash_key(k) & mask;
    for (unsigned int i = 0; i <= mask; i++) {
        unsigned long long c = keys[h];
        if (c == k) return vals[h];
        if (c == EMPTY_KEY) return -1;
        h = (h + 1) & mask;
    }
    return -1;
}

// counters: [0] blocks allocated, [1] overflow flags (1 pool, 2 table).
__device__ void hash_insert(unsigned long long* keys, int* vals, unsigned int mask, int* coords,
                            unsigned int* counters, unsigned int max_blocks, int x, int y, int z) {
    unsigned long long k = pack_key(x, y, z);
    unsigned int h = hash_key(k) & mask;
    for (unsigned int i = 0; i <= mask; i++) {
        unsigned long long c = keys[h];
        if (c == k) return;
        if (c == EMPTY_KEY) {
            c = atomicCAS(&keys[h], EMPTY_KEY, k);
            if (c == EMPTY_KEY) {
                unsigned int idx = atomicAdd(&counters[0], 1u);
                if (idx < max_blocks) {
                    coords[3 * idx + 0] = x;
                    coords[3 * idx + 1] = y;
                    coords[3 * idx + 2] = z;
                    vals[h] = (int)idx;
                } else {
                    atomicOr(&counters[1], 1u);
                }
                return;
            }
            if (c == k) return;
        }
        h = (h + 1) & mask;
    }
    atomicOr(&counters[1], 2u);
}

// One thread per pixel: allocates every block the band [d - trunc, d + trunc]
// along the pixel's ray passes through (steps of half a block).
extern "C" __global__ void tsdf_alloc(const float* __restrict__ depth, TsdfCam cam, float voxel,
                                      float trunc, float ox, float oy, float oz,
                                      unsigned long long* keys, int* vals, unsigned int mask,
                                      int* coords, unsigned int* counters,
                                      unsigned int max_blocks) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= cam.width || v >= cam.height) return;
    float d = depth[v * cam.width + u];
    if (!(d > cam.depth_min && d < cam.depth_max)) return;
    float xc = (u + 0.5f - cam.cx) / cam.fx, yc = (v + 0.5f - cam.cy) / cam.fy;
    float len = sqrtf(xc * xc + yc * yc + 1.f);
    float bsz = voxel * BLK;
    float z0 = fmaxf(d - trunc, cam.depth_min), z1 = d + trunc;
    float step = 0.5f * bsz / len;
    int n = (int)ceilf((z1 - z0) / step) + 1;
    const float* R = cam.r;
    int px = -2147483647 - 1, py = px, pz = px;
    for (int i = 0; i < n; i++) {
        float z = fminf(z0 + i * step, z1);
        float a = xc * z - cam.t[0], b = yc * z - cam.t[1], c = z - cam.t[2];
        // p_w = R^T (p_c - t)
        float wx = R[0] * a + R[3] * b + R[6] * c - ox;
        float wy = R[1] * a + R[4] * b + R[7] * c - oy;
        float wz = R[2] * a + R[5] * b + R[8] * c - oz;
        // Block of the nearest sample, so the 8 cube corners around a point
        // are covered when the band straddles a block face.
        int bx = (int)floorf(wx / bsz), by = (int)floorf(wy / bsz), bz = (int)floorf(wz / bsz);
        if (bx == px && by == py && bz == pz) continue;
        px = bx; py = by; pz = bz;
        hash_insert(keys, vals, mask, coords, counters, max_blocks, bx, by, bz);
    }
}

// One CUDA block per voxel block, one thread per sample.
extern "C" __global__ void tsdf_integrate(const float* __restrict__ depth, TsdfCam cam, float voxel,
                                          float trunc, float max_weight, float ox, float oy,
                                          float oz, const int* __restrict__ coords,
                                          float* tsdf, float* weight) {
    unsigned int b = blockIdx.x;
    int t = threadIdx.x;
    int gx = coords[3 * b] * BLK + (t & 7);
    int gy = coords[3 * b + 1] * BLK + ((t >> 3) & 7);
    int gz = coords[3 * b + 2] * BLK + (t >> 6);
    float x = ox + voxel * gx, y = oy + voxel * gy, z = oz + voxel * gz;
    const float* R = cam.r;
    float zc = R[6] * x + R[7] * y + R[8] * z + cam.t[2];
    if (!(zc > cam.depth_min)) return;
    float xc = R[0] * x + R[1] * y + R[2] * z + cam.t[0];
    float yc = R[3] * x + R[4] * y + R[5] * z + cam.t[1];
    float uf = cam.fx * xc / zc + cam.cx, vf = cam.fy * yc / zc + cam.cy;
    if (!(uf >= 0.f && vf >= 0.f && uf < (float)cam.width && vf < (float)cam.height)) return;
    float d = depth[(int)vf * cam.width + (int)uf];
    if (!(d > cam.depth_min && d < cam.depth_max)) return;
    float sdf = d - zc;
    if (sdf < -trunc) return;
    float val = fminf(1.f, sdf / trunc);
    size_t i = (size_t)b * BLK3 + t;
    float w = weight[i];
    tsdf[i] = (tsdf[i] * w + val) / (w + 1.f);
    weight[i] = fminf(w + 1.f, max_weight);
}

// ---------------------------------------------------------------- marching cubes

struct Vol {
    const unsigned long long* keys;
    const int* vals;
    unsigned int mask;
    const float* tsdf;
    const float* weight;
    float min_w;
};

// Sample g; false when unallocated or unobserved.
__device__ __forceinline__ bool sample(const Vol& V, int own, int3 bc, int gx, int gy, int gz,
                                       float& out) {
    int bx = gx >> 3, by = gy >> 3, bz = gz >> 3;
    int b = (bx == bc.x && by == bc.y && bz == bc.z) ? own : hash_find(V.keys, V.vals, V.mask, bx, by, bz);
    if (b < 0) return false;
    size_t i = (size_t)b * BLK3 + ((gz & 7) * 64 + (gy & 7) * 8 + (gx & 7));
    if (!(V.weight[i] >= V.min_w)) return false;
    out = V.tsdf[i];
    return true;
}

__device__ __forceinline__ int3 block_of(const int* coords, unsigned int b) {
    return make_int3(coords[3 * b], coords[3 * b + 1], coords[3 * b + 2]);
}

// The 8 corner samples of the cube at g; false if any is missing/saturated.
__device__ bool cube(const Vol& V, int own, int3 bc, int gx, int gy, int gz, float max_abs,
                     float c[8]) {
    for (int k = 0; k < 8; k++) {
        if (!sample(V, own, bc, gx + (k & 1), gy + ((k >> 1) & 1), gz + ((k >> 2) & 1), c[k]))
            return false;
        if (!(fabsf(c[k]) < max_abs)) return false;
    }
    return true;
}

__device__ bool edge(const Vol& V, int own, int3 bc, int gx, int gy, int gz, int axis,
                     float max_abs, float& a, float& b) {
    if (!sample(V, own, bc, gx, gy, gz, a) || !(fabsf(a) < max_abs)) return false;
    if (!sample(V, own, bc, gx + (axis == 0), gy + (axis == 1), gz + (axis == 2), b) ||
        !(fabsf(b) < max_abs))
        return false;
    return (a < 0.f) != (b < 0.f);
}

#define MC_ARGS                                                                     \
    const unsigned long long *keys, const int *vals, unsigned int mask,             \
        const int *__restrict__ coords, const float *tsdf, const float *weight,     \
        float max_abs, float min_w

#define MC_SETUP                                         \
    Vol V{keys, vals, mask, tsdf, weight, min_w};        \
    unsigned int b = blockIdx.x;                         \
    int t = threadIdx.x;                                 \
    int3 bc = block_of(coords, b);                       \
    int gx = bc.x * BLK + (t & 7);                       \
    int gy = bc.y * BLK + ((t >> 3) & 7);                \
    int gz = bc.z * BLK + (t >> 6);

// counters: [0] vertices, [1] faces.
extern "C" __global__ void mc_count(MC_ARGS, const unsigned char* __restrict__ tri_count,
                                    unsigned int* counters) {
    MC_SETUP
    __shared__ unsigned int sv, sf;
    if (t == 0) { sv = 0; sf = 0; }
    __syncthreads();
    float c[8], a, e;
    unsigned int nv = 0, nf = 0;
    for (int ax = 0; ax < 3; ax++) nv += edge(V, b, bc, gx, gy, gz, ax, max_abs, a, e);
    if (cube(V, b, bc, gx, gy, gz, max_abs, c)) {
        int cfg = 0;
        for (int k = 0; k < 8; k++) cfg |= (c[k] < 0.f) << k;
        nf = tri_count[cfg];
    }
    if (nv) atomicAdd(&sv, nv);
    if (nf) atomicAdd(&sf, nf);
    __syncthreads();
    if (t == 0) {
        if (sv) atomicAdd(&counters[0], sv);
        if (sf) atomicAdd(&counters[1], sf);
    }
}

__device__ float3 grad(const Vol& V, int own, int3 bc, int gx, int gy, int gz, float c) {
    float g[3];
    for (int ax = 0; ax < 3; ax++) {
        int dx = ax == 0, dy = ax == 1, dz = ax == 2;
        float p, m;
        bool hp = sample(V, own, bc, gx + dx, gy + dy, gz + dz, p);
        bool hm = sample(V, own, bc, gx - dx, gy - dy, gz - dz, m);
        g[ax] = hp && hm ? 0.5f * (p - m) : hp ? p - c : hm ? c - m : 0.f;
    }
    return make_float3(g[0], g[1], g[2]);
}

// counters: [0] vertices emitted. edge_vid[b][sample][axis] = vertex or -1.
extern "C" __global__ void mc_vertices(MC_ARGS, float voxel, float ox, float oy, float oz,
                                       int* edge_vid, float* pos, float* nrm, unsigned int cap,
                                       unsigned int* counters) {
    MC_SETUP
    for (int ax = 0; ax < 3; ax++) {
        size_t slot = ((size_t)b * BLK3 + t) * 3 + ax;
        float a, e;
        if (!edge(V, b, bc, gx, gy, gz, ax, max_abs, a, e)) {
            edge_vid[slot] = -1;
            continue;
        }
        unsigned int vi = atomicAdd(&counters[0], 1u);
        if (vi >= cap) {
            edge_vid[slot] = -1;
            continue;
        }
        edge_vid[slot] = (int)vi;
        float s = a / (a - e);
        int dx = ax == 0, dy = ax == 1, dz = ax == 2;
        pos[3 * vi + 0] = ox + voxel * (gx + s * dx);
        pos[3 * vi + 1] = oy + voxel * (gy + s * dy);
        pos[3 * vi + 2] = oz + voxel * (gz + s * dz);
        float3 g0 = grad(V, b, bc, gx, gy, gz, a);
        float3 g1 = grad(V, b, bc, gx + dx, gy + dy, gz + dz, e);
        float nx = g0.x + s * (g1.x - g0.x), ny = g0.y + s * (g1.y - g0.y),
              nz = g0.z + s * (g1.z - g0.z);
        float inv = rsqrtf(fmaxf(nx * nx + ny * ny + nz * nz, 1e-30f));
        nrm[3 * vi + 0] = nx * inv;
        nrm[3 * vi + 1] = ny * inv;
        nrm[3 * vi + 2] = nz * inv;
    }
}

// counters: [1] faces emitted, [2] edges whose vertex could not be found.
extern "C" __global__ void mc_faces(MC_ARGS, const signed char* __restrict__ tris,
                                    const unsigned char* __restrict__ tri_count,
                                    const int* __restrict__ edges, const int* __restrict__ edge_vid,
                                    unsigned int* faces, unsigned int cap,
                                    unsigned int* counters) {
    MC_SETUP
    float c[8];
    if (!cube(V, b, bc, gx, gy, gz, max_abs, c)) return;
    int cfg = 0;
    for (int k = 0; k < 8; k++) cfg |= (c[k] < 0.f) << k;
    int n = tri_count[cfg];
    for (int k = 0; k < n; k++) {
        int id[3];
        bool ok = true;
        for (int j = 0; j < 3; j++) {
            const int* e = edges + 4 * tris[cfg * 32 + 3 * k + j];
            int ex = gx + e[0], ey = gy + e[1], ez = gz + e[2];
            int bx = ex >> 3, by = ey >> 3, bz = ez >> 3;
            int ob = (bx == bc.x && by == bc.y && bz == bc.z) ? (int)b
                                                               : hash_find(keys, vals, mask, bx, by, bz);
            id[j] = ob < 0 ? -1
                           : edge_vid[((size_t)ob * BLK3 + ((ez & 7) * 64 + (ey & 7) * 8 + (ex & 7))) * 3 + e[3]];
            ok &= id[j] >= 0;
        }
        if (!ok) {
            atomicAdd(&counters[2], 1u);
            continue;
        }
        unsigned int fi = atomicAdd(&counters[1], 1u);
        if (fi >= cap) continue;
        faces[3 * fi + 0] = id[0];
        faces[3 * fi + 1] = id[1];
        faces[3 * fi + 2] = id[2];
    }
}
