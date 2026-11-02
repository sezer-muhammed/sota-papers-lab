// Texture baking kernels for the mesh pipeline. Owner: mesh-tex.
// Compilation unit: common.cuh + mesh_bake.cu (no splat_common.cuh needed).
//
// Two kernels:
//   render_depth  — rasterise triangle depth into a per-view z-buffer
//   bake_texels   — for each atlas texel, project to cameras, z-test, blend

// ─── Structs ────────────────────────────────────────────────────────────────

struct BakeCamera {
    float r_cw[9];   // row-major 3×3
    float t_cw[3];
    float fx, fy, cx, cy;
    int   width, height;
};

struct BakeTriangle {
    float p0[3], p1[3], p2[3];
    float n[3];       // face or vertex normal (averaged)
    float uv0[2], uv1[2], uv2[2];
};

// ─── Helpers ────────────────────────────────────────────────────────────────

__device__ void world_to_cam(const float* __restrict__ pos,
                             const BakeCamera& cam,
                             float& cx, float& cy, float& cz) {
    cx = cam.r_cw[0]*pos[0] + cam.r_cw[1]*pos[1] + cam.r_cw[2]*pos[2] + cam.t_cw[0];
    cy = cam.r_cw[3]*pos[0] + cam.r_cw[4]*pos[1] + cam.r_cw[5]*pos[2] + cam.t_cw[1];
    cz = cam.r_cw[6]*pos[0] + cam.r_cw[7]*pos[1] + cam.r_cw[8]*pos[2] + cam.t_cw[2];
}

__device__ void cam_to_pixel(float cx, float cy, float cz,
                             const BakeCamera& cam,
                             float& px, float& py) {
    float inv_z = 1.0f / cz;
    px = cam.fx * cx * inv_z + cam.cx;
    py = cam.fy * cy * inv_z + cam.cy;
}

__device__ bool barycentric_2d(float ax, float ay, float bx, float by,
                               float ccx, float ccy, float px, float py,
                               float& u, float& v, float& w) {
    float v0x = bx - ax, v0y = by - ay;
    float v1x = ccx - ax, v1y = ccy - ay;
    float v2x = px - ax,  v2y = py - ay;
    float d00 = v0x*v0x + v0y*v0y;
    float d01 = v0x*v1x + v0y*v1y;
    float d11 = v1x*v1x + v1y*v1y;
    float d20 = v2x*v0x + v2y*v0y;
    float d21 = v2x*v1x + v2y*v1y;
    float denom = d00*d11 - d01*d01;
    if (fabsf(denom) < 1e-12f) return false;
    float inv = 1.0f / denom;
    v = (d11*d20 - d01*d21) * inv;
    w = (d00*d21 - d01*d20) * inv;
    u = 1.0f - v - w;
    return true;
}

// ─── render_depth ───────────────────────────────────────────────────────────

// One thread per triangle: rasterise its screen-space footprint into a depth
// buffer with atomicMin on the bit-cast float (positive floats sort like u32).

extern "C" __global__
void render_depth(const float* __restrict__ positions,    // [V][3]
                  const unsigned int* __restrict__ indices, // [T*3]
                  int n_triangles,
                  const BakeCamera cam,
                  unsigned int* __restrict__ depth_u32,    // [H][W], init 0x7F7FFFFF
                  int img_w, int img_h)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_triangles) return;

    int i0 = indices[tid*3],   i1 = indices[tid*3+1], i2 = indices[tid*3+2];
    float c0x, c0y, c0z, c1x, c1y, c1z, c2x, c2y, c2z;
    world_to_cam(&positions[i0*3], cam, c0x, c0y, c0z);
    world_to_cam(&positions[i1*3], cam, c1x, c1y, c1z);
    world_to_cam(&positions[i2*3], cam, c2x, c2y, c2z);

    if (c0z <= 0.0f && c1z <= 0.0f && c2z <= 0.0f) return;

    float px0, py0, px1, py1, px2, py2;
    cam_to_pixel(c0x, c0y, c0z, cam, px0, py0);
    cam_to_pixel(c1x, c1y, c1z, cam, px1, py1);
    cam_to_pixel(c2x, c2y, c2z, cam, px2, py2);

    int min_x = max(0, (int)floorf(fminf(fminf(px0, px1), px2)));
    int max_x = min(img_w, (int)ceilf(fmaxf(fmaxf(px0, px1), px2)));
    int min_y = max(0, (int)floorf(fminf(fminf(py0, py1), py2)));
    int max_y = min(img_h, (int)ceilf(fmaxf(fmaxf(py0, py1), py2)));

    for (int y = min_y; y < max_y; y++) {
        for (int x = min_x; x < max_x; x++) {
            float ppx = (float)x + 0.5f, ppy = (float)y + 0.5f;
            float u, v, w;
            if (!barycentric_2d(px0, py0, px1, py1, px2, py2, ppx, ppy, u, v, w))
                continue;
            if (u < 0.0f || v < 0.0f || w < 0.0f) continue;
            float z = u*c0z + v*c1z + w*c2z;
            if (z <= 0.0f) continue;
            unsigned int z_bits = __float_as_uint(z);
            atomicMin(&depth_u32[y * img_w + x], z_bits);
        }
    }
}

// ─── bake_texels ────────────────────────────────────────────────────────────

// One thread per atlas texel. For each texel, find which triangle it belongs
// to (pre-mapped via a triangle-texel table built on CPU), compute 3D world
// position, project to each camera, z-test, blend.

extern "C" __global__
void bake_texels(const BakeTriangle* __restrict__ triangles,
                 int n_triangles,
                 const BakeCamera* __restrict__ cameras,
                 const unsigned char* __restrict__* __restrict__ images,
                 const int* __restrict__ img_strides,
                 const unsigned int* __restrict__* __restrict__ depth_maps,
                 int n_cameras,
                 unsigned char* __restrict__ atlas,       // [AH][AW][4] RGBA
                 int atlas_w, int atlas_h,
                 const int* __restrict__ texel_tri_map,   // [AH][AW] -> tri index, -1 if none
                 const float* __restrict__ texel_bary)    // [AH][AW][3] barycentric
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = atlas_w * atlas_h;
    if (idx >= total) return;

    int tri_id = texel_tri_map[idx];
    if (tri_id < 0) return;

    float b0 = texel_bary[idx*3],   b1 = texel_bary[idx*3+1], b2 = texel_bary[idx*3+2];
    const BakeTriangle& tri = triangles[tri_id];

    float wx = b0*tri.p0[0] + b1*tri.p1[0] + b2*tri.p2[0];
    float wy = b0*tri.p0[1] + b1*tri.p1[1] + b2*tri.p2[1];
    float wz = b0*tri.p0[2] + b1*tri.p1[2] + b2*tri.p2[2];

    float r_acc = 0.0f, g_acc = 0.0f, b_acc = 0.0f, w_acc = 0.0f;

    for (int ci = 0; ci < n_cameras; ci++) {
        const BakeCamera& cam = cameras[ci];
        float pos[3] = {wx, wy, wz};
        float cx, cy, cz;
        world_to_cam(pos, cam, cx, cy, cz);
        if (cz <= 0.01f) continue;

        float px, py;
        cam_to_pixel(cx, cy, cz, cam, px, py);
        int ix = (int)px, iy = (int)py;
        if (ix < 0 || iy < 0 || ix >= cam.width || iy >= cam.height) continue;

        unsigned int stored_z_bits = depth_maps[ci][iy * cam.width + ix];
        float stored_z = __uint_as_float(stored_z_bits);
        float bias = 0.01f * cz;
        if (cz > stored_z + bias) continue;

        // View direction = camera centre - world pos
        float eye_x = -(cam.r_cw[0]*cam.t_cw[0] + cam.r_cw[3]*cam.t_cw[1] + cam.r_cw[6]*cam.t_cw[2]);
        float eye_y = -(cam.r_cw[1]*cam.t_cw[0] + cam.r_cw[4]*cam.t_cw[1] + cam.r_cw[7]*cam.t_cw[2]);
        float eye_z = -(cam.r_cw[2]*cam.t_cw[0] + cam.r_cw[5]*cam.t_cw[1] + cam.r_cw[8]*cam.t_cw[2]);
        float vx = eye_x - wx, vy = eye_y - wy, vz = eye_z - wz;
        float vlen = sqrtf(vx*vx + vy*vy + vz*vz);
        if (vlen < 1e-8f) continue;
        vx /= vlen; vy /= vlen; vz /= vlen;

        float cos_a = tri.n[0]*vx + tri.n[1]*vy + tri.n[2]*vz;
        if (cos_a <= 0.0f) continue;

        float weight = cos_a;
        int pix_off = iy * img_strides[ci] + ix * 3;
        r_acc += (float)images[ci][pix_off]     * weight;
        g_acc += (float)images[ci][pix_off + 1] * weight;
        b_acc += (float)images[ci][pix_off + 2] * weight;
        w_acc += weight;
    }

    if (w_acc > 0.0f) {
        int off = idx * 4;
        atlas[off]     = (unsigned char)fminf(r_acc / w_acc, 255.0f);
        atlas[off + 1] = (unsigned char)fminf(g_acc / w_acc, 255.0f);
        atlas[off + 2] = (unsigned char)fminf(b_acc / w_acc, 255.0f);
        atlas[off + 3] = 255;
    }
}
