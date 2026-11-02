// CUB reference for the image-to-3d sort and binning (apps/image-to-3d/src/kernels/sort.rs).
//
// Reads one workload dumped by the Rust bench (`SPL_SORT_DUMP=1 ... sort::bench`): depth
// keys and per-Gaussian tile counts (v order), the emitted (unsorted) instances, and our
// results. On exactly that data it times, with CUDA events (median of 20 after 3 warm-ups):
//
//   Faster-GS sequence (nerficg-project/faster-gaussian-splatting@090e8ad
//   FasterGSCudaBackend/rasterization/src/forward.cu, Apache-2.0; the two small kernels below
//   follow its kernels_forward.cuh): CUB SortPairs of the 32-bit depth keys, the depth
//   ordering gather + CUB ExclusiveSum, CUB SortPairs of the u16 tile keys (13 bits), the
//   instance-range memset + extract_instance_ranges + extract_bucket_counts + CUB InclusiveSum,
//   and the n_buckets readback it does before blending;
//   gsplat sequence (isect_tiles' sort + isect_offset_encode, gsplat 1.5.3): one CUB SortPairs
//   of 64-bit tile << 32 | depth keys over the same instances (end bit 32 + 13), then the
//   offset encode;
//   CUB onesweep on the same tile keys widened to u32.
//
// Each result is checked against ours (same sorted Gaussian ids, same tile ranges).
//
// Build and run in the CUDA 12.8 devel container (spl-cuda-devel):
//   nvcc -O3 -std=c++17 -arch=sm_89 -o cub_ref cub_ref.cu
//   ./cub_ref <dump dir>...
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#define CK(x)                                                                                 \
    do {                                                                                      \
        cudaError_t e_ = (x);                                                                 \
        if (e_ != cudaSuccess) {                                                              \
            fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                          \
        }                                                                                     \
    } while (0)

template <typename T>
static std::vector<T> read_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    size_t bytes = f.tellg();
    std::vector<T> v(bytes / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

template <typename T>
static T* dev(const std::vector<T>& h) {
    T* d;
    CK(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)));
    if (!h.empty()) CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template <typename T>
static T* dev_alloc(size_t n) {
    T* d;
    CK(cudaMalloc(&d, std::max<size_t>(n, 1) * sizeof(T)));
    return d;
}

template <typename T>
static std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    if (n) CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

struct Timer {
    cudaEvent_t a, b;
    Timer() {
        cudaEventCreate(&a);
        cudaEventCreate(&b);
    }
    void start() { cudaEventRecord(a); }
    float stop() {
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    }
};

// Median of `reps` timings of f() after `warm` runs; reset() runs untimed before each.
template <typename R, typename F>
static float bench(R reset, F f, int warm = 3, int reps = 20) {
    Timer t;
    std::vector<float> v;
    for (int i = 0; i < warm + reps; ++i) {
        reset();
        CK(cudaDeviceSynchronize());
        t.start();
        f();
        float ms = t.stop();
        if (i >= warm) v.push_back(ms);
    }
    CK(cudaGetLastError());
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ---- Faster-GS's small kernels (Apache-2.0, see the header)
__global__ void apply_depth_ordering(const uint32_t* sorted, const uint32_t* n_touched, uint32_t* offset, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) offset[i] = n_touched[sorted[i]];
}

__global__ void extract_instance_ranges(const uint16_t* keys, uint2* ranges, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint16_t t = keys[i];
    if (i == 0)
        ranges[t].x = 0;
    else {
        uint16_t p = keys[i - 1];
        if (t != p) {
            ranges[p].y = i;
            ranges[t].x = i;
        }
    }
    if (i == n - 1) ranges[t].y = n;
}

__global__ void extract_bucket_counts(const uint2* ranges, uint32_t* n_buckets, uint32_t n_tiles) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tiles) return;
    uint2 r = ranges[t];
    n_buckets[t] = (r.y - r.x + 31) / 32;
}

// ---- gsplat-style 64-bit keys and offset encode
__global__ void make_isect_ids(const uint16_t* tiles, const uint32_t* vals, const uint32_t* depth_keys,
                               uint64_t* ids, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ids[i] = ((uint64_t)tiles[i] << 32) | depth_keys[vals[i]];
}

// gsplat's isect_offset_encode: offsets[t] = first instance of tile t (empty tiles get the next).
__global__ void isect_offset_encode(const uint64_t* ids, uint32_t n, uint32_t n_tiles, uint32_t* offsets) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t t = (uint32_t)(ids[i] >> 32);
    if (i == 0)
        for (uint32_t j = 0; j <= t; ++j) offsets[j] = 0;
    if (i == n - 1)
        for (uint32_t j = t + 1; j < n_tiles; ++j) offsets[j] = n;
    if (i > 0) {
        uint32_t p = (uint32_t)(ids[i - 1] >> 32);
        if (p != t)
            for (uint32_t j = p + 1; j <= t; ++j) offsets[j] = i;
    }
}

__global__ void widen(const uint16_t* a, uint32_t* b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}

static uint32_t blocks(uint32_t n, uint32_t b = 256) { return std::max<uint32_t>((n + b - 1) / b, 1); }

static int check(const char* what, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    if (a.size() != b.size()) {
        printf("  MISMATCH %s: sizes %zu vs %zu\n", what, a.size(), b.size());
        return 1;
    }
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            printf("  MISMATCH %s at %zu: %u vs %u\n", what, i, a[i], b[i]);
            return 1;
        }
    printf("  ok %s (%zu entries identical to ours)\n", what, a.size());
    return 0;
}

static int run(const std::string& dir) {
    std::ifstream m(dir + "/meta.json");
    std::string meta((std::istreambuf_iterator<char>(m)), std::istreambuf_iterator<char>());
    auto field = [&](const char* k) {
        size_t p = meta.find(std::string("\"") + k + "\"");
        return (uint32_t)std::stoul(meta.substr(meta.find(':', p) + 1));
    };
    const uint32_t V = field("visible"), I = field("instances"), T = field("tiles"), bits = field("tile_bits");
    auto h_dk = read_raw<uint32_t>(dir + "/depth_keys.u32");
    auto h_nt = read_raw<uint32_t>(dir + "/n_tiles.u32");
    auto h_ek = read_raw<uint16_t>(dir + "/emit_keys.u16");
    auto h_ev = read_raw<uint32_t>(dir + "/emit_vals.u32");
    auto ours_vals = read_raw<uint32_t>(dir + "/ours_sorted_vals.u32");
    auto ours_start = read_raw<uint32_t>(dir + "/ours_tile_start.u32");
    auto ours_order = read_raw<uint32_t>(dir + "/ours_depth_order.u32");
    printf("%s: V %u I %u tiles %u\n", dir.c_str(), V, I, T);
    int bad = 0;

    // ---------------- Faster-GS sequence
    std::vector<uint32_t> iota(V);
    for (uint32_t i = 0; i < V; ++i) iota[i] = i;
    uint32_t *dk0 = dev(h_dk), *dk1 = dev_alloc<uint32_t>(V), *dv0 = dev(iota), *dv1 = dev_alloc<uint32_t>(V);
    uint32_t *pk = dev(h_dk), *pv = dev(iota);
    uint32_t *nt = dev(h_nt), *off = dev_alloc<uint32_t>(V);
    cub::DoubleBuffer<uint32_t> dkey(dk0, dk1), dval(dv0, dv1);
    size_t tmp_bytes = 0, t2 = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, tmp_bytes, dkey, dval, (int)V);
    cub::DeviceScan::ExclusiveSum(nullptr, t2, off, off, (int)V);
    tmp_bytes = std::max(tmp_bytes, t2);
    uint16_t *tk0 = dev(h_ek), *tk1 = dev_alloc<uint16_t>(I), *ptk = dev(h_ek);
    uint32_t *tv0 = dev(h_ev), *tv1 = dev_alloc<uint32_t>(I), *ptv = dev(h_ev);
    cub::DoubleBuffer<uint16_t> tkey(tk0, tk1);
    cub::DoubleBuffer<uint32_t> tval(tv0, tv1);
    cub::DeviceRadixSort::SortPairs(nullptr, t2, tkey, tval, (int)I, 0, (int)bits);
    tmp_bytes = std::max(tmp_bytes, t2);
    uint2* ranges = dev_alloc<uint2>(T);
    uint32_t *nb = dev_alloc<uint32_t>(T), *nb_off = dev_alloc<uint32_t>(T);
    cub::DeviceScan::InclusiveSum(nullptr, t2, nb, nb_off, (int)T);
    tmp_bytes = std::max(tmp_bytes, t2);
    void* tmp;
    CK(cudaMalloc(&tmp, tmp_bytes));

    auto reset_depth = [&] {
        CK(cudaMemcpyAsync(dk0, pk, V * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(dv0, pv, V * 4, cudaMemcpyDeviceToDevice));
        dkey = cub::DoubleBuffer<uint32_t>(dk0, dk1);
        dval = cub::DoubleBuffer<uint32_t>(dv0, dv1);
    };
    auto reset_tiles = [&] {
        CK(cudaMemcpyAsync(tk0, ptk, (size_t)I * 2, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(tv0, ptv, (size_t)I * 4, cudaMemcpyDeviceToDevice));
        tkey = cub::DoubleBuffer<uint16_t>(tk0, tk1);
        tval = cub::DoubleBuffer<uint32_t>(tv0, tv1);
    };
    float fgs_depth = bench(reset_depth, [&] { cub::DeviceRadixSort::SortPairs(tmp, tmp_bytes, dkey, dval, (int)V); });
    bad += check("Faster-GS depth order", host(dval.Current(), V), ours_order);
    float fgs_scan = bench([] {}, [&] {
        apply_depth_ordering<<<blocks(V), 256>>>(dval.Current(), nt, off, V);
        cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, off, off, (int)V);
    });
    float fgs_tile = bench(reset_tiles, [&] { cub::DeviceRadixSort::SortPairs(tmp, tmp_bytes, tkey, tval, (int)I, 0, (int)bits); });
    bad += check("Faster-GS tile sort (CUB u16)", host(tval.Current(), I), ours_vals);
    uint32_t n_buckets = 0;
    float fgs_ranges = bench([] {}, [&] {
        CK(cudaMemsetAsync(ranges, 0, T * sizeof(uint2)));
        extract_instance_ranges<<<blocks(I), 256>>>(tkey.Current(), ranges, I);
        extract_bucket_counts<<<blocks(T), 256>>>(ranges, nb, T);
        cub::DeviceScan::InclusiveSum(tmp, tmp_bytes, nb, nb_off, (int)T);
        CK(cudaMemcpy(&n_buckets, nb_off + T - 1, 4, cudaMemcpyDeviceToHost));
    });
    {
        auto r = host(ranges, T);
        std::vector<uint32_t> starts(T + 1);
        uint32_t run = 0;
        for (uint32_t t = 0; t < T; ++t) {
            starts[t] = r[t].y > r[t].x ? r[t].x : run;
            run = r[t].y > r[t].x ? r[t].y : run;
        }
        starts[T] = I;
        bad += check("Faster-GS tile ranges", starts, ours_start);
    }

    // ---------------- gsplat sequence: one 64-bit sort + offset encode
    uint64_t *ids0 = dev_alloc<uint64_t>(I), *ids1 = dev_alloc<uint64_t>(I), *pids = dev_alloc<uint64_t>(I);
    uint32_t *gv0 = dev(h_ev), *gv1 = dev_alloc<uint32_t>(I), *goff = dev_alloc<uint32_t>(T);
    make_isect_ids<<<blocks(I), 256>>>(ptk, ptv, pk, pids, I);
    cub::DoubleBuffer<uint64_t> gkey(ids0, ids1);
    cub::DoubleBuffer<uint32_t> gval(gv0, gv1);
    size_t gtmp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, gtmp_bytes, gkey, gval, (int)I, 0, 32 + (int)bits);
    void* gtmp;
    CK(cudaMalloc(&gtmp, gtmp_bytes));
    auto reset_g = [&] {
        CK(cudaMemcpyAsync(ids0, pids, (size_t)I * 8, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(gv0, ptv, (size_t)I * 4, cudaMemcpyDeviceToDevice));
        gkey = cub::DoubleBuffer<uint64_t>(ids0, ids1);
        gval = cub::DoubleBuffer<uint32_t>(gv0, gv1);
    };
    float gs_sort = bench(reset_g, [&] { cub::DeviceRadixSort::SortPairs(gtmp, gtmp_bytes, gkey, gval, (int)I, 0, 32 + (int)bits); });
    bad += check("gsplat 64-bit sort", host(gval.Current(), I), ours_vals);
    float gs_enc = bench([] {}, [&] { isect_offset_encode<<<blocks(I), 256>>>(gkey.Current(), I, T, goff); });
    {
        auto o = host(goff, T);
        o.push_back(I);
        bad += check("gsplat offsets", o, ours_start);
    }
    CK(cudaFree(gtmp));
    CK(cudaFree(ids0));
    CK(cudaFree(ids1));
    CK(cudaFree(pids));

    // ---------------- CUB onesweep on u32 tile keys
    uint32_t *wk0 = dev_alloc<uint32_t>(I), *wk1 = dev_alloc<uint32_t>(I), *pwk = dev_alloc<uint32_t>(I);
    widen<<<blocks(I), 256>>>(ptk, pwk, I);
    cub::DoubleBuffer<uint32_t> wkey(wk0, wk1), wval(tv0, tv1);
    size_t wtmp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, wtmp_bytes, wkey, wval, (int)I, 0, (int)bits);
    void* wtmp;
    CK(cudaMalloc(&wtmp, wtmp_bytes));
    auto reset_w = [&] {
        CK(cudaMemcpyAsync(wk0, pwk, (size_t)I * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(tv0, ptv, (size_t)I * 4, cudaMemcpyDeviceToDevice));
        wkey = cub::DoubleBuffer<uint32_t>(wk0, wk1);
        wval = cub::DoubleBuffer<uint32_t>(tv0, tv1);
    };
    float cub32 = bench(reset_w, [&] { cub::DeviceRadixSort::SortPairs(wtmp, wtmp_bytes, wkey, wval, (int)I, 0, (int)bits); });
    bad += check("CUB onesweep u32 tile sort", host(wval.Current(), I), ours_vals);

    printf("  Faster-GS: depth sort %.3f, depth gather+scan %.3f, tile sort (u16) %.3f, ranges+buckets+readback %.3f ms; total %.3f ms\n",
           fgs_depth, fgs_scan, fgs_tile, fgs_ranges, fgs_depth + fgs_scan + fgs_tile + fgs_ranges);
    printf("  gsplat: 64-bit sort %.3f, offset encode %.3f ms; total %.3f ms\n", gs_sort, gs_enc, gs_sort + gs_enc);
    printf("  CUB onesweep u32 tile keys: %.3f ms\n", cub32);
    printf("JSON {\"dir\":\"%s\",\"V\":%u,\"I\":%u,\"fgs_depth\":%.4f,\"fgs_scan\":%.4f,\"fgs_tile\":%.4f,\"fgs_ranges\":%.4f,"
           "\"gs_sort\":%.4f,\"gs_encode\":%.4f,\"cub_u32_tile\":%.4f,\"mismatches\":%d}\n",
           dir.c_str(), V, I, fgs_depth, fgs_scan, fgs_tile, fgs_ranges, gs_sort, gs_enc, cub32, bad);
    for (void* p : {(void*)dk0, (void*)dk1, (void*)dv0, (void*)dv1, (void*)pk, (void*)pv, (void*)nt, (void*)off,
                    (void*)tk0, (void*)tk1, (void*)ptk, (void*)tv0, (void*)tv1, (void*)ptv, (void*)ranges, (void*)nb,
                    (void*)nb_off, tmp, (void*)gv0, (void*)gv1, (void*)goff, (void*)wk0, (void*)wk1, (void*)pwk, wtmp})
        CK(cudaFree(p));
    return bad;
}

int main(int argc, char** argv) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("GPU %s, %d SMs, CUB %d.%d.%d\n", p.name, p.multiProcessorCount, CUB_MAJOR_VERSION, CUB_MINOR_VERSION,
           CUB_SUBMINOR_VERSION);
    int bad = 0;
    for (int i = 1; i < argc; ++i) bad += run(argv[i]);
    return bad ? 1 : 0;
}
