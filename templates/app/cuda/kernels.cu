// Kernels for __NAME__. Compiled at runtime by NVRTC (see spl_gpu::load_cuda_source),
// so no nvcc is needed to build. extern "C" keeps the names unmangled.

extern "C" __global__ void scale(float* x, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= factor;
}
