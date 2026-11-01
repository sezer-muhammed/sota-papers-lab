// End-to-end check for a GPU host: NVRTC compiles this at runtime, then the
// host launches it and verifies the result on the CPU.

extern "C" __global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
