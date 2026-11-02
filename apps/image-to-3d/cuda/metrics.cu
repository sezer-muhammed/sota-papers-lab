// Evaluation metrics on 8-bit images: PSNR and SSIM ("same" zero padding, and "valid"
// averaged over pixels >= 5 from every border). Owner: loss (E4).
// Spec: spec-train Section 7.4. f64 sums; no clamping of sigma^2 or S (the reference
// scripts do not clamp either). Must equal the parity scripts to 1e-4 dB / 1e-5.
//
// STUBS: every kernel below has its final signature and an empty body.

// Interleaved RGB8 render and ground truth, w x h. sums[0] += sum of squared differences.
extern "C" __global__ void psnr_u8(const uint8_t* __restrict__ render, const uint8_t* __restrict__ gt, uint32_t w,
                                   uint32_t h, double* __restrict__ sums) {}

// sums[0] += SSIM over all pixels and channels, sums[1] += SSIM over the valid region,
// sums[2] += valid pixel count.
extern "C" __global__ void ssim_u8(const uint8_t* __restrict__ render, const uint8_t* __restrict__ gt, uint32_t w,
                                   uint32_t h, double* __restrict__ sums) {}
