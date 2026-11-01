// Parametric tone curve (Lightroom's Tone Curve panel, Region mode).
// Owner: tone curve (with src/render/tone_curve.rs, which packs T and holds
// the curve's shape and its CPU tests).
//
// src/render.rs splices this file into cuda/render.cu at its HOOKS line, so
// render.cu's #defines and helpers above that line are in scope. render_core
// calls tone_curve_apply once per pixel right after the Basic panel's global
// tone stage, on display-linear Rec.2020 in [0, 1]; the colour stages
// (Oklab, Vibrance, Saturation, Color Mixer, gamut mapping) follow.
//
// T is the candidate's CURVE region: T[0] is 1 when any region slider is set
// (0: the stage is skipped, so neutral sliders change nothing), then
// CURVE_N entries of sqrt(output) sampled at sqrt(input) = i / (CURVE_N - 1).
// The curve itself (sRGB-encoded axis, splits at 25/50/75, monotone) is baked
// on the CPU; here it is only interpolated, and applied like the Basic tone
// stage: to the largest and smallest channel, the middle one keeping its
// relative place, so hue holds.

#define TONE_CURVE_HOOK
#define CURVE_N 256

// Display-linear in [0, 1] through the baked curve.
__device__ __forceinline__ float tone_curve_lut(const float* __restrict__ T, float v) {
    float w = lut(T + 1, CURVE_N, 0.0f, 1.0f, sqrtf(clamp01(v)));
    return w * w;
}

__device__ __forceinline__ void tone_curve_apply(const float* __restrict__ T, float& x, float& y, float& z) {
    if (T[0] == 0.0f) return;
    float hi = fmaxf(x, fmaxf(y, z)), lo = fminf(x, fminf(y, z));
    float thi = tone_curve_lut(T, hi);
    if (hi - lo > 1e-20f) {
        float tlo = tone_curve_lut(T, lo);
        float k = (thi - tlo) / (hi - lo);
        x = tlo + (x - lo) * k;
        y = tlo + (y - lo) * k;
        z = tlo + (z - lo) * k;
    } else {
        x = y = z = thi;
    }
}
