// Parametric tone curve (Lightroom's Tone Curve panel, Region mode).
// Owner: tone curve (with src/render/tone_curve.rs, which packs T).
//
// src/render.rs splices this file into cuda/render.cu at its HOOKS line, so
// render.cu's #defines and helpers above that line are in scope. When
// TONE_CURVE_HOOK is defined, render_core calls tone_curve_apply once per
// pixel right after the Basic panel's global tone stage, on display-linear
// Rec.2020 in [0, 1]; the colour stages (Oklab, Vibrance, Saturation, Color
// Mixer, gamut mapping) follow.
//
// T is the candidate's CURVE region (CURVE_LEN floats). As packed today:
// T[0..4] = Shadows, Darks, Lights, Highlights region sliders / 100 (-1..1),
// T[4..7] = the splits 0.25, 0.5, 0.75; the rest is zero and free.
//
// Stub: off (TONE_CURVE_HOOK undefined), so output is unchanged bit for bit.
// To become: Lightroom's parametric curve on the perceptual (gamma-encoded)
// value, applied hue-preservingly like the Basic tone stage, most likely as
// a lookup table the Rust side bakes into T so the kernel only interpolates.
// Then define TONE_CURVE_HOOK. Neutral sliders must leave pixels unchanged
// (to rounding; compare `render::fingerprint` before and after).

// #define TONE_CURVE_HOOK

__device__ __forceinline__ void tone_curve_apply(const float* __restrict__ T, float& x, float& y, float& z) {
    (void)T;
    (void)x;
    (void)y;
    (void)z;
}
