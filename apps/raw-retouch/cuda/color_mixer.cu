// Color Mixer (HSL per hue band) and Calibration's shadows tint.
// Owner: colour mixer (with src/render/color_mixer.rs, which packs M and
// folds Calibration's red/green/blue primaries into the camera matrices).
//
// src/render.rs splices this file into cuda/render.cu at its HOOKS line, so
// render.cu's #defines, `Lab` and helpers above that line are in scope.
// When COLOR_MIXER_HOOK is defined, render_core calls color_mixer_apply per
// pixel on Oklab after Vibrance and Saturation, before gamut mapping: hue
// angle atan2(b, a), chroma, lightness L.
//
// M is the candidate's MIXER region (MIXER_LEN floats). As packed today:
// M[0..8] hue, M[8..16] saturation, M[16..24] luminance of the red, orange,
// yellow, green, aqua, blue, purple and magenta bands (sliders / 100,
// -1..1); M[24] shadows tint; M[25..31] red, green, blue primary hue and
// saturation, interleaved (sliders / 100). The rest is zero and free (for
// per-band tables baked on the CPU, say).
//
// Stub: off (COLOR_MIXER_HOOK undefined), so output is unchanged bit for
// bit. To become: Lightroom-like band weights over hue with smooth overlap
// (skin tones sit in orange), hue shifts, chroma scaling and lightness
// changes that keep neutrals neutral, and the shadows tint (green-magenta,
// weighted toward dark pixels). Then define COLOR_MIXER_HOOK. Neutral
// sliders must leave pixels unchanged (to rounding; compare
// `render::fingerprint` before and after).

// #define COLOR_MIXER_HOOK

__device__ __forceinline__ void color_mixer_apply(const float* __restrict__ M, Lab& o) {
    (void)M;
    (void)o;
}
