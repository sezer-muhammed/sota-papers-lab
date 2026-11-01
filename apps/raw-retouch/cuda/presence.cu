// Texture, Clarity and Dehaze. Owner: presence (with src/render/presence.rs).
//
// src/render.rs splices this file into cuda/render.cu at its HOOKS line, so
// render.cu's #defines and helpers above that line are in scope, and it is
// compiled into the same module: kernels added here (blur passes, haze
// estimation) are found by name from presence.rs (`gpu.module("render",
// &SOURCE)`).
//
// Two parts:
// - Planes, computed once per image (search and display previews when the
//   engine is built, every strip of a full-resolution export) by kernels in
//   this file, launched from src/render/presence.rs: four floats per pixel
//   (a float4), e.g. blurred log luminance at Texture's and Clarity's scales
//   and a haze transmission estimate. They do not depend on the sliders, so
//   the search's thousands of renders read them for free.
// - presence_apply, called by render_core (when PRESENCE_HOOK is defined)
//   per pixel on scene-linear Rec.2020 after exposure, Highlights/Shadows
//   and Whites/Blacks, before the global tone curve. P is the whole
//   candidate; the presence values are at P[P_PRES..P_PRES + P_PRES_N],
//   packed by presence.rs (today: Texture, Clarity, Dehaze / 100, then
//   zeros). q is the pixel's planes, all zero when the image has none. base
//   is its base layer (smoothed log2 luminance at exposure 0, as-shot white
//   balance; see render_core).
//
// Stub: off (PRESENCE_HOOK undefined) and presence.rs makes no planes, so
// output is unchanged bit for bit. Then define PRESENCE_HOOK. Neutral
// sliders must leave pixels unchanged (to rounding; compare
// `render::fingerprint` before and after).

// #define PRESENCE_HOOK

__device__ __forceinline__ void presence_apply(const float* __restrict__ P, float4 q, float base, float& x, float& y,
                                               float& z) {
    (void)P;
    (void)q;
    (void)base;
    (void)x;
    (void)y;
    (void)z;
}
