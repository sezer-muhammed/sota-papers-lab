// Black & White: Lightroom's B&W treatment and its gray (B&W) mixer.
// Owner: bw (with src/render/bw.rs, which packs B and holds the slider
// logic and its CPU twin for the tests).
//
// src/render.rs splices this file into cuda/render.cu at its HOOKS line, so
// render.cu's #defines, `Lab` and helpers above that line are in scope.
// render_core calls bw_apply per pixel on linear Rec.2020 right after white
// balance and the camera matrix, before exposure: the pixel becomes its
// luminance Y, scaled by the gray mixer as a function of its Oklab hue and
// colourfulness, so Exposure, Highlights/Shadows, Whites/Blacks, Presence,
// the tone curves and Contrast all work on that grey as they would on any.
// bw_finish clears Oklab a/b at the end of render_core, so nothing (Oklab
// rounding, Calibration's shadows tint) can tint the result.
//
// B is the candidate's BW region (BW_LEN floats), packed by bw.rs:
// B[BW_ON]       1 in black and white, else 0 (and the stage is skipped);
// B[BW_MIX]      1 when any gray mixer slider is set;
// B[BW_LUT..]    per hue bin k (Oklab hue 360 k / BW_N degrees; bin BW_N
//                repeats bin 0): the stops a fully colourful pixel of that
//                hue moves.
//
// Greys stay exact: the move fades in with colourfulness s = C / L (which
// exposure does not change) as q / (1 + q), q = (s / BW_S0)^2, so s = 0
// keeps Y as it is. It also fades out in deep shadows (scene luminance
// after exposure between BW_DARK0 and BW_DARK1, middle grey 0.18), where a
// pixel's hue is mostly colour noise that would otherwise turn into grain.

#define BW_HOOK

#define BW_ON 0
#define BW_MIX 1
#define BW_LUT 8
#define BW_N 72
#define BW_S0 (0.12f)
#define BW_DARK0 (0.001f)
#define BW_DARK1 (0.006f)

__device__ __forceinline__ void bw_apply(const float* __restrict__ B, float gain, float& x, float& y, float& z) {
    if (B[BW_ON] == 0.0f) return;
    float Y = Y_R * x + Y_G * y + Y_B * z;
    if (B[BW_MIX] != 0.0f && Y > 0.0f) {
        Lab o;
        rec2020_to_oklab(x, y, z, o);
        float c2 = o.a * o.a + o.b * o.b;
        if (c2 > 1e-14f && o.L > 1e-6f) {
            float f = atan2f(o.b, o.a) * ((float)BW_N / 6.28318530718f);
            if (f < 0.0f) f += (float)BW_N;
            f = fminf(fmaxf(f, 0.0f), (float)BW_N);
            int i = min((int)f, BW_N - 1);
            float t = f - (float)i;
            float e = B[BW_LUT + i] + t * (B[BW_LUT + i + 1] - B[BW_LUT + i]);
            float q = c2 / (o.L * o.L * (BW_S0 * BW_S0));
            float w = q / (1.0f + q);
            w *= smooth01((Y * gain - BW_DARK0) * (1.0f / (BW_DARK1 - BW_DARK0)));
            Y *= exp2f(w * e);
        }
    }
    x = y = z = Y;
}

__device__ __forceinline__ void bw_finish(const float* __restrict__ B, Lab& o) {
    if (B[BW_ON] != 0.0f) {
        o.a = 0.0f;
        o.b = 0.0f;
    }
}
