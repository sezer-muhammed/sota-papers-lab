// Color Mixer (HSL per hue band) and Calibration's shadows tint.
// Owner: colour mixer (with src/render/color_mixer.rs, which packs M and
// folds Calibration's red/green/blue primaries into the camera matrices).
//
// src/render.rs splices this file into cuda/render.cu at its HOOKS line, so
// render.cu's #defines, `Lab` and helpers above that line are in scope.
// render_core calls color_mixer_apply per pixel on Oklab after Vibrance and
// Saturation, before gamut mapping. It works in OkLCh: hue angle atan2(b, a),
// chroma, lightness L.
//
// M is the candidate's MIXER region (MIXER_LEN floats), packed by
// color_mixer.rs, which holds all the slider logic (and its CPU tests):
// M[0..31]       the sliders / 100, for reading only: red..magenta hue,
//                saturation, luminance, shadows tint, primaries;
// M[MIX_ON]      1 when any Color Mixer slider is set (else HSL is skipped);
// M[MIX_TINT..]  the shadows tint: Oklab (a, b) offset per unit of L at
//                full shadow weight (zero when off);
// M[MIX_LUT..]   per hue bin k (Oklab hue 360 k / MIX_N degrees; bin MIX_N
//                repeats bin 0): hue shift (radians), chroma gain, and the
//                log2 of the lightness scale at black.
//
// Greys stay grey: hue and saturation act on chroma, so C = 0 maps to 0,
// and every change fades in with chroma as (C/C0)^4 / (1 + (C/C0)^4): hue
// and saturation over MIX_CN, luminance over MIX_CL. Near neutrals (noise,
// a white wall's or a grey card's faint cast, up to C of about 0.015) keep
// their colour and tone. Luminance scales lightness by g at black, rolling
// off to 1 at white (L g / (1 + L (g - 1)): monotonic, black and white stay
// put).

#define COLOR_MIXER_HOOK

#define MIX_ON 31
#define MIX_TINT 32
#define MIX_LUT 36
#define MIX_N 72
#define MIX_CN (0.025f)
#define MIX_CL (0.05f)
// Shadows tint fades out from black to this Oklab lightness.
#define MIX_TINT_L1 (0.75f)

__device__ __forceinline__ void color_mixer_apply(const float* __restrict__ M, Lab& o) {
    if (M[MIX_ON] != 0.0f) {
        float c2 = o.a * o.a + o.b * o.b;
        if (c2 > 1e-14f) {
            // Hue bin: atan2 in (-pi, pi], bins from hue 0.
            float f = atan2f(o.b, o.a) * ((float)MIX_N / 6.28318530718f);
            if (f < 0.0f) f += (float)MIX_N;
            f = fminf(fmaxf(f, 0.0f), (float)MIX_N);
            int i = min((int)f, MIX_N - 1);
            float t = f - (float)i;
            const float* T = M + MIX_LUT + 3 * i;
            float dh = T[0] + t * (T[3] - T[0]);
            float gain = T[1] + t * (T[4] - T[1]);
            float e = T[2] + t * (T[5] - T[2]);
            float q = c2 * (1.0f / (MIX_CN * MIX_CN));
            q *= q;
            float wn = q / (1.0f + q);
            float k = 1.0f + wn * (gain - 1.0f);
            float s, c;
            sincosf(wn * dh, &s, &c);
            float u = k * c, v = k * s;
            float a = u * o.a - v * o.b;
            float b = v * o.a + u * o.b;
            o.a = a;
            o.b = b;
            q = c2 * (1.0f / (MIX_CL * MIX_CL));
            q *= q;
            e *= q / (1.0f + q);
            if (e != 0.0f) {
                float x = clamp01(o.L);
                float g = exp2f(e);
                o.L = x * g / (1.0f + x * (g - 1.0f));
            }
        }
    }
    float ta = M[MIX_TINT], tb = M[MIX_TINT + 1];
    if (ta != 0.0f || tb != 0.0f) {
        float L = fmaxf(o.L, 0.0f);
        float w = L * (1.0f - smooth01(L * (1.0f / MIX_TINT_L1)));
        o.a += w * ta;
        o.b += w * tb;
    }
}
