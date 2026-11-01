//! Memory layouts shared with `cuda/render.cu`. Each constant mirrors a
//! `#define` of the same name there; `layout_matches_cuda` checks them.

/// Floats per packed parameter set.
pub(crate) const CAND_STRIDE: usize = 928;
/// Camera -> Rec.2020 matrix with the white-balance delta folded in (3x3, row-major).
pub(crate) const P_A1: usize = 0;
/// The same matrix for fully clipped pixels, which stay neutral.
pub(crate) const P_A2: usize = 9;
/// Linear exposure gain, 2^(exposure + BASELINE_EV).
pub(crate) const P_GAIN: usize = 18;
/// Added to the base layer to get its EV relative to middle grey after exposure.
pub(crate) const P_BSHIFT: usize = 19;
pub(crate) const P_VIB: usize = 20;
pub(crate) const P_SAT: usize = 21;
/// Global tone curve: display-linear output at scene EV TONE_U0..=TONE_U1 (relative to grey).
pub(crate) const TONE_OFF: usize = 32;
pub(crate) const TONE_N: usize = 512;
pub(crate) const TONE_U0: f32 = -16.0;
pub(crate) const TONE_U1: f32 = 6.0;
/// Highlights/shadows: EV added at base-layer EV LOCAL_B0..=LOCAL_B1 (relative to grey).
pub(crate) const LOCAL_OFF: usize = 544;
pub(crate) const LOCAL_N: usize = 192;
pub(crate) const LOCAL_B0: f32 = -16.0;
pub(crate) const LOCAL_B1: f32 = 8.0;
/// Whites/Blacks: EV added at the pixel's own luminance EV, sampled like LOCAL.
pub(crate) const ENDS_OFF: usize = 736;

/// Bilateral grid: padding nodes on every side, and the log2-luminance axis.
pub(crate) const GRID_PAD: usize = 2;
pub(crate) const GRID_Z0: f32 = -14.0;
pub(crate) const GRID_Z1: f32 = 1.0;
pub(crate) const GRID_DZ: f32 = 0.75;
#[cfg(test)]
pub(crate) const GRID_PRIOR: f32 = 0.001;

/// Per-(candidate, chunk) statistics record: luma and chroma histograms then
/// counts (u32), and sums (f64).
pub(crate) const LUMA_BINS: usize = 1024;
pub(crate) const CHROMA_BINS: usize = 256;
pub(crate) const CHROMA_MAX: f32 = 0.4;
pub(crate) const N_COUNTS: usize = 8;
pub(crate) const N_SUMS: usize = 10;
pub(crate) const PART_U32: usize = LUMA_BINS + CHROMA_BINS + N_COUNTS;
pub(crate) const STATS_THREADS: usize = 256;

pub(crate) const C_PX: usize = 0;
pub(crate) const C_HI: usize = 1;
pub(crate) const C_HI_NEW: usize = 2;
pub(crate) const C_LO: usize = 3;
pub(crate) const C_OOG: usize = 4;
pub(crate) const C_CAST: usize = 5;
pub(crate) const C_SBLK: usize = 6;
pub(crate) const C_LCBLK: usize = 7;

pub(crate) const S_LUMA: usize = 0;
pub(crate) const S_RG: usize = 1;
pub(crate) const S_YB: usize = 2;
pub(crate) const S_RG2: usize = 3;
pub(crate) const S_YB2: usize = 4;
pub(crate) const S_CHROMA: usize = 5;
pub(crate) const S_CA: usize = 6;
pub(crate) const S_CB: usize = 7;
pub(crate) const S_STRUCT: usize = 8;
pub(crate) const S_LC: usize = 9;

#[cfg(test)]
mod tests {
    use super::*;

    fn defines() -> std::collections::HashMap<String, String> {
        include_str!("../../cuda/render.cu")
            .lines()
            .filter_map(|line| {
                let mut words = line.strip_prefix("#define ")?.splitn(2, ' ');
                let name = words.next()?.to_string();
                let value = words
                    .next()?
                    .trim()
                    .trim_start_matches('(')
                    .trim_end_matches(')');
                Some((name, value.trim_end_matches('f').to_string()))
            })
            .collect()
    }

    #[test]
    fn layout_matches_cuda() {
        let d = defines();
        let ints: &[(&str, usize)] = &[
            ("CAND_STRIDE", CAND_STRIDE),
            ("P_A1", P_A1),
            ("P_A2", P_A2),
            ("P_GAIN", P_GAIN),
            ("P_BSHIFT", P_BSHIFT),
            ("P_VIB", P_VIB),
            ("P_SAT", P_SAT),
            ("TONE_OFF", TONE_OFF),
            ("TONE_N", TONE_N),
            ("LOCAL_OFF", LOCAL_OFF),
            ("LOCAL_N", LOCAL_N),
            ("ENDS_OFF", ENDS_OFF),
            ("GRID_PAD", GRID_PAD),
            ("LUMA_BINS", LUMA_BINS),
            ("CHROMA_BINS", CHROMA_BINS),
            ("N_COUNTS", N_COUNTS),
            ("N_SUMS", N_SUMS),
            ("STATS_THREADS", STATS_THREADS),
            ("C_PX", C_PX),
            ("C_HI", C_HI),
            ("C_HI_NEW", C_HI_NEW),
            ("C_LO", C_LO),
            ("C_OOG", C_OOG),
            ("C_CAST", C_CAST),
            ("C_SBLK", C_SBLK),
            ("C_LCBLK", C_LCBLK),
            ("S_LUMA", S_LUMA),
            ("S_RG", S_RG),
            ("S_YB", S_YB),
            ("S_RG2", S_RG2),
            ("S_YB2", S_YB2),
            ("S_CHROMA", S_CHROMA),
            ("S_CA", S_CA),
            ("S_CB", S_CB),
            ("S_STRUCT", S_STRUCT),
            ("S_LC", S_LC),
        ];
        for (name, value) in ints {
            let got: usize = d[*name].parse().unwrap_or_else(|_| panic!("{name}"));
            assert_eq!(got, *value, "{name}");
        }
        let floats: &[(&str, f32)] = &[
            ("TONE_U0", TONE_U0),
            ("TONE_U1", TONE_U1),
            ("LOCAL_B0", LOCAL_B0),
            ("LOCAL_B1", LOCAL_B1),
            ("GRID_Z0", GRID_Z0),
            ("GRID_Z1", GRID_Z1),
            ("GRID_DZ", GRID_DZ),
            ("GRID_PRIOR", GRID_PRIOR),
            ("CHROMA_MAX", CHROMA_MAX),
        ];
        for (name, value) in floats {
            let got: f32 = d[*name].parse().unwrap_or_else(|_| panic!("{name}"));
            assert_eq!(got, *value, "{name}");
        }
        const {
            assert!(
                TONE_OFF + TONE_N <= LOCAL_OFF
                    && LOCAL_OFF + LOCAL_N <= ENDS_OFF
                    && ENDS_OFF + LOCAL_N <= CAND_STRIDE
            )
        };
        const { assert!(P_SAT < TONE_OFF) };
    }
}
