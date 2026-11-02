//! Constants shared with the CUDA headers. Owner: preprocess (E1).
//!
//! Each constant mirrors a `#define` of the same name in `cuda/common.cuh`,
//! `cuda/splat_common.cuh` or `cuda/adam.cuh`; `layout_matches_cuda` keeps
//! them in step (the raw-retouch pattern). Variant defaults (`TILE_H`,
//! `CHAIN_K`, ...) are the `#ifndef` fallbacks; the real values come from
//! [`super::RasterVariant::defines`].

// ---- RNG purposes (common.cuh)
pub const RNG_SPLIT: u32 = 1;
pub const RNG_MCMC_NOISE: u32 = 2;
pub const RNG_MCMC_DRAW: u32 = 3;
pub const RNG_VCP_GUMBEL: u32 = 4;
pub const RNG_BF16_SR: u32 = 5;
pub const RNG_RANDOM_BG: u32 = 6;
pub const RNG_VIEW_SAMPLE: u32 = 7;

// ---- variant defaults (splat_common.cuh)
pub const TILE_H: u32 = 16;
pub const CHAIN_K: u32 = 4;
pub const BLEND_GSPLAT: u32 = 0;
pub const BLEND_FASTERGS: u32 = 1;

// ---- tiles and passes
pub const TILE_W: u32 = 16;
pub const MAX_TILES: u32 = 8192;
pub const BATCH: u32 = 256;
pub const BUCKET: u32 = 32;
pub const RASTER_THREADS: u32 = 256;
pub const PRE_THREADS: u32 = 256;
pub const PRE_BLOCKS_PER_SM: u32 = 6;
pub const BWD_WARPS: u32 = 4;
pub const SORT_THREADS: u32 = 384;
pub const SORT_IPT: u32 = 16;
pub const RADIX_BITS: u32 = 8;
pub const RADIX_BINS: u32 = 256;
pub const TILE_KEY_BITS: u32 = 13;
pub const SH_REST: u32 = 45;
pub const MAX_SH_DEGREE: u32 = 3;
pub const CAM_GRAD_N: u32 = 10;
pub const RING_SLOTS: u32 = 8;

// ---- numerics (spec-raster 6.1)
pub const ALPHA_MIN: f32 = 1.0 / 255.0;
pub const ALPHA_MAX: f32 = 0.99;
pub const T_MIN: f32 = 1e-4;
pub const ONE_MINUS_ALPHA_MIN: f32 = 1e-6;
pub const MIN_COMPENSATION: f32 = 0.005;
pub const EPS2D_DEFAULT: f32 = 0.3;
pub const NEAR_DEFAULT: f32 = 0.01;
pub const FAR_DEFAULT: f32 = 1e4;
pub const CLIP_MARGIN: f32 = 0.15;
pub const QUAT_MIN_NORM2: f32 = 1e-8;
pub const KBIN_REL: f32 = 1e-4;
pub const KBIN_ABS: f32 = 1e-4;
pub const LOG2E: f32 = std::f32::consts::LOG2_E;
pub const SH_C0: f32 = 0.282_094_8;
pub const SH_C1: f32 = 0.488_602_52;
pub const SH_C2: [f32; 5] = [
    1.092_548_5,
    -1.092_548_5,
    0.315_391_57,
    -1.092_548_5,
    0.546_274_24,
];
pub const SH_C3: [f32; 7] = [
    -0.590_043_6,
    2.890_611_4,
    -0.457_045_8,
    0.373_176_34,
    -0.457_045_8,
    1.445_305_7,
    -0.590_043_6,
];
/// `SplatRec::{ca, cc}` = A·CONIC_SCALE_AC, `cb` = B·CONIC_SCALE_B (exp2-ready conic).
pub const CONIC_SCALE_AC: f32 = -(0.5 * std::f32::consts::LOG2_E);
pub const CONIC_SCALE_B: f32 = -std::f32::consts::LOG2_E;

// ---- status words
pub const STATUS_LEN: usize = 16;
pub const ST_OVERFLOW: usize = 0;
pub const ST_N_VIS: usize = 1;
pub const ST_N_INST: usize = 2;
pub const ST_N_CHAINS: usize = 3;
pub const ST_ITER: usize = 4;
pub const ST_OVERFLOW_ITER: usize = 5;
pub const ST_N_TILES_HIT: usize = 6;
pub const OVF_VISIBLE: u32 = 1;
pub const OVF_INSTANCES: u32 = 2;
pub const OVF_CHAINS: u32 = 4;

// ---- FrameParams.flags
pub const FLAG_AA: u32 = 1;
pub const FLAG_FILTER3D: u32 = 2;
pub const FLAG_DEPTH: u32 = 4;
pub const FLAG_ABS_GRAD: u32 = 8;
pub const FLAG_DENSIFY_STATS: u32 = 16;
pub const FLAG_CAM_GRAD: u32 = 32;
pub const FLAG_RANDOM_BG: u32 = 64;
pub const FLAG_AA_GRAD: u32 = 128;

// ---- raster_fwd mode
pub const MODE_TRAIN: u32 = 0;
pub const MODE_INFER: u32 = 1;
pub const MODE_COUNT: u32 = 2;

// ---- optimizer (adam.cuh)
pub const N_GROUPS: usize = 6;
pub const GROUP_MEANS: usize = 0;
pub const GROUP_SCALES: usize = 1;
pub const GROUP_ROTATIONS: usize = 2;
pub const GROUP_OPACITIES: usize = 3;
pub const GROUP_SH0: usize = 4;
pub const GROUP_SH_REST: usize = 5;
pub const WRITE_ACCUMULATE: u32 = 0;
pub const WRITE_OVERWRITE: u32 = 1;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// Every `#define NAME VALUE` of the headers (first definition wins, so
    /// `#ifndef` fallbacks read as the defaults).
    fn defines() -> HashMap<String, String> {
        let mut out = HashMap::new();
        for text in [
            include_str!("../../cuda/common.cuh"),
            include_str!("../../cuda/splat_common.cuh"),
            include_str!("../../cuda/adam.cuh"),
        ] {
            for line in text.lines() {
                let Some(rest) = line.trim_start().strip_prefix("#define ") else {
                    continue;
                };
                let mut words = rest.splitn(2, ' ');
                let (Some(name), Some(value)) = (words.next(), words.next()) else {
                    continue;
                };
                let value = value.trim().trim_start_matches('(').trim_end_matches(')');
                out.entry(name.to_string())
                    .or_insert_with(|| value.trim_end_matches('f').to_string());
            }
        }
        out
    }

    #[test]
    fn layout_matches_cuda() {
        let d = defines();
        let ints: &[(&str, u64)] = &[
            ("RNG_SPLIT", RNG_SPLIT.into()),
            ("RNG_MCMC_NOISE", RNG_MCMC_NOISE.into()),
            ("RNG_MCMC_DRAW", RNG_MCMC_DRAW.into()),
            ("RNG_VCP_GUMBEL", RNG_VCP_GUMBEL.into()),
            ("RNG_BF16_SR", RNG_BF16_SR.into()),
            ("RNG_RANDOM_BG", RNG_RANDOM_BG.into()),
            ("RNG_VIEW_SAMPLE", RNG_VIEW_SAMPLE.into()),
            ("TILE_H", TILE_H.into()),
            ("CHAIN_K", CHAIN_K.into()),
            ("BLEND_GSPLAT", BLEND_GSPLAT.into()),
            ("BLEND_FASTERGS", BLEND_FASTERGS.into()),
            ("TILE_W", TILE_W.into()),
            ("MAX_TILES", MAX_TILES.into()),
            ("BATCH", BATCH.into()),
            ("BUCKET", BUCKET.into()),
            ("RASTER_THREADS", RASTER_THREADS.into()),
            ("PRE_THREADS", PRE_THREADS.into()),
            ("PRE_BLOCKS_PER_SM", PRE_BLOCKS_PER_SM.into()),
            ("BWD_WARPS", BWD_WARPS.into()),
            ("SORT_THREADS", SORT_THREADS.into()),
            ("SORT_IPT", SORT_IPT.into()),
            ("RADIX_BITS", RADIX_BITS.into()),
            ("RADIX_BINS", RADIX_BINS.into()),
            ("TILE_KEY_BITS", TILE_KEY_BITS.into()),
            ("SH_REST", SH_REST.into()),
            ("MAX_SH_DEGREE", MAX_SH_DEGREE.into()),
            ("CAM_GRAD_N", CAM_GRAD_N.into()),
            ("RING_SLOTS", RING_SLOTS.into()),
            ("STATUS_LEN", STATUS_LEN as u64),
            ("ST_OVERFLOW", ST_OVERFLOW as u64),
            ("ST_N_VIS", ST_N_VIS as u64),
            ("ST_N_INST", ST_N_INST as u64),
            ("ST_N_CHAINS", ST_N_CHAINS as u64),
            ("ST_ITER", ST_ITER as u64),
            ("ST_OVERFLOW_ITER", ST_OVERFLOW_ITER as u64),
            ("ST_N_TILES_HIT", ST_N_TILES_HIT as u64),
            ("OVF_VISIBLE", OVF_VISIBLE.into()),
            ("OVF_INSTANCES", OVF_INSTANCES.into()),
            ("OVF_CHAINS", OVF_CHAINS.into()),
            ("FLAG_AA", FLAG_AA.into()),
            ("FLAG_FILTER3D", FLAG_FILTER3D.into()),
            ("FLAG_DEPTH", FLAG_DEPTH.into()),
            ("FLAG_ABS_GRAD", FLAG_ABS_GRAD.into()),
            ("FLAG_DENSIFY_STATS", FLAG_DENSIFY_STATS.into()),
            ("FLAG_CAM_GRAD", FLAG_CAM_GRAD.into()),
            ("FLAG_RANDOM_BG", FLAG_RANDOM_BG.into()),
            ("FLAG_AA_GRAD", FLAG_AA_GRAD.into()),
            ("MODE_TRAIN", MODE_TRAIN.into()),
            ("MODE_INFER", MODE_INFER.into()),
            ("MODE_COUNT", MODE_COUNT.into()),
            ("N_GROUPS", N_GROUPS as u64),
            ("GROUP_MEANS", GROUP_MEANS as u64),
            ("GROUP_SCALES", GROUP_SCALES as u64),
            ("GROUP_ROTATIONS", GROUP_ROTATIONS as u64),
            ("GROUP_OPACITIES", GROUP_OPACITIES as u64),
            ("GROUP_SH0", GROUP_SH0 as u64),
            ("GROUP_SH_REST", GROUP_SH_REST as u64),
            ("WRITE_ACCUMULATE", WRITE_ACCUMULATE.into()),
            ("WRITE_OVERWRITE", WRITE_OVERWRITE.into()),
        ];
        for (name, value) in ints {
            let got: u64 = d
                .get(*name)
                .unwrap_or_else(|| panic!("{name} missing"))
                .parse()
                .unwrap_or_else(|_| panic!("{name}"));
            assert_eq!(got, *value, "{name}");
        }
        let floats: &[(&str, f32)] = &[
            ("ALPHA_MIN", ALPHA_MIN),
            ("ALPHA_MAX", ALPHA_MAX),
            ("T_MIN", T_MIN),
            ("ONE_MINUS_ALPHA_MIN", ONE_MINUS_ALPHA_MIN),
            ("MIN_COMPENSATION", MIN_COMPENSATION),
            ("EPS2D_DEFAULT", EPS2D_DEFAULT),
            ("NEAR_DEFAULT", NEAR_DEFAULT),
            ("FAR_DEFAULT", FAR_DEFAULT),
            ("CLIP_MARGIN", CLIP_MARGIN),
            ("QUAT_MIN_NORM2", QUAT_MIN_NORM2),
            ("KBIN_REL", KBIN_REL),
            ("KBIN_ABS", KBIN_ABS),
            ("LOG2E", LOG2E),
            ("SH_C0", SH_C0),
            ("SH_C1", SH_C1),
            ("SH_C2_0", SH_C2[0]),
            ("SH_C2_1", SH_C2[1]),
            ("SH_C2_2", SH_C2[2]),
            ("SH_C2_3", SH_C2[3]),
            ("SH_C2_4", SH_C2[4]),
            ("SH_C3_0", SH_C3[0]),
            ("SH_C3_1", SH_C3[1]),
            ("SH_C3_2", SH_C3[2]),
            ("SH_C3_3", SH_C3[3]),
            ("SH_C3_4", SH_C3[4]),
            ("SH_C3_5", SH_C3[5]),
            ("SH_C3_6", SH_C3[6]),
            ("CONIC_SCALE_AC", CONIC_SCALE_AC),
            ("CONIC_SCALE_B", CONIC_SCALE_B),
        ];
        for (name, value) in floats {
            let got: f32 = d
                .get(*name)
                .unwrap_or_else(|| panic!("{name} missing"))
                .parse()
                .unwrap_or_else(|_| panic!("{name}"));
            assert_eq!(got, *value, "{name}");
        }
        // 2 ln 255 < 3.33^2: the opacity-aware bound always binds (spec-raster 2.8).
        const { assert!(TILE_W * TILE_H == RASTER_THREADS && MAX_TILES == 1 << TILE_KEY_BITS) };
        assert!(2.0 * 255f32.ln() < 3.33 * 3.33);
    }
}
