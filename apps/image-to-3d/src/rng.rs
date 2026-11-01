//! Counter-based randomness shared by host and device. Owner: trainer (E7).
//!
//! Philox-4x32-10 (Salmon et al., SC 2011), bit-identical to `philox4x32_10` in
//! `cuda/common.cuh`: any kernel's draws are a pure function of (run seed,
//! element, iteration, purpose), so CPU references and replays after an
//! overflow reproduce them exactly (spec-train 1).

/// One Philox-4x32-10 block.
pub fn philox4x32_10(mut c: [u32; 4], mut k: [u32; 2]) -> [u32; 4] {
    for r in 0..10 {
        if r > 0 {
            k[0] = k[0].wrapping_add(0x9E37_79B9);
            k[1] = k[1].wrapping_add(0xBB67_AE85);
        }
        let p0 = 0xD251_1F53u64 * c[0] as u64;
        let p1 = 0xCD9E_8D57u64 * c[2] as u64;
        let (hi0, lo0) = ((p0 >> 32) as u32, p0 as u32);
        let (hi1, lo1) = ((p1 >> 32) as u32, p1 as u32);
        c = [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0];
    }
    c
}

/// Uniform in the open interval (0, 1), as `u01_open` on the device: 23
/// bits, so `k + 0.5` is exact in f32 and the top value stays below 1.
pub fn u01_open(x: u32) -> f32 {
    ((x >> 9) as f32 + 0.5) * (1.0 / 8_388_608.0)
}

/// Two standard normals (Box-Muller), as `normal2` on the device (which uses
/// fast-math `__logf`/`__sincosf`, so compare with a tolerance).
pub fn normal2(a: u32, b: u32) -> (f32, f32) {
    let r = (-2.0 * u01_open(a).ln()).sqrt();
    let (s, c) = (std::f32::consts::TAU * u01_open(b)).sin_cos();
    (r * c, r * s)
}

/// The run's key plus the counter convention `(element, iteration, purpose, 0)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Philox {
    pub seed: [u32; 2],
}

impl Philox {
    pub fn new(seed: u64) -> Self {
        Self {
            seed: [seed as u32, (seed >> 32) as u32],
        }
    }

    /// Four u32 draws for (element, iteration, purpose); purposes are the
    /// `RNG_*` constants of `kernels::layout`.
    pub fn draw4(&self, elem: u32, iteration: u32, purpose: u32) -> [u32; 4] {
        philox4x32_10([elem, iteration, purpose, 0], self.seed)
    }

    pub fn uniform(&self, elem: u32, iteration: u32, purpose: u32) -> f32 {
        u01_open(self.draw4(elem, iteration, purpose)[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known-answer vectors of the Random123 distribution (kat_vectors).
    #[test]
    fn philox_matches_random123_known_answers() {
        assert_eq!(
            philox4x32_10([0; 4], [0; 2]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        assert_eq!(
            philox4x32_10([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
        assert_eq!(
            philox4x32_10(
                [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
                [0xa409_3822, 0x299f_31d0]
            ),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }

    #[test]
    fn uniforms_are_open_and_normals_are_standard() {
        assert!(u01_open(0) > 0.0 && u01_open(u32::MAX) < 1.0);
        let p = Philox::new(42);
        let n = 20_000;
        let (mut sum, mut sq) = (0f64, 0f64);
        for i in 0..n {
            let d = p.draw4(i, 7, 1);
            let (a, b) = normal2(d[0], d[1]);
            sum += (a + b) as f64;
            sq += (a * a + b * b) as f64;
        }
        let mean = sum / (2 * n) as f64;
        let var = sq / (2 * n) as f64 - mean * mean;
        assert!(
            mean.abs() < 0.02 && (var - 1.0).abs() < 0.03,
            "mean {mean} var {var}"
        );
        assert_ne!(
            p.draw4(0, 0, 1),
            p.draw4(0, 0, 2),
            "purposes are independent streams"
        );
    }
}
