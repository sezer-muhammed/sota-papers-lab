//! Camera colour: matrices and white balance ⇄ Lightroom temperature/tint.
//! Owner: develop.
//!
//! The temperature/tint math is a port of Adobe's DNG SDK
//! (`dng_temperature.cpp`, `dng_color_spec.cpp`), so with the same camera
//! matrices our Temperature and Tint mean what Lightroom's do and an exported
//! XMP lands on the same white balance.

type M3 = [[f64; 3]; 3];
type V3 = [f64; 3];

/// Calibration temperatures the DNG SDK assigns to illuminant A and D65
/// (`dng_camera_profile::IlluminantToTemperature`); interpolating at Adobe's
/// values keeps our kelvin numbers equal to Lightroom's.
const TEMP_A: f64 = 2850.0;
const TEMP_D65: f64 = 6500.0;

const D50_XY: (f64, f64) = (0.3457, 0.3585);
const D65_XY: (f64, f64) = (0.3127, 0.3290);

/// uv distance → Lightroom tint units (negative: green is negative tint).
const TINT_SCALE: f64 = -3000.0;

/// Robertson isotherms (Wyszecki & Stiles, Color Science, 2nd ed., p. 228)
/// exactly as the DNG SDK lists them: reciprocal megakelvin, u, v, slope.
const ROBERTSON: [[f64; 4]; 31] = [
    [0.0, 0.18006, 0.26352, -0.24341],
    [10.0, 0.18066, 0.26589, -0.25479],
    [20.0, 0.18133, 0.26846, -0.26876],
    [30.0, 0.18208, 0.27119, -0.28539],
    [40.0, 0.18293, 0.27407, -0.30470],
    [50.0, 0.18388, 0.27709, -0.32675],
    [60.0, 0.18494, 0.28021, -0.35156],
    [70.0, 0.18611, 0.28342, -0.37915],
    [80.0, 0.18740, 0.28668, -0.40955],
    [90.0, 0.18880, 0.28997, -0.44278],
    [100.0, 0.19032, 0.29326, -0.47888],
    [125.0, 0.19462, 0.30141, -0.58204],
    [150.0, 0.19962, 0.30921, -0.70471],
    [175.0, 0.20525, 0.31647, -0.84901],
    [200.0, 0.21142, 0.32312, -1.0182],
    [225.0, 0.21807, 0.32909, -1.2168],
    [250.0, 0.22511, 0.33439, -1.4512],
    [275.0, 0.23247, 0.33904, -1.7298],
    [300.0, 0.24010, 0.34308, -2.0637],
    [325.0, 0.24702, 0.34655, -2.4681],
    [350.0, 0.25591, 0.34951, -2.9641],
    [375.0, 0.26400, 0.35200, -3.5814],
    [400.0, 0.27218, 0.35407, -4.3633],
    [425.0, 0.28039, 0.35577, -5.3762],
    [450.0, 0.28863, 0.35714, -6.7262],
    [475.0, 0.29685, 0.35823, -8.5955],
    [500.0, 0.30505, 0.35907, -11.324],
    [525.0, 0.31320, 0.35968, -15.628],
    [550.0, 0.32129, 0.36011, -23.325],
    [575.0, 0.32931, 0.36038, -40.770],
    [600.0, 0.33724, 0.36051, -116.45],
];

/// Built from the RAW's DNG-style ColorMatrix (XYZ → camera) for D65 and,
/// when present, standard illuminant A.
#[derive(Debug, Clone)]
pub struct ColorModel {
    pub xyz_to_cam_d65: [[f32; 3]; 3],
    pub xyz_to_cam_a: Option<[[f32; 3]; 3]>,
}

impl ColorModel {
    pub fn new(xyz_to_cam_d65: [[f32; 3]; 3], xyz_to_cam_a: Option<[[f32; 3]; 3]>) -> Self {
        Self {
            xyz_to_cam_d65,
            xyz_to_cam_a,
        }
    }

    /// Camera RGB, white balanced so that neutral is (1, 1, 1), → linear
    /// Rec.2020 (D65). `temperature` selects the matrix when the camera has two.
    ///
    /// dcraw's construction with Rec.2020 primaries: the camera's response to
    /// the Rec.2020 primaries, rows scaled so Rec.2020 white maps to camera
    /// (1, 1, 1), inverted. White balance multipliers do the chromatic
    /// adaptation, so camera neutral always lands on D65 white.
    pub fn rec2020_from_cam(&self, temperature: f32) -> [[f32; 3]; 3] {
        let mut cam_from_rec2020 = mul(&self.xyz_to_cam(f64::from(temperature)), &rec2020_to_xyz());
        for row in &mut cam_from_rec2020 {
            let sum: f64 = row.iter().sum();
            if sum.abs() > 1e-9 {
                row.iter_mut().for_each(|v| *v /= sum);
            }
        }
        to_f32(&invert(&cam_from_rec2020).unwrap_or(IDENTITY))
    }

    /// White balance multipliers (green = 1) that neutralize light of this
    /// Lightroom temperature (kelvin) and tint.
    pub fn wb_for(&self, temperature: f32, tint: f32) -> [f32; 3] {
        let xy = temp_tint_to_xy(f64::from(temperature), f64::from(tint));
        let neutral = self.camera_neutral(xy);
        [
            (neutral[1] / neutral[0]) as f32,
            1.0,
            (neutral[1] / neutral[2]) as f32,
        ]
    }

    /// The Lightroom temperature and tint that these multipliers correspond to.
    pub fn temp_tint_for(&self, wb: [f32; 3]) -> (f32, f32) {
        if wb.iter().any(|&m| !m.is_finite() || m <= 0.0) {
            return (TEMP_D65 as f32, 0.0);
        }
        let neutral = wb.map(|m| 1.0 / f64::from(m));
        let (x, y) = self.neutral_to_xy(neutral);
        let (temperature, tint) = xy_to_temp_tint(x, y);
        (temperature as f32, tint as f32)
    }

    /// Multipliers that neutralize D65 daylight: the fallback when a file
    /// carries no usable as-shot white balance.
    pub fn daylight_wb(&self) -> [f32; 3] {
        let neutral = self.camera_neutral(D65_XY);
        [
            (neutral[1] / neutral[0]) as f32,
            1.0,
            (neutral[1] / neutral[2]) as f32,
        ]
    }

    /// XYZ → camera at this colour temperature: the two calibrations blended
    /// by inverse temperature (`dng_color_spec::FindXYZtoCamera`).
    fn xyz_to_cam(&self, temperature: f64) -> M3 {
        let d65 = to_f64(&self.xyz_to_cam_d65);
        let Some(a) = self.xyz_to_cam_a.as_ref().map(to_f64) else {
            return d65;
        };
        let g = if temperature <= TEMP_A {
            1.0
        } else if temperature >= TEMP_D65 || !temperature.is_finite() {
            0.0
        } else {
            (1.0 / temperature - 1.0 / TEMP_D65) / (1.0 / TEMP_A - 1.0 / TEMP_D65)
        };
        let mut m = [[0.0; 3]; 3];
        for (r, row) in m.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = g * a[r][c] + (1.0 - g) * d65[r][c];
            }
        }
        m
    }

    fn xyz_to_cam_at(&self, xy: (f64, f64)) -> M3 {
        self.xyz_to_cam(xy_to_temp_tint(xy.0, xy.1).0)
    }

    /// Camera response to a white of chromaticity `xy`, largest channel 1
    /// (`dng_color_spec::SetWhiteXY`).
    fn camera_neutral(&self, xy: (f64, f64)) -> V3 {
        let white = mul_v(&self.xyz_to_cam_at(xy), &xy_to_xyz(xy));
        let max = white.iter().copied().fold(f64::MIN, f64::max);
        if max <= 0.0 || !max.is_finite() {
            return [1.0; 3];
        }
        white.map(|v| (v / max).clamp(0.001, 1.0))
    }

    /// The white chromaticity whose camera response is `neutral`. The matrix
    /// depends on the answer, so iterate to a fixed point
    /// (`dng_color_spec::NeutralToXY`).
    fn neutral_to_xy(&self, neutral: V3) -> (f64, f64) {
        const MAX_PASSES: usize = 30;
        let mut last = D50_XY;
        for pass in 0..MAX_PASSES {
            let Some(cam_to_xyz) = invert(&self.xyz_to_cam_at(last)) else {
                return D65_XY;
            };
            let mut next = xyz_to_xy(mul_v(&cam_to_xyz, &neutral));
            if (next.0 - last.0).abs() + (next.1 - last.1).abs() < 1e-7 {
                return next;
            }
            // Out of passes means a two-value oscillation: settle in between.
            if pass == MAX_PASSES - 1 {
                next = ((last.0 + next.0) * 0.5, (last.1 + next.1) * 0.5);
            }
            last = next;
        }
        last
    }
}

/// Chromaticity → Lightroom temperature and tint (`dng_temperature::Set_xy_coord`).
pub fn xy_to_temp_tint(x: f64, y: f64) -> (f64, f64) {
    let denom = 1.5 - x + 6.0 * y;
    let u = 2.0 * x / denom;
    let v = 3.0 * y / denom;

    let (mut last_dt, mut last_du, mut last_dv) = (0.0, 0.0, 0.0);
    for index in 1..ROBERTSON.len() {
        let [_, iu, iv, slope] = ROBERTSON[index];
        let len = (1.0 + slope * slope).sqrt();
        let (du, dv) = (1.0 / len, slope / len);
        let (uu, vv) = (u - iu, v - iv);
        // Signed distance from this isotherm.
        let dt = -uu * dv + vv * du;
        if dt <= 0.0 || index == ROBERTSON.len() - 1 {
            let dt = -dt.min(0.0);
            let f = if index == 1 { 0.0 } else { dt / (last_dt + dt) };
            let [r0, u0, v0, _] = ROBERTSON[index - 1];
            let r1 = ROBERTSON[index][0];
            let temperature = 1.0e6 / (r0 * f + r1 * (1.0 - f));
            let uu = u - (u0 * f + iu * (1.0 - f));
            let vv = v - (v0 * f + iv * (1.0 - f));
            let du = du * (1.0 - f) + last_du * f;
            let dv = dv * (1.0 - f) + last_dv * f;
            let len = (du * du + dv * dv).sqrt();
            let tint = (uu * du / len + vv * dv / len) * TINT_SCALE;
            return (temperature, tint);
        }
        last_dt = dt;
        last_du = du;
        last_dv = dv;
    }
    unreachable!("the last isotherm always terminates the search")
}

/// Lightroom temperature and tint → chromaticity (`dng_temperature::Get_xy_coord`).
pub fn temp_tint_to_xy(temperature: f64, tint: f64) -> (f64, f64) {
    let r = 1.0e6 / temperature.max(1.0);
    let offset = tint / TINT_SCALE;
    let last = ROBERTSON.len() - 2;
    let index = (0..=last)
        .find(|&i| r < ROBERTSON[i + 1][0])
        .unwrap_or(last);
    let [r0, u0, v0, t0] = ROBERTSON[index];
    let [r1, u1, v1, t1] = ROBERTSON[index + 1];
    let f = (r1 - r) / (r1 - r0);
    let mut u = u0 * f + u1 * (1.0 - f);
    let mut v = v0 * f + v1 * (1.0 - f);
    let len0 = (1.0 + t0 * t0).sqrt();
    let len1 = (1.0 + t1 * t1).sqrt();
    let du = f / len0 + (1.0 - f) / len1;
    let dv = t0 / len0 * f + t1 / len1 * (1.0 - f);
    let len = (du * du + dv * dv).sqrt();
    u += du / len * offset;
    v += dv / len * offset;
    let denom = u - 4.0 * v + 2.0;
    (1.5 * u / denom, v / denom)
}

/// Rec.2020 RGB → XYZ (D65), from the primaries so it matches the white
/// point the white balance math uses to the last digit.
pub fn rec2020_to_xyz() -> M3 {
    rgb_to_xyz([(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)], D65_XY)
}

fn rgb_to_xyz(primaries: [(f64, f64); 3], white: (f64, f64)) -> M3 {
    let columns = primaries.map(xy_to_xyz);
    let p = [0, 1, 2].map(|r| columns.map(|xyz| xyz[r]));
    let s = mul_v(&invert(&p).unwrap_or(IDENTITY), &xy_to_xyz(white));
    let mut m = p;
    for row in &mut m {
        for (c, v) in row.iter_mut().enumerate() {
            *v *= s[c];
        }
    }
    m
}

fn xy_to_xyz((x, y): (f64, f64)) -> V3 {
    let y = y.max(1e-9);
    [x / y, 1.0, (1.0 - x - y) / y]
}

fn xyz_to_xy(xyz: V3) -> (f64, f64) {
    let sum = xyz[0] + xyz[1] + xyz[2];
    if sum > 0.0 && sum.is_finite() {
        (xyz[0] / sum, xyz[1] / sum)
    } else {
        D50_XY
    }
}

const IDENTITY: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn mul(a: &M3, b: &M3) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    m
}

fn mul_v(a: &M3, v: &V3) -> V3 {
    [0, 1, 2].map(|r| a[r][0] * v[0] + a[r][1] * v[1] + a[r][2] * v[2])
}

pub(crate) fn invert(m: &M3) -> Option<M3> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 || !det.is_finite() {
        return None;
    }
    let cof =
        |r0: usize, r1: usize, c0: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    Some([
        [
            cof(1, 2, 1, 2) / det,
            -cof(0, 2, 1, 2) / det,
            cof(0, 1, 1, 2) / det,
        ],
        [
            -cof(1, 2, 0, 2) / det,
            cof(0, 2, 0, 2) / det,
            -cof(0, 1, 0, 2) / det,
        ],
        [
            cof(1, 2, 0, 1) / det,
            -cof(0, 2, 0, 1) / det,
            cof(0, 1, 0, 1) / det,
        ],
    ])
}

fn to_f64(m: &[[f32; 3]; 3]) -> M3 {
    m.map(|row| row.map(f64::from))
}

fn to_f32(m: &M3) -> [[f32; 3]; 3] {
    m.map(|row| row.map(|v| v as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sony ILCE-7M3 as rawler ships it (Adobe's matrices).
    fn a7m3() -> ColorModel {
        ColorModel::new(
            [
                [0.7374, -0.2389, -0.0551],
                [-0.5435, 1.3162, 0.2519],
                [-0.1006, 0.1795, 0.6552],
            ],
            Some([
                [0.8664, -0.4056, 0.0403],
                [-0.4565, 1.2378, 0.2453],
                [-0.0301, 0.0889, 0.7157],
            ]),
        )
    }

    #[test]
    fn xy_temperature_round_trip() {
        let (mut worst_t, mut worst_tint) = (0.0f64, 0.0f64);
        for t in (2000..=50000).step_by(250) {
            for tint in [-150.0, -60.0, 0.0, 35.0, 150.0] {
                let (x, y) = temp_tint_to_xy(f64::from(t), tint);
                let (t2, tint2) = xy_to_temp_tint(x, y);
                worst_t = worst_t.max((t2 - f64::from(t)).abs() / f64::from(t));
                worst_tint = worst_tint.max((tint2 - tint).abs());
            }
        }
        assert!(
            worst_t < 1e-4 && worst_tint < 0.05,
            "{worst_t} {worst_tint}"
        );
    }

    #[test]
    fn planckian_reference_points() {
        // D65 sits about 6500 K on Lightroom's scale, slightly toward magenta.
        let (t, tint) = xy_to_temp_tint(D65_XY.0, D65_XY.1);
        assert!((t - 6504.0).abs() < 20.0, "D65 = {t} K");
        assert!(tint.abs() < 10.0, "D65 tint {tint}");
        let (t, _) = xy_to_temp_tint(0.44757, 0.40745);
        assert!((t - 2856.0).abs() < 15.0, "illuminant A = {t} K");
    }

    #[test]
    fn wb_round_trip_over_lightroom_range() {
        for model in [a7m3(), ColorModel::new(a7m3().xyz_to_cam_d65, None)] {
            for t in (2500..=12000).step_by(100) {
                for tint in (-50..=50).step_by(5) {
                    let wb = model.wb_for(t as f32, tint as f32);
                    assert!((wb[1] - 1.0).abs() < 1e-6);
                    let (t2, tint2) = model.temp_tint_for(wb);
                    assert!((t2 - t as f32).abs() < 1.0, "{t} K {tint}: got {t2} K");
                    assert!(
                        (tint2 - tint as f32).abs() < 0.5,
                        "{t} K {tint}: got tint {tint2}"
                    );
                }
            }
        }
    }

    #[test]
    fn warmer_light_needs_more_blue_gain() {
        let model = a7m3();
        let tungsten = model.wb_for(3000.0, 0.0);
        let shade = model.wb_for(7500.0, 0.0);
        assert!(tungsten[2] > shade[2] && tungsten[0] < shade[0]);
        // Magenta tint means the light was green: less green gain, so R and B rise.
        let magenta = model.wb_for(5500.0, 30.0);
        let neutral = model.wb_for(5500.0, 0.0);
        assert!(magenta[0] > neutral[0] && magenta[2] > neutral[2]);
    }

    #[test]
    fn rec2020_maps_neutral_to_neutral() {
        let model = a7m3();
        for t in [2000.0, 2850.0, 4000.0, 5500.0, 6500.0, 10000.0] {
            let m = model.rec2020_from_cam(t);
            for row in m {
                let sum: f32 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-4, "row sum {sum} at {t} K");
            }
        }
        // The A calibration rules below 2850 K, D65 above 6500 K, a blend between.
        let only_d65 = ColorModel::new(model.xyz_to_cam_d65, None);
        assert_eq!(
            model.rec2020_from_cam(7000.0),
            only_d65.rec2020_from_cam(3000.0)
        );
        assert_eq!(
            model.rec2020_from_cam(2000.0),
            model.rec2020_from_cam(2850.0)
        );
        assert_ne!(
            model.rec2020_from_cam(4000.0),
            model.rec2020_from_cam(2850.0)
        );
        assert_ne!(
            model.rec2020_from_cam(4000.0),
            model.rec2020_from_cam(6500.0)
        );
    }

    #[test]
    fn rec2020_primaries_are_standard() {
        let m = rec2020_to_xyz();
        let expected = [
            [0.636958, 0.144617, 0.168881],
            [0.262700, 0.677998, 0.059302],
            [0.000000, 0.028073, 1.060985],
        ];
        let close = m
            .iter()
            .flatten()
            .zip(expected.iter().flatten())
            .all(|(a, b)| (a - b).abs() < 2e-6);
        assert!(close, "{m:?}");
    }

    #[test]
    fn daylight_wb_reads_as_daylight() {
        let model = a7m3();
        let wb = model.daylight_wb();
        let (t, tint) = model.temp_tint_for(wb);
        assert!(
            (t - 6504.0).abs() < 25.0 && tint.abs() < 10.0,
            "{t} K {tint}"
        );
    }

    #[test]
    fn bad_multipliers_do_not_panic() {
        let model = a7m3();
        assert_eq!(model.temp_tint_for([f32::NAN, 1.0, 1.5]), (6500.0, 0.0));
        let (t, tint) = model.temp_tint_for([1.0, 1.0, 1.0]);
        assert!(t.is_finite() && tint.is_finite());
    }
}
