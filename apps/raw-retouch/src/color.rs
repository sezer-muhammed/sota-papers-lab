//! Camera colour: matrices and white balance ⇄ Lightroom temperature/tint.
//! Owner: develop.

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
    pub fn rec2020_from_cam(&self, temperature: f32) -> [[f32; 3]; 3] {
        let _ = temperature;
        unimplemented!("rec2020_from_cam")
    }

    /// White balance multipliers (green = 1) that neutralize light of this
    /// Lightroom temperature (kelvin) and tint.
    pub fn wb_for(&self, temperature: f32, tint: f32) -> [f32; 3] {
        let _ = (temperature, tint);
        unimplemented!("wb_for")
    }

    /// The Lightroom temperature and tint that these multipliers correspond to.
    pub fn temp_tint_for(&self, wb: [f32; 3]) -> (f32, f32) {
        let _ = wb;
        unimplemented!("temp_tint_for")
    }
}
