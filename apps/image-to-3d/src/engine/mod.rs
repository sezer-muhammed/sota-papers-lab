//! Stage A front ends behind one [`Engine`] trait, and the licence profile
//! that picks one (`reports/image-to-3d/spec-commercial.md` §4; Phase 7).
//! Owner: commercial (this directory).
//!
//! | Profile | Engine | Licence of code / weights |
//! |---|---|---|
//! | [`LicenceProfile::Research`] (default) | native VGGT-Ω ([`crate::vggt`]) | FAIR non-commercial research licence: research use only |
//! | [`LicenceProfile::Commercial`] | DA3-BASE on ONNX Runtime ([`da3`]) | Apache-2.0 / Apache-2.0 |
//!
//! Every engine returns the same [`EngineOut`]: OpenCV world-to-camera
//! extrinsics with world = frame 0, pinhole intrinsics in preprocessed
//! pixels, depth, confidence and world points per pixel, in model units
//! (relative scale). The commercial profile refuses any artifact whose
//! manifest licence is not in [`COMMERCIAL_LICENCES`] and records the
//! engine, licence and weights hash in the run (`docs/LICENSING.md`).

pub mod da3;
pub mod vggt;

#[cfg(test)]
mod gpu_tests;

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{CancellationToken, Context, DataDir, Result, RunCx, anyhow, bail};
use ts_rs::TS;

use crate::io::images::Rgb8Image;
use crate::vggt::{DenseHost, Prepared, VggtCameras};

/// Which licence the result must be usable under.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export, export_to = "image-to-3d/")]
pub enum LicenceProfile {
    /// Best quality: VGGT-Ω (FAIR non-commercial research licence; its
    /// outputs may not be used commercially).
    #[default]
    Research,
    /// Only Apache / BSD / MIT models: DA3-BASE (Apache-2.0).
    Commercial,
}

/// Weights licences (SPDX, lower case) the commercial profile may load.
pub const COMMERCIAL_LICENCES: [&str; 4] = ["apache-2.0", "bsd-2-clause", "bsd-3-clause", "mit"];

impl LicenceProfile {
    /// Errors unless an artifact under `licence` may be loaded in this
    /// profile.
    pub fn allow(self, what: &str, licence: &str) -> Result<()> {
        let l = licence.trim().to_ascii_lowercase();
        if self == Self::Commercial && !COMMERCIAL_LICENCES.contains(&l.as_str()) {
            bail!(
                "{what} is licensed `{licence}`, which the commercial profile does not allow (allowed: {})",
                COMMERCIAL_LICENCES.join(", ")
            );
        }
        Ok(())
    }
}

/// Stage times of one inference (wall clock around synchronous GPU work).
#[derive(Debug, Clone, Default, Serialize)]
pub struct EngineTimes {
    pub preprocess_ms: f32,
    /// Trunk / backbone (DA3: with its camera decoder, which is in the same
    /// graph).
    pub trunk_ms: f32,
    /// Heads (VGGT-Ω: camera and dense; DA3: the depth head).
    pub dense_ms: f32,
    /// Forward pass (trunk + heads + transfers).
    pub forward_ms: f32,
}

/// What every engine returns.
pub struct EngineOut {
    /// Frames at the model resolution (colours of the points).
    pub prepared: Prepared,
    pub cameras: VggtCameras,
    /// Depth, confidence and world points per pixel, `[N][H][W]`.
    pub dense: DenseHost,
    pub times: EngineTimes,
}

/// Identity of the loaded model, recorded with every run.
#[derive(Debug, Clone, Serialize)]
pub struct EngineInfo {
    /// e.g. `da3-base`, `vggt-omega-1b-512`.
    pub engine: String,
    /// SPDX id of the weights licence (or the licence name).
    pub licence: String,
    /// SHA-256 of the source weights file, when known.
    pub weights_sha256: Option<String>,
}

/// A Stage A front end: unposed frames → cameras, depth, points.
pub trait Engine: Send {
    fn info(&self) -> EngineInfo;
    fn infer(
        &mut self,
        frames: &[Rgb8Image],
        cancel: &CancellationToken,
        progress: &mut dyn FnMut(&str, f32),
    ) -> Result<EngineOut>;
}

/// The DA3 model directory: `$SPL_DA3_DIR`, else
/// `<data dir>/models/da3-base`.
pub fn da3_dir(cx: &RunCx) -> Result<PathBuf> {
    let dir = match std::env::var_os("SPL_DA3_DIR") {
        Some(p) => PathBuf::from(p),
        None => da3::Da3Engine::default_dir(
            &cx.resources
                .get::<DataDir>()
                .map(|d| d.0.join("models"))
                .ok_or_else(|| anyhow!("this host has no data directory"))?,
        ),
    };
    if !dir.join("manifest.json").is_file() {
        bail!(
            "DA3-BASE graphs not found in {} (set SPL_DA3_DIR; export with apps/image-to-3d/tools/onnx/export_da3.py)",
            dir.display()
        );
    }
    Ok(dir)
}

type Da3Slot = Mutex<Option<da3::Da3Engine>>;

/// The process-wide commercial engine: sessions stay loaded between runs.
fn da3_slot() -> &'static Da3Slot {
    static SLOT: OnceLock<Da3Slot> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// Runs the commercial-profile engine on `frames`, loading it (and checking
/// its licence) on first use. Returns the result, the load time in seconds
/// and what ran.
pub fn infer_commercial(
    dir: &Path,
    frames: &[Rgb8Image],
    cancel: &CancellationToken,
    progress: &mut dyn FnMut(&str, f32),
) -> Result<(EngineOut, f64, EngineInfo)> {
    let mut slot = da3_slot()
        .lock()
        .map_err(|_| anyhow!("engine lock poisoned"))?;
    let t = std::time::Instant::now();
    let engine = match slot.take() {
        Some(e) if e.dir == dir => slot.insert(e),
        _ => slot.insert(
            da3::Da3Engine::load(dir, LicenceProfile::Commercial)
                .with_context(|| format!("load DA3-BASE from {}", dir.display()))?,
        ),
    };
    let load_s = t.elapsed().as_secs_f64();
    let out = engine.infer(frames, cancel, progress)?;
    Ok((out, load_s + engine.session_s, engine.info()))
}

/// Frees the commercial engine (before the research engine takes the card).
pub fn unload_commercial() {
    if let Ok(mut s) = da3_slot().lock() {
        *s = None;
    }
}

/// Row-major 3×3 times 3-vector.
fn mat3_vec(r: &[f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        r[0] * v[0] + r[1] * v[1] + r[2] * v[2],
        r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
        r[6] * v[0] + r[7] * v[1] + r[8] * v[2],
    ]
}

/// Rotation of a quaternion `[x, y, z, w]` (scalar last, not necessarily
/// unit), row-major; `quat_to_mat` of DA3 / VGGT.
pub fn quat_xyzw_to_mat(q: [f32; 4]) -> [f32; 9] {
    let [i, j, k, r] = q;
    let two_s = 2.0 / (i * i + j * j + k * k + r * r);
    [
        1.0 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1.0 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1.0 - two_s * (i * i + j * j),
    ]
}

/// `[R | t]` (row-major 3×4) inverted.
pub fn invert_rt(m: &[f32; 12]) -> [f32; 12] {
    let r = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
    let rt = [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];
    let t = mat3_vec(&rt, [m[3], m[7], m[11]]);
    [
        rt[0], rt[1], rt[2], -t[0], rt[3], rt[4], rt[5], -t[1], rt[6], rt[7], rt[8], -t[2],
    ]
}

/// `a · b` for 3×4 rigid transforms (implicit last row `0 0 0 1`).
pub fn compose_rt(a: &[f32; 12], b: &[f32; 12]) -> [f32; 12] {
    let mut o = [0.0f32; 12];
    for row in 0..3 {
        for col in 0..4 {
            let mut s = if col == 3 { a[row * 4 + 3] } else { 0.0 };
            for k in 0..3 {
                s += a[row * 4 + k] * b[k * 4 + col];
            }
            o[row * 4 + col] = s;
        }
    }
    o
}

/// Cameras from DA3 pose encodings `[t, quat_xyzw, fov_h, fov_w]` of the
/// camera-to-world pose (`pose_encoding_to_extri_intri` + `affine_inverse`),
/// re-based so that world = frame 0.
pub fn cameras_from_c2w_pose_enc(
    pose_enc: &[[f32; 9]],
    width: usize,
    height: usize,
) -> VggtCameras {
    let c2w: Vec<[f32; 12]> = pose_enc
        .iter()
        .map(|p| {
            let r = quat_xyzw_to_mat([p[3], p[4], p[5], p[6]]);
            [
                r[0], r[1], r[2], p[0], r[3], r[4], r[5], p[1], r[6], r[7], r[8], p[2],
            ]
        })
        .collect();
    let base = c2w
        .first()
        .copied()
        .unwrap_or([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let (w, h) = (width as f32, height as f32);
    VggtCameras {
        pose_enc: pose_enc.to_vec(),
        // w2c_i · c2w_0: frame 0 becomes the identity.
        extrinsics: c2w
            .iter()
            .map(|m| compose_rt(&invert_rt(m), &base))
            .collect(),
        intrinsics: pose_enc
            .iter()
            .map(|p| {
                let fy = (h / 2.0) / (p[7] / 2.0).tan().max(1e-6);
                let fx = (w / 2.0) / (p[8] / 2.0).tan().max(1e-6);
                [fx, 0.0, w / 2.0, 0.0, fy, h / 2.0, 0.0, 0.0, 1.0]
            })
            .collect(),
        width,
        height,
    }
}

/// World points of every pixel: `c2w · depth · K⁻¹ [u, v, 1]` with integer
/// pixel coordinates (VGGT's `unproject_depth_map_to_point_map`).
pub fn unproject(c: &VggtCameras, depth: &[f32]) -> Vec<f32> {
    let (w, h) = (c.width, c.height);
    let mut out = vec![0.0f32; depth.len() * 3];
    for (i, (e, k)) in c.extrinsics.iter().zip(&c.intrinsics).enumerate() {
        let c2w = invert_rt(e);
        let r = [
            c2w[0], c2w[1], c2w[2], c2w[4], c2w[5], c2w[6], c2w[8], c2w[9], c2w[10],
        ];
        let t = [c2w[3], c2w[7], c2w[11]];
        for v in 0..h {
            for u in 0..w {
                let p = (i * h + v) * w + u;
                let Some(&d) = depth.get(p) else { return out };
                let cam = [
                    d * (u as f32 - k[2]) / k[0],
                    d * (v as f32 - k[5]) / k[4],
                    d,
                ];
                let x = mat3_vec(&r, cam);
                out[p * 3] = x[0] + t[0];
                out[p * 3 + 1] = x[1] + t[1];
                out[p * 3 + 2] = x[2] + t[2];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn commercial_profile_allows_only_permissive_licences() {
        let c = LicenceProfile::Commercial;
        assert!(c.allow("DA3-BASE", "apache-2.0").is_ok());
        assert!(c.allow("Pi3", "BSD-2-Clause").is_ok());
        assert!(c.allow("DA3-LARGE", "cc-by-nc-4.0").is_err());
        assert!(
            c.allow("VGGT-Omega", "fair-noncommercial-research")
                .is_err()
        );
        assert!(
            LicenceProfile::Research
                .allow("VGGT-Omega", "fair-noncommercial-research")
                .is_ok()
        );
    }

    #[test]
    fn profile_serialises_snake_case_and_defaults_to_research() {
        assert_eq!(
            serde_json::to_string(&LicenceProfile::Commercial).unwrap(),
            "\"commercial\""
        );
        assert_eq!(LicenceProfile::default(), LicenceProfile::Research);
    }

    #[test]
    fn quaternion_matches_axis_angle() {
        // 90° about z: x → y.
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let r = quat_xyzw_to_mat([0.0, 0.0, s, s]);
        assert!(close(
            &mat3_vec(&r, [1.0, 0.0, 0.0]),
            &[0.0, 1.0, 0.0],
            1e-6
        ));
        // Scale of the quaternion does not matter.
        assert!(close(&quat_xyzw_to_mat([0.0, 0.0, 2.0, 2.0]), &r, 1e-6));
    }

    #[test]
    fn rigid_inverse_and_compose() {
        let r = quat_xyzw_to_mat([0.1, -0.3, 0.2, 0.9]);
        let m = [
            r[0], r[1], r[2], 1.0, r[3], r[4], r[5], -2.0, r[6], r[7], r[8], 0.5,
        ];
        let id = compose_rt(&m, &invert_rt(&m));
        assert!(close(
            &id,
            &[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
            1e-5
        ));
    }

    #[test]
    fn cameras_rebase_to_frame_zero() {
        let a = [0.5, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.2];
        let b = [1.5, 0.2, -1.0, 0.0, 0.38268343, 0.0, 0.9238795, 1.0, 1.2];
        let c = cameras_from_c2w_pose_enc(&[a, b], 504, 336);
        assert!(close(
            &c.extrinsics[0],
            &[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
            1e-6
        ));
        // Frame 1 is 45° about y and 1 unit along x (plus 0.2 in y) from frame 0.
        let c2w1 = invert_rt(&c.extrinsics[1]);
        assert!(close(&[c2w1[3], c2w1[7], c2w1[11]], &[1.0, 0.2, 0.0], 1e-5));
        let k = c.intrinsics[0];
        assert!((k[0] - 252.0 / (1.2f32 / 2.0).tan()).abs() < 1e-3);
        assert!((k[4] - 168.0 / 0.5f32.tan()).abs() < 1e-3);
        assert_eq!((k[2], k[5]), (252.0, 168.0));
    }

    #[test]
    fn unprojected_points_reproject_to_their_pixels() {
        let pe = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.2],
            [0.3, -0.1, 0.2, 0.05, 0.1, 0.0, 0.99, 1.0, 1.2],
        ];
        let (w, h) = (6, 4);
        let c = cameras_from_c2w_pose_enc(&pe, w, h);
        let depth: Vec<f32> = (0..2 * w * h).map(|i| 1.0 + i as f32 * 0.1).collect();
        let pts = unproject(&c, &depth);
        for f in 0..2 {
            let (e, k) = (&c.extrinsics[f], &c.intrinsics[f]);
            for p in [0usize, 7, 23] {
                let x = &pts[(f * w * h + p) * 3..][..3];
                let cam = [
                    e[0] * x[0] + e[1] * x[1] + e[2] * x[2] + e[3],
                    e[4] * x[0] + e[5] * x[1] + e[6] * x[2] + e[7],
                    e[8] * x[0] + e[9] * x[1] + e[10] * x[2] + e[11],
                ];
                let u = k[0] * cam[0] / cam[2] + k[2];
                let v = k[4] * cam[1] / cam[2] + k[5];
                assert!((u - (p % w) as f32).abs() < 1e-3 && (v - (p / w) as f32).abs() < 1e-3);
                assert!((cam[2] - depth[f * w * h + p]).abs() < 1e-4);
            }
        }
    }
}
