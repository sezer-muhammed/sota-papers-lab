//! Array comparison with the tolerances of spec-raster 6.4: max/mean
//! absolute error, relative L2, cosine, and per-element relative error over
//! the significant elements (|ref| > frac · max |ref|).

use std::fmt;

use spl_core::{Result, bail};

/// Anything comparable as f64.
pub trait Real: Copy {
    fn f64(self) -> f64;
}

impl Real for f32 {
    fn f64(self) -> f64 {
        self as f64
    }
}

impl Real for f64 {
    fn f64(self) -> f64 {
        self
    }
}

impl Real for u8 {
    fn f64(self) -> f64 {
        self as f64
    }
}

impl Real for u32 {
    fn f64(self) -> f64 {
        self as f64
    }
}

impl Real for i32 {
    fn f64(self) -> f64 {
        self as f64
    }
}

/// Bounds a comparison must meet; `None` skips that check.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tol {
    pub max_abs: Option<f64>,
    pub mean_abs: Option<f64>,
    pub rel_l2: Option<f64>,
    pub min_cosine: Option<f64>,
    /// (max relative error, significance fraction of max |ref|).
    pub elem_rel: Option<(f64, f64)>,
}

impl Tol {
    /// Rendered image vs reference, fp32 records.
    pub const IMAGE: Tol = Tol {
        max_abs: Some(2e-5),
        mean_abs: Some(1e-6),
        rel_l2: None,
        min_cosine: None,
        elem_rel: None,
    };
    /// Rendered image with the 32 B fp16 record.
    pub const IMAGE_REC32: Tol = Tol {
        max_abs: Some(1e-3),
        ..Tol::NONE
    };
    /// Final transmittance.
    pub const TRANSMITTANCE: Tol = Tol {
        max_abs: Some(1e-6),
        ..Tol::NONE
    };
    /// means2d (px) and depth.
    pub const MEANS2D: Tol = Tol {
        max_abs: Some(1e-4),
        ..Tol::NONE
    };
    /// Conics and compensation: relative 1e-5.
    pub const CONIC: Tol = Tol {
        elem_rel: Some((1e-5, 0.0)),
        ..Tol::NONE
    };
    /// 2D gradients (mean2d, conic, opacity, colour) per group.
    pub const GRAD_2D: Tol = Tol {
        rel_l2: Some(1e-4),
        min_cosine: Some(0.99999),
        elem_rel: Some((1e-3, 1e-3)),
        ..Tol::NONE
    };
    /// 3D gradients (μ, ŝ, q, ô, SH).
    pub const GRAD_3D: Tol = Tol {
        rel_l2: Some(1e-3),
        min_cosine: Some(0.9999),
        ..Tol::NONE
    };
    /// Pose and focal gradients vs converted references or finite differences.
    pub const POSE: Tol = Tol {
        elem_rel: Some((1e-3, 0.0)),
        ..Tol::NONE
    };
    pub const EXACT: Tol = Tol {
        max_abs: Some(0.0),
        ..Tol::NONE
    };
    pub const NONE: Tol = Tol {
        max_abs: None,
        mean_abs: None,
        rel_l2: None,
        min_cosine: None,
        elem_rel: None,
    };
}

/// Error statistics of `ours` against `reference`.
#[derive(Clone, Debug, PartialEq)]
pub struct Diff {
    pub name: String,
    pub n: usize,
    pub max_abs: f64,
    /// Index of the largest absolute error.
    pub worst: usize,
    pub mean_abs: f64,
    /// ‖ours − ref‖ / ‖ref‖ (0 when both are 0).
    pub rel_l2: f64,
    /// ⟨ours, ref⟩ / (‖ours‖‖ref‖); 1 when both are 0.
    pub cosine: f64,
    /// Largest |ours − ref| / |ref| over |ref| > significance · max |ref|.
    pub max_elem_rel: f64,
    pub significance: f64,
    /// Elements where exactly one side is NaN or infinite.
    pub non_finite: usize,
}

impl Diff {
    pub fn new<A: Real, B: Real>(
        name: &str,
        ours: &[A],
        reference: &[B],
        significance: f64,
    ) -> Self {
        assert_eq!(ours.len(), reference.len(), "{name}: length mismatch");
        let n = ours.len();
        let ref_max = reference
            .iter()
            .map(|r| r.f64().abs())
            .filter(|v| v.is_finite())
            .fold(0.0, f64::max);
        let threshold = significance * ref_max;
        let (mut max_abs, mut worst, mut sum_abs) = (0f64, 0usize, 0f64);
        let (mut dd, mut oo, mut rr, mut or) = (0f64, 0f64, 0f64, 0f64);
        let (mut max_rel, mut non_finite) = (0f64, 0usize);
        for (i, (a, b)) in ours.iter().zip(reference).enumerate() {
            let (a, b) = (a.f64(), b.f64());
            if !a.is_finite() || !b.is_finite() {
                if a.is_finite() != b.is_finite() || (a.is_nan() != b.is_nan()) {
                    non_finite += 1;
                }
                continue;
            }
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
                worst = i;
            }
            sum_abs += d;
            dd += d * d;
            oo += a * a;
            rr += b * b;
            or += a * b;
            if b.abs() > threshold && b != 0.0 {
                max_rel = max_rel.max(d / b.abs());
            }
        }
        let rel_l2 = if rr > 0.0 {
            (dd / rr).sqrt()
        } else if dd > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        let cosine = if oo > 0.0 && rr > 0.0 {
            or / (oo.sqrt() * rr.sqrt())
        } else if oo == 0.0 && rr == 0.0 {
            1.0
        } else {
            0.0
        };
        Self {
            name: name.to_string(),
            n,
            max_abs,
            worst,
            mean_abs: if n > 0 { sum_abs / n as f64 } else { 0.0 },
            rel_l2,
            cosine,
            max_elem_rel: max_rel,
            significance,
            non_finite,
        }
    }

    /// Ok when every bound of `tol` holds; the error lists each violation.
    pub fn check(&self, tol: &Tol) -> Result<()> {
        let mut bad = Vec::new();
        if self.non_finite > 0 {
            bad.push(format!("{} non-finite mismatches", self.non_finite));
        }
        if let Some(t) = tol.max_abs
            && self.max_abs > t
        {
            bad.push(format!(
                "max_abs {:.3e} > {t:.1e} (at {})",
                self.max_abs, self.worst
            ));
        }
        if let Some(t) = tol.mean_abs
            && self.mean_abs > t
        {
            bad.push(format!("mean_abs {:.3e} > {t:.1e}", self.mean_abs));
        }
        if let Some(t) = tol.rel_l2
            && self.rel_l2 > t
        {
            bad.push(format!("rel_l2 {:.3e} > {t:.1e}", self.rel_l2));
        }
        if let Some(t) = tol.min_cosine
            && self.cosine < t
        {
            bad.push(format!("cosine {:.7} < {t}", self.cosine));
        }
        if let Some((t, _)) = tol.elem_rel
            && self.max_elem_rel > t
        {
            bad.push(format!("elem rel {:.3e} > {t:.1e}", self.max_elem_rel));
        }
        if bad.is_empty() {
            Ok(())
        } else {
            bail!("{}: {} [{self}]", self.name, bad.join("; "))
        }
    }
}

impl fmt::Display for Diff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} n={} max_abs={:.3e}@{} mean_abs={:.3e} rel_l2={:.3e} cos={:.8} elem_rel={:.3e}",
            self.name,
            self.n,
            self.max_abs,
            self.worst,
            self.mean_abs,
            self.rel_l2,
            self.cosine,
            self.max_elem_rel
        )
    }
}

/// Compares and checks in one call, using `tol`'s significance fraction.
pub fn check_close<A: Real, B: Real>(
    name: &str,
    ours: &[A],
    reference: &[B],
    tol: &Tol,
) -> Result<Diff> {
    let d = Diff::new(name, ours, reference, tol.elem_rel.map_or(0.0, |e| e.1));
    d.check(tol)?;
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statistics_are_correct() {
        let d = Diff::new("x", &[1.0f32, 2.0, 3.0], &[1.0f64, 2.0, 3.5], 0.0);
        assert_eq!((d.max_abs, d.worst), (0.5, 2));
        assert!((d.mean_abs - 0.5 / 3.0).abs() < 1e-12);
        assert!((d.max_elem_rel - 0.5 / 3.5).abs() < 1e-12);
        assert!(d.cosine < 1.0 && d.cosine > 0.99);
        let same = Diff::new("z", &[0.0f32; 4], &[0.0f32; 4], 0.0);
        assert_eq!((same.rel_l2, same.cosine), (0.0, 1.0));
    }

    #[test]
    fn check_reports_each_violation() {
        assert!(check_close("img", &[0.5f32, 0.25], &[0.5f32, 0.25], &Tol::IMAGE).is_ok());
        let err = check_close("img", &[0.5f32, 0.3], &[0.5f32, 0.25], &Tol::IMAGE).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("max_abs") && msg.contains("mean_abs"), "{msg}");
        // Insignificant elements do not count for the element-wise bound.
        let tol = Tol {
            elem_rel: Some((1e-3, 1e-2)),
            ..Tol::NONE
        };
        assert!(check_close("g", &[1.0f64, 2e-5], &[1.0f64, 1e-5], &tol).is_ok());
        assert!(check_close("nan", &[f32::NAN], &[1.0f32], &Tol::NONE).is_err());
    }
}
