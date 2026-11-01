//! The search box: Lightroom units to and from the unit cube CMA-ES works
//! in. Temperature is laid out linearly in mireds, the scale on which equal
//! steps look equally strong at any colour temperature; every other slider
//! is linear in its own units.
//!
//! The cube's dimensions are a chosen set of [`Params`] fields: by default
//! those [`PARAM_SPECS`] marks `searched` (all but Detail and Lens), whole
//! groups ([`Space::with_groups`]), or one search stage's fields with the
//! rest held at a given point ([`Space::fixing`]). Every other field is
//! held where the box fixes it (`lo`, which equals `hi` there), and a
//! field the box fixes is never a dimension.

use super::priors::{FIRST, PRIORS};
use crate::types::{LinearImage, PARAM_SPECS, ParamGroup, ParamSpec, Params};

#[derive(Debug, Clone)]
pub(crate) struct Space {
    pub(crate) lo: Params,
    pub(crate) hi: Params,
    /// Indices into [`Params::to_array`] of the searched fields, ascending.
    dims: Vec<usize>,
    /// Mireds of the coolest (`lo`) and warmest (`hi`) temperature.
    cool_mired: f64,
    warm_mired: f64,
}

fn mired(kelvin: f32) -> f64 {
    1.0e6 / f64::from(kelvin.max(1.0))
}

impl Space {
    /// The box over every field [`PARAM_SPECS`] marks `searched`.
    pub(crate) fn new(lo: Params, hi: Params) -> Self {
        let dims = (0..Params::COUNT)
            .filter(|&i| PARAM_SPECS[i].searched)
            .collect();
        Self::with_dims(lo, hi, dims)
    }

    /// The box over every field of `groups`.
    pub(crate) fn with_groups(lo: Params, hi: Params, groups: &[ParamGroup]) -> Self {
        Self::with_dims(lo, hi, ParamSpec::indices_of(groups))
    }

    /// The box over the fields `dims` (indices into [`Params::to_array`])
    /// with every other field fixed at `at`, clamped into this box: one
    /// search stage around a point.
    pub(crate) fn fixing(&self, dims: &[usize], at: &Params) -> Self {
        let at = self.clamp(at).to_array();
        let (mut lo, mut hi) = (self.lo.to_array(), self.hi.to_array());
        for i in (0..Params::COUNT).filter(|i| !dims.contains(i)) {
            lo[i] = at[i];
            hi[i] = at[i];
        }
        Self::with_dims(
            Params::from_array(lo),
            Params::from_array(hi),
            dims.to_vec(),
        )
    }

    /// This box narrowed, for every field with an expert prior, to `stds`
    /// of its standard deviation around `at` (on Lightroom's whole-unit
    /// grid): the prior's trust region. In such a box one unit of the cube
    /// is about as far, in prior cost, along every new setting, so CMA-ES's
    /// round step size suits them all, and no evaluation goes to settings
    /// the prior would reject anyway.
    pub(crate) fn around(&self, at: &Params, stds: f32) -> Self {
        let at = self.clamp(at).to_array();
        let (mut lo, mut hi) = (self.lo.to_array(), self.hi.to_array());
        for (k, prior) in PRIORS.iter().enumerate() {
            let i = FIRST + k;
            let reach = stds * prior.std;
            lo[i] = lo[i].max((at[i] - reach).ceil()).min(at[i].max(lo[i]));
            hi[i] = hi[i].min((at[i] + reach).floor()).max(at[i].min(hi[i]));
        }
        Self::with_dims(
            Params::from_array(lo),
            Params::from_array(hi),
            self.dims.clone(),
        )
    }

    /// The box over the fields `dims`, less those it fixes (`lo == hi`).
    fn with_dims(lo: Params, hi: Params, mut dims: Vec<usize>) -> Self {
        let (l, h) = (lo.to_array(), hi.to_array());
        dims.retain(|&i| i < Params::COUNT && l[i] < h[i]);
        dims.sort_unstable();
        dims.dedup();
        Self {
            lo,
            hi,
            dims,
            cool_mired: mired(lo.temperature),
            warm_mired: mired(hi.temperature),
        }
    }

    /// Indices into [`Params::to_array`] of the cube's dimensions, in order.
    pub(crate) fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Unit-cube coordinates of `p`, clamped into the box: one per
    /// dimension.
    pub(crate) fn unit_of(&self, p: &Params) -> Vec<f64> {
        let (v, lo, hi) = (p.to_array(), self.lo.to_array(), self.hi.to_array());
        let u = self.dims.iter().map(|&i| {
            if i == 0 {
                let span = self.cool_mired - self.warm_mired;
                return if span.abs() < 1e-9 {
                    0.5
                } else {
                    (self.cool_mired - mired(v[0])) / span
                };
            }
            let span = f64::from(hi[i] - lo[i]);
            if span.abs() < 1e-9 {
                0.5
            } else {
                f64::from(v[i] - lo[i]) / span
            }
        });
        u.map(|x| {
            if x.is_finite() {
                x.clamp(0.0, 1.0)
            } else {
                0.5
            }
        })
        .collect()
    }

    /// The parameter set at unit-cube coordinates `u` (clamped into the
    /// box); fields outside the cube's dimensions take `lo`.
    pub(crate) fn params_at(&self, u: &[f64]) -> Params {
        let (lo, hi) = (self.lo.to_array(), self.hi.to_array());
        let at = |d: usize| {
            u.get(d)
                .copied()
                .filter(|x| x.is_finite())
                .unwrap_or(0.5)
                .clamp(0.0, 1.0)
        };
        let mut a = lo;
        for (d, &i) in self.dims.iter().enumerate() {
            a[i] = if i == 0 {
                let m = self.cool_mired - at(d) * (self.cool_mired - self.warm_mired);
                ((1.0e6 / m) as f32).max(lo[0]).min(hi[0])
            } else {
                let v = f64::from(lo[i]) + at(d) * f64::from(hi[i] - lo[i]);
                (v as f32).max(lo[i]).min(hi[i])
            };
        }
        Params::from_array(a)
    }

    pub(crate) fn clamp(&self, p: &Params) -> Params {
        super::clamp_params(p, &self.lo, &self.hi)
    }

    /// Euclidean distance in the unit cube: 0.1 is a tenth of every
    /// slider's search range (about 0.4 EV, or 10 slider points).
    pub(crate) fn distance(&self, a: &Params, b: &Params) -> f32 {
        let (a, b) = (self.unit_of(a), self.unit_of(b));
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt() as f32
    }
}

/// Brightness facts of the linear search image (1.0 = RAW clip), used to
/// widen the box for scenes PPR10K's portraits do not cover.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LinearKey {
    pub(crate) median: f32,
    pub(crate) p05: f32,
    pub(crate) p995: f32,
}

impl LinearKey {
    /// Percentiles of Rec.709-weighted luminance over at most ~64k pixels.
    pub(crate) fn measure(img: &LinearImage) -> Option<Self> {
        let pixels = img.width * img.height;
        if pixels == 0 || img.data.len() < pixels * 3 {
            return None;
        }
        let step = (pixels / 65_536).max(1);
        let mut lum: Vec<f32> = img.data[..pixels * 3]
            .as_chunks::<3>()
            .0
            .iter()
            .step_by(step)
            .map(|c| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
            .filter(|v| v.is_finite())
            .map(|v| v.max(0.0))
            .collect();
        if lum.is_empty() {
            return None;
        }
        let mut at = |q: f32| {
            let k = ((lum.len() - 1) as f32 * q).round() as usize;
            *lum.select_nth_unstable_by(k, f32::total_cmp).1
        };
        Some(Self {
            median: at(0.5),
            p05: at(0.05),
            p995: at(0.995),
        })
    }
}
