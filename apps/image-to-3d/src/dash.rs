//! DashGaussian resolution warm-up inputs: image magnitude spectra (spec-train 6.5
//! step 1) and the hook that installs the schedule into a resolved preset.
//! Owner: trainer (E7). Written from the DashGaussian paper's description; no code
//! copied (its code is CC-BY-NC-SA).
//!
//! Spectra use an in-crate radix-2 FFT on the image zero-padded to powers of two.
//! Padding interpolates the spectrum but keeps each window's share of the energy
//! (the window is a fraction 1/s of each axis either way), which is all the
//! schedule reads.

use rayon::prelude::*;
use spl_core::Result;

use crate::io::images::{Rgb8Image, load_views};
use crate::io::scene::Scene;
use crate::kernels::adam::LrSchedule;
use crate::model::Group;
use crate::presets::{DashSchedule, MagnitudeSpectrum, Preset, Schedule};

/// Spectra come from at most this many evenly spaced training images.
pub const MAX_SPECTRUM_IMAGES: usize = 32;

/// In-place iterative radix-2 FFT of (re, im); `re.len()` is a power of two.
fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (ang * k as f64).sin_cos();
                let (c, s) = (c as f32, s as f32);
                let (a, b) = (start + k, start + k + len / 2);
                let tr = re[b] * c - im[b] * s;
                let ti = re[b] * s + im[b] * c;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// `|fftshift(FFT2(I))|` of the channel-mean image, zero-padded to powers of two.
pub fn magnitude_spectrum(img: &Rgb8Image) -> MagnitudeSpectrum {
    let (w, h) = (img.width as usize, img.height as usize);
    let (pw, ph) = (w.next_power_of_two(), h.next_power_of_two());
    let mut re = vec![0f32; pw * ph];
    let mut im = vec![0f32; pw * ph];
    for y in 0..h {
        for x in 0..w {
            let p = &img.data[3 * (y * w + x)..3 * (y * w + x) + 3];
            re[y * pw + x] = (p[0] as f32 + p[1] as f32 + p[2] as f32) / (3.0 * 255.0);
        }
    }
    re.par_chunks_mut(pw)
        .zip(im.par_chunks_mut(pw))
        .for_each(|(r, i)| fft(r, i));
    // Columns: transpose, FFT rows, keep transposed (magnitude is read by index).
    let mut tr = vec![0f32; pw * ph];
    let mut ti = vec![0f32; pw * ph];
    for y in 0..ph {
        for x in 0..pw {
            tr[x * ph + y] = re[y * pw + x];
            ti[x * ph + y] = im[y * pw + x];
        }
    }
    tr.par_chunks_mut(ph)
        .zip(ti.par_chunks_mut(ph))
        .for_each(|(r, i)| fft(r, i));
    let mut data = vec![0f32; pw * ph];
    for x in 0..pw {
        for y in 0..ph {
            let (sx, sy) = ((x + pw / 2) % pw, (y + ph / 2) % ph);
            let k = x * ph + y;
            data[sy * pw + sx] = (tr[k] * tr[k] + ti[k] * ti[k]).sqrt();
        }
    }
    MagnitudeSpectrum {
        width: pw as u32,
        height: ph as u32,
        data,
    }
}

/// Builds the warm-up schedule from the scene's training images and installs it:
/// `schedule.dash`, and the means LR held until the first level below scale 2
/// (spec-train 6.5 step 6). No-op unless `preset.warmup`.
pub fn install(preset: &mut Preset, schedule: &mut Schedule, scene: &Scene) -> Result<()> {
    if !preset.warmup || scene.train.is_empty() {
        return Ok(());
    }
    let started = std::time::Instant::now();
    let n = scene.train.len().min(MAX_SPECTRUM_IMAGES);
    let pick: Vec<usize> = (0..n)
        .map(|i| scene.train[i * scene.train.len() / n])
        .collect();
    let spectra: Vec<MagnitudeSpectrum> = load_views(&scene.views, &pick)?
        .iter()
        .map(magnitude_spectrum)
        .collect();
    let t_res = preset.densify_end().min(schedule.total);
    let dash = DashSchedule::from_spectra(&spectra, t_res)?;
    let hold = dash.hold_until();
    if let LrSchedule::ExpDecay { hold_until, .. } = &mut preset.groups[Group::Means.index()].lr {
        *hold_until = hold;
    }
    tracing::info!(
        "DashGaussian warm-up: s_max {:.2}, full res from {t_res}, means LR held to {hold} ({:.1} s)",
        dash.levels.first().map_or(1.0, |l| l.1),
        started.elapsed().as_secs_f64()
    );
    schedule.means_hold_until = hold;
    schedule.dash = Some(dash);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_a_direct_dft() {
        let x: Vec<f32> = (0..16).map(|i| ((i * 7) % 5) as f32 - 1.5).collect();
        let (mut re, mut im) = (x.clone(), vec![0f32; 16]);
        fft(&mut re, &mut im);
        for k in 0..16 {
            let (mut a, mut b) = (0f64, 0f64);
            for (n, &v) in x.iter().enumerate() {
                let ang = -2.0 * std::f64::consts::PI * (k * n) as f64 / 16.0;
                a += v as f64 * ang.cos();
                b += v as f64 * ang.sin();
            }
            assert!((re[k] as f64 - a).abs() < 1e-4 && (im[k] as f64 - b).abs() < 1e-4);
        }
    }

    #[test]
    fn spectrum_of_a_constant_is_a_centred_spike() {
        let img = Rgb8Image {
            width: 6,
            height: 4,
            data: vec![255; 6 * 4 * 3],
        };
        let s = magnitude_spectrum(&img);
        assert_eq!((s.width, s.height), (8, 4));
        let centre = s.data[2 * 8 + 4];
        assert!((centre - 24.0).abs() < 1e-4, "{centre}");
    }
}
