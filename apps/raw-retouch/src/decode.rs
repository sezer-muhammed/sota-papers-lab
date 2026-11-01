//! RAW file → [`RawFrame`] with rawler (CPU). Owner: develop.

use std::panic::{AssertUnwindSafe, catch_unwind};

use rawler::decoders::{FormatHint, RawDecodeParams, RawMetadata};
use rawler::formats::tiff::Rational;
use rawler::imgop::xyz::Illuminant;
use rawler::rawimage::{BlackLevel, RawImage, RawImageData, RawPhotometricInterpretation};
use rawler::rawsource::RawSource;
use spl_core::{Result, anyhow, bail};

use crate::color::ColorModel;
use crate::types::RawInfo;

/// A decoded RAW, cropped to the camera's default crop, still mosaiced (or
/// three-channel for linear RAWs such as Sony's M/S lossless sizes).
pub struct RawFrame {
    /// `as_shot_temperature`/`as_shot_tint` filled in from [`ColorModel`];
    /// `raw_clipped_fraction` is filled in by develop.
    pub info: RawInfo,
    pub color: ColorModel,
    /// As-shot white balance multipliers, green = 1.
    pub wb_as_shot: [f32; 3],
    pub width: usize,
    pub height: usize,
    /// 1 = Bayer mosaic, 3 = linear RGB.
    pub channels: usize,
    /// Bayer colours at (0,0), (0,1), (1,0), (1,1) of the cropped image: 0 = R, 1 = G, 2 = B.
    pub cfa: [u8; 4],
    /// Samples, row-major (interleaved when `channels` is 3), cropped.
    pub data: Vec<u16>,
    /// Black level per CFA position (same order as `cfa`), or per channel for linear RAWs.
    pub black: [f32; 4],
    pub white: f32,
}

/// Smallest cropped image we accept; the demosaic reads 6 pixels around each one.
const MIN_EDGE: usize = 16;

/// Decodes any format rawler supports (every Sony ARW variant, DNG, ...).
pub fn decode(bytes: &[u8]) -> Result<RawFrame> {
    if bytes.len() < 1024 {
        bail!("not a RAW file: only {} bytes", bytes.len());
    }
    // rawler panics on some malformed or unexpected files; a bad upload must
    // fail its run, not take the worker thread down.
    let decoded = catch_unwind(AssertUnwindSafe(|| read_raw(bytes))).map_err(|panic| {
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("unknown panic");
        anyhow!("the RAW decoder failed on this file: {message}")
    })??;
    build_frame(decoded, bytes)
}

struct Decoded {
    image: RawImage,
    metadata: Option<RawMetadata>,
    format: FormatHint,
}

fn read_raw(bytes: &[u8]) -> Result<Decoded> {
    let src = RawSource::new_from_slice(bytes);
    let decoder = rawler::get_decoder(&src)
        .map_err(|e| anyhow!("unsupported or unrecognized RAW file: {e}"))?;
    let params = RawDecodeParams::default();
    let image = decoder
        .raw_image(&src, &params, false)
        .map_err(|e| anyhow!("decode RAW image data: {e}"))?;
    // EXIF is a nicety; a file whose image decodes still develops without it.
    let metadata = decoder.raw_metadata(&src, &params).ok();
    Ok(Decoded {
        image,
        metadata,
        format: decoder.format_hint(),
    })
}

fn build_frame(decoded: Decoded, bytes: &[u8]) -> Result<RawFrame> {
    let Decoded {
        image: img,
        metadata,
        format,
    } = decoded;
    let model_name = format!("{} {}", img.clean_make, img.clean_model);
    let RawImageData::Integer(samples) = &img.data else {
        bail!("{model_name}: floating-point RAW data (HDR DNG) is not supported");
    };
    let channels = match &img.photometric {
        RawPhotometricInterpretation::Cfa(_) => 1,
        RawPhotometricInterpretation::LinearRaw => 3,
        RawPhotometricInterpretation::BlackIsZero => {
            bail!("{model_name}: monochrome RAW files are not supported")
        }
    };
    if img.cpp != channels || samples.len() < img.width * img.height * channels {
        bail!(
            "{model_name}: inconsistent RAW layout ({}x{}, {} samples per pixel, {} samples)",
            img.width,
            img.height,
            img.cpp,
            samples.len()
        );
    }

    let (x0, y0, width, height) = crop_rect(&img);
    let cfa = match &img.photometric {
        RawPhotometricInterpretation::Cfa(config) => {
            if config.cfa.width != 2 || config.cfa.height != 2 {
                bail!(
                    "{model_name}: {} colour filter array is not supported (Bayer only)",
                    config.cfa
                );
            }
            let cfa = [(0, 0), (0, 1), (1, 0), (1, 1)]
                .map(|(dy, dx)| config.cfa.color_at(y0 + dy, x0 + dx) as u8);
            if !is_bayer(cfa) {
                bail!(
                    "{model_name}: {} colour filter array is not supported (RGB Bayer only)",
                    config.cfa
                );
            }
            cfa
        }
        _ => [0, 1, 1, 2],
    };
    let black = black_levels(&img.blacklevel, x0, y0, channels);
    let mut white = img
        .whitelevel
        .0
        .iter()
        .copied()
        .min()
        .unwrap_or(u16::MAX.into()) as f32;
    if white <= black.iter().copied().fold(0.0, f32::max) + 16.0 {
        bail!("{model_name}: white level {white} is not above the black level {black:?}");
    }

    let color =
        color_model(&img).ok_or_else(|| anyhow!("{model_name}: no camera colour matrix"))?;
    let mut data = crop(samples, img.width, channels, x0, y0, width, height);

    // Sony's M and S lossless sizes are YCbCr the camera derived from white
    // balanced RGB: rawler hands them back balanced with multipliers of one.
    // Recover the as-shot multipliers from the SR2 block and undo them
    // losslessly, so these frames mean what every other frame means.
    let wb_as_shot = if channels == 3 && format == FormatHint::ARW {
        let wb = sony_sr2_wb(bytes).unwrap_or_else(|| color.daylight_wb());
        white = unbalance(&mut data, [black[0], black[1], black[2]], white, wb);
        wb
    } else {
        normalized_wb(img.wb_coeffs).unwrap_or_else(|| color.daylight_wb())
    };
    white = effective_white(&data, width, channels, cfa, black, white, wb_as_shot);

    let exif = metadata.as_ref().map(|m| &m.exif);
    let (as_shot_temperature, as_shot_tint) = color.temp_tint_for(wb_as_shot);
    let text = |s: Option<&String>| s.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    let info = RawInfo {
        make: non_empty(&img.clean_make, &img.make),
        model: non_empty(&img.clean_model, &img.model),
        lens: text(exif.and_then(|e| e.lens_model.as_ref())).or_else(|| {
            metadata
                .as_ref()
                .and_then(|m| m.lens.as_ref())
                .map(|lens| lens.lens_name.clone())
        }),
        width: width as u32,
        height: height as u32,
        orientation: exif
            .and_then(|e| e.orientation)
            .filter(|o| (1..=8).contains(o))
            .unwrap_or(1) as u8,
        iso: exif.and_then(|e| {
            e.iso_speed_ratings
                .filter(|&iso| iso > 0 && iso < u16::MAX)
                .map(u32::from)
                .or(e.recommended_exposure_index.filter(|&iso| iso > 0))
                .or(e.iso_speed.filter(|&iso| iso > 0))
                .or(e.iso_speed_ratings.map(u32::from))
        }),
        exposure_time_s: exif.and_then(|e| ratio(e.exposure_time)),
        f_number: exif.and_then(|e| ratio(e.fnumber)),
        focal_length_mm: exif.and_then(|e| ratio(e.focal_length)),
        captured_at: exif
            .and_then(|e| text(e.date_time_original.as_ref()).or(text(e.create_date.as_ref()))),
        as_shot_temperature,
        as_shot_tint,
        raw_clipped_fraction: 0.0,
    };

    Ok(RawFrame {
        info,
        color,
        wb_as_shot,
        width,
        height,
        channels,
        cfa,
        data,
        black,
        white,
    })
}

/// The camera's default crop (falling back to the active area, then to the
/// whole buffer), clipped to the buffer: (x, y, width, height).
fn crop_rect(img: &RawImage) -> (usize, usize, usize, usize) {
    let full = (0, 0, img.width, img.height);
    [img.crop_area, img.active_area]
        .into_iter()
        .flatten()
        .map(|r| {
            let x = r.p.x.min(img.width);
            let y = r.p.y.min(img.height);
            (x, y, r.d.w.min(img.width - x), r.d.h.min(img.height - y))
        })
        .find(|&(_, _, w, h)| w >= MIN_EDGE && h >= MIN_EDGE)
        .unwrap_or(full)
}

fn crop(
    samples: &[u16],
    full_width: usize,
    channels: usize,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
) -> Vec<u16> {
    let mut out = Vec::with_capacity(width * height * channels);
    for row in y0..y0 + height {
        let start = (row * full_width + x0) * channels;
        out.extend_from_slice(&samples[start..start + width * channels]);
    }
    out
}

/// One red, two greens, one blue.
fn is_bayer(cfa: [u8; 4]) -> bool {
    let count = |c| cfa.iter().filter(|&&v| v == c).count();
    count(0) == 1 && count(1) == 2 && count(2) == 1
}

/// Black per CFA position of the cropped image, or per channel for linear RAWs.
fn black_levels(levels: &BlackLevel, x0: usize, y0: usize, channels: usize) -> [f32; 4] {
    let values = levels.as_vec();
    let (w, h, cpp) = (levels.width.max(1), levels.height.max(1), levels.cpp.max(1));
    let at = |row: usize, col: usize, ch: usize| {
        let i = ((row % h) * w + col % w) * cpp + ch.min(cpp - 1);
        values.get(i).or(values.first()).copied().unwrap_or(0.0)
    };
    if channels == 3 {
        [at(y0, x0, 0), at(y0, x0, 1), at(y0, x0, 2), 0.0]
    } else {
        [(0, 0), (0, 1), (1, 0), (1, 1)].map(|(dy, dx)| at(y0 + dy, x0 + dx, 0))
    }
}

/// rawler's multipliers as R, G, B with green = 1, or None when unusable.
fn normalized_wb(coeffs: [f32; 4]) -> Option<[f32; 3]> {
    let [r, g, b, _] = coeffs;
    let valid = |v: f32| v.is_finite() && v > 0.0;
    (valid(r) && valid(g) && valid(b)).then(|| [r / g, 1.0, b / g])
}

/// DNG-style XYZ → camera matrices: the daylight one and, when the camera
/// has it, the tungsten one.
fn color_model(img: &RawImage) -> Option<ColorModel> {
    let matrix = |illuminants: &[Illuminant]| {
        illuminants.iter().find_map(|illuminant| {
            let m = img.color_matrix.get(illuminant)?;
            (m.len() == 9 && m.iter().all(|v| v.is_finite()))
                .then(|| [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]])
        })
    };
    let daylight = matrix(&[
        Illuminant::D65,
        Illuminant::D55,
        Illuminant::Daylight,
        Illuminant::FineWeather,
        Illuminant::Flash,
        Illuminant::D50,
        Illuminant::D75,
        Illuminant::CloudyWeather,
        Illuminant::Shade,
    ]);
    let tungsten = matrix(&[
        Illuminant::A,
        Illuminant::Tungsten,
        Illuminant::IsoStudioTungsten,
    ]);
    match (daylight, tungsten) {
        (Some(d65), a) => Some(ColorModel::new(d65, a)),
        (None, Some(a)) => Some(ColorModel::new(a, None)),
        (None, None) => None,
    }
}

/// Divides linear RGB that the camera already white balanced by `wb`,
/// scaling every channel up by the largest multiplier first so no code value
/// is lost. Returns the white level that keeps `(v - black) / (white - black)
/// * wb` equal to the camera's balanced value.
fn unbalance(data: &mut [u16], black: [f32; 3], white: f32, wb: [f32; 3]) -> f32 {
    let mut max = [0u16; 3];
    for px in data.as_chunks::<3>().0 {
        for (m, &v) in max.iter_mut().zip(px) {
            *m = (*m).max(v);
        }
    }
    let mut k = wb.iter().copied().fold(1.0, f32::max);
    for ((&m, &b), &mult) in max.iter().zip(&black).zip(&wb) {
        let span = f32::from(m) - b;
        if span > 0.0 {
            k = k.min((f32::from(u16::MAX) - b) * mult / span);
        }
    }
    let gain = wb.map(|m| k / m);
    for px in data.as_chunks_mut::<3>().0 {
        for ((v, &b), &g) in px.iter_mut().zip(&black).zip(&gain) {
            *v = (b + (f32::from(*v) - b) * g)
                .round()
                .clamp(0.0, f32::from(u16::MAX)) as u16;
        }
    }
    let mean_black = (black[0] + black[1] + black[2]) / 3.0;
    mean_black + (white - mean_black) * k
}

/// The declared white level, lowered when a channel visibly saturates below
/// it at a level that, after white balance, would not reach the neutral clip
/// point: otherwise clipped highlights would come out pink or green.
///
/// Saturation shows as a spike: many samples at exactly the channel maximum
/// and few just below it (a smooth highlight never looks like that).
fn effective_white(
    data: &[u16],
    width: usize,
    channels: usize,
    cfa: [u8; 4],
    black: [f32; 4],
    white: f32,
    wb: [f32; 3],
) -> f32 {
    const SPIKE_MIN: u32 = 64;
    const SPIKE_RATIO: u32 = 8;
    const WINDOW: u16 = 16;

    // Slots are CFA positions (Bayer) or channels (linear).
    let slots = if channels == 1 { 4 } else { 3 };
    let mut max = [0u16; 4];
    for_each_slot(data, width, channels, |slot, v| {
        max[slot] = max[slot].max(v)
    });
    let needs_check: Vec<usize> = (0..slots).filter(|&s| f32::from(max[s]) < white).collect();
    if needs_check.is_empty() {
        return white;
    }
    // Counts of the top WINDOW + 1 code values per slot.
    let floor = max.map(|m| m.saturating_sub(WINDOW));
    let mut counts = [[0u32; WINDOW as usize + 1]; 4];
    for_each_slot(data, width, channels, |slot, v| {
        if v >= floor[slot] {
            counts[slot][usize::from(max[slot] - v)] += 1;
        }
    });
    let clip = wb.iter().copied().fold(f32::MAX, f32::min);
    let mut effective = white;
    for s in needs_check {
        let at_max = counts[s][0];
        let near = counts[s][1..].iter().copied().max().unwrap_or(0);
        if at_max < SPIKE_MIN || at_max < SPIKE_RATIO * near {
            continue;
        }
        let (color, slot_black) = if channels == 1 {
            (cfa[s] as usize, black[s])
        } else {
            (s, black[s])
        };
        // This slot saturates at max[s]; it must still reach the clip level
        // after its white balance gain.
        let saturated = (f32::from(max[s]) - slot_black) * wb[color] / clip;
        if slot_black + saturated < effective {
            effective = (slot_black + saturated).max(slot_black + MIN_EDGE as f32);
        }
    }
    effective
}

/// Calls `f(slot, sample)` for every sample, where slots are CFA positions
/// (Bayer, row-major) or channels (linear RGB). Row-wise so no sample costs
/// a division.
fn for_each_slot(data: &[u16], width: usize, channels: usize, mut f: impl FnMut(usize, u16)) {
    if channels == 3 {
        for &[r, g, b] in data.as_chunks::<3>().0 {
            f(0, r);
            f(1, g);
            f(2, b);
        }
        return;
    }
    for (row, line) in data.chunks_exact(width.max(1)).enumerate() {
        let base = (row & 1) * 2;
        let (pairs, rest) = line.as_chunks::<2>();
        for &[even, odd] in pairs {
            f(base, even);
            f(base + 1, odd);
        }
        if let Some(&v) = rest.first() {
            f(base, v);
        }
    }
}

fn ratio(r: Option<Rational>) -> Option<f32> {
    r.filter(|r| r.d != 0 && r.n != 0)
        .map(|r| r.n as f32 / r.d as f32)
}

fn non_empty(clean: &str, raw: &str) -> String {
    let s = if clean.trim().is_empty() { raw } else { clean };
    s.trim().to_string()
}

/// As-shot multipliers from a Sony ARW's encrypted SR2 private block
/// (WB_GRBGLevels 0x7303 or WB_RGGBLevels 0x7313), green = 1. rawler reads
/// the same tags for Bayer files but drops them for YCbCr ones.
fn sony_sr2_wb(file: &[u8]) -> Option<[f32; 3]> {
    let tiff = Tiff::new(file)?;
    let ifd0 = tiff.u32(4)? as usize;
    let private = tiff.entry(ifd0, 0xc634)?;
    let private_ifd = tiff.u32(private.value_at)? as usize;
    let offset = tiff
        .entry(private_ifd, 0x7200)
        .and_then(|e| tiff.u32(e.value_at))? as usize;
    let length = tiff
        .entry(private_ifd, 0x7201)
        .and_then(|e| tiff.u32(e.value_at))? as usize;
    let key = tiff
        .entry(private_ifd, 0x7221)
        .and_then(|e| tiff.u32(e.value_at))?;
    let encrypted = file.get(offset..offset.checked_add(length)?)?;
    let sr2 = sony_decrypt(encrypted, key);
    let sr2 = Tiff {
        bytes: &sr2,
        little_endian: tiff.little_endian,
    };
    // Entry values that do not fit inline point at absolute file offsets.
    let levels = |tag: u16| {
        let e = sr2.entry(0, tag)?;
        if e.count != 4 || !matches!(e.kind, 3 | 8) {
            return None;
        }
        let at = (sr2.u32(e.value_at)? as usize).checked_sub(offset)?;
        Some([0, 1, 2, 3].map(|i| sr2.u16(at + 2 * i).map(f32::from)))
    };
    let [r, g, b] = match (levels(0x7303), levels(0x7313)) {
        (Some([Some(g1), Some(r), Some(b), Some(g2)]), _) => [r, (g1 + g2) / 2.0, b],
        (_, Some([Some(r), Some(g1), Some(g2), Some(b)])) => [r, (g1 + g2) / 2.0, b],
        _ => return None,
    };
    normalized_wb([r, g, b, f32::NAN])
}

/// Sony's SR2 cipher (dcraw `sony_decrypt`): a keystream XOR, so applying it
/// twice gives the input back.
fn sony_decrypt(data: &[u8], key: u32) -> Vec<u8> {
    let mut pad = [0u32; 128];
    let mut k = key;
    for p in pad.iter_mut().take(4) {
        k = k.wrapping_mul(48_828_125).wrapping_add(1);
        *p = k;
    }
    pad[3] = (pad[3] << 1) | ((pad[0] ^ pad[2]) >> 31);
    for p in 4..127 {
        pad[p] = ((pad[p - 4] ^ pad[p - 2]) << 1) | ((pad[p - 3] ^ pad[p - 1]) >> 31);
    }
    for p in pad.iter_mut().take(127) {
        *p = u32::from_be(*p);
    }
    let mut out = Vec::with_capacity(data.len());
    for (i, word) in data.as_chunks::<4>().0.iter().enumerate() {
        let p = i + 127;
        pad[p & 127] = pad[(p + 1) & 127] ^ pad[(p + 65) & 127];
        let w = u32::from_le_bytes(*word) ^ pad[p & 127];
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// Just enough TIFF to walk IFD entries.
struct Tiff<'a> {
    bytes: &'a [u8],
    little_endian: bool,
}

struct Entry {
    kind: u16,
    count: u32,
    /// Offset of the 4-byte value/offset field.
    value_at: usize,
}

impl<'a> Tiff<'a> {
    fn new(bytes: &'a [u8]) -> Option<Self> {
        let little_endian = match bytes.get(0..4)? {
            [b'I', b'I', 42, 0] => true,
            [b'M', b'M', 0, 42] => false,
            _ => return None,
        };
        Some(Self {
            bytes,
            little_endian,
        })
    }

    fn u16(&self, at: usize) -> Option<u16> {
        let b: [u8; 2] = self.bytes.get(at..at + 2)?.try_into().ok()?;
        Some(if self.little_endian {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        })
    }

    fn u32(&self, at: usize) -> Option<u32> {
        let b: [u8; 4] = self.bytes.get(at..at + 4)?.try_into().ok()?;
        Some(if self.little_endian {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    }

    fn entry(&self, ifd: usize, tag: u16) -> Option<Entry> {
        let count = self.u16(ifd)? as usize;
        (0..count.min(4096)).find_map(|i| {
            let at = ifd + 2 + 12 * i;
            (self.u16(at)? == tag).then_some(())?;
            Some(Entry {
                kind: self.u16(at + 2)?,
                count: self.u32(at + 4)?,
                value_at: at + 8,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayer_patterns() {
        assert!(is_bayer([0, 1, 1, 2]));
        assert!(is_bayer([1, 2, 0, 1]));
        assert!(!is_bayer([0, 1, 1, 1]));
        assert!(!is_bayer([0, 1, 3, 2]));
    }

    #[test]
    fn black_follows_the_crop_origin() {
        let levels = BlackLevel::new(&[500u16, 510, 520, 530], 2, 2, 1);
        assert_eq!(black_levels(&levels, 0, 0, 1), [500.0, 510.0, 520.0, 530.0]);
        assert_eq!(black_levels(&levels, 1, 0, 1), [510.0, 500.0, 530.0, 520.0]);
        assert_eq!(
            black_levels(&levels, 13, 13, 1),
            [530.0, 520.0, 510.0, 500.0]
        );
        let linear = BlackLevel::new(&[1024u16, 1024, 1024], 1, 1, 3);
        assert_eq!(
            black_levels(&linear, 8, 4, 3),
            [1024.0, 1024.0, 1024.0, 0.0]
        );
    }

    #[test]
    fn wb_is_normalized_to_green() {
        assert_eq!(
            normalized_wb([2.0, 1.0, 1.5, f32::NAN]),
            Some([2.0, 1.0, 1.5])
        );
        assert_eq!(
            normalized_wb([4.0, 2.0, 3.0, f32::NAN]),
            Some([2.0, 1.0, 1.5])
        );
        assert_eq!(normalized_wb([f32::NAN, 1.0, 1.0, 1.0]), None);
        assert_eq!(normalized_wb([2.0, 0.0, 1.0, 1.0]), None);
    }

    #[test]
    fn unbalance_round_trips_balanced_values() {
        let black = [1024.0; 3];
        let wb = [2.4, 1.0, 1.7];
        let original: Vec<u16> = vec![
            1024, 1024, 1024, // black
            3000, 3000, 3000, // balanced grey
            20000, 19000, 21000, // clipped, overshooting white
            1000, 1100, 1010, // below black
        ];
        let mut data = original.clone();
        let white = unbalance(&mut data, black, 15360.0, wb);
        for (px, orig) in data
            .as_chunks::<3>()
            .0
            .iter()
            .zip(original.as_chunks::<3>().0)
        {
            for (((&v, &o), &b), &m) in px.iter().zip(orig).zip(&black).zip(&wb) {
                let balanced = (f32::from(v) - b) / (white - b) * m;
                let expected = (f32::from(o) - b) / (15360.0 - b);
                assert!((balanced - expected).abs() < 1e-4, "{px:?} vs {orig:?}");
            }
        }
    }

    #[test]
    fn white_drops_only_for_a_saturation_spike_below_it() {
        let (w, h) = (64, 64);
        let wb = [2.0, 1.0, 1.5];
        let black = [512.0; 4];
        let cfa = [0, 1, 1, 2];
        // Smooth data, no clipping: the declared white stays.
        let smooth: Vec<u16> = (0..w * h).map(|i| 600 + (i % 4000) as u16).collect();
        assert_eq!(
            effective_white(&smooth, w, 1, cfa, black, 15360.0, wb),
            15360.0
        );
        // Green saturates at 14000 over a large area: white must come down to it.
        let mut clipped = smooth.clone();
        for (i, v) in clipped.iter_mut().enumerate().take(w * 32) {
            // The greens of an RGGB mosaic pile up at 14000; red and blue stay smooth.
            *v = if (i / w + i % w) % 2 == 1 {
                14000
            } else {
                3000 + (i % 997) as u16
            };
        }
        let white = effective_white(&clipped, w, 1, cfa, black, 15360.0, wb);
        assert!((white - 14000.0).abs() < 0.5, "{white}");
        // Data reaching the declared white needs no change.
        let mut over = smooth;
        over[0] = 16383;
        over[1] = 16383;
        over[w] = 16383;
        over[w + 1] = 16383;
        assert_eq!(
            effective_white(&over, w, 1, cfa, black, 15360.0, wb),
            15360.0
        );
    }

    #[test]
    fn sr2_white_balance_is_found_and_decrypted() {
        // A little-endian TIFF: IFD0 → DNGPrivateData → SR2Private IFD →
        // encrypted SR2SubIFD holding WB_RGGBLevels at an absolute offset.
        let key = 0x1234_5678;
        let sr2_offset = 256usize;
        let mut sr2 = vec![0u8; 64];
        sr2[0..2].copy_from_slice(&1u16.to_le_bytes());
        sr2[2..4].copy_from_slice(&0x7313u16.to_le_bytes());
        sr2[4..6].copy_from_slice(&8u16.to_le_bytes());
        sr2[6..10].copy_from_slice(&4u32.to_le_bytes());
        sr2[10..14].copy_from_slice(&((sr2_offset + 32) as u32).to_le_bytes());
        for (i, v) in [2425u16, 1024, 1024, 1816].iter().enumerate() {
            sr2[32 + 2 * i..34 + 2 * i].copy_from_slice(&v.to_le_bytes());
        }
        let encrypted = sony_decrypt(&sr2, key);
        assert_eq!(sony_decrypt(&encrypted, key), sr2);

        let mut file = vec![0u8; 2048];
        file[0..4].copy_from_slice(b"II*\0");
        file[4..8].copy_from_slice(&8u32.to_le_bytes());
        let entry = |file: &mut Vec<u8>, at: usize, tag: u16, kind: u16, count: u32, value: u32| {
            file[at..at + 2].copy_from_slice(&tag.to_le_bytes());
            file[at + 2..at + 4].copy_from_slice(&kind.to_le_bytes());
            file[at + 4..at + 8].copy_from_slice(&count.to_le_bytes());
            file[at + 8..at + 12].copy_from_slice(&value.to_le_bytes());
        };
        file[8..10].copy_from_slice(&1u16.to_le_bytes());
        entry(&mut file, 10, 0xc634, 1, 4, 100);
        file[100..102].copy_from_slice(&3u16.to_le_bytes());
        entry(&mut file, 102, 0x7200, 4, 1, sr2_offset as u32);
        entry(&mut file, 114, 0x7201, 4, 1, sr2.len() as u32);
        entry(&mut file, 126, 0x7221, 4, 1, key);
        file[sr2_offset..sr2_offset + encrypted.len()].copy_from_slice(&encrypted);

        let wb = sony_sr2_wb(&file).expect("white balance");
        assert!((wb[0] - 2425.0 / 1024.0).abs() < 1e-6 && wb[1] == 1.0);
        assert!((wb[2] - 1816.0 / 1024.0).abs() < 1e-6);
        assert!(sony_sr2_wb(&file[..200]).is_none());
    }

    #[test]
    fn garbage_is_a_clear_error() {
        let err = decode(&[0u8; 4096]).err().expect("error");
        assert!(err.to_string().contains("RAW"), "{err}");
        assert!(decode(b"tiny").is_err());
    }
}
