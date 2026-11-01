//! Image files: orientation, resizing, JPEG and 16-bit TIFF. Owner: judge.

use std::io::Cursor;
use std::sync::OnceLock;

use jpeg_encoder::{ChromaSubsamplingMethod, ColorType, Encoder, SamplingFactor};
use spl_core::{Result, anyhow, bail};
use tiff::encoder::{Compression, DeflateLevel, Predictor, TiffEncoder, colortype};

use crate::types::{Rgb8Image, Rgb16Image};

/// Applies EXIF orientation (1 to 8) so the result is upright.
pub fn orient(img: &Rgb8Image, orientation: u8) -> Rgb8Image {
    let (width, height, data) = orient_pixels(&img.data, img.width, img.height, orientation);
    Rgb8Image {
        width,
        height,
        data,
    }
}

/// [`orient`] for 16-bit images.
pub fn orient16(img: &Rgb16Image, orientation: u8) -> Rgb16Image {
    let (width, height, data) = orient_pixels(&img.data, img.width, img.height, orientation);
    Rgb16Image {
        width,
        height,
        data,
    }
}

/// Where output pixel (x, y) comes from, as a pixel index affine in x and
/// y: `base + x * dx + y * dy`. Returns the output size too.
fn orientation_map(w: usize, h: usize, orientation: u8) -> (usize, usize, isize, isize, isize) {
    let (wi, hi) = (w as isize, h as isize);
    let last_row = (hi - 1) * wi;
    match orientation {
        2 => (w, h, wi - 1, -1, wi),
        3 => (w, h, last_row + wi - 1, -1, -wi),
        4 => (w, h, last_row, 1, -wi),
        5 => (h, w, 0, wi, 1),
        6 => (h, w, last_row, -wi, 1),
        7 => (h, w, last_row + wi - 1, -wi, -1),
        8 => (h, w, wi - 1, wi, -1),
        // 1, and the 0 or out-of-range values some writers emit, mean upright.
        _ => (w, h, 0, 1, wi),
    }
}

fn orient_pixels<T: Copy>(
    src: &[T],
    w: usize,
    h: usize,
    orientation: u8,
) -> (usize, usize, Vec<T>) {
    let (ow, oh, base, dx, dy) = orientation_map(w, h, orientation);
    // Upright images, and malformed buffers rather than a panic, pass through.
    if w == 0 || h == 0 || src.len() < w * h * 3 || (dx, dy) == (1, w as isize) {
        return (w, h, src.to_vec());
    }
    let mut out = src[..w * h * 3].to_vec();
    // Tiles keep the strided reads of the rotating orientations inside the cache.
    const TILE: usize = 64;
    for ty in (0..oh).step_by(TILE) {
        for tx in (0..ow).step_by(TILE) {
            for y in ty..(ty + TILE).min(oh) {
                let row_src = base + y as isize * dy;
                let row_out = y * ow;
                for x in tx..(tx + TILE).min(ow) {
                    let s = (row_src + x as isize * dx) as usize * 3;
                    let d = (row_out + x) * 3;
                    out[d..d + 3].copy_from_slice(&src[s..s + 3]);
                }
            }
        }
    }
    (ow, oh, out)
}

/// The 8x8 Bayer threshold matrix (values 0 to 63).
const BAYER8: [[u8; 8]; 8] = [
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
];

/// 16 → 8 bits with light dithering.
///
/// Ordered dithering of one 8-bit step: each sample becomes
/// `floor(v * 255 / 65535 + t)` with a Bayer threshold `t` in (0, 1), so
/// smooth skies do not band, values that are exact in 8 bits stay exact, and
/// the local mean is preserved. All channels of a pixel share the threshold,
/// so greys stay grey.
pub fn to_rgb8(img: &Rgb16Image) -> Rgb8Image {
    let (w, h) = (img.width, img.height);
    if w == 0 || h == 0 {
        return Rgb8Image {
            width: w,
            height: h,
            data: Vec::new(),
        };
    }
    // t * 65535 for t = (b + 0.5) / 64, so floor((v * 255 + T) / 65535) is the dithered value.
    let thresholds = BAYER8.map(|row| row.map(|b| (u32::from(b) * 2 + 1) * 65535 / 128));
    let mut data = Vec::with_capacity(w * h * 3);
    for (y, row) in img.data.chunks_exact(w * 3).take(h).enumerate() {
        let t_row = &thresholds[y & 7];
        for (x, px) in row.as_chunks::<3>().0.iter().enumerate() {
            let t = t_row[x & 7];
            for &v in px {
                data.push(((u32::from(v) * 255 + t) / 65535).min(255) as u8);
            }
        }
    }
    Rgb8Image {
        width: w,
        height: h,
        data,
    }
}

/// Area-average downscale so the long edge is at most `long_edge`; never upscales.
///
/// Every output pixel is the exact average of the source area it covers,
/// fractional edge pixels weighted by coverage, computed in linear light so
/// fine bright detail keeps its brightness.
pub fn resize(img: &Rgb8Image, long_edge: usize) -> Rgb8Image {
    let (w, h) = (img.width, img.height);
    let target = long_edge.max(1);
    if w == 0 || h == 0 || w.max(h) <= target || img.data.len() < w * h * 3 {
        return img.clone();
    }
    let (ow, oh) = if w >= h {
        (target, scaled(h, target, w))
    } else {
        (scaled(w, target, h), target)
    };
    let cols = coverage(w, ow);
    let rows = coverage(h, oh);
    let lut = srgb_to_linear_lut();

    let mut out = Vec::with_capacity(ow * oh * 3);
    let mut acc = vec![0f32; ow * 3];
    // The source row on the boundary between two output rows is used twice.
    let mut cached: Option<(usize, Vec<f32>)> = None;
    for span in &rows {
        acc.fill(0.0);
        for (k, &wy) in span.weights.iter().enumerate() {
            let sy = span.start + k;
            let hrow = match cached.take() {
                Some((y, row)) if y == sy => row,
                _ => resample_row(&img.data[sy * w * 3..(sy + 1) * w * 3], &cols, lut),
            };
            for (a, v) in acc.iter_mut().zip(&hrow) {
                *a += wy * v;
            }
            cached = Some((sy, hrow));
        }
        out.extend(acc.iter().map(|&v| linear_to_srgb8(v)));
    }
    Rgb8Image {
        width: ow,
        height: oh,
        data: out,
    }
}

fn scaled(short: usize, target: usize, long: usize) -> usize {
    ((short as f64 * target as f64 / long as f64).round() as usize).max(1)
}

/// The source pixels an output pixel covers and their weights, which sum to 1.
struct Span {
    start: usize,
    weights: Vec<f32>,
}

/// Exact box coverage of `n_out` equal output intervals over `n_in` source pixels.
fn coverage(n_in: usize, n_out: usize) -> Vec<Span> {
    let scale = n_in as f64 / n_out as f64;
    (0..n_out)
        .map(|o| {
            let (lo, hi) = (o as f64 * scale, ((o + 1) as f64 * scale).min(n_in as f64));
            let start = lo.floor() as usize;
            let end = (hi.ceil() as usize).clamp(start + 1, n_in);
            let weights = (start..end)
                .map(|i| {
                    let overlap = hi.min((i + 1) as f64) - lo.max(i as f64);
                    (overlap.max(0.0) / scale) as f32
                })
                .collect();
            Span { start, weights }
        })
        .collect()
}

fn resample_row(row: &[u8], cols: &[Span], lut: &[f32; 256]) -> Vec<f32> {
    let mut out = Vec::with_capacity(cols.len() * 3);
    for span in cols {
        let mut sum = [0f32; 3];
        for (k, &wx) in span.weights.iter().enumerate() {
            let p = (span.start + k) * 3;
            for c in 0..3 {
                sum[c] += wx * lut[row[p + c] as usize];
            }
        }
        out.extend_from_slice(&sum);
    }
    out
}

fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_to_linear_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| std::array::from_fn(|i| srgb_to_linear(i as f64 / 255.0) as f32))
}

/// Linear light → the nearest 8-bit sRGB code: the number of decision
/// thresholds (midpoints between codes, in linear light) at or below `v`.
fn linear_to_srgb8(v: f32) -> u8 {
    static THRESHOLDS: OnceLock<[f32; 255]> = OnceLock::new();
    let t = THRESHOLDS
        .get_or_init(|| std::array::from_fn(|k| srgb_to_linear((k as f64 + 0.5) / 255.0) as f32));
    t.partition_point(|&th| th <= v) as u8
}

/// Baseline sRGB JPEG with 4:4:4 chroma at quality >= 90.
///
/// Below 90 the chroma is 4:2:0, box-averaged rather than point-sampled.
pub fn jpeg(img: &Rgb8Image, quality: u8) -> Result<Vec<u8>> {
    let (w, h) = check_size(img.width, img.height, img.data.len(), 65535)?;
    let quality = quality.clamp(1, 100);
    let mut out = Vec::with_capacity(img.width * img.height / 3 + 4096);
    let mut encoder = Encoder::new(&mut out, quality);
    encoder.set_sampling_factor(if quality >= 90 {
        SamplingFactor::F_1_1
    } else {
        SamplingFactor::F_2_2
    });
    encoder.set_chroma_subsampling_method(ChromaSubsamplingMethod::Average);
    encoder
        .encode(&img.data, w as u16, h as u16, ColorType::Rgb)
        .map_err(|e| anyhow!("encode JPEG ({w}x{h}): {e}"))?;
    Ok(out)
}

/// 16-bit RGB TIFF, losslessly compressed with Deflate and horizontal
/// prediction (every photo editor reads it): about 30% smaller than raw
/// samples for under a second at 24 MP.
pub fn tiff16(img: &Rgb16Image) -> Result<Vec<u8>> {
    let (w, h) = check_size(img.width, img.height, img.data.len(), u32::MAX as usize)?;
    let mut buf = Cursor::new(Vec::with_capacity(img.data.len() * 2 / 3));
    let mut encoder = TiffEncoder::new(&mut buf)
        .map_err(|e| anyhow!("start TIFF: {e}"))?
        .with_compression(Compression::Deflate(DeflateLevel::Fast))
        .with_predictor(Predictor::Horizontal);
    encoder
        .write_image::<colortype::RGB16>(w, h, &img.data[..img.width * img.height * 3])
        .map_err(|e| anyhow!("encode TIFF ({w}x{h}): {e}"))?;
    Ok(buf.into_inner())
}

fn check_size(w: usize, h: usize, len: usize, max: usize) -> Result<(u32, u32)> {
    if w == 0 || h == 0 {
        bail!("image is empty ({w}x{h})");
    }
    if w > max || h > max {
        bail!("image is too large for the format ({w}x{h}, at most {max} per side)");
    }
    if len < w * h * 3 {
        bail!(
            "image data has {len} samples, {w}x{h} RGB needs {}",
            w * h * 3
        );
    }
    Ok((w as u32, h as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image(w: usize, h: usize) -> Rgb8Image {
        let data = (0..w * h)
            .flat_map(|i| [(i % 251) as u8, (i / 251 % 251) as u8, (i * 7 % 253) as u8])
            .collect();
        Rgb8Image {
            width: w,
            height: h,
            data,
        }
    }

    fn px(img: &Rgb8Image, x: usize, y: usize) -> [u8; 3] {
        let i = (y * img.width + x) * 3;
        [img.data[i], img.data[i + 1], img.data[i + 2]]
    }

    /// The camera's view of an upright scene under each EXIF orientation,
    /// built from the tag's definition (where the stored 0th row and 0th
    /// column sit in the upright picture), independently of `orient`.
    fn stored_from_upright(up: &Rgb8Image, orientation: u8) -> Rgb8Image {
        let (vw, vh) = (up.width, up.height);
        // (row 0 is at: top/bottom/left/right, column 0 is at: left/right/top/bottom)
        let (row0, col0) = match orientation {
            1 => ("top", "left"),
            2 => ("top", "right"),
            3 => ("bottom", "right"),
            4 => ("bottom", "left"),
            5 => ("left", "top"),
            6 => ("right", "top"),
            7 => ("right", "bottom"),
            8 => ("left", "bottom"),
            _ => unreachable!(),
        };
        let rows_are_rows = matches!(row0, "top" | "bottom");
        let (sw, sh) = if rows_are_rows { (vw, vh) } else { (vh, vw) };
        let mut data = vec![0u8; sw * sh * 3];
        for sy in 0..sh {
            for sx in 0..sw {
                let (vx, vy) = if rows_are_rows {
                    let vy = if row0 == "top" { sy } else { vh - 1 - sy };
                    let vx = if col0 == "left" { sx } else { vw - 1 - sx };
                    (vx, vy)
                } else {
                    let vx = if row0 == "left" { sy } else { vw - 1 - sy };
                    let vy = if col0 == "top" { sx } else { vh - 1 - sx };
                    (vx, vy)
                };
                let d = (sy * sw + sx) * 3;
                data[d..d + 3].copy_from_slice(&px(up, vx, vy));
            }
        }
        Rgb8Image {
            width: sw,
            height: sh,
            data,
        }
    }

    fn same(a: &Rgb8Image, b: &Rgb8Image) -> bool {
        a.width == b.width && a.height == b.height && a.data == b.data
    }

    #[test]
    fn orient_makes_every_exif_orientation_upright() {
        // Odd, non-square and larger than a tile, so tiles have ragged edges.
        for (w, h) in [(3, 2), (1, 5), (67, 130), (131, 70)] {
            let upright = test_image(w, h);
            for o in 1..=8 {
                let stored = stored_from_upright(&upright, o);
                let shown = orient(&stored, o);
                assert!(same(&shown, &upright), "orientation {o} at {w}x{h}");
            }
        }
    }

    #[test]
    fn orient_round_trips_through_the_inverse() {
        let img = test_image(70, 33);
        for (o, inverse) in [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 8),
            (7, 7),
            (8, 6),
        ] {
            let back = orient(&orient(&img, o), inverse);
            assert!(same(&back, &img), "orientation {o}");
        }
        assert!(same(&orient(&img, 0), &img));
        assert!(same(&orient(&img, 9), &img));
    }

    #[test]
    fn orient_passes_malformed_images_through() {
        let short = Rgb8Image {
            width: 4,
            height: 3,
            data: vec![7; 20],
        };
        for o in 1..=8 {
            let out = orient(&short, o);
            assert_eq!((out.width, out.height, out.data.len()), (4, 3, 20));
        }
        let empty = orient(&Rgb8Image::default(), 6);
        assert_eq!((empty.width, empty.height), (0, 0));
    }

    #[test]
    fn orient16_matches_orient() {
        let img8 = test_image(41, 17);
        let img16 = Rgb16Image {
            width: 41,
            height: 17,
            data: img8.data.iter().map(|&v| u16::from(v) * 257 + 3).collect(),
        };
        for o in 1..=8 {
            let a = orient(&img8, o);
            let b = orient16(&img16, o);
            assert_eq!((a.width, a.height), (b.width, b.height));
            let back: Vec<u8> = b.data.iter().map(|&v| ((v - 3) / 257) as u8).collect();
            assert_eq!(a.data, back, "orientation {o}");
        }
    }

    #[test]
    fn rotations_compose_like_rotations() {
        let img = test_image(23, 9);
        // Two quarter turns clockwise are a half turn; transpose is its own inverse.
        assert!(same(&orient(&orient(&img, 6), 6), &orient(&img, 3)));
        assert!(same(&orient(&orient(&img, 8), 8), &orient(&img, 3)));
        assert!(same(&orient(&orient(&img, 2), 4), &orient(&img, 3)));
        assert!(same(&orient(&orient(&img, 5), 2), &orient(&img, 6)));
    }

    #[test]
    fn to_rgb8_keeps_exact_values_and_means() {
        // Values exact in 8 bits stay exact under any threshold.
        let exact = Rgb16Image {
            width: 256,
            height: 8,
            data: (0..256 * 8)
                .flat_map(|i| {
                    let v = (i % 256) as u16 * 257;
                    [v, v, v]
                })
                .collect(),
        };
        let out = to_rgb8(&exact);
        for (i, px) in out.data.as_chunks::<3>().0.iter().enumerate() {
            assert_eq!(*px, [(i % 256) as u8; 3]);
        }

        // An in-between value dithers between its two neighbours, unbiased,
        // and greys stay grey.
        let v = (100.3f64 * 257.0).round() as u16;
        let flat = Rgb16Image {
            width: 64,
            height: 64,
            data: vec![v; 64 * 64 * 3],
        };
        let out = to_rgb8(&flat);
        assert!(out.data.iter().all(|&c| c == 100 || c == 101));
        assert!(
            out.data
                .as_chunks::<3>()
                .0
                .iter()
                .all(|p| p[0] == p[1] && p[1] == p[2])
        );
        let mean = out.data.iter().map(|&c| f64::from(c)).sum::<f64>() / out.data.len() as f64;
        let want = f64::from(v) * 255.0 / 65535.0;
        assert!((mean - want).abs() < 0.01, "mean {mean} want {want}");
    }

    #[test]
    fn resize_area_averages_in_linear_light() {
        // A 2x2 checkerboard of black and white averages to half the light,
        // which is sRGB 188, not 128.
        let mut img = Rgb8Image {
            width: 64,
            height: 32,
            data: vec![0; 64 * 32 * 3],
        };
        for y in 0..32 {
            for x in 0..64 {
                if (x + y) % 2 == 0 {
                    let i = (y * 64 + x) * 3;
                    img.data[i..i + 3].copy_from_slice(&[255, 255, 255]);
                }
            }
        }
        let small = resize(&img, 32);
        assert_eq!((small.width, small.height), (32, 16));
        assert!(
            small.data.iter().all(|&v| v == 188),
            "{:?}",
            &small.data[..6]
        );
    }

    #[test]
    fn resize_keeps_flat_colour_and_aspect() {
        let img = Rgb8Image {
            width: 1500,
            height: 1000,
            data: [12u8, 200, 77].repeat(1500 * 1000),
        };
        let small = resize(&img, 320);
        assert_eq!((small.width, small.height), (320, 213));
        assert!(
            small
                .data
                .as_chunks::<3>()
                .0
                .iter()
                .all(|p| *p == [12, 200, 77])
        );

        let tall = resize(&test_image(300, 1000), 100);
        assert_eq!((tall.width, tall.height), (30, 100));
    }

    #[test]
    fn resize_handles_fractional_coverage() {
        // 5 → 2 columns: each output column covers 2.5 source columns.
        let row = [0u8, 0, 255, 255, 255];
        let img = Rgb8Image {
            width: 5,
            height: 1,
            data: row.iter().flat_map(|&v| [v, v, v]).collect(),
        };
        let out = resize(&img, 2);
        assert_eq!((out.width, out.height), (2, 1));
        // Left: (0 + 0 + 0.5 * 1) / 2.5 = 0.2 of the light; right: all white.
        assert_eq!(out.data[0], linear_to_srgb8(0.2));
        assert_eq!(out.data[3], 255);
    }

    #[test]
    fn resize_never_upscales() {
        let img = test_image(40, 30);
        assert!(same(&resize(&img, 40), &img));
        assert!(same(&resize(&img, 4000), &img));
    }

    #[test]
    fn linear_to_srgb8_inverts_the_lut() {
        let lut = srgb_to_linear_lut();
        for (code, &lin) in lut.iter().enumerate() {
            assert_eq!(linear_to_srgb8(lin), code as u8);
        }
        assert_eq!(linear_to_srgb8(-1.0), 0);
        assert_eq!(linear_to_srgb8(2.0), 255);
    }

    /// Start-of-frame fields: (width, height, luma sampling byte, chroma sampling byte).
    fn jpeg_frame(bytes: &[u8]) -> (usize, usize, u8, u8) {
        assert_eq!(&bytes[..2], &[0xFF, 0xD8], "SOI");
        assert_eq!(&bytes[bytes.len() - 2..], &[0xFF, 0xD9], "EOI");
        let mut i = 2;
        loop {
            assert_eq!(bytes[i], 0xFF);
            let marker = bytes[i + 1];
            let len = usize::from(u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]));
            if marker == 0xC0 {
                let s = &bytes[i + 4..];
                assert_eq!(s[0], 8, "precision");
                let h = usize::from(u16::from_be_bytes([s[1], s[2]]));
                let w = usize::from(u16::from_be_bytes([s[3], s[4]]));
                assert_eq!(s[5], 3, "components");
                return (w, h, s[7], s[10]);
            }
            assert_ne!(marker, 0xC2, "progressive");
            i += 2 + len;
        }
    }

    #[test]
    fn jpeg_is_baseline_with_the_right_subsampling() {
        let img = test_image(97, 61);
        let hi = jpeg(&img, 95).unwrap();
        assert_eq!(jpeg_frame(&hi), (97, 61, 0x11, 0x11));
        let lo = jpeg(&img, 88).unwrap();
        assert_eq!(jpeg_frame(&lo), (97, 61, 0x22, 0x11));
        // Quality 0 is clamped rather than rejected.
        assert!(jpeg(&img, 0).is_ok());
    }

    #[test]
    fn jpeg_rejects_bad_images() {
        assert!(jpeg(&Rgb8Image::default(), 90).is_err());
        let short = Rgb8Image {
            width: 10,
            height: 10,
            data: vec![0; 10],
        };
        assert!(jpeg(&short, 90).is_err());
        let huge = Rgb8Image {
            width: 70000,
            height: 1,
            data: vec![0; 70000 * 3],
        };
        assert!(jpeg(&huge, 90).is_err());
    }

    #[test]
    fn tiff16_round_trips() {
        use tiff::decoder::{Decoder, DecodingResult};

        let img = Rgb16Image {
            width: 131,
            height: 77,
            data: (0..131 * 77 * 3)
                .map(|i| (i * 7919 % 65536) as u16)
                .collect(),
        };
        let bytes = tiff16(&img).unwrap();
        let mut decoder = Decoder::new(Cursor::new(bytes)).unwrap();
        assert_eq!(decoder.dimensions().unwrap(), (131, 77));
        assert_eq!(decoder.colortype().unwrap(), tiff::ColorType::RGB(16));
        match decoder.read_image().unwrap() {
            DecodingResult::U16(data) => assert_eq!(data, img.data),
            _ => panic!("not 16-bit"),
        }
        assert!(tiff16(&Rgb16Image::default()).is_err());
    }

    /// Export cost at 24 and 61 MP. Run in release on the GPU host; set
    /// SPL_TEST_OUT to keep a display-size JPEG of the result.
    #[test]
    #[ignore = "full-resolution timings; run with --release"]
    fn full_resolution_timings() {
        use std::time::Instant;

        for (w, h) in [(6000, 4000), (9504, 6336)] {
            // Smooth colour gradients under a moderate noise floor.
            let mut seed = 0x9E37_79B9u32;
            let mut data = Vec::with_capacity(w * h * 3);
            for y in 0..h {
                for x in 0..w {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    let noise = (seed % 512) as f32 - 256.0;
                    let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
                    let wave = 0.5 + 0.4 * ((u * 9.0).sin() * (v * 7.0).cos());
                    for c in [0.15 + 0.7 * u, 0.2 + 0.6 * v, wave] {
                        data.push((c * 60000.0 + noise).clamp(0.0, 65535.0) as u16);
                    }
                }
            }
            let img = Rgb16Image {
                width: w,
                height: h,
                data,
            };
            let t = Instant::now();
            let upright = orient16(&img, 6);
            let orient_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let rgb8 = to_rgb8(&upright);
            let dither_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let q95 = jpeg(&rgb8, 95).unwrap();
            let jpeg95_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let q88 = jpeg(&rgb8, 88).unwrap();
            let jpeg88_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let tiff = tiff16(&upright).unwrap();
            let tiff_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let display = resize(&rgb8, 1536);
            let resize_ms = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "{w}x{h}: orient16 {orient_ms:.0} ms, to_rgb8 {dither_ms:.0} ms, \
                 jpeg q95 {jpeg95_ms:.0} ms ({:.1} MB), jpeg q88 {jpeg88_ms:.0} ms ({:.1} MB), \
                 tiff16 {tiff_ms:.0} ms ({:.0} MB, {:.0}% of raw), resize to 1536 {resize_ms:.0} ms",
                q95.len() as f64 / 1e6,
                q88.len() as f64 / 1e6,
                tiff.len() as f64 / 1e6,
                100.0 * tiff.len() as f64 / (w * h * 6) as f64,
            );
            assert_eq!((upright.width, upright.height), (h, w));
            if let Some(dir) = std::env::var_os("SPL_TEST_OUT") {
                let dir = std::path::PathBuf::from(dir).join("judge");
                std::fs::create_dir_all(&dir).unwrap();
                let name = format!("encode-{w}x{h}-display.jpg");
                std::fs::write(dir.join(name), jpeg(&display, 90).unwrap()).unwrap();
            }
        }
    }
}
