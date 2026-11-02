//! Training images: decode, area downscale, parallel loading and the GPU u8
//! cache. Owner: trainer (E7), scene I/O. Spec: spec-train 8.3-8.4.
//!
//! The cache is one `CudaSlice<u8>` pool holding every cached view as
//! interleaved RGB8 (3 B/px) at its training size, a host table of
//! (offset, width, height) and the same table on the device (`CacheEntryDev`),
//! so kernels find this iteration's target through `IterParams.target_offset`
//! (or the table) without host involvement.

use std::path::Path;

use anyhow::ensure;
use rayon::prelude::*;
use spl_core::{Context, Result};
use spl_gpu::cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};

use super::scene::{Scene, View};
use crate::gpu::{CudaResultExt, Gpu};

/// Interleaved 8-bit RGB, row-major from the top-left.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rgb8Image {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Rgb8Image {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0; width as usize * height as usize * 3],
        }
    }

    pub fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        let i = (y as usize * self.width as usize + x as usize) * 3;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    pub fn save_png(&self, path: &Path) -> Result<()> {
        image::save_buffer(
            path,
            &self.data,
            self.width,
            self.height,
            image::ExtendedColorType::Rgb8,
        )
        .with_context(|| format!("write {}", path.display()))
    }
}

/// Decodes a JPEG or PNG to RGB8; alpha and 16-bit samples are dropped or
/// reduced, greyscale is expanded.
///
/// JPEGs go through libjpeg (mozjpeg, a libjpeg-turbo fork: ISLOW IDCT and
/// fancy upsampling, as PIL decodes them), so ground-truth pixels equal what
/// every reference implementation trains and evaluates on. Measured on 10
/// garden `images_4` JPEGs (spec-train 8.4): zune-jpeg (the `image` crate)
/// differed from PIL 11.3 in 23.7% of samples, 9.07% by more than 1 LSB (max 5).
pub fn decode_file(path: &Path) -> Result<Rgb8Image> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    decode_bytes(&bytes).with_context(|| format!("decode {}", path.display()))
}

pub fn decode_bytes(bytes: &[u8]) -> Result<Rgb8Image> {
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return decode_jpeg_libjpeg(bytes);
    }
    decode_bytes_image_crate(bytes)
}

/// The `image` crate's decoders (zune-jpeg for JPEG), for formats libjpeg
/// does not read and for comparisons.
pub fn decode_bytes_image_crate(bytes: &[u8]) -> Result<Rgb8Image> {
    let img = image::load_from_memory(bytes)?.into_rgb8();
    Ok(Rgb8Image {
        width: img.width(),
        height: img.height(),
        data: img.into_raw(),
    })
}

/// libjpeg decode with PIL's defaults. libjpeg reports errors by unwinding,
/// so the call runs under `catch_unwind`.
pub fn decode_jpeg_libjpeg(bytes: &[u8]) -> Result<Rgb8Image> {
    let decoded = std::panic::catch_unwind(|| -> std::io::Result<Rgb8Image> {
        let mut d = mozjpeg::Decompress::new_mem(bytes)?;
        d.dct_method(mozjpeg::DctMethod::IntegerSlow);
        d.do_fancy_upsampling(true);
        let mut started = d.rgb()?;
        let (width, height) = (started.width() as u32, started.height() as u32);
        let data = started.read_scanlines::<u8>()?;
        started.finish()?;
        Ok(Rgb8Image {
            width,
            height,
            data,
        })
    });
    match decoded {
        Ok(Ok(img)) => {
            ensure!(
                img.data.len() == img.width as usize * img.height as usize * 3,
                "libjpeg returned {} bytes for {}x{}",
                img.data.len(),
                img.width,
                img.height
            );
            Ok(img)
        }
        Ok(Err(e)) => Err(e).context("libjpeg"),
        Err(_) => Err(spl_core::anyhow!("libjpeg failed to decode the JPEG")),
    }
}

/// Area (box) resampling to `w × h` for any ratio ≤ 1 per axis: each output
/// pixel is the coverage-weighted mean of the input pixels under it, rounded
/// half up. Separable, f32 accumulation. Upscaling is not supported.
pub fn downscale_area(src: &Rgb8Image, w: u32, h: u32) -> Rgb8Image {
    assert!(
        w >= 1 && h >= 1 && w <= src.width && h <= src.height,
        "area filter only downscales"
    );
    if (w, h) == (src.width, src.height) {
        return src.clone();
    }
    let weights = |n_in: u32, n_out: u32| -> Vec<Vec<(usize, f32)>> {
        let scale = n_in as f64 / n_out as f64;
        (0..n_out)
            .map(|o| {
                let (a, b) = (o as f64 * scale, (o + 1) as f64 * scale);
                let mut taps = Vec::new();
                let mut i = a.floor() as usize;
                while (i as f64) < b && i < n_in as usize {
                    let cover = (b.min(i as f64 + 1.0) - a.max(i as f64)) / scale;
                    if cover > 0.0 {
                        taps.push((i, cover as f32));
                    }
                    i += 1;
                }
                taps
            })
            .collect()
    };
    let (wx, wy) = (weights(src.width, w), weights(src.height, h));
    let sw = src.width as usize;
    // Horizontal pass: src.height rows of w pixels, f32.
    let rows: Vec<f32> = (0..src.height as usize)
        .into_par_iter()
        .flat_map_iter(|y| {
            let row = &src.data[y * sw * 3..(y + 1) * sw * 3];
            wx.iter().flat_map(move |taps| {
                let mut acc = [0f32; 3];
                for &(x, wgt) in taps {
                    for c in 0..3 {
                        acc[c] += wgt * row[x * 3 + c] as f32;
                    }
                }
                acc
            })
        })
        .collect();
    let ow = w as usize;
    let data: Vec<u8> = wy
        .par_iter()
        .flat_map_iter(|taps| {
            let rows = &rows;
            (0..ow * 3).map(move |i| {
                let v: f32 = taps
                    .iter()
                    .map(|&(y, wgt)| wgt * rows[y * ow * 3 + i])
                    .sum();
                (v + 0.5).floor().clamp(0.0, 255.0) as u8
            })
        })
        .collect();
    Rgb8Image {
        width: w,
        height: h,
        data,
    }
}

/// The view's image at its training size (decoded, area-downscaled when
/// `max_width` shrank it).
pub fn load_view_image(view: &View) -> Result<Rgb8Image> {
    let img = decode_file(&view.image_path)?;
    if (img.width, img.height) == (view.width, view.height) {
        return Ok(img);
    }
    ensure!(
        img.width >= view.width && img.height >= view.height,
        "{} is {}x{}, smaller than its view size {}x{}",
        view.name,
        img.width,
        img.height,
        view.width,
        view.height
    );
    Ok(downscale_area(&img, view.width, view.height))
}

/// Decodes `indices` of `views` in parallel, in order.
pub fn load_views(views: &[View], indices: &[usize]) -> Result<Vec<Rgb8Image>> {
    indices
        .par_iter()
        .map(|&i| load_view_image(&views[i]))
        .collect()
}

/// One cached image on the device. 16 B.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheEntryDev {
    /// Byte offset into the pool.
    pub offset: u64,
    pub width: u32,
    pub height: u32,
}

const _: () = assert!(std::mem::size_of::<CacheEntryDev>() == 16);
// SAFETY: #[repr(C)] u64 + 2 × u32, no padding; zero is valid.
unsafe impl DeviceRepr for CacheEntryDev {}
unsafe impl ValidAsZeroBits for CacheEntryDev {}

/// All cached views in one device pool (spec-train 8.4).
pub struct ImageCache {
    /// Interleaved RGB8 of every cached view, back to back.
    pub pool: CudaSlice<u8>,
    /// Scene view index of each entry.
    pub views: Vec<usize>,
    pub entries: Vec<CacheEntryDev>,
    pub table: CudaSlice<CacheEntryDev>,
    /// Host wall time spent decoding, seconds.
    pub decode_s: f64,
}

impl ImageCache {
    /// Decodes `views` of `scene` in parallel and uploads them in one copy.
    pub fn upload(gpu: &Gpu, scene: &Scene, views: &[usize]) -> Result<Self> {
        let started = std::time::Instant::now();
        let images = load_views(&scene.views, views)?;
        let decode_s = started.elapsed().as_secs_f64();
        Self::from_images(gpu, views.to_vec(), &images, decode_s)
    }

    pub fn from_images(
        gpu: &Gpu,
        views: Vec<usize>,
        images: &[Rgb8Image],
        decode_s: f64,
    ) -> Result<Self> {
        ensure!(views.len() == images.len(), "one image per view");
        let mut entries = Vec::with_capacity(images.len());
        let mut offset = 0u64;
        for img in images {
            entries.push(CacheEntryDev {
                offset,
                width: img.width,
                height: img.height,
            });
            offset += img.data.len() as u64;
        }
        let mut host = Vec::with_capacity(offset as usize);
        images
            .iter()
            .for_each(|img| host.extend_from_slice(&img.data));
        let s = &gpu.stream;
        let pool = if host.is_empty() {
            s.alloc_zeros::<u8>(1).cuda("alloc image pool")?
        } else {
            s.clone_htod(&host).cuda("upload image pool")?
        };
        let table = if entries.is_empty() {
            s.alloc_zeros::<CacheEntryDev>(1)
                .cuda("alloc image table")?
        } else {
            s.clone_htod(&entries).cuda("upload image table")?
        };
        Ok(Self {
            pool,
            views,
            entries,
            table,
            decode_s,
        })
    }

    /// Bytes of pixels held.
    pub fn bytes(&self) -> u64 {
        self.entries
            .iter()
            .map(|e| e.width as u64 * e.height as u64 * 3)
            .sum()
    }

    /// The cache entry of scene view `view`.
    pub fn entry(&self, view: usize) -> Option<&CacheEntryDev> {
        self.views
            .iter()
            .position(|&v| v == view)
            .map(|i| &self.entries[i])
    }

    /// Copies one cached view back (tests, eval).
    pub fn download(&self, gpu: &Gpu, view: usize) -> Result<Rgb8Image> {
        let e = *self
            .entry(view)
            .with_context(|| format!("view {view} is not cached"))?;
        let len = e.width as usize * e.height as usize * 3;
        let mut data = vec![0u8; len];
        let start = e.offset as usize;
        gpu.stream
            .memcpy_dtoh(&self.pool.slice(start..start + len), &mut data)
            .cuda("download cached image")?;
        gpu.sync()?;
        Ok(Rgb8Image {
            width: e.width,
            height: e.height,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: u32, h: u32) -> Rgb8Image {
        let mut img = Rgb8Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                img.data[i] = (x * 255 / (w - 1).max(1)) as u8;
                img.data[i + 1] = (y * 255 / (h - 1).max(1)) as u8;
                img.data[i + 2] = 100;
            }
        }
        img
    }

    #[test]
    fn integer_area_downscale_is_the_box_mean() {
        let mut img = Rgb8Image::new(4, 2);
        img.data = vec![
            0, 0, 0, 10, 10, 10, 20, 20, 20, 31, 31, 31, //
            0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30,
        ];
        let out = downscale_area(&img, 2, 1);
        assert_eq!(
            out.data,
            vec![5, 5, 5, 25, 25, 25],
            "(20 + 31 + 20 + 30) / 4 = 25.25"
        );
    }

    #[test]
    fn fractional_downscale_keeps_flat_colours_and_mean() {
        let mut flat = Rgb8Image::new(37, 29);
        flat.data
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = [200, 7, 64][i % 3]);
        let out = downscale_area(&flat, 16, 11);
        assert!(out.data.chunks(3).all(|p| p == [200, 7, 64]));

        let src = ramp(1237, 822);
        let out = downscale_area(&src, 1000, 665);
        let mean = |img: &Rgb8Image, c: usize| {
            img.data
                .iter()
                .skip(c)
                .step_by(3)
                .map(|&v| v as f64)
                .sum::<f64>()
                / (img.width * img.height) as f64
        };
        for c in 0..3 {
            assert!((mean(&src, c) - mean(&out, c)).abs() < 0.6, "channel {c}");
        }
    }

    #[test]
    fn decode_reads_png_and_jpeg() {
        let dir = tempfile::tempdir().unwrap();
        let img = ramp(33, 17);
        img.save_png(&dir.path().join("a.png")).unwrap();
        assert_eq!(decode_file(&dir.path().join("a.png")).unwrap(), img);
        let jpg = dir.path().join("a.jpg");
        image::save_buffer(&jpg, &img.data, 33, 17, image::ExtendedColorType::Rgb8).unwrap();
        let back = decode_file(&jpg).unwrap();
        assert_eq!((back.width, back.height), (33, 17));
        let err = back
            .data
            .iter()
            .zip(&img.data)
            .map(|(a, b)| a.abs_diff(*b) as u32)
            .max()
            .unwrap();
        assert!(err < 40, "JPEG round trip max error {err}");
        assert!(decode_file(&dir.path().join("missing.jpg")).is_err());
        // Both decoders agree up to their IDCT/upsampling differences.
        let bytes = std::fs::read(&jpg).unwrap();
        let zune = decode_bytes_image_crate(&bytes).unwrap();
        let diff = back
            .data
            .iter()
            .zip(&zune.data)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(diff <= 8, "libjpeg vs zune max diff {diff}");
    }

    #[test]
    fn corrupt_jpegs_fail_without_panicking() {
        let mut bytes = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut bytes)
            .encode(&ramp(40, 30).data, 40, 30, image::ExtendedColorType::Rgb8)
            .unwrap();
        // Truncated data: libjpeg warns and fills grey, or fails; it must not abort.
        let _ = decode_bytes(&bytes[..bytes.len() / 3]);
        assert!(decode_bytes(&[0xff, 0xd8, 0xff, 0x00, 0x01]).is_err());
        assert!(decode_bytes(b"not an image").is_err());
    }
}
