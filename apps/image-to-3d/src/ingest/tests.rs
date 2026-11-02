//! CPU tests of Stage A0 (no GPU). Tests that need `ffmpeg` skip without it.

use std::path::Path;

use super::video::{ffmpeg_args, parse_probe};
use super::*;

#[test]
fn fit_size_keeps_aspect_rounds_and_never_upscales() {
    assert_eq!(fit_size(3840, 2160, 512, 16), (512, 288));
    assert_eq!(fit_size(6000, 4000, 512, 16), (512, 336));
    assert_eq!(fit_size(2160, 3840, 512, 16), (288, 512));
    assert_eq!(fit_size(3840, 2160, 1600, 1), (1600, 900));
    assert_eq!(fit_size(800, 600, 1600, 1), (800, 600));
    assert_eq!(fit_size(10, 5, 512, 16), (16, 16));
}

#[test]
fn bins_keep_the_sharpest_frame_per_equal_time_slice() {
    let mut b = KeyframeBins::new(10.0, 5);
    assert_eq!(b.bin_of(0.0), 0);
    assert_eq!(b.bin_of(1.99), 0);
    assert_eq!(b.bin_of(2.0), 1);
    assert_eq!(b.bin_of(10.5), 4);
    assert_eq!(b.offer(0.1, 1.0, 0), Some(0));
    assert_eq!(b.offer(0.2, 0.5, 1), None);
    assert_eq!(b.offer(0.3, 1.0, 2), None, "ties keep the earlier frame");
    assert_eq!(b.offer(1.5, 3.0, 3), Some(0));
    assert_eq!(b.offer(9.0, 0.1, 4), Some(4));
    assert_eq!(b.kept(), vec![Some(3), None, None, None, Some(4)]);
}

#[test]
fn focal_prior_uses_the_full_frame_diagonal() {
    // 36×24 mm at 36 px/mm: a 26 mm equivalent lens gives f = 26·36 px.
    let p = FocalPrior { focal_35mm: 26.0 };
    assert!((p.focal_px(1296, 864) - 26.0 * 36.0).abs() < 0.1);
}

#[test]
fn sharpness_cpu_is_zero_for_flat_and_grows_with_detail() {
    let flat = Rgb8 {
        width: 8,
        height: 8,
        data: vec![128; 8 * 8 * 3],
    };
    assert_eq!(sharpness_cpu(&flat), 0.0);
    let checker = Rgb8 {
        width: 8,
        height: 8,
        data: (0..64)
            .flat_map(|i| [if (i % 8 + i / 8) % 2 == 0 { 255u8 } else { 0 }; 3])
            .collect(),
    };
    // Laplacian of a ±1 checker is ±4 everywhere: variance 16.
    assert!((sharpness_cpu(&checker) - 16.0).abs() < 1e-9);
}

#[test]
fn resize_area_cpu_averages_blocks() {
    let src = Rgb8 {
        width: 4,
        height: 2,
        data: [0u8, 100, 200, 50]
            .iter()
            .chain(&[0u8, 100, 200, 50])
            .flat_map(|v| [*v; 3])
            .collect(),
    };
    let d = resize_area_cpu(&src, 2, 1);
    assert_eq!(d.data, vec![50, 50, 50, 125, 125, 125]);
}

#[test]
fn probe_json_applies_rotation_and_rational_rates() {
    let j = r#"{"streams":[{"width":3840,"height":2160,"avg_frame_rate":"30000/1001",
        "side_data_list":[{"rotation":-90}]}],"format":{"duration":"60.06"}}"#;
    let i = parse_probe(j).unwrap();
    assert_eq!((i.width, i.height), (2160, 3840));
    assert!((i.fps - 29.97).abs() < 0.01);
    assert!((i.duration_s - 60.06).abs() < 1e-9);
    let j = r#"{"streams":[{"width":640,"height":360,"avg_frame_rate":"0/0",
        "r_frame_rate":"25/1","tags":{"rotate":"180"},"duration":"2.0"}],"format":{}}"#;
    let i = parse_probe(j).unwrap();
    assert_eq!(
        (i.width, i.height, i.fps, i.duration_s),
        (640, 360, 25.0, 2.0)
    );
    assert!(parse_probe(r#"{"streams":[]}"#).is_err());
}

#[test]
fn ffmpeg_args_sample_or_pass_through() {
    let cfg = IngestConfig::default();
    let a = ffmpeg_args(Path::new("in.mp4"), &cfg).join(" ");
    assert!(a.contains("-i in.mp4 -map 0:v:0"), "{a}");
    assert!(a.contains("-vf fps=10 "), "{a}");
    assert!(a.ends_with("-f rawvideo -pix_fmt rgb24 pipe:1"), "{a}");
    let cfg = IngestConfig {
        sample_fps: None,
        ffmpeg_input_args: vec!["-hwaccel".into(), "cuda".into()],
        ..cfg
    };
    let a = ffmpeg_args(Path::new("in.mp4"), &cfg).join(" ");
    assert!(a.contains("-hwaccel cuda -i in.mp4"), "{a}");
    assert!(a.contains("-fps_mode passthrough"), "{a}");
}

/// A JPEG with an APP1 EXIF segment spliced in after SOI.
pub(super) fn jpeg_with_exif(w: u32, h: u32, tiff: &[u8]) -> Vec<u8> {
    let img =
        image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 40) as u8, (y * 40) as u8, 7]));
    let mut jpg = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(
            &mut std::io::Cursor::new(&mut jpg),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
    let mut app1 = b"Exif\0\0".to_vec();
    app1.extend_from_slice(tiff);
    let mut out = jpg[..2].to_vec();
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&app1);
    out.extend_from_slice(&jpg[2..]);
    out
}

/// Little-endian TIFF: IFD0 = {Orientation = `orientation`, ExifIFD →
/// {FocalLengthIn35mmFilm = `f35`}}.
pub(super) fn tiff_orientation_focal(orientation: u16, f35: u16) -> Vec<u8> {
    let mut b = b"II*\0".to_vec();
    b.extend_from_slice(&8u32.to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    for (tag, ty, v) in [(0x0112u16, 3u16, orientation as u32), (0x8769, 4, 38)] {
        b.extend_from_slice(&tag.to_le_bytes());
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(b.len(), 38);
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&0xA405u16.to_le_bytes());
    b.extend_from_slice(&3u16.to_le_bytes());
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&(f35 as u32).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b
}

#[test]
fn decode_applies_exif_orientation_and_reads_the_focal_prior() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("a.jpg");
    std::fs::write(&p, jpeg_with_exif(8, 4, &tiff_orientation_focal(6, 26))).unwrap();
    let (img, focal) = decode_image(&p).unwrap();
    assert_eq!((img.width, img.height), (4, 8), "orientation 6 rotates 90°");
    assert_eq!(focal, Some(FocalPrior { focal_35mm: 26.0 }));

    let p = dir.path().join("b.png");
    image::RgbImage::new(5, 3).save(&p).unwrap();
    let (img, focal) = decode_image(&p).unwrap();
    assert_eq!((img.width, img.height, focal), (5, 3, None));
}

#[test]
fn decodes_webp() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("c.webp");
    image::RgbImage::from_pixel(6, 2, image::Rgb([1, 2, 3]))
        .save(&p)
        .unwrap();
    let (img, _) = decode_image(&p).unwrap();
    assert_eq!((img.width, img.height), (6, 2));
    assert_eq!(&img.data[..3], &[1, 2, 3]);
}

fn have_ffmpeg() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Writes a `secs`-second `w`×`h` testsrc2 clip at `fps` where every frame
/// except n % 5 == 2 is box-blurred; returns false when ffmpeg is missing.
pub(super) fn synth_video(path: &Path, w: u32, h: u32, fps: u32, secs: u32) -> bool {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not found; skipping");
        return false;
    }
    let src = format!("testsrc2=size={w}x{h}:rate={fps}:duration={secs}");
    let out = std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            &src,
        ])
        .args(["-vf", "boxblur=6:1:enable='not(eq(mod(n\\,5)\\,2))'"])
        .args(["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    true
}

#[tokio::test]
async fn probes_a_synthetic_clip() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("s.mp4");
    if !synth_video(&p, 320, 180, 30, 2) {
        return;
    }
    let i = probe_video(&p).await.unwrap();
    assert_eq!((i.width, i.height), (320, 180));
    assert!((i.fps - 30.0).abs() < 1e-6);
    assert!((i.duration_s - 2.0).abs() < 0.1, "{}", i.duration_s);
}

#[test]
fn every_ingest_kernel_is_declared() {
    let text = super::gpu::SOURCE.text();
    let mut declared: Vec<&str> = text
        .split("extern \"C\" __global__ void ")
        .skip(1)
        .filter_map(|rest| rest.split('(').next())
        .collect();
    declared.sort_unstable();
    let mut listed = super::gpu::SOURCE.kernels.to_vec();
    listed.sort_unstable();
    assert_eq!(declared, listed);
}
