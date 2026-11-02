//! COLMAP sparse models: binary and text, read and write. Owner: trainer (E7),
//! scene I/O. Spec: spec-train 8.1.
//!
//! Binary files are little-endian with no padding, as COLMAP's
//! `Reconstruction::Write*Binary` writes them:
//!
//! ```text
//! cameras.bin:  u64 n; n × { u32 camera_id; i32 model_id; u64 width; u64 height; f64 params[num_params(model)] }
//! images.bin:   u64 n; n × { u32 image_id; f64 qw, qx, qy, qz; f64 tx, ty, tz; u32 camera_id;
//!                            name bytes up to and excluding 0x00; u64 n2d; n2d × { f64 x; f64 y; i64 point3d_id } }
//! points3D.bin: u64 n; n × { u64 point3d_id; f64 x, y, z; u8 r, g, b; f64 error;
//!                            u64 track_len; track_len × { u32 image_id; u32 point2d_idx } }
//! ```
//!
//! Parsing reads whole files and walks them with a bounds-checked cursor;
//! every count is checked against the bytes left before anything is
//! allocated. COLMAP ≥ 3.12's `rigs.*` and `frames.*` files are ignored.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::ensure;
use spl_core::{Context, Result, bail};

/// COLMAP camera models: (id, name, parameter count).
pub const CAMERA_MODELS: [(i32, &str, usize); 12] = [
    (0, "SIMPLE_PINHOLE", 3),
    (1, "PINHOLE", 4),
    (2, "SIMPLE_RADIAL", 4),
    (3, "RADIAL", 5),
    (4, "OPENCV", 8),
    (5, "OPENCV_FISHEYE", 8),
    (6, "FULL_OPENCV", 12),
    (7, "FOV", 5),
    (8, "SIMPLE_RADIAL_FISHEYE", 4),
    (9, "RADIAL_FISHEYE", 5),
    (10, "THIN_PRISM_FISHEYE", 12),
    (11, "RAD_TAN_THIN_PRISM_FISHEYE", 16),
];

pub fn model_name(id: i32) -> Option<&'static str> {
    CAMERA_MODELS.iter().find(|m| m.0 == id).map(|m| m.1)
}

pub fn model_id(name: &str) -> Option<i32> {
    CAMERA_MODELS.iter().find(|m| m.1 == name).map(|m| m.0)
}

pub fn model_params(id: i32) -> Option<usize> {
    CAMERA_MODELS.iter().find(|m| m.0 == id).map(|m| m.2)
}

#[derive(Clone, Debug, PartialEq)]
pub struct ColmapCamera {
    pub id: u32,
    pub model_id: i32,
    pub width: u64,
    pub height: u64,
    pub params: Vec<f64>,
}

impl ColmapCamera {
    /// (fx, fy, cx, cy) for PINHOLE and SIMPLE_PINHOLE; `None` for models
    /// training does not accept (they need `colmap image_undistorter` first).
    pub fn pinhole(&self) -> Option<[f64; 4]> {
        match (self.model_id, self.params.as_slice()) {
            (0, &[f, cx, cy]) => Some([f, f, cx, cy]),
            (1, &[fx, fy, cx, cy]) => Some([fx, fy, cx, cy]),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ColmapImage {
    pub id: u32,
    /// World-to-camera rotation (w, x, y, z).
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
    pub camera_id: u32,
    pub name: String,
    /// (x, y, point3d_id); −1 = no 3D point.
    pub points2d: Vec<(f64, f64, i64)>,
}

impl ColmapImage {
    /// Row-major world-to-camera rotation of the normalized `qvec`.
    pub fn rotation(&self) -> [[f64; 3]; 3] {
        let [w, x, y, z] = self.qvec;
        let n = (w * w + x * x + y * y + z * z).sqrt();
        let (w, x, y, z) = (w / n, x / n, y / n, z / n);
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }

    /// Camera centre C = −Rᵀ t.
    pub fn center(&self) -> [f64; 3] {
        let r = self.rotation();
        let t = self.tvec;
        std::array::from_fn(|j| -(r[0][j] * t[0] + r[1][j] * t[1] + r[2][j] * t[2]))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ColmapPoint {
    pub id: u64,
    pub xyz: [f64; 3],
    pub rgb: [u8; 3],
    pub error: f64,
    /// (image_id, point2d_idx).
    pub track: Vec<(u32, u32)>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ColmapModel {
    pub cameras: BTreeMap<u32, ColmapCamera>,
    pub images: BTreeMap<u32, ColmapImage>,
    /// In file order.
    pub points: Vec<ColmapPoint>,
}

impl ColmapModel {
    /// `sparse_dir/*.bin` when present, else `*.txt`.
    pub fn read_dir(sparse_dir: &Path) -> Result<Self> {
        if sparse_dir.join("cameras.bin").is_file() {
            Self::read_bin(sparse_dir)
        } else if sparse_dir.join("cameras.txt").is_file() {
            Self::read_txt(sparse_dir)
        } else {
            bail!(
                "no COLMAP model (cameras.bin or cameras.txt) in {}",
                sparse_dir.display()
            )
        }
    }

    pub fn read_bin(sparse_dir: &Path) -> Result<Self> {
        let read = |name: &str| {
            let p = sparse_dir.join(name);
            std::fs::read(&p).with_context(|| format!("read {}", p.display()))
        };
        Self::parse_bin(
            &read("cameras.bin")?,
            &read("images.bin")?,
            &read("points3D.bin")?,
        )
    }

    pub fn read_txt(sparse_dir: &Path) -> Result<Self> {
        let read = |name: &str| {
            let p = sparse_dir.join(name);
            std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))
        };
        Self::parse_txt(
            &read("cameras.txt")?,
            &read("images.txt")?,
            &read("points3D.txt")?,
        )
    }

    /// Parses the three binary files (testable without files).
    pub fn parse_bin(cameras: &[u8], images: &[u8], points: &[u8]) -> Result<Self> {
        let mut model = Self::default();

        let mut c = Cursor::new(cameras, "cameras.bin");
        let n = c.count(4 + 4 + 8 + 8)?;
        for _ in 0..n {
            let id = c.u32()?;
            let model_id = c.i32()?;
            let width = c.u64()?;
            let height = c.u64()?;
            let np = model_params(model_id).with_context(|| {
                format!("cameras.bin: camera {id} has unknown model id {model_id}")
            })?;
            c.need(np * 8)?;
            let params = (0..np).map(|_| c.f64()).collect::<Result<Vec<_>>>()?;
            let cam = ColmapCamera {
                id,
                model_id,
                width,
                height,
                params,
            };
            ensure!(
                model.cameras.insert(id, cam).is_none(),
                "cameras.bin: duplicate camera id {id}"
            );
        }
        c.finish()?;

        let mut c = Cursor::new(images, "images.bin");
        let n = c.count(4 + 7 * 8 + 4 + 1 + 8)?;
        for _ in 0..n {
            let id = c.u32()?;
            let qvec = [c.f64()?, c.f64()?, c.f64()?, c.f64()?];
            let tvec = [c.f64()?, c.f64()?, c.f64()?];
            let camera_id = c.u32()?;
            let name = c.cstr()?;
            let n2d = c.count(24)?;
            let mut points2d = Vec::with_capacity(n2d);
            for _ in 0..n2d {
                points2d.push((c.f64()?, c.f64()?, c.i64()?));
            }
            let img = ColmapImage {
                id,
                qvec,
                tvec,
                camera_id,
                name,
                points2d,
            };
            ensure!(
                model.images.insert(id, img).is_none(),
                "images.bin: duplicate image id {id}"
            );
        }
        c.finish()?;

        let mut c = Cursor::new(points, "points3D.bin");
        let n = c.count(8 + 24 + 3 + 8 + 8)?;
        model.points.reserve(n);
        for _ in 0..n {
            let id = c.u64()?;
            let xyz = [c.f64()?, c.f64()?, c.f64()?];
            let rgb = [c.u8()?, c.u8()?, c.u8()?];
            let error = c.f64()?;
            let len = c.count(8)?;
            let mut track = Vec::with_capacity(len);
            for _ in 0..len {
                track.push((c.u32()?, c.u32()?));
            }
            model.points.push(ColmapPoint {
                id,
                xyz,
                rgb,
                error,
                track,
            });
        }
        c.finish()?;
        model.validate()?;
        Ok(model)
    }

    /// Parses the three text files.
    pub fn parse_txt(cameras: &str, images: &str, points: &str) -> Result<Self> {
        let mut model = Self::default();

        for (no, line) in data_lines(cameras) {
            let w: Vec<&str> = line.split_whitespace().collect();
            ensure!(
                w.len() >= 4,
                "cameras.txt:{no}: expected ID MODEL WIDTH HEIGHT PARAMS..."
            );
            let id: u32 = parse(w[0], "cameras.txt", no)?;
            let model_id = model_id(w[1])
                .with_context(|| format!("cameras.txt:{no}: unknown model {}", w[1]))?;
            let np = model_params(model_id).unwrap_or(0);
            ensure!(
                w.len() == 4 + np,
                "cameras.txt:{no}: {} takes {np} parameters, found {}",
                w[1],
                w.len() - 4
            );
            let cam = ColmapCamera {
                id,
                model_id,
                width: parse(w[2], "cameras.txt", no)?,
                height: parse(w[3], "cameras.txt", no)?,
                params: w[4..]
                    .iter()
                    .map(|s| parse(s, "cameras.txt", no))
                    .collect::<Result<_>>()?,
            };
            ensure!(
                model.cameras.insert(id, cam).is_none(),
                "cameras.txt:{no}: duplicate camera id {id}"
            );
        }

        // Two lines per image; the second (2D points) may be empty, so only the
        // header line skips blanks and comments (as COLMAP's reader does).
        let mut lines = images.lines().enumerate();
        while let Some((i, line)) = lines.next() {
            let no = i + 1;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut w = line.split_whitespace();
            let mut next = |what: &str| {
                w.next()
                    .with_context(|| format!("images.txt:{no}: missing {what}"))
            };
            let id: u32 = parse(next("IMAGE_ID")?, "images.txt", no)?;
            let mut q = [0f64; 4];
            for (k, x) in q.iter_mut().enumerate() {
                *x = parse(next(["QW", "QX", "QY", "QZ"][k])?, "images.txt", no)?;
            }
            let mut t = [0f64; 3];
            for (k, x) in t.iter_mut().enumerate() {
                *x = parse(next(["TX", "TY", "TZ"][k])?, "images.txt", no)?;
            }
            let camera_id: u32 = parse(next("CAMERA_ID")?, "images.txt", no)?;
            let name =
                image_name(line).with_context(|| format!("images.txt:{no}: missing NAME"))?;
            let pts = lines.next().map_or("", |(_, l)| l).trim();
            let v: Vec<&str> = pts.split_whitespace().collect();
            ensure!(
                v.len().is_multiple_of(3),
                "images.txt:{}: 2D points must be X Y POINT3D_ID triples",
                no + 1
            );
            let points2d = v
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| {
                    Ok((
                        parse(p[0], "images.txt", no + 1)?,
                        parse(p[1], "images.txt", no + 1)?,
                        parse(p[2], "images.txt", no + 1)?,
                    ))
                })
                .collect::<Result<_>>()?;
            let img = ColmapImage {
                id,
                qvec: q,
                tvec: t,
                camera_id,
                name,
                points2d,
            };
            ensure!(
                model.images.insert(id, img).is_none(),
                "images.txt:{no}: duplicate image id {id}"
            );
        }

        for (no, line) in data_lines(points) {
            let w: Vec<&str> = line.split_whitespace().collect();
            ensure!(
                w.len() >= 8 && (w.len() - 8).is_multiple_of(2),
                "points3D.txt:{no}: expected ID X Y Z R G B ERROR (IMAGE_ID POINT2D_IDX)..."
            );
            let p = |i: usize| -> Result<f64> { parse(w[i], "points3D.txt", no) };
            let c = |i: usize| -> Result<u8> { parse(w[i], "points3D.txt", no) };
            model.points.push(ColmapPoint {
                id: parse(w[0], "points3D.txt", no)?,
                xyz: [p(1)?, p(2)?, p(3)?],
                rgb: [c(4)?, c(5)?, c(6)?],
                error: p(7)?,
                track: w[8..]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|t| {
                        Ok((
                            parse(t[0], "points3D.txt", no)?,
                            parse(t[1], "points3D.txt", no)?,
                        ))
                    })
                    .collect::<Result<_>>()?,
            });
        }
        model.validate()?;
        Ok(model)
    }

    /// Every image refers to a known camera.
    pub fn validate(&self) -> Result<()> {
        for img in self.images.values() {
            ensure!(
                self.cameras.contains_key(&img.camera_id),
                "image {} ({}) refers to unknown camera {}",
                img.id,
                img.name,
                img.camera_id
            );
        }
        Ok(())
    }

    /// The three binary files, in id order (cameras, images) and stored order (points).
    pub fn to_bin(&self) -> [Vec<u8>; 3] {
        let mut cams = Vec::new();
        put_u64(&mut cams, self.cameras.len() as u64);
        for c in self.cameras.values() {
            cams.extend_from_slice(&c.id.to_le_bytes());
            cams.extend_from_slice(&c.model_id.to_le_bytes());
            put_u64(&mut cams, c.width);
            put_u64(&mut cams, c.height);
            c.params.iter().for_each(|p| put_f64(&mut cams, *p));
        }
        let mut imgs = Vec::new();
        put_u64(&mut imgs, self.images.len() as u64);
        for i in self.images.values() {
            imgs.extend_from_slice(&i.id.to_le_bytes());
            i.qvec
                .iter()
                .chain(&i.tvec)
                .for_each(|x| put_f64(&mut imgs, *x));
            imgs.extend_from_slice(&i.camera_id.to_le_bytes());
            imgs.extend_from_slice(i.name.as_bytes());
            imgs.push(0);
            put_u64(&mut imgs, i.points2d.len() as u64);
            for &(x, y, id) in &i.points2d {
                put_f64(&mut imgs, x);
                put_f64(&mut imgs, y);
                imgs.extend_from_slice(&id.to_le_bytes());
            }
        }
        let mut pts = Vec::new();
        put_u64(&mut pts, self.points.len() as u64);
        for p in &self.points {
            put_u64(&mut pts, p.id);
            p.xyz.iter().for_each(|x| put_f64(&mut pts, *x));
            pts.extend_from_slice(&p.rgb);
            put_f64(&mut pts, p.error);
            put_u64(&mut pts, p.track.len() as u64);
            for &(img, idx) in &p.track {
                pts.extend_from_slice(&img.to_le_bytes());
                pts.extend_from_slice(&idx.to_le_bytes());
            }
        }
        [cams, imgs, pts]
    }

    /// The three text files in COLMAP's layout (shortest round-trip floats).
    pub fn to_txt(&self) -> [String; 3] {
        let mut cams = String::from(
            "# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n",
        );
        let _ = writeln!(cams, "# Number of cameras: {}", self.cameras.len());
        for c in self.cameras.values() {
            let _ = write!(
                cams,
                "{} {} {} {}",
                c.id,
                model_name(c.model_id).unwrap_or("UNKNOWN"),
                c.width,
                c.height
            );
            c.params.iter().for_each(|p| {
                let _ = write!(cams, " {p}");
            });
            cams.push('\n');
        }
        let mut imgs = String::from(
            "# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
        );
        let _ = writeln!(imgs, "# Number of images: {}", self.images.len());
        for i in self.images.values() {
            let [qw, qx, qy, qz] = i.qvec;
            let [tx, ty, tz] = i.tvec;
            let _ = writeln!(
                imgs,
                "{} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {} {}",
                i.id, i.camera_id, i.name
            );
            let pts: Vec<String> = i
                .points2d
                .iter()
                .map(|(x, y, id)| format!("{x} {y} {id}"))
                .collect();
            imgs.push_str(&pts.join(" "));
            imgs.push('\n');
        }
        let mut pts = String::from(
            "# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n",
        );
        let _ = writeln!(pts, "# Number of points: {}", self.points.len());
        for p in &self.points {
            let [x, y, z] = p.xyz;
            let [r, g, b] = p.rgb;
            let _ = write!(pts, "{} {x} {y} {z} {r} {g} {b} {}", p.id, p.error);
            for (img, idx) in &p.track {
                let _ = write!(pts, " {img} {idx}");
            }
            pts.push('\n');
        }
        [cams, imgs, pts]
    }

    pub fn write_bin(&self, sparse_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(sparse_dir)
            .with_context(|| format!("create {}", sparse_dir.display()))?;
        for (name, bytes) in ["cameras.bin", "images.bin", "points3D.bin"]
            .iter()
            .zip(self.to_bin())
        {
            let p = sparse_dir.join(name);
            std::fs::write(&p, bytes).with_context(|| format!("write {}", p.display()))?;
        }
        Ok(())
    }

    pub fn write_txt(&self, sparse_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(sparse_dir)
            .with_context(|| format!("create {}", sparse_dir.display()))?;
        for (name, text) in ["cameras.txt", "images.txt", "points3D.txt"]
            .iter()
            .zip(self.to_txt())
        {
            let p = sparse_dir.join(name);
            std::fs::write(&p, text).with_context(|| format!("write {}", p.display()))?;
        }
        Ok(())
    }
}

/// Non-empty, non-comment lines with 1-based line numbers.
fn data_lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.trim()))
        .filter(|(_, l)| !l.is_empty() && !l.starts_with('#'))
}

/// NAME is the rest of the line after the 9th token (it may contain spaces).
fn image_name(line: &str) -> Option<String> {
    let mut rest = line.trim_start();
    for _ in 0..9 {
        let end = rest.find(char::is_whitespace)?;
        rest = rest[end..].trim_start();
    }
    let name = rest.trim();
    (!name.is_empty()).then(|| name.to_string())
}

fn parse<T: std::str::FromStr>(s: &str, file: &str, line: usize) -> Result<T> {
    s.parse()
        .ok()
        .with_context(|| format!("{file}:{line}: cannot parse {s:?}"))
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_f64(out: &mut Vec<u8>, v: f64) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Bounds-checked little-endian reader.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
    file: &'static str,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8], file: &'static str) -> Self {
        Self { buf, pos: 0, file }
    }

    fn need(&self, n: usize) -> Result<()> {
        let left = self.buf.len() - self.pos;
        ensure!(
            n <= left,
            "{}: truncated at byte {} (need {n}, {left} left)",
            self.file,
            self.pos
        );
        Ok(())
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.need(N)?;
        let out = self.buf[self.pos..self.pos + N]
            .try_into()
            .expect("length checked");
        self.pos += N;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take::<1>()?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take()?))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take()?))
    }

    /// A u64 count of items that take at least `min_item` bytes each,
    /// validated against the bytes left before anyone allocates for it.
    fn count(&mut self, min_item: usize) -> Result<usize> {
        let n = self.u64()?;
        let left = (self.buf.len() - self.pos) as u64;
        ensure!(
            n.checked_mul(min_item as u64).is_some_and(|b| b <= left),
            "{}: count {n} at byte {} exceeds the {left} bytes left",
            self.file,
            self.pos - 8
        );
        Ok(n as usize)
    }

    fn cstr(&mut self) -> Result<String> {
        let rest = &self.buf[self.pos..];
        let end = rest
            .iter()
            .position(|&b| b == 0)
            .with_context(|| format!("{}: unterminated name at byte {}", self.file, self.pos))?;
        let s = std::str::from_utf8(&rest[..end])
            .with_context(|| format!("{}: name at byte {} is not UTF-8", self.file, self.pos))?
            .to_string();
        self.pos += end + 1;
        Ok(s)
    }

    fn finish(&self) -> Result<()> {
        ensure!(
            self.pos == self.buf.len(),
            "{}: {} trailing bytes",
            self.file,
            self.buf.len() - self.pos
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model with every edge the formats have: two camera models, a name
    /// with a space, an image with no 2D points, invalid point ids, awkward floats.
    pub(crate) fn synthetic() -> ColmapModel {
        let mut m = ColmapModel::default();
        m.cameras.insert(
            1,
            ColmapCamera {
                id: 1,
                model_id: 1,
                width: 4946,
                height: 3286,
                params: vec![4000.123456789, 4001.5, 2473.0000001, 1643.25],
            },
        );
        m.cameras.insert(
            7,
            ColmapCamera {
                id: 7,
                model_id: 4,
                width: 640,
                height: 480,
                params: vec![500.0, 500.0, 320.0, 240.0, 0.1, -0.01, 1e-5, -3e-17],
            },
        );
        m.images.insert(
            3,
            ColmapImage {
                id: 3,
                qvec: [0.9, 0.1, -0.3, 0.2],
                tvec: [1.0 / 3.0, -2.5, 1e-300],
                camera_id: 1,
                name: "DSC 0001.JPG".into(),
                points2d: vec![(10.5, 20.25, 0), (1e9, -0.0, -1)],
            },
        );
        m.images.insert(
            4,
            ColmapImage {
                id: 4,
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [0.0, 0.0, 5.0],
                camera_id: 7,
                name: "sub/b.png".into(),
                points2d: vec![],
            },
        );
        m.points.push(ColmapPoint {
            id: 0,
            xyz: [0.1, 0.2, std::f64::consts::PI],
            rgb: [255, 0, 17],
            error: 0.75,
            track: vec![(3, 0)],
        });
        m.points.push(ColmapPoint {
            id: 99,
            xyz: [-1.0, 2.0, -3.0],
            rgb: [1, 2, 3],
            error: 0.0,
            track: vec![],
        });
        m
    }

    #[test]
    fn binary_round_trip_is_exact() {
        let m = synthetic();
        let [c, i, p] = m.to_bin();
        let back = ColmapModel::parse_bin(&c, &i, &p).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.to_bin(), [c, i, p]);
    }

    #[test]
    fn text_round_trip_reproduces_the_binary_bytes() {
        let m = synthetic();
        let [c, i, p] = m.to_txt();
        let back = ColmapModel::parse_txt(&c, &i, &p).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.to_bin(), m.to_bin(), "bin -> txt -> bin is byte-exact");
        assert_eq!(back.to_txt(), [c, i, p]);
    }

    #[test]
    fn files_round_trip_through_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        let m = synthetic();
        m.write_bin(&dir.path().join("bin")).unwrap();
        m.write_txt(&dir.path().join("txt")).unwrap();
        assert_eq!(ColmapModel::read_dir(&dir.path().join("bin")).unwrap(), m);
        assert_eq!(ColmapModel::read_dir(&dir.path().join("txt")).unwrap(), m);
        assert!(ColmapModel::read_dir(dir.path()).is_err());
    }

    #[test]
    fn corrupt_binaries_fail_cleanly() {
        let [c, i, p] = synthetic().to_bin();
        for cut in [0, 7, 8, 20, c.len() - 1] {
            assert!(
                ColmapModel::parse_bin(&c[..cut], &i, &p).is_err(),
                "cameras cut at {cut}"
            );
        }
        for cut in [9, 60, i.len() - 1] {
            assert!(
                ColmapModel::parse_bin(&c, &i[..cut], &p).is_err(),
                "images cut at {cut}"
            );
        }
        // A count claiming more items than bytes must fail before allocating.
        let mut huge = p.clone();
        huge[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(ColmapModel::parse_bin(&c, &i, &huge).is_err());
        let mut trailing = p.clone();
        trailing.push(0);
        assert!(ColmapModel::parse_bin(&c, &i, &trailing).is_err());
        let mut bad_model = c.clone();
        bad_model[8 + 4..8 + 8].copy_from_slice(&42i32.to_le_bytes());
        let err = ColmapModel::parse_bin(&bad_model, &i, &p).unwrap_err();
        assert!(format!("{err:#}").contains("unknown model id 42"));
    }

    #[test]
    fn text_reader_handles_names_with_spaces_and_empty_point_lines() {
        let images =
            "# header\n\n3 1 0 0 0 0 0 0 1 my image.jpg\n\n4 1 0 0 0 1 2 3 1 b.jpg\n1.5 2.5 -1\n";
        let cams = "1 SIMPLE_PINHOLE 100 80 90 50 40\n";
        let m = ColmapModel::parse_txt(cams, images, "").unwrap();
        assert_eq!(m.images[&3].name, "my image.jpg");
        assert!(m.images[&3].points2d.is_empty());
        assert_eq!(m.images[&4].points2d, vec![(1.5, 2.5, -1)]);
        assert_eq!(m.cameras[&1].pinhole(), Some([90.0, 90.0, 50.0, 40.0]));
        assert!(
            ColmapModel::parse_txt("1 PINHOLE 1 1 1 1 1\n", "", "").is_err(),
            "wrong param count"
        );
        assert!(
            ColmapModel::parse_txt(cams, "3 1 0 0 0 0 0 0 2 a.jpg\n\n", "").is_err(),
            "unknown camera"
        );
    }

    #[test]
    fn rotation_and_center_follow_colmap() {
        let img = ColmapImage {
            id: 1,
            // 90° about z: x -> y.
            qvec: [
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
                0.0,
                std::f64::consts::FRAC_1_SQRT_2,
            ],
            tvec: [1.0, 0.0, 0.0],
            camera_id: 1,
            name: String::new(),
            points2d: vec![],
        };
        let r = img.rotation();
        assert!((r[1][0] - 1.0).abs() < 1e-12 && (r[0][1] + 1.0).abs() < 1e-12);
        // C = -R^T t = -(R row 0 as column) = (0, 1, 0).
        let c = img.center();
        assert!((c[0]).abs() < 1e-12 && (c[1] - 1.0).abs() < 1e-12, "{c:?}");
    }
}
