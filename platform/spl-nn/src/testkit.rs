//! Test helpers: a seeded RNG, error statistics, `.npy` files for the
//! PyTorch fixtures, CUDA-event timing and the output directory.

use std::path::{Path, PathBuf};

/// SplitMix64: deterministic, dependency-free.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = (self.uniform() as f64).max(1e-12);
        let u2 = self.uniform() as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }

    pub fn normals(&mut self, n: usize, std: f32) -> Vec<f32> {
        (0..n).map(|_| self.normal() * std).collect()
    }
}

/// Error of `got` against `want`: max absolute, RMS, and RMS relative to
/// the RMS of `want`.
#[derive(Debug, Clone, Copy)]
pub struct Err {
    pub max_abs: f64,
    pub rms: f64,
    pub rel_rms: f64,
}

pub fn err(got: &[f32], want: &[f64]) -> Err {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let mut max_abs = 0.0f64;
    let (mut se, mut sw) = (0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want) {
        let d = g as f64 - w;
        assert!(d.is_finite(), "non-finite output {g} vs {w}");
        max_abs = max_abs.max(d.abs());
        se += d * d;
        sw += w * w;
    }
    let n = got.len().max(1) as f64;
    Err {
        max_abs,
        rms: (se / n).sqrt(),
        rel_rms: (se / sw.max(1e-30)).sqrt(),
    }
}

pub fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

pub fn out_dir() -> PathBuf {
    let dir =
        std::env::var_os("SPL_OUT").map_or_else(|| PathBuf::from("/out/nn-dense"), PathBuf::from);
    std::fs::create_dir_all(&dir).expect("create output dir");
    dir
}

/// A loaded `.npy` array: shape and little-endian payload.
pub struct Npy {
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Npy {
    pub fn read(path: &Path) -> Self {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        assert_eq!(&bytes[..6], b"\x93NUMPY", "{} is not .npy", path.display());
        let (hlen, start) = match bytes[6] {
            1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
            _ => (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            ),
        };
        let header = std::str::from_utf8(&bytes[start..start + hlen]).expect("utf8 header");
        let field = |name: &str| {
            let at = header
                .find(name)
                .unwrap_or_else(|| panic!("{name} in {header}"));
            &header[at + name.len()..]
        };
        let descr = field("'descr': '").split('\'').next().unwrap().to_string();
        assert!(
            field("'fortran_order': ").starts_with("False"),
            "C order only"
        );
        let shape_txt = field("'shape': (").split(')').next().unwrap();
        let shape = shape_txt
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().expect("shape"))
            .collect();
        Self {
            descr,
            shape,
            data: bytes[start + hlen..].to_vec(),
        }
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn f32(&self) -> Vec<f32> {
        assert_eq!(self.descr, "<f4");
        let (chunks, _) = self.data.as_chunks::<4>();
        chunks.iter().map(|c| f32::from_le_bytes(*c)).collect()
    }

    /// bf16 stored as raw `<u2` bits.
    pub fn u16(&self) -> Vec<u16> {
        assert_eq!(self.descr, "<u2");
        let (chunks, _) = self.data.as_chunks::<2>();
        chunks.iter().map(|c| u16::from_le_bytes(*c)).collect()
    }

    /// FP8 stored as raw `|u1` bits.
    pub fn u8(&self) -> Vec<u8> {
        assert_eq!(self.descr, "|u1");
        self.data.clone()
    }
}

pub fn fixtures_dir() -> PathBuf {
    std::env::var_os("SPL_NN_FIXTURES")
        .map_or_else(|| PathBuf::from("/out/nn-dense/fixtures"), PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_header_parses() {
        let dir = std::env::temp_dir().join("spl-nn-npy-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.npy");
        let mut header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }".to_string();
        while !(10 + header.len() + 1).is_multiple_of(64) {
            header.push(' ');
        }
        header.push('\n');
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend((header.len() as u16).to_le_bytes());
        bytes.extend(header.as_bytes());
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            bytes.extend(v.to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();
        let a = Npy::read(&path);
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.len(), 6);
        assert_eq!(a.f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn rng_is_deterministic_and_roughly_normal() {
        let a = Rng::new(7).normals(20000, 1.0);
        assert_eq!(a, Rng::new(7).normals(20000, 1.0));
        let mean = a.iter().map(|&x| x as f64).sum::<f64>() / a.len() as f64;
        let var = a.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / a.len() as f64;
        assert!(
            mean.abs() < 0.03 && (var - 1.0).abs() < 0.05,
            "{mean} {var}"
        );
    }
}
