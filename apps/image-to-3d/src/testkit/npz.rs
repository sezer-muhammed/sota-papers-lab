//! NumPy `.npy` / `.npz` files, the exchange format of `gs-oracle`
//! (benchmarks/image-to-3d/ORACLE.md). Writes stored (uncompressed) zips of
//! v1.0 `.npy` entries; reads stored and deflated entries, v1/v2 headers,
//! little-endian C-order arrays of the dtypes below.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::ensure;
use spl_core::{Context, Result, bail};

/// Array payloads by dtype (`<f4`, `<f8`, `<i4`, `<i8`, `<u4`, `<u2`, `|u1`, `|b1`).
#[derive(Clone, Debug, PartialEq)]
pub enum NpyData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U32(Vec<u32>),
    U16(Vec<u16>),
    U8(Vec<u8>),
    Bool(Vec<bool>),
}

impl NpyData {
    fn descr(&self) -> &'static str {
        match self {
            NpyData::F32(_) => "<f4",
            NpyData::F64(_) => "<f8",
            NpyData::I32(_) => "<i4",
            NpyData::I64(_) => "<i8",
            NpyData::U32(_) => "<u4",
            NpyData::U16(_) => "<u2",
            NpyData::U8(_) => "|u1",
            NpyData::Bool(_) => "|b1",
        }
    }

    pub fn len(&self) -> usize {
        match self {
            NpyData::F32(v) => v.len(),
            NpyData::F64(v) => v.len(),
            NpyData::I32(v) => v.len(),
            NpyData::I64(v) => v.len(),
            NpyData::U32(v) => v.len(),
            NpyData::U16(v) => v.len(),
            NpyData::U8(v) => v.len(),
            NpyData::Bool(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn bytes(&self) -> Vec<u8> {
        fn le<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
            v.iter().flat_map(|x| f(*x)).collect()
        }
        match self {
            NpyData::F32(v) => le(v, f32::to_le_bytes),
            NpyData::F64(v) => le(v, f64::to_le_bytes),
            NpyData::I32(v) => le(v, i32::to_le_bytes),
            NpyData::I64(v) => le(v, i64::to_le_bytes),
            NpyData::U32(v) => le(v, u32::to_le_bytes),
            NpyData::U16(v) => le(v, u16::to_le_bytes),
            NpyData::U8(v) => v.clone(),
            NpyData::Bool(v) => v.iter().map(|&b| u8::from(b)).collect(),
        }
    }

    fn parse(descr: &str, raw: &[u8], n: usize) -> Result<Self> {
        fn take<const N: usize, T>(
            raw: &[u8],
            n: usize,
            f: impl Fn([u8; N]) -> T,
        ) -> Result<Vec<T>> {
            ensure!(
                raw.len() == n * N,
                "payload has {} bytes, expected {}",
                raw.len(),
                n * N
            );
            Ok(raw.as_chunks::<N>().0.iter().map(|c| f(*c)).collect())
        }
        Ok(match descr {
            "<f4" => NpyData::F32(take(raw, n, f32::from_le_bytes)?),
            "<f8" => NpyData::F64(take(raw, n, f64::from_le_bytes)?),
            "<i4" => NpyData::I32(take(raw, n, i32::from_le_bytes)?),
            "<i8" => NpyData::I64(take(raw, n, i64::from_le_bytes)?),
            "<u4" => NpyData::U32(take(raw, n, u32::from_le_bytes)?),
            "<u2" => NpyData::U16(take(raw, n, u16::from_le_bytes)?),
            "|u1" => NpyData::U8(take(raw, n, |b: [u8; 1]| b[0])?),
            "|b1" => NpyData::Bool(take(raw, n, |b: [u8; 1]| b[0] != 0)?),
            other => bail!("unsupported npy dtype {other}"),
        })
    }
}

/// One array: shape (empty for a 0-d scalar) and C-order data.
#[derive(Clone, Debug, PartialEq)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: NpyData,
}

impl NpyArray {
    pub fn new(shape: &[usize], data: NpyData) -> Result<Self> {
        let n: usize = shape.iter().product();
        ensure!(
            n == data.len(),
            "shape {shape:?} holds {n} elements, data has {}",
            data.len()
        );
        Ok(Self {
            shape: shape.to_vec(),
            data,
        })
    }

    /// Element values as f64 (any numeric dtype), for comparisons.
    pub fn to_f64(&self) -> Vec<f64> {
        match &self.data {
            NpyData::F32(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::F64(v) => v.clone(),
            NpyData::I32(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::I64(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::U32(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::U16(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::U8(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::Bool(v) => v.iter().map(|&x| f64::from(u8::from(x))).collect(),
        }
    }

    /// The `.npy` v1.0 bytes (header padded to a multiple of 64).
    pub fn to_npy(&self) -> Vec<u8> {
        let shape = match self.shape.len() {
            0 => "()".to_string(),
            1 => format!("({},)", self.shape[0]),
            _ => format!(
                "({})",
                self.shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        let mut header = format!(
            "{{'descr': '{}', 'fortran_order': False, 'shape': {shape}, }}",
            self.data.descr()
        );
        let unpadded = 10 + header.len() + 1;
        header.push_str(&" ".repeat(unpadded.next_multiple_of(64) - unpadded));
        header.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&self.data.bytes());
        out
    }

    pub fn from_npy(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY",
            "not an .npy file"
        );
        let (hlen, start) = match bytes[6] {
            1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
            2 | 3 => {
                ensure!(bytes.len() >= 12, "truncated .npy header");
                (
                    u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) as usize,
                    12,
                )
            }
            v => bail!("unsupported .npy version {v}"),
        };
        ensure!(bytes.len() >= start + hlen, "truncated .npy header");
        let header =
            std::str::from_utf8(&bytes[start..start + hlen]).context(".npy header is not UTF-8")?;
        let field = |key: &str| -> Result<&str> {
            let at = header
                .find(&format!("'{key}'"))
                .with_context(|| format!(".npy header lacks {key}: {header}"))?;
            Ok(header[at + key.len() + 2..]
                .trim_start()
                .trim_start_matches(':')
                .trim_start())
        };
        let descr = field("descr")?;
        let descr = descr
            .strip_prefix('\'')
            .and_then(|d| d.split('\'').next())
            .context("bad descr")?;
        let descr = match descr {
            "<b1" => "|b1",
            "<u1" => "|u1",
            d => d,
        };
        ensure!(
            field("fortran_order")?.starts_with("False"),
            "Fortran-order arrays are not supported"
        );
        let shape_text = field("shape")?;
        let shape_text = shape_text
            .strip_prefix('(')
            .and_then(|s| s.split(')').next())
            .context("bad shape")?;
        let shape = shape_text
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<usize>()
                    .with_context(|| format!("bad shape entry {s}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let n = shape.iter().product();
        let data = NpyData::parse(descr, &bytes[start + hlen..], n)?;
        Ok(Self { shape, data })
    }
}

/// A named set of arrays (`np.savez`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Npz(pub BTreeMap<String, NpyArray>);

impl Npz {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: &str, shape: &[usize], data: NpyData) -> Result<&mut Self> {
        self.0.insert(name.to_string(), NpyArray::new(shape, data)?);
        Ok(self)
    }

    pub fn f32(&mut self, name: &str, shape: &[usize], v: Vec<f32>) -> Result<&mut Self> {
        self.insert(name, shape, NpyData::F32(v))
    }

    pub fn i32_scalar(&mut self, name: &str, v: i32) -> Result<&mut Self> {
        self.insert(name, &[], NpyData::I32(vec![v]))
    }

    pub fn f32_scalar(&mut self, name: &str, v: f32) -> Result<&mut Self> {
        self.insert(name, &[], NpyData::F32(vec![v]))
    }

    pub fn get(&self, name: &str) -> Result<&NpyArray> {
        self.0.get(name).with_context(|| {
            format!(
                "npz has no array {name:?} (has {:?})",
                self.0.keys().collect::<Vec<_>>()
            )
        })
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(true);
        for (name, arr) in &self.0 {
            zip.start_file(format!("{name}.npy"), opts)?;
            zip.write_all(&arr.to_npy())?;
        }
        Ok(zip.finish()?.into_inner())
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut zip =
            zip::ZipArchive::new(std::io::Cursor::new(bytes)).context("not a zip (.npz) file")?;
        let mut out = BTreeMap::new();
        for i in 0..zip.len() {
            let mut f = zip.by_index(i)?;
            let name = f.name()?.trim_end_matches(".npy").to_string();
            let mut buf = Vec::with_capacity(f.size().min(1 << 30) as usize);
            f.read_to_end(&mut buf)?;
            out.insert(
                name.clone(),
                NpyArray::from_npy(&buf).with_context(|| format!("array {name}"))?,
            );
        }
        Ok(Self(out))
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        std::fs::write(path, self.to_bytes()?).with_context(|| format!("write {}", path.display()))
    }

    pub fn read(path: &Path) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path).with_context(|| format!("read {}", path.display()))?)
            .with_context(|| format!("parse {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_header_is_numpy_v1_aligned() {
        let a = NpyArray::new(&[2, 3], NpyData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
        let b = a.to_npy();
        let hlen = u16::from_le_bytes([b[8], b[9]]) as usize;
        assert_eq!((10 + hlen) % 64, 0);
        let header = std::str::from_utf8(&b[10..10 + hlen]).unwrap();
        assert!(header.starts_with("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }"));
        assert!(header.ends_with('\n'));
        assert_eq!(b.len(), 10 + hlen + 24);
    }

    #[test]
    fn npz_round_trips_every_dtype_and_scalars() {
        let mut z = Npz::new();
        z.f32("means", &[2, 3], vec![0.5; 6]).unwrap();
        z.insert("ids", &[3], NpyData::I64(vec![-1, 0, i64::MAX]))
            .unwrap();
        z.insert("mask", &[2], NpyData::Bool(vec![true, false]))
            .unwrap();
        z.insert("px", &[1, 2], NpyData::U8(vec![0, 255])).unwrap();
        z.insert("d", &[1], NpyData::F64(vec![std::f64::consts::E]))
            .unwrap();
        z.insert("k", &[2], NpyData::U16(vec![8191, 0])).unwrap();
        z.insert("v", &[1], NpyData::U32(vec![u32::MAX])).unwrap();
        z.i32_scalar("width", 64).unwrap();
        z.f32_scalar("eps2d", 0.3).unwrap();
        let back = Npz::from_bytes(&z.to_bytes().unwrap()).unwrap();
        assert_eq!(back, z);
        assert_eq!(back.get("width").unwrap().shape, Vec::<usize>::new());
        assert!(back.get("nope").is_err());
        assert!(NpyArray::new(&[2, 2], NpyData::F32(vec![0.0; 3])).is_err());
    }

    /// The bytes numpy 2.x writes for `np.save(f, np.arange(3, dtype='<i4'))`.
    #[test]
    fn reads_numpy_written_npy() {
        let mut b = b"\x93NUMPY\x01\x00\x76\x00".to_vec();
        let mut h = "{'descr': '<i4', 'fortran_order': False, 'shape': (3,), }".to_string();
        h.push_str(&" ".repeat(118 - h.len() - 1));
        h.push('\n');
        b.extend_from_slice(h.as_bytes());
        for i in 0..3i32 {
            b.extend_from_slice(&i.to_le_bytes());
        }
        let a = NpyArray::from_npy(&b).unwrap();
        assert_eq!(
            (a.shape.clone(), a.data),
            (vec![3], NpyData::I32(vec![0, 1, 2]))
        );
    }

    #[test]
    fn reads_deflated_npz() {
        let a = NpyArray::new(&[4], NpyData::F32(vec![1.0, 2.0, 3.0, 4.0])).unwrap();
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zip.start_file("x.npy", opts).unwrap();
        zip.write_all(&a.to_npy()).unwrap();
        let bytes = zip.finish().unwrap().into_inner();
        assert_eq!(Npz::from_bytes(&bytes).unwrap().get("x").unwrap(), &a);
    }
}
