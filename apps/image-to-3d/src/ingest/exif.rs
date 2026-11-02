//! Minimal EXIF (TIFF) reader for the focal prior. Owner: ingest (A0).
//!
//! Reads `FocalLengthIn35mmFilm` (0xA405) and `FocalLength` (0x920A) from the
//! Exif sub-IFD. Orientation is handled by the `image` crate's decoders.

/// Focal length tags of one image.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExifFocal {
    /// 35 mm equivalent focal length in mm (0xA405); 0 in the file means unknown.
    pub focal_35mm: Option<f32>,
    /// Physical focal length in mm (0x920A).
    pub focal_mm: Option<f32>,
}

const TAG_EXIF_IFD: u16 = 0x8769;
const TAG_FOCAL_35MM: u16 = 0xA405;
const TAG_FOCAL: u16 = 0x920A;

struct Tiff<'a> {
    b: &'a [u8],
    le: bool,
}

impl Tiff<'_> {
    fn u16(&self, at: usize) -> Option<u16> {
        let s: [u8; 2] = self.b.get(at..at + 2)?.try_into().ok()?;
        Some(if self.le {
            u16::from_le_bytes(s)
        } else {
            u16::from_be_bytes(s)
        })
    }

    fn u32(&self, at: usize) -> Option<u32> {
        let s: [u8; 4] = self.b.get(at..at + 4)?.try_into().ok()?;
        Some(if self.le {
            u32::from_le_bytes(s)
        } else {
            u32::from_be_bytes(s)
        })
    }

    /// (type, count, offset of the 4-byte value field) of `tag` in the IFD at `ifd`.
    fn find(&self, ifd: usize, tag: u16) -> Option<(u16, u32, usize)> {
        let n = self.u16(ifd)? as usize;
        (0..n.min(512)).find_map(|i| {
            let e = ifd + 2 + 12 * i;
            (self.u16(e)? == tag).then_some(())?;
            Some((self.u16(e + 2)?, self.u32(e + 4)?, e + 8))
        })
    }

    /// First value of a SHORT, LONG or RATIONAL entry as f64.
    fn number(&self, (ty, count, field): (u16, u32, usize)) -> Option<f64> {
        if count == 0 {
            return None;
        }
        match ty {
            3 => self.u16(field).map(f64::from),
            4 => self.u32(field).map(f64::from),
            5 => {
                let at = self.u32(field)? as usize;
                let (n, d) = (self.u32(at)?, self.u32(at + 4)?);
                (d != 0).then(|| n as f64 / d as f64)
            }
            _ => None,
        }
    }
}

/// Parses an EXIF block (raw TIFF, optionally prefixed by `Exif\0\0`).
pub fn parse_focal(exif: &[u8]) -> ExifFocal {
    let b = exif.strip_prefix(b"Exif\0\0").unwrap_or(exif);
    let le = match b.get(0..4) {
        Some([b'I', b'I', 42, 0]) => true,
        Some([b'M', b'M', 0, 42]) => false,
        _ => return ExifFocal::default(),
    };
    let t = Tiff { b, le };
    let Some(ifd0) = t.u32(4) else {
        return ExifFocal::default();
    };
    let Some(exif_ifd) = t
        .find(ifd0 as usize, TAG_EXIF_IFD)
        .and_then(|(_, _, f)| t.u32(f))
    else {
        return ExifFocal::default();
    };
    let positive = |v: Option<f64>| v.filter(|v| *v > 0.0 && v.is_finite()).map(|v| v as f32);
    ExifFocal {
        focal_35mm: positive(
            t.find(exif_ifd as usize, TAG_FOCAL_35MM)
                .and_then(|e| t.number(e)),
        ),
        focal_mm: positive(
            t.find(exif_ifd as usize, TAG_FOCAL)
                .and_then(|e| t.number(e)),
        ),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A TIFF block with IFD0 → Exif IFD holding 0xA405 = `f35` and 0x920A = `fmm`/100.
    pub(crate) fn build(le: bool, f35: u16, fmm_x100: u32) -> Vec<u8> {
        let w16 = |v: u16| if le { v.to_le_bytes() } else { v.to_be_bytes() };
        let w32 = |v: u32| if le { v.to_le_bytes() } else { v.to_be_bytes() };
        let mut b = Vec::new();
        b.extend_from_slice(if le { b"II" } else { b"MM" });
        b.extend_from_slice(&w16(42));
        b.extend_from_slice(&w32(8));
        // IFD0 at 8: one entry (Exif pointer), next IFD = 0. Size 2 + 12 + 4 = 18.
        b.extend_from_slice(&w16(1));
        b.extend_from_slice(&w16(TAG_EXIF_IFD));
        b.extend_from_slice(&w16(4));
        b.extend_from_slice(&w32(1));
        b.extend_from_slice(&w32(26));
        b.extend_from_slice(&w32(0));
        // Exif IFD at 26: two entries, size 2 + 24 + 4 = 30; rational at 56.
        b.extend_from_slice(&w16(2));
        b.extend_from_slice(&w16(TAG_FOCAL));
        b.extend_from_slice(&w16(5));
        b.extend_from_slice(&w32(1));
        b.extend_from_slice(&w32(56));
        b.extend_from_slice(&w16(TAG_FOCAL_35MM));
        b.extend_from_slice(&w16(3));
        b.extend_from_slice(&w32(1));
        b.extend_from_slice(&w16(f35));
        b.extend_from_slice(&[0, 0]);
        b.extend_from_slice(&w32(0));
        assert_eq!(b.len(), 56);
        b.extend_from_slice(&w32(fmm_x100));
        b.extend_from_slice(&w32(100));
        b
    }

    #[test]
    fn reads_both_byte_orders() {
        for le in [true, false] {
            let f = parse_focal(&build(le, 26, 565));
            assert_eq!(f.focal_35mm, Some(26.0));
            assert!((f.focal_mm.unwrap() - 5.65).abs() < 1e-6);
            let mut prefixed = b"Exif\0\0".to_vec();
            prefixed.extend(build(le, 26, 565));
            assert_eq!(parse_focal(&prefixed), f);
        }
    }

    #[test]
    fn zero_and_garbage_are_unknown() {
        assert_eq!(parse_focal(&build(true, 0, 0)), ExifFocal::default());
        assert_eq!(parse_focal(b"not exif"), ExifFocal::default());
        let mut cut = build(true, 26, 565);
        cut.truncate(30);
        assert_eq!(parse_focal(&cut).focal_35mm, None);
    }
}
