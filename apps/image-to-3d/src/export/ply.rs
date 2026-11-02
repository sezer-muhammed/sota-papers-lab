//! INRIA 3DGS PLY (spec-train 9.1). Owner: loss (E4).
//!
//! `binary_little_endian 1.0`, one `vertex` element of 62 f32 properties in
//! the order x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2
//! rot_0..3; `f_rest_{c·15 + k}` is coefficient k of channel c
//! (channel-major, the transpose of our `[15][3]`); always the degree-3
//! layout (zeros above the trained degree); normals 0; opacity is the logit,
//! scales are logs, rotation is normalized (w, x, y, z). Viewers treat the
//! file as RDF.

use std::io::Write;

use anyhow::ensure;
use spl_core::Result;

use crate::model::GaussiansHost;

/// Floats per vertex in the INRIA layout.
pub const PLY_FLOATS: usize = 62;

pub fn write_inria_ply(g: &GaussiansHost, out: &mut dyn Write) -> Result<()> {
    g.validate()?;
    let n = g.len();
    let header = format!(
        "ply\n\
         format binary_little_endian 1.0\n\
         element vertex {n}\n\
         property float x\n\
         property float y\n\
         property float z\n\
         property float nx\n\
         property float ny\n\
         property float nz\n\
         property float f_dc_0\n\
         property float f_dc_1\n\
         property float f_dc_2\n\
         {f_rest}\
         property float opacity\n\
         property float scale_0\n\
         property float scale_1\n\
         property float scale_2\n\
         property float rot_0\n\
         property float rot_1\n\
         property float rot_2\n\
         property float rot_3\n\
         end_header\n",
        f_rest = (0..45)
            .map(|i| format!("property float f_rest_{i}\n"))
            .collect::<String>()
    );
    out.write_all(header.as_bytes())?;

    let mut row = [0f32; PLY_FLOATS];
    for i in 0..n {
        row[0] = g.means[i * 3];
        row[1] = g.means[i * 3 + 1];
        row[2] = g.means[i * 3 + 2];
        row[3] = 0.0;
        row[4] = 0.0;
        row[5] = 0.0;
        row[6] = g.sh0[i * 3];
        row[7] = g.sh0[i * 3 + 1];
        row[8] = g.sh0[i * 3 + 2];
        // f_rest: channel-major (c·15 + k), our storage is [15][3] per Gaussian
        for k in 0..15 {
            for c in 0..3 {
                row[9 + c * 15 + k] = g.sh_rest[i * 45 + k * 3 + c];
            }
        }
        row[54] = g.opacities[i];
        row[55] = g.scales[i * 3];
        row[56] = g.scales[i * 3 + 1];
        row[57] = g.scales[i * 3 + 2];
        let qn = quat_normalize(&g.rotations[i * 4..i * 4 + 4]);
        row[58] = qn[0];
        row[59] = qn[1];
        row[60] = qn[2];
        row[61] = qn[3];

        // SAFETY: f32 is repr(transparent) over [u8; 4] on LE platforms.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(row.as_ptr().cast::<u8>(), PLY_FLOATS * 4) };
        out.write_all(bytes)?;
    }
    Ok(())
}

fn quat_normalize(q: &[f32]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1e-12 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

pub fn read_inria_ply(bytes: &[u8]) -> Result<GaussiansHost> {
    let text = std::str::from_utf8(bytes)
        .unwrap_or_else(|e| std::str::from_utf8(&bytes[..e.valid_up_to()]).unwrap_or(""));
    let header_end = text
        .find("end_header\n")
        .map(|i| i + "end_header\n".len())
        .or_else(|| {
            text.find("end_header\r\n")
                .map(|i| i + "end_header\r\n".len())
        });
    let header_end = header_end.ok_or_else(|| spl_core::anyhow!("no end_header in PLY"))?;

    let mut n_vertices = 0usize;
    for line in text[..header_end].lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("element vertex ") {
            n_vertices = rest.trim().parse()?;
        }
    }

    let body = &bytes[header_end..];
    let expected = n_vertices * PLY_FLOATS * 4;
    ensure!(
        body.len() >= expected,
        "PLY body too short: {expected} bytes expected, {} found",
        body.len()
    );

    let n = n_vertices;
    let mut g = GaussiansHost {
        means: vec![0.0; n * 3],
        scales: vec![0.0; n * 3],
        rotations: vec![0.0; n * 4],
        opacities: vec![0.0; n],
        sh0: vec![0.0; n * 3],
        sh_rest: vec![0.0; n * 45],
        sh_degree: 3,
        antialiased: false,
    };

    for i in 0..n {
        let off = i * PLY_FLOATS * 4;
        let row: Vec<f32> = (0..PLY_FLOATS)
            .map(|j| {
                let b = &body[off + j * 4..off + j * 4 + 4];
                f32::from_le_bytes([b[0], b[1], b[2], b[3]])
            })
            .collect();

        g.means[i * 3] = row[0];
        g.means[i * 3 + 1] = row[1];
        g.means[i * 3 + 2] = row[2];
        g.sh0[i * 3] = row[6];
        g.sh0[i * 3 + 1] = row[7];
        g.sh0[i * 3 + 2] = row[8];
        // f_rest: channel-major → [15][3]
        for k in 0..15 {
            for c in 0..3 {
                g.sh_rest[i * 45 + k * 3 + c] = row[9 + c * 15 + k];
            }
        }
        g.opacities[i] = row[54];
        g.scales[i * 3] = row[55];
        g.scales[i * 3 + 1] = row[56];
        g.scales[i * 3 + 2] = row[57];
        g.rotations[i * 4] = row[58];
        g.rotations[i * 4 + 1] = row[59];
        g.rotations[i * 4 + 2] = row[60];
        g.rotations[i * 4 + 3] = row[61];
    }
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_gaussians(n: usize) -> GaussiansHost {
        let mut g = GaussiansHost::zeros(n);
        for i in 0..n {
            g.means[i * 3] = i as f32;
            g.means[i * 3 + 1] = i as f32 * 0.5;
            g.means[i * 3 + 2] = -(i as f32);
            g.scales[i * 3] = -2.0 + i as f32 * 0.1;
            g.scales[i * 3 + 1] = -1.5;
            g.scales[i * 3 + 2] = -3.0;
            g.opacities[i] = -2.0 + i as f32 * 0.5;
            g.sh0[i * 3] = 0.1;
            g.sh0[i * 3 + 1] = -0.2;
            g.sh0[i * 3 + 2] = 0.3;
            for j in 0..45 {
                g.sh_rest[i * 45 + j] = (j as f32 - 22.0) * 0.01;
            }
        }
        g
    }

    #[test]
    fn ply_round_trip() {
        let g = sample_gaussians(7);
        let mut buf = Vec::new();
        write_inria_ply(&g, &mut buf).unwrap();
        let back = read_inria_ply(&buf).unwrap();
        assert_eq!(back.len(), g.len());
        for i in 0..g.len() {
            for a in 0..3 {
                assert!(
                    (back.means[i * 3 + a] - g.means[i * 3 + a]).abs() < 1e-6,
                    "means[{i}][{a}]"
                );
                assert!(
                    (back.scales[i * 3 + a] - g.scales[i * 3 + a]).abs() < 1e-6,
                    "scales[{i}][{a}]"
                );
                assert!(
                    (back.sh0[i * 3 + a] - g.sh0[i * 3 + a]).abs() < 1e-6,
                    "sh0[{i}][{a}]"
                );
            }
            assert!(
                (back.opacities[i] - g.opacities[i]).abs() < 1e-6,
                "opacity[{i}]"
            );
            for j in 0..45 {
                assert!(
                    (back.sh_rest[i * 45 + j] - g.sh_rest[i * 45 + j]).abs() < 1e-6,
                    "sh_rest[{i}][{j}]"
                );
            }
            let qn = super::quat_normalize(&g.rotations[i * 4..i * 4 + 4]);
            for (a, &expected) in qn.iter().enumerate() {
                assert!(
                    (back.rotations[i * 4 + a] - expected).abs() < 1e-6,
                    "rot[{i}][{a}]"
                );
            }
        }
    }

    #[test]
    fn ply_empty() {
        let g = GaussiansHost::zeros(0);
        let mut buf = Vec::new();
        write_inria_ply(&g, &mut buf).unwrap();
        let back = read_inria_ply(&buf).unwrap();
        assert_eq!(back.len(), 0);
    }

    #[test]
    fn ply_header_contains_62_properties() {
        let g = sample_gaussians(1);
        let mut buf = Vec::new();
        write_inria_ply(&g, &mut buf).unwrap();
        let end = buf
            .windows(b"end_header\n".len())
            .position(|w| w == b"end_header\n")
            .unwrap()
            + b"end_header\n".len();
        let header = std::str::from_utf8(&buf[..end]).unwrap();
        let props = header
            .lines()
            .filter(|l| l.starts_with("property float"))
            .count();
        assert_eq!(props, PLY_FLOATS);
    }
}
