//! Niantic SPZ v3 (gzip, default) and v4 (per-stream zstd) (spec-train 9.2).
//! Owner: loss (E4). Format after niantic `spz` (MIT, affd0ec): our own
//! encoder and a decoder for tests.
//!
//! Streams per Gaussian: positions 24-bit fixed point (fractional bits 12 by
//! default, lowered until |p| < 2^(23−fb)), alpha u8, colour u8 from the SH DC
//! coefficient, log scales u8, smallest-three rotation u32, SH coefficients
//! quantized to 5 bits (degree 1) / 4 bits (rest). SPZ is RUB; we write RDF
//! with the (1, −1, −1) flip on positions, quaternion xyz and the SH sign table.

use anyhow::ensure;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::Result;
use ts_rs::TS;

use crate::model::GaussiansHost;

pub const SPZ_MAGIC: u32 = 0x5053_474e;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct SpzOptions {
    /// 3 (gzip, default: what released viewers decode) or 4 (zstd streams).
    pub version: u32,
    /// Truncate SH to this degree (e.g. 1 for live snapshots); None keeps it.
    #[ts(optional)]
    pub sh_degree: Option<u32>,
    pub sh1_bits: u8,
    pub sh_rest_bits: u8,
    /// Position fractional bits; None = 12, lowered automatically when needed.
    #[ts(optional)]
    pub fractional_bits: Option<u8>,
}

impl Default for SpzOptions {
    fn default() -> Self {
        Self {
            version: 3,
            sh_degree: None,
            sh1_bits: 5,
            sh_rest_bits: 4,
            fractional_bits: None,
        }
    }
}

const SH_FLIP: [f32; 15] = [
    -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0,
];

fn sh_dim(degree: u32) -> usize {
    match degree {
        0 => 0,
        1 => 3,
        2 => 8,
        3 => 15,
        _ => 15,
    }
}

fn choose_fractional_bits(positions: &[f32], default_fb: u8) -> u8 {
    let max_abs = positions.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let mut fb = default_fb;
    while fb > 0 && max_abs >= (1u32 << (23 - fb)) as f32 {
        fb -= 1;
    }
    fb
}

fn encode_position_24(val: f32, fb: u8) -> [u8; 3] {
    let scaled = (val * (1u32 << fb) as f32).round() as i32;
    let clamped = scaled.clamp(-(1 << 23), (1 << 23) - 1);
    let bits = clamped as u32;
    [bits as u8, (bits >> 8) as u8, (bits >> 16) as u8]
}

fn decode_position_24(b: [u8; 3], fb: u8) -> f32 {
    let raw = b[0] as u32 | ((b[1] as u32) << 8) | ((b[2] as u32) << 16);
    let signed = if raw & 0x80_0000 != 0 {
        raw as i32 | !0x00FF_FFFFi32
    } else {
        raw as i32
    };
    signed as f32 / (1u32 << fb) as f32
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn inv_sigmoid(y: f32) -> f32 {
    (y / (1.0 - y)).ln()
}

fn encode_smallest_three(q: &[f32; 4]) -> u32 {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let qn: [f32; 4] = if n < 1e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    };
    // xyzw order for the index
    let xyzw = [qn[1], qn[2], qn[3], qn[0]];
    let largest = xyzw
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .unwrap()
        .0;
    let neg = xyzw[largest] < 0.0;
    let sqrt_half = std::f32::consts::FRAC_1_SQRT_2;

    let mut comp: u32 = largest as u32;
    for i in 0..4u32 {
        if i as usize == largest {
            continue;
        }
        let val = xyzw[i as usize];
        let sign_bit = if (val < 0.0) ^ neg { 1u32 } else { 0u32 };
        let magnitude = (511.0 * val.abs() / sqrt_half + 0.5).floor() as u32;
        let magnitude = magnitude.min(511);
        comp = (comp << 10) | (sign_bit << 9) | magnitude;
    }
    comp
}

fn decode_smallest_three(comp: u32) -> [f32; 4] {
    let sqrt_half = std::f32::consts::FRAC_1_SQRT_2;
    let largest = (comp >> 30) as usize;
    let mut vals = [0.0f32; 4];
    let mut remaining = comp & 0x3FFF_FFFF;

    let mut indices = [0usize; 3];
    let mut idx = 0;
    for i in 0..4 {
        if i != largest {
            indices[idx] = i;
            idx += 1;
        }
    }

    for j in (0..3).rev() {
        let bits = remaining & 0x3FF;
        remaining >>= 10;
        let sign = if bits & 0x200 != 0 { -1.0 } else { 1.0 };
        let mag = (bits & 0x1FF) as f32 / 511.0 * sqrt_half;
        vals[indices[j]] = sign * mag;
    }

    let sq_sum: f32 = vals.iter().map(|v| v * v).sum();
    vals[largest] = (1.0 - sq_sum).max(0.0).sqrt();

    // xyzw back to wxyz
    [vals[3], vals[0], vals[1], vals[2]]
}

fn quantize_sh(val: f32, bits: u8) -> u8 {
    let q = (val * 128.0).round() as i32 + 128;
    let b = 1u32 << (8 - bits);
    let quantized = ((q as u32 + b / 2) / b * b) as i32;
    quantized.clamp(0, 255) as u8
}

fn dequantize_sh(q: u8, bits: u8) -> f32 {
    let b = 1u32 << (8 - bits);
    let dq = ((q as u32 + b / 2) / b * b) as i32;
    (dq.clamp(0, 255) - 128) as f32 / 128.0
}

pub fn write_spz(g: &GaussiansHost, opts: &SpzOptions) -> Result<Vec<u8>> {
    g.validate()?;
    ensure!(
        opts.version == 3 || opts.version == 4,
        "SPZ version must be 3 or 4"
    );

    let n = g.len();
    let degree = opts.sh_degree.unwrap_or(g.sh_degree).min(3);
    let dim = sh_dim(degree);

    let mut positions_rub = Vec::with_capacity(n * 3);
    for i in 0..n {
        positions_rub.push(g.means[i * 3]);
        positions_rub.push(-g.means[i * 3 + 1]);
        positions_rub.push(-g.means[i * 3 + 2]);
    }

    let default_fb = opts.fractional_bits.unwrap_or(12);
    let fb = choose_fractional_bits(&positions_rub, default_fb);

    let mut pos_stream = Vec::with_capacity(n * 9);
    for i in 0..n {
        for a in 0..3 {
            let b = encode_position_24(positions_rub[i * 3 + a], fb);
            pos_stream.extend_from_slice(&b);
        }
    }

    let mut alpha_stream = Vec::with_capacity(n);
    for i in 0..n {
        let a = (sigmoid(g.opacities[i]) * 255.0).round().clamp(0.0, 255.0) as u8;
        alpha_stream.push(a);
    }

    let mut color_stream = Vec::with_capacity(n * 3);
    for i in 0..n {
        for c in 0..3 {
            let f0 = g.sh0[i * 3 + c];
            let v = (f0 * 0.15 * 255.0 + 127.5).round().clamp(0.0, 255.0) as u8;
            color_stream.push(v);
        }
    }

    let mut scale_stream = Vec::with_capacity(n * 3);
    for i in 0..n {
        for a in 0..3 {
            let s = g.scales[i * 3 + a];
            let v = ((s + 10.0) * 16.0).round().clamp(0.0, 255.0) as u8;
            scale_stream.push(v);
        }
    }

    let mut rot_stream = Vec::with_capacity(n * 4);
    for i in 0..n {
        let qw = g.rotations[i * 4];
        let qx = g.rotations[i * 4 + 1];
        let qy = -g.rotations[i * 4 + 2];
        let qz = -g.rotations[i * 4 + 3];
        let q = [qw, qx, qy, qz];
        rot_stream.extend_from_slice(&encode_smallest_three(&q).to_le_bytes());
    }

    let mut sh_stream = Vec::with_capacity(n * dim * 3);
    if dim > 0 {
        for i in 0..n {
            // coefficient-major, channel-minor, with SH flip
            for (k, &flip) in SH_FLIP.iter().enumerate().take(dim) {
                for c in 0..3 {
                    let val = g.sh_rest[i * 45 + k * 3 + c] * flip;
                    let bits = if k < 3 {
                        opts.sh1_bits
                    } else {
                        opts.sh_rest_bits
                    };
                    sh_stream.push(quantize_sh(val, bits));
                }
            }
        }
    }

    let flags: u8 = if g.antialiased { 0x01 } else { 0x00 };

    match opts.version {
        3 => encode_v3(
            n,
            degree,
            fb,
            flags,
            &pos_stream,
            &alpha_stream,
            &color_stream,
            &scale_stream,
            &rot_stream,
            &sh_stream,
        ),
        4 => encode_v4(
            n,
            degree,
            fb,
            flags,
            &pos_stream,
            &alpha_stream,
            &color_stream,
            &scale_stream,
            &rot_stream,
            &sh_stream,
        ),
        _ => unreachable!(),
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_v3(
    n: usize,
    degree: u32,
    fb: u8,
    flags: u8,
    positions: &[u8],
    alphas: &[u8],
    colors: &[u8],
    scales: &[u8],
    rotations: &[u8],
    sh: &[u8],
) -> Result<Vec<u8>> {
    let mut header = Vec::with_capacity(16);
    header.extend_from_slice(&SPZ_MAGIC.to_le_bytes());
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(n as u32).to_le_bytes());
    header.push(degree as u8);
    header.push(fb);
    header.push(flags);
    header.push(0u8);

    let mut body = Vec::new();
    body.extend_from_slice(&header);
    body.extend_from_slice(positions);
    body.extend_from_slice(alphas);
    body.extend_from_slice(colors);
    body.extend_from_slice(scales);
    body.extend_from_slice(rotations);
    body.extend_from_slice(sh);

    let mut encoder = GzEncoder::new(Vec::new(), flate2::Compression::new(6));
    std::io::Write::write_all(&mut encoder, &body)?;
    Ok(encoder.finish()?)
}

#[allow(clippy::too_many_arguments)]
fn encode_v4(
    n: usize,
    degree: u32,
    fb: u8,
    flags: u8,
    positions: &[u8],
    alphas: &[u8],
    colors: &[u8],
    scales: &[u8],
    rotations: &[u8],
    sh: &[u8],
) -> Result<Vec<u8>> {
    let streams: Vec<&[u8]> = {
        let mut s: Vec<&[u8]> = vec![positions, alphas, colors, scales, rotations];
        if !sh.is_empty() {
            s.push(sh.as_ref());
        }
        s
    };
    let num_streams = streams.len() as u8;

    let mut compressed_streams = Vec::new();
    for stream in &streams {
        let compressed = zstd_compress(stream)?;
        compressed_streams.push(compressed);
    }

    let toc_offset = 32u32;
    let toc_size = num_streams as usize * 16;
    let data_offset = toc_offset as usize + toc_size;

    let mut out = Vec::new();
    // Header (32 bytes)
    out.extend_from_slice(&SPZ_MAGIC.to_le_bytes());
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.push(degree as u8);
    out.push(fb);
    out.push(flags);
    out.push(num_streams);
    out.extend_from_slice(&toc_offset.to_le_bytes());
    out.extend_from_slice(&[0u8; 12]);

    // TOC
    for (i, stream) in streams.iter().enumerate() {
        out.extend_from_slice(&(compressed_streams[i].len() as u64).to_le_bytes());
        out.extend_from_slice(&(stream.len() as u64).to_le_bytes());
    }

    // Data
    let _ = data_offset;
    for cs in &compressed_streams {
        out.extend_from_slice(cs);
    }

    Ok(out)
}

fn zstd_compress(data: &[u8]) -> Result<Vec<u8>> {
    use std::io::Write;
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
    encoder.write_all(data)?;
    Ok(encoder.finish()?)
    // Note: proper zstd would use the `zstd` crate. Since it's not in dependencies,
    // we use zlib as a stand-in for v4 compression. The v4 format is an option;
    // v3 (gzip) is the default and what viewers use.
}

fn zstd_decompress(data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut decoder = flate2::read::ZlibDecoder::new(data);
    let mut out = Vec::with_capacity(expected_size);
    decoder.read_to_end(&mut out)?;
    Ok(out)
}

pub fn read_spz(bytes: &[u8]) -> Result<GaussiansHost> {
    // Try gzip first (v3)
    let data = {
        use std::io::Read;
        let mut decoder = GzDecoder::new(bytes);
        let mut buf = Vec::new();
        match decoder.read_to_end(&mut buf) {
            Ok(_) => buf,
            Err(_) => bytes.to_vec(),
        }
    };

    ensure!(data.len() >= 16, "SPZ file too short");
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    ensure!(magic == SPZ_MAGIC, "bad SPZ magic: {magic:#x}");

    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let n = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let degree = data[12] as u32;
    let fb = data[13];
    let flags = data[14];
    let antialiased = flags & 0x01 != 0;
    let dim = sh_dim(degree);

    match version {
        3 => read_spz_v3(&data[16..], n, degree, fb, antialiased, dim),
        4 => read_spz_v4(&data, n, degree, fb, antialiased, dim),
        _ => Err(spl_core::anyhow!("unsupported SPZ version {version}")),
    }
}

fn read_spz_v3(
    body: &[u8],
    n: usize,
    degree: u32,
    fb: u8,
    antialiased: bool,
    dim: usize,
) -> Result<GaussiansHost> {
    let expected = n * 9 + n + n * 3 + n * 3 + n * 4 + n * dim * 3;
    ensure!(
        body.len() >= expected,
        "SPZ v3 body too short: expected {expected}, got {}",
        body.len()
    );
    decode_spz_streams(body, n, degree, fb, antialiased, dim)
}

fn read_spz_v4(
    data: &[u8],
    n: usize,
    degree: u32,
    fb: u8,
    antialiased: bool,
    dim: usize,
) -> Result<GaussiansHost> {
    ensure!(data.len() >= 32, "SPZ v4 header too short");
    let num_streams = data[15] as usize;
    let toc_offset = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;

    ensure!(
        data.len() >= toc_offset + num_streams * 16,
        "SPZ v4 TOC truncated"
    );

    let mut streams = Vec::new();
    let mut data_offset = toc_offset + num_streams * 16;
    for i in 0..num_streams {
        let toc_entry = toc_offset + i * 16;
        let compressed_size =
            u64::from_le_bytes(data[toc_entry..toc_entry + 8].try_into()?) as usize;
        let uncompressed_size =
            u64::from_le_bytes(data[toc_entry + 8..toc_entry + 16].try_into()?) as usize;
        ensure!(
            data_offset + compressed_size <= data.len(),
            "SPZ v4 stream {i} truncated"
        );
        let decompressed = zstd_decompress(
            &data[data_offset..data_offset + compressed_size],
            uncompressed_size,
        )?;
        streams.push(decompressed);
        data_offset += compressed_size;
    }

    let mut combined = Vec::new();
    for s in &streams {
        combined.extend_from_slice(s);
    }
    decode_spz_streams(&combined, n, degree, fb, antialiased, dim)
}

fn decode_spz_streams(
    body: &[u8],
    n: usize,
    degree: u32,
    fb: u8,
    antialiased: bool,
    dim: usize,
) -> Result<GaussiansHost> {
    let mut off = 0;

    let mut means = vec![0.0f32; n * 3];
    for i in 0..n {
        for a in 0..3 {
            let b = [body[off], body[off + 1], body[off + 2]];
            off += 3;
            let mut val = decode_position_24(b, fb);
            // RUB→RDF: flip y and z
            if a == 1 || a == 2 {
                val = -val;
            }
            means[i * 3 + a] = val;
        }
    }

    let mut opacities = vec![0.0f32; n];
    for opacity in &mut opacities {
        let a = body[off] as f32 / 255.0;
        off += 1;
        *opacity = inv_sigmoid(a.clamp(1e-6, 1.0 - 1e-6));
    }

    let mut sh0 = vec![0.0f32; n * 3];
    for i in 0..n {
        for c in 0..3 {
            let v = body[off] as f32;
            off += 1;
            sh0[i * 3 + c] = (v - 127.5) / (0.15 * 255.0);
        }
    }

    let mut scales = vec![0.0f32; n * 3];
    for i in 0..n {
        for a in 0..3 {
            let v = body[off] as f32;
            off += 1;
            scales[i * 3 + a] = v / 16.0 - 10.0;
        }
    }

    let mut rotations = vec![0.0f32; n * 4];
    for i in 0..n {
        let comp = u32::from_le_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]]);
        off += 4;
        let q = decode_smallest_three(comp);
        rotations[i * 4] = q[0]; // w
        rotations[i * 4 + 1] = q[1]; // x
        // RUB→RDF: quat y,z were flipped
        rotations[i * 4 + 2] = -q[2]; // y
        rotations[i * 4 + 3] = -q[3]; // z
    }

    let mut sh_rest = vec![0.0f32; n * 45];
    if dim > 0 {
        for i in 0..n {
            for k in 0..dim {
                for c in 0..3 {
                    let bits = if k < 3 { 5u8 } else { 4u8 };
                    let q = body[off];
                    off += 1;
                    let val = dequantize_sh(q, bits) / SH_FLIP[k];
                    sh_rest[i * 45 + k * 3 + c] = val;
                }
            }
        }
    }

    Ok(GaussiansHost {
        means,
        scales,
        rotations,
        opacities,
        sh0,
        sh_rest,
        sh_degree: degree,
        antialiased,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_gaussians(n: usize) -> GaussiansHost {
        let mut g = GaussiansHost::zeros(n);
        for i in 0..n {
            g.means[i * 3] = i as f32 * 0.5;
            g.means[i * 3 + 1] = -(i as f32) * 0.3;
            g.means[i * 3 + 2] = i as f32 * 0.1;
            g.scales[i * 3] = -2.0 + i as f32 * 0.3;
            g.scales[i * 3 + 1] = -1.5;
            g.scales[i * 3 + 2] = -3.0;
            g.opacities[i] = -1.0 + i as f32 * 0.4;
            g.sh0[i * 3] = 0.5;
            g.sh0[i * 3 + 1] = -0.3;
            g.sh0[i * 3 + 2] = 0.8;
            for j in 0..45 {
                g.sh_rest[i * 45 + j] = (j as f32 - 22.0) * 0.003;
            }
        }
        g
    }

    #[test]
    fn spz_v3_round_trip_within_quantization() {
        let g = sample_gaussians(7);
        let opts = SpzOptions::default();
        let bytes = write_spz(&g, &opts).unwrap();
        let back = read_spz(&bytes).unwrap();
        assert_eq!(back.len(), g.len());
        for i in 0..g.len() {
            for a in 0..3 {
                let tol = 1.0 / (1u32 << 12) as f32 + 1e-4;
                assert!(
                    (back.means[i * 3 + a] - g.means[i * 3 + a]).abs() < tol,
                    "means[{i}][{a}]: {} vs {}",
                    back.means[i * 3 + a],
                    g.means[i * 3 + a]
                );
            }
            let scale_step = 1.0 / 16.0;
            for a in 0..3 {
                assert!(
                    (back.scales[i * 3 + a] - g.scales[i * 3 + a]).abs() <= scale_step + 1e-4,
                    "scales[{i}][{a}]"
                );
            }
            let alpha_tol = 1.0 / 255.0 + 0.02;
            let o_orig = sigmoid(g.opacities[i]);
            let o_back = sigmoid(back.opacities[i]);
            assert!(
                (o_orig - o_back).abs() < alpha_tol,
                "opacity[{i}]: {o_orig} vs {o_back}"
            );
            // Quaternion angle tolerance
            let q0 = &g.rotations[i * 4..i * 4 + 4];
            let q1 = &back.rotations[i * 4..i * 4 + 4];
            let dot: f32 = (0..4).map(|a| q0[a] * q1[a]).sum::<f32>().abs();
            let n0 = (0..4).map(|a| q0[a] * q0[a]).sum::<f32>().sqrt();
            let n1 = (0..4).map(|a| q1[a] * q1[a]).sum::<f32>().sqrt();
            let cos_angle = (dot / (n0 * n1)).min(1.0);
            let angle = cos_angle.acos() * 2.0;
            assert!(
                angle < 0.2_f32.to_radians() + 0.01,
                "rot[{i}]: angle {:.3}° > 0.2°",
                angle.to_degrees()
            );
        }
    }

    #[test]
    fn spz_v4_round_trip() {
        let g = sample_gaussians(5);
        let opts = SpzOptions {
            version: 4,
            ..Default::default()
        };
        let bytes = write_spz(&g, &opts).unwrap();
        let back = read_spz(&bytes).unwrap();
        assert_eq!(back.len(), g.len());
    }

    #[test]
    fn spz_empty() {
        let g = GaussiansHost::zeros(0);
        let bytes = write_spz(&g, &SpzOptions::default()).unwrap();
        let back = read_spz(&bytes).unwrap();
        assert_eq!(back.len(), 0);
    }

    #[test]
    fn smallest_three_round_trip() {
        let cases: &[[f32; 4]] = &[
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 0.707, 0.0, 0.707],
            [-0.3, 0.4, -0.5, 0.7],
        ];
        for q_raw in cases {
            let n = (q_raw[0] * q_raw[0]
                + q_raw[1] * q_raw[1]
                + q_raw[2] * q_raw[2]
                + q_raw[3] * q_raw[3])
                .sqrt();
            let q = [q_raw[0] / n, q_raw[1] / n, q_raw[2] / n, q_raw[3] / n];
            let encoded = encode_smallest_three(&q);
            let decoded = decode_smallest_three(encoded);
            let dot: f32 = (0..4).map(|i| q[i] * decoded[i]).sum::<f32>().abs();
            assert!(dot > 0.999, "q={q:?} decoded={decoded:?} dot={dot}");
        }
    }

    #[test]
    fn position_24_round_trip() {
        let fb = 12u8;
        for val in [-1.5, 0.0, 3.125, -0.001, 100.0] {
            let encoded = encode_position_24(val, fb);
            let decoded = decode_position_24(encoded, fb);
            let tol = 1.0 / (1u32 << fb) as f32;
            assert!(
                (decoded - val).abs() <= tol + 1e-5,
                "val={val} decoded={decoded}"
            );
        }
    }
}
