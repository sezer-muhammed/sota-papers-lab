//! GLB (glTF 2.0 binary) writer with PBR metallic-roughness material and
//! embedded texture. Owner: mesh-tex.
//!
//! Spec: <https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html>
//! The binary is one JSON chunk + one BIN chunk; the texture is a PNG or JPEG
//! embedded in the BIN chunk and referenced via a bufferView.

use std::io::Write;
use std::time::Instant;

use anyhow::ensure;
use spl_core::Result;

use super::{TextureAtlas, UvMesh};

/// Write a valid GLB file containing the mesh with a base-colour texture.
///
/// Returns (glb_bytes, time_ms).
pub fn write_glb(
    mesh: &UvMesh,
    texture: &TextureAtlas,
    texture_format: TextureFormat,
) -> Result<(Vec<u8>, f64)> {
    let started = Instant::now();

    let tex_bytes = encode_texture(texture, texture_format)?;

    let (bin_data, accessors_json, buffer_views_json) = build_bin_chunk(mesh, &tex_bytes)?;

    let json = build_json(
        &accessors_json,
        &buffer_views_json,
        texture_format,
        bin_data.len(),
    );
    let json_bytes = json.into_bytes();

    let json_padded_len = align4(json_bytes.len());
    let bin_padded_len = align4(bin_data.len());
    let total_len = 12 + 8 + json_padded_len + 8 + bin_padded_len;

    let mut out = Vec::with_capacity(total_len);

    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total_len as u32).to_le_bytes());

    out.extend_from_slice(&(json_padded_len as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
    out.extend_from_slice(&json_bytes);
    out.extend(std::iter::repeat_n(
        b' ',
        json_padded_len - json_bytes.len(),
    ));

    out.extend_from_slice(&(bin_padded_len as u32).to_le_bytes());
    out.extend_from_slice(&0x004E4942u32.to_le_bytes());
    out.extend_from_slice(&bin_data);
    out.extend(std::iter::repeat_n(0u8, bin_padded_len - bin_data.len()));

    let ms = started.elapsed().as_secs_f64() * 1e3;
    Ok((out, ms))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureFormat {
    Png,
    Jpeg,
}

fn align4(n: usize) -> usize {
    (n + 3) & !3
}

fn encode_texture(atlas: &TextureAtlas, format: TextureFormat) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    match format {
        TextureFormat::Png => {
            let encoder = image::codecs::png::PngEncoder::new(&mut buf);
            image::ImageEncoder::write_image(
                encoder,
                &atlas.pixels,
                atlas.width,
                atlas.height,
                image::ExtendedColorType::Rgba8,
            )
            .map_err(|e| spl_core::anyhow!("PNG encode: {e}"))?;
        }
        TextureFormat::Jpeg => {
            let rgb: Vec<u8> = atlas
                .pixels
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|px| [px[0], px[1], px[2]])
                .collect();
            let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 90);
            image::ImageEncoder::write_image(
                encoder,
                &rgb,
                atlas.width,
                atlas.height,
                image::ExtendedColorType::Rgb8,
            )
            .map_err(|e| spl_core::anyhow!("JPEG encode: {e}"))?;
        }
    }
    Ok(buf)
}

struct BufView {
    offset: usize,
    length: usize,
    target: Option<u32>,
}

fn build_bin_chunk(mesh: &UvMesh, tex_bytes: &[u8]) -> Result<(Vec<u8>, String, String)> {
    let mut bin = Vec::new();
    let mut views = Vec::new();

    let pos_bytes: Vec<u8> = mesh
        .positions
        .iter()
        .flat_map(|p| {
            let mut b = [0u8; 12];
            b[0..4].copy_from_slice(&p[0].to_le_bytes());
            b[4..8].copy_from_slice(&p[1].to_le_bytes());
            b[8..12].copy_from_slice(&p[2].to_le_bytes());
            b
        })
        .collect();
    let pos_offset = bin.len();
    bin.extend_from_slice(&pos_bytes);
    pad_to_4(&mut bin);
    views.push(BufView {
        offset: pos_offset,
        length: pos_bytes.len(),
        target: Some(34962),
    });

    let norm_bytes: Vec<u8> = mesh
        .normals
        .iter()
        .flat_map(|n| {
            let mut b = [0u8; 12];
            b[0..4].copy_from_slice(&n[0].to_le_bytes());
            b[4..8].copy_from_slice(&n[1].to_le_bytes());
            b[8..12].copy_from_slice(&n[2].to_le_bytes());
            b
        })
        .collect();
    let norm_offset = bin.len();
    bin.extend_from_slice(&norm_bytes);
    pad_to_4(&mut bin);
    views.push(BufView {
        offset: norm_offset,
        length: norm_bytes.len(),
        target: Some(34962),
    });

    let uv_bytes: Vec<u8> = mesh
        .uvs
        .iter()
        .flat_map(|uv| {
            let mut b = [0u8; 8];
            b[0..4].copy_from_slice(&uv[0].to_le_bytes());
            b[4..8].copy_from_slice(&uv[1].to_le_bytes());
            b
        })
        .collect();
    let uv_offset = bin.len();
    bin.extend_from_slice(&uv_bytes);
    pad_to_4(&mut bin);
    views.push(BufView {
        offset: uv_offset,
        length: uv_bytes.len(),
        target: Some(34962),
    });

    let idx_bytes: Vec<u8> = mesh.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    let idx_offset = bin.len();
    bin.extend_from_slice(&idx_bytes);
    pad_to_4(&mut bin);
    views.push(BufView {
        offset: idx_offset,
        length: idx_bytes.len(),
        target: Some(34963),
    });

    let tex_offset = bin.len();
    bin.extend_from_slice(tex_bytes);
    pad_to_4(&mut bin);
    views.push(BufView {
        offset: tex_offset,
        length: tex_bytes.len(),
        target: None,
    });

    let views_json: Vec<String> = views
        .iter()
        .map(|v| {
            let target = match v.target {
                Some(t) => format!(",\"target\":{t}"),
                None => String::new(),
            };
            format!(
                "{{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{}{}}}",
                v.offset, v.length, target
            )
        })
        .collect();
    let views_str = views_json.join(",");

    let (pos_min, pos_max) = compute_bounds(&mesh.positions);
    let accessors = format!(
        concat!(
            "{{\"bufferView\":0,\"componentType\":5126,\"count\":{},\"type\":\"VEC3\",",
            "\"min\":[{},{},{}],\"max\":[{},{},{}]}},",
            "{{\"bufferView\":1,\"componentType\":5126,\"count\":{},\"type\":\"VEC3\"}},",
            "{{\"bufferView\":2,\"componentType\":5126,\"count\":{},\"type\":\"VEC2\"}},",
            "{{\"bufferView\":3,\"componentType\":5125,\"count\":{},\"type\":\"SCALAR\"}}"
        ),
        mesh.positions.len(),
        pos_min[0],
        pos_min[1],
        pos_min[2],
        pos_max[0],
        pos_max[1],
        pos_max[2],
        mesh.normals.len(),
        mesh.uvs.len(),
        mesh.indices.len()
    );

    Ok((bin, accessors, views_str))
}

fn compute_bounds(positions: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in positions {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }
    (min, max)
}

fn build_json(
    accessors: &str,
    buffer_views: &str,
    format: TextureFormat,
    buffer_byte_length: usize,
) -> String {
    let mime = match format {
        TextureFormat::Png => "image/png",
        TextureFormat::Jpeg => "image/jpeg",
    };

    format!(
        concat!(
            "{{",
            "\"asset\":{{\"version\":\"2.0\",\"generator\":\"spl-image-to-3d\"}},",
            "\"scene\":0,",
            "\"scenes\":[{{\"nodes\":[0]}}],",
            "\"nodes\":[{{\"mesh\":0}}],",
            "\"meshes\":[{{\"primitives\":[{{",
            "\"attributes\":{{\"POSITION\":0,\"NORMAL\":1,\"TEXCOORD_0\":2}},",
            "\"indices\":3,",
            "\"material\":0",
            "}}]}}],",
            "\"materials\":[{{",
            "\"pbrMetallicRoughness\":{{",
            "\"baseColorTexture\":{{\"index\":0}},",
            "\"metallicFactor\":0.0,",
            "\"roughnessFactor\":1.0",
            "}},",
            "\"doubleSided\":true",
            "}}],",
            "\"textures\":[{{\"source\":0}}],",
            "\"images\":[{{\"bufferView\":4,\"mimeType\":\"{}\"}}],",
            "\"accessors\":[{}],",
            "\"bufferViews\":[{}],",
            "\"buffers\":[{{\"byteLength\":{}}}]",
            "}}"
        ),
        mime, accessors, buffer_views, buffer_byte_length,
    )
}

fn pad_to_4(buf: &mut Vec<u8>) {
    let rem = buf.len() % 4;
    if rem != 0 {
        buf.extend(std::iter::repeat_n(0u8, 4 - rem));
    }
}

/// Validate GLB magic and structure (basic sanity check).
pub fn validate_glb(data: &[u8]) -> Result<GlbInfo> {
    ensure!(data.len() >= 12, "GLB too short");
    ensure!(&data[0..4] == b"glTF", "not a GLB (bad magic)");
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    ensure!(version == 2, "unsupported glTF version {version}");
    let total = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    ensure!(
        total <= data.len(),
        "GLB header claims {total} bytes but file is {}",
        data.len()
    );

    ensure!(data.len() >= 20, "GLB missing JSON chunk header");
    let json_len = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    let json_type = u32::from_le_bytes(data[16..20].try_into().unwrap());
    ensure!(json_type == 0x4E4F534A, "first chunk is not JSON");

    let json_str = std::str::from_utf8(&data[20..20 + json_len])
        .map_err(|e| spl_core::anyhow!("GLB JSON is not UTF-8: {e}"))?;
    let json: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| spl_core::anyhow!("GLB JSON parse error: {e}"))?;

    let asset = json.get("asset").and_then(|a| a.get("version"));
    ensure!(
        asset.is_some_and(|v| v.as_str() == Some("2.0")),
        "missing or wrong asset.version"
    );

    let bin_start = 20 + align4(json_len);
    let has_bin = if bin_start + 8 <= data.len() {
        let bin_type = u32::from_le_bytes(data[bin_start + 4..bin_start + 8].try_into().unwrap());
        bin_type == 0x004E4942
    } else {
        false
    };

    let meshes = json
        .get("meshes")
        .and_then(|m| m.as_array())
        .map(|a| a.len())
        .unwrap_or(0);

    Ok(GlbInfo {
        version,
        json_bytes: json_len,
        has_bin_chunk: has_bin,
        mesh_count: meshes,
        total_bytes: total,
    })
}

#[derive(Clone, Debug)]
pub struct GlbInfo {
    pub version: u32,
    pub json_bytes: usize,
    pub has_bin_chunk: bool,
    pub mesh_count: usize,
    pub total_bytes: usize,
}

/// Write a GLB to a file (convenience).
pub fn write_glb_file(
    path: &std::path::Path,
    mesh: &UvMesh,
    texture: &TextureAtlas,
    format: TextureFormat,
) -> Result<(usize, f64)> {
    let (bytes, ms) = write_glb(mesh, texture, format)?;
    let len = bytes.len();
    let mut f = std::fs::File::create(path)
        .map_err(|e| spl_core::anyhow!("create {}: {e}", path.display()))?;
    f.write_all(&bytes)
        .map_err(|e| spl_core::anyhow!("write {}: {e}", path.display()))?;
    Ok((len, ms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::tests::cube_mesh;
    use crate::mesh::uv::generate_atlas;

    fn make_test_atlas(w: u32, h: u32) -> TextureAtlas {
        let mut atlas = TextureAtlas::new(w, h);
        for px in atlas.pixels.as_chunks_mut::<4>().0 {
            px[0] = 128;
            px[1] = 64;
            px[2] = 32;
            px[3] = 255;
        }
        atlas
    }

    #[test]
    fn glb_roundtrip_validates() {
        let mesh = cube_mesh();
        let (uv_mesh, _) = generate_atlas(&mesh, 64, 1).unwrap();
        let atlas = make_test_atlas(uv_mesh.atlas_width, uv_mesh.atlas_height);
        let (glb, ms) = write_glb(&uv_mesh, &atlas, TextureFormat::Png).unwrap();

        assert!(ms >= 0.0);
        let info = validate_glb(&glb).unwrap();
        assert_eq!(info.version, 2);
        assert!(info.has_bin_chunk);
        assert_eq!(info.mesh_count, 1);
        assert!(info.total_bytes > 0);
    }

    #[test]
    fn glb_jpeg_validates() {
        let mesh = cube_mesh();
        let (uv_mesh, _) = generate_atlas(&mesh, 64, 1).unwrap();
        let atlas = make_test_atlas(uv_mesh.atlas_width, uv_mesh.atlas_height);
        let (glb, _) = write_glb(&uv_mesh, &atlas, TextureFormat::Jpeg).unwrap();
        let info = validate_glb(&glb).unwrap();
        assert_eq!(info.version, 2);
        assert!(info.has_bin_chunk);
    }

    #[test]
    fn glb_magic_check() {
        assert!(validate_glb(b"NOT_GLB").is_err());
        assert!(validate_glb(b"").is_err());
    }

    #[test]
    fn glb_contains_pbr_material() {
        let mesh = cube_mesh();
        let (uv_mesh, _) = generate_atlas(&mesh, 64, 1).unwrap();
        let atlas = make_test_atlas(uv_mesh.atlas_width, uv_mesh.atlas_height);
        let (glb, _) = write_glb(&uv_mesh, &atlas, TextureFormat::Png).unwrap();

        let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let json_str = std::str::from_utf8(&glb[20..20 + json_len]).unwrap();
        assert!(json_str.contains("pbrMetallicRoughness"));
        assert!(json_str.contains("\"metallicFactor\":0"));
        assert!(json_str.contains("baseColorTexture"));
        assert!(json_str.contains("doubleSided"));
    }

    #[test]
    fn glb_file_write_and_validate() {
        let mesh = cube_mesh();
        let (uv_mesh, _) = generate_atlas(&mesh, 64, 1).unwrap();
        let atlas = make_test_atlas(uv_mesh.atlas_width, uv_mesh.atlas_height);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.glb");
        let (size, _ms) = write_glb_file(&path, &uv_mesh, &atlas, TextureFormat::Png).unwrap();
        assert!(size > 0);

        let data = std::fs::read(&path).unwrap();
        let info = validate_glb(&data).unwrap();
        assert_eq!(info.version, 2);
    }
}
