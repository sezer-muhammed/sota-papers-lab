//! Media types the lab's apps exchange, and the file extensions that go
//! with them: hosts name downloads with these, `spl put` guesses with them.

pub const OCTET_STREAM: &str = "application/octet-stream";
pub const JSON: &str = "application/json";
pub const ZIP: &str = "application/zip";
/// Niantic's compressed splat format (MIT). Gzip inside: never serve it
/// with `Content-Encoding`, or browsers unpack it before the viewer does.
pub const SPZ: &str = "application/x-spz";
/// Stanford PLY, binary little endian: splats in the INRIA 3DGS layout
/// (SH degree 3), or meshes.
pub const PLY: &str = "application/ply";
/// glTF 2.0 binary (`.glb`): meshes with PBR textures.
pub const GLB: &str = "model/gltf-binary";
pub const GLTF: &str = "model/gltf+json";
pub const JPEG: &str = "image/jpeg";
pub const PNG: &str = "image/png";
pub const WEBP: &str = "image/webp";
pub const TIFF: &str = "image/tiff";
pub const HEIC: &str = "image/heic";
pub const MP4: &str = "video/mp4";
/// `.mov`, as iPhones record.
pub const QUICKTIME: &str = "video/quicktime";
pub const WEBM: &str = "video/webm";

/// `(extension, media type)`; the first extension listed for a media type
/// is the one downloads get.
const TYPES: &[(&str, &str)] = &[
    ("spz", SPZ),
    ("ply", PLY),
    ("glb", GLB),
    ("gltf", GLTF),
    ("json", JSON),
    ("zip", ZIP),
    ("jpg", JPEG),
    ("jpeg", JPEG),
    ("png", PNG),
    ("webp", WEBP),
    ("tif", TIFF),
    ("tiff", TIFF),
    ("heic", HEIC),
    ("arw", "image/x-sony-arw"),
    ("dng", "image/x-adobe-dng"),
    ("xmp", "application/rdf+xml"),
    ("mp4", MP4),
    ("m4v", MP4),
    ("mov", QUICKTIME),
    ("webm", WEBM),
    ("txt", "text/plain"),
    ("csv", "text/csv"),
    ("bin", OCTET_STREAM),
];

/// The media type of a file extension (without the dot, any case).
pub fn from_extension(extension: &str) -> Option<&'static str> {
    TYPES
        .iter()
        .find(|(ext, _)| ext.eq_ignore_ascii_case(extension))
        .map(|&(_, media_type)| media_type)
}

/// The usual file extension (without the dot) for a media type; parameters
/// such as `; charset=utf-8` and case are ignored.
pub fn extension(media_type: &str) -> Option<&'static str> {
    let essence = media_type.split(';').next().unwrap_or_default().trim();
    TYPES
        .iter()
        .find(|(_, known)| known.eq_ignore_ascii_case(essence))
        .map(|&(ext, _)| ext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_d_formats_have_their_extensions() {
        assert_eq!(extension(SPZ), Some("spz"));
        assert_eq!(extension(PLY), Some("ply"));
        assert_eq!(extension(GLB), Some("glb"));
        assert_eq!(extension("Application/JSON; charset=utf-8"), Some("json"));
        assert_eq!(extension(JPEG), Some("jpg"));
        assert_eq!(extension(QUICKTIME), Some("mov"));
        assert_eq!(extension("application/x-unknown"), None);
    }

    #[test]
    fn extensions_name_their_media_types() {
        assert_eq!(from_extension("SPZ"), Some(SPZ));
        assert_eq!(from_extension("glb"), Some(GLB));
        assert_eq!(from_extension("MOV"), Some(QUICKTIME));
        assert_eq!(from_extension("ARW"), Some("image/x-sony-arw"));
        assert_eq!(from_extension("exe"), None);
        for &(ext, media_type) in TYPES {
            assert_eq!(from_extension(ext), Some(media_type));
            assert!(extension(media_type).is_some(), "{media_type}");
        }
    }
}
