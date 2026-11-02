//! DA3-BASE (ByteDance-Seed Depth Anything 3, Apache-2.0 code and weights)
//! on ONNX Runtime's CUDA execution provider: the commercial profile's
//! Stage A engine (`reports/image-to-3d/spec-commercial.md` §3-5).
//!
//! Model directory (`$SPL_DA3_DIR`, else `<data dir>/models/da3-base`),
//! written by `tools/onnx/export_da3.py`:
//! `manifest.json` (licence, source hash, resolution buckets), one
//! `backbone_<W>x<H>.onnx` and `head_<W>x<H>.onnx` per bucket, and the fp16
//! weights they share, `da3_base.weights` (206 MB).
//!
//! Inference: frames are centre-cropped to the bucket with the nearest
//! aspect ratio (longest side 504, both sides multiples of 14) and resized
//! with the same PIL-exact bicubic as the VGGT-Ω path. The backbone graph
//! (DINOv2 ViT-B with alternating frame / global attention and the camera
//! decoder) runs once over all N frames with its four feature levels bound
//! to device memory (IoBinding); the DPT depth head then reads them in place,
//! 8 frames per call, like the official code. Pose encodings, depth and
//! confidence come back to the host, where cameras and world points are
//! formed ([`super::cameras_from_c2w_pose_enc`], [`super::unproject`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::ensure;
use rayon::prelude::*;
use serde::Deserialize;
use spl_core::{CancellationToken, Context, Result, anyhow, bail};
use spl_ort::f16;
use spl_ort::ort::session::Session;
use spl_ort::ort::value::{DynTensorValueType, Tensor};

use super::{
    Engine, EngineInfo, EngineOut, EngineTimes, LicenceProfile, cameras_from_c2w_pose_enc,
    unproject,
};
use crate::io::images::Rgb8Image;
use crate::vggt::preprocess::{aspect_crop, resize_bicubic};
use crate::vggt::{DenseHost, Geometry, Prepared, VggtCameras};

/// Frames per head call (the official `chunk_size`).
pub const HEAD_CHUNK: usize = 8;
/// The backbone graph selects a reference view, which needs 3 frames;
/// shorter inputs are padded by repeating the last frame.
pub const MIN_FRAMES: usize = 3;
/// DINOv2 patch size.
pub const PATCH: usize = 14;
const FEATURES: usize = 1536;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone, Deserialize)]
pub struct Bucket {
    pub width: usize,
    pub height: usize,
    pub backbone: String,
    pub head: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub model: String,
    pub licence: String,
    pub source_sha256: String,
    pub precision: String,
    pub buckets: Vec<Bucket>,
}

impl Manifest {
    pub fn read(dir: &Path) -> Result<Self> {
        let p = dir.join("manifest.json");
        let m: Self = serde_json::from_slice(
            &std::fs::read(&p).with_context(|| format!("read {}", p.display()))?,
        )
        .with_context(|| format!("parse {}", p.display()))?;
        ensure!(
            !m.buckets.is_empty(),
            "{}: no resolution buckets",
            p.display()
        );
        ensure!(
            m.precision == "fp16",
            "{}: expected fp16 graphs",
            p.display()
        );
        Ok(m)
    }

    /// The bucket whose aspect ratio (`h / w`) is nearest to `aspect`
    /// (in log scale, so 2:1 and 1:2 are equally far from 1:1).
    pub fn bucket(&self, aspect: f64) -> &Bucket {
        let d = |b: &Bucket| ((b.height as f64 / b.width as f64) / aspect).ln().abs();
        self.buckets
            .iter()
            .min_by(|a, b| d(a).total_cmp(&d(b)))
            .expect("manifest has buckets")
    }
}

/// Centre crop `(left, top, width, height)` of a `w × h` image to aspect
/// `a = h / w`.
pub fn crop_to_aspect(w: usize, h: usize, a: f64) -> (usize, usize, usize, usize) {
    let cur = h as f64 / w as f64;
    if (cur - a).abs() < 1e-9 {
        (0, 0, w, h)
    } else if cur > a {
        let ch = ((w as f64 * a).round() as usize).clamp(1, h);
        (0, (h - ch) / 2, w, ch)
    } else {
        let cw = ((h as f64 / a).round() as usize).clamp(1, w);
        ((w - cw) / 2, 0, cw, h)
    }
}

fn crop(f: &Rgb8Image, (x0, y0, w, h): (usize, usize, usize, usize)) -> Vec<u8> {
    let fw = f.width as usize;
    let mut out = Vec::with_capacity(w * h * 3);
    for y in y0..y0 + h {
        out.extend_from_slice(&f.data[(y * fw + x0) * 3..(y * fw + x0 + w) * 3]);
    }
    out
}

/// Frames at the bucket resolution: the first frame's aspect (brought into
/// [0.5, 2] like the VGGT-Ω path) picks the bucket; every frame is
/// centre-cropped to the bucket's aspect and resized.
pub fn prepare(frames: &[Rgb8Image], m: &Manifest) -> Result<Prepared> {
    ensure!(!frames.is_empty(), "no frames");
    let f0 = &frames[0];
    let b0 = aspect_crop(f0.width as usize, f0.height as usize);
    let bucket = m.bucket(b0.3 as f64 / b0.2.max(1) as f64);
    let (bw, bh) = (bucket.width, bucket.height);
    let a = bh as f64 / bw as f64;
    let per: Vec<Vec<u8>> = frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| -> Result<Vec<u8>> {
            let (w, h) = (f.width as usize, f.height as usize);
            ensure!(
                w > 0 && h > 0 && f.data.len() == w * h * 3,
                "frame {i}: bad image"
            );
            let bx = crop_to_aspect(w, h, a);
            Ok(resize_bicubic(&crop(f, bx), bx.2, bx.3, bw, bh))
        })
        .collect::<Result<_>>()?;
    Ok(Prepared {
        geom: Geometry::new(frames.len(), bh, bw),
        pixels: per.concat(),
    })
}

/// RGB8 `[N][H][W][3]` → ImageNet-normalised fp16 `[N][3][H][W]`, padded to
/// at least `min_frames` frames by repeating the last one.
pub fn normalise(p: &Prepared, min_frames: usize) -> (usize, Vec<f16>) {
    let g = p.geom;
    let hw = g.height * g.width;
    let n = g.frames.max(min_frames);
    let mut out = vec![f16::ZERO; n * 3 * hw];
    out.par_chunks_mut(3 * hw).enumerate().for_each(|(i, dst)| {
        let src = &p.pixels[i.min(g.frames - 1) * hw * 3..][..hw * 3];
        for c in 0..3 {
            let (m, s) = (MEAN[c], STD[c]);
            for (d, px) in dst[c * hw..(c + 1) * hw]
                .iter_mut()
                .zip(src.as_chunks::<3>().0)
            {
                *d = f16::from_f32((px[c] as f32 / 255.0 - m) / s);
            }
        }
    });
    (n, out)
}

/// Pose encodings `[N][9]` (camera-to-world), depth and confidence
/// `[N][H][W]`.
pub type Forward = (Vec<[f32; 9]>, Vec<f32>, Vec<f32>);

struct Graphs {
    backbone: Session,
    head: Session,
}

pub struct Da3Engine {
    pub dir: PathBuf,
    pub manifest: Manifest,
    pub device: i32,
    sessions: HashMap<(usize, usize), Graphs>,
    pub times: EngineTimes,
    /// Seconds the last call spent creating sessions (first use of a
    /// bucket), not counted in `times`.
    pub session_s: f64,
}

impl Da3Engine {
    pub fn default_dir(models: &Path) -> PathBuf {
        models.join("da3-base")
    }

    /// Reads the manifest and checks the licence against `profile`; graphs
    /// load lazily per bucket.
    pub fn load(dir: &Path, profile: LicenceProfile) -> Result<Self> {
        let manifest = Manifest::read(dir)?;
        profile.allow(&manifest.model, &manifest.licence)?;
        spl_ort::environment()?;
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
            device: 0,
            sessions: HashMap::new(),
            times: EngineTimes::default(),
            session_s: 0.0,
        })
    }

    pub fn preprocess(&self, frames: &[Rgb8Image]) -> Result<Prepared> {
        prepare(frames, &self.manifest)
    }

    fn graphs(&mut self, w: usize, h: usize) -> Result<&mut Graphs> {
        if !self.sessions.contains_key(&(w, h)) {
            let b = self
                .manifest
                .buckets
                .iter()
                .find(|b| (b.width, b.height) == (w, h))
                .ok_or_else(|| anyhow!("no DA3 graph for {w}x{h}"))?;
            let opts = spl_ort::CudaOptions {
                device: self.device,
                ..Default::default()
            };
            let g = Graphs {
                backbone: spl_ort::cuda_session(&self.dir.join(&b.backbone), &opts)?,
                head: spl_ort::cuda_session(&self.dir.join(&b.head), &opts)?,
            };
            self.sessions.insert((w, h), g);
        }
        Ok(self.sessions.get_mut(&(w, h)).expect("inserted"))
    }

    /// Frees every session (and the CUDA EP's memory arena with it).
    pub fn unload_graphs(&mut self) {
        self.sessions.clear();
    }

    /// Forward pass: pose encodings `[N][9]` (camera-to-world), depth and
    /// confidence `[N][H][W]`.
    pub fn forward(
        &mut self,
        p: &Prepared,
        cancel: &CancellationToken,
        progress: &mut dyn FnMut(&str, f32),
    ) -> Result<Forward> {
        let g = p.geom;
        ensure!(
            g.width.is_multiple_of(PATCH) && g.height.is_multiple_of(PATCH),
            "frames must be multiples of {PATCH} pixels"
        );
        let (h, w) = (g.height, g.width);
        let device = self.device;
        let t_load = Instant::now();
        self.graphs(w, h)?;
        self.session_s = t_load.elapsed().as_secs_f64();
        let started = Instant::now();
        let (n, x) = normalise(p, MIN_FRAMES);
        let graphs = self.graphs(w, h)?;
        let t_upload = Instant::now();
        progress("backbone", 0.0);

        let images = Tensor::<f16>::from_array(([n, 3, h, w], x)).map_err(|e| anyhow!("{e}"))?;
        let mut binding = graphs
            .backbone
            .create_binding()
            .map_err(|e| anyhow!("{e}"))?;
        binding
            .bind_input("images", &images)
            .map_err(|e| anyhow!("{e}"))?;
        let gpu = spl_ort::cuda_memory(device)?;
        for i in 0..4 {
            binding
                .bind_output_to_device(format!("f{i}"), &gpu)
                .map_err(|e| anyhow!("{e}"))?;
        }
        binding
            .bind_output_to_device("pose_enc", &spl_ort::cpu_memory()?)
            .map_err(|e| anyhow!("{e}"))?;
        let outs = graphs
            .backbone
            .run_binding(&binding)
            .map_err(|e| anyhow!("DA3 backbone: {e}"))?;
        let trunk_ms = t_upload.elapsed().as_secs_f32() * 1e3;
        if cancel.is_cancelled() {
            bail!("cancelled");
        }
        let (_, pe) = outs["pose_enc"]
            .try_extract_tensor::<f16>()
            .map_err(|e| anyhow!("pose_enc: {e}"))?;
        let pose_enc: Vec<[f32; 9]> = pe
            .as_chunks::<9>()
            .0
            .iter()
            .take(g.frames)
            .map(|c| std::array::from_fn(|k| c[k].to_f32()))
            .collect();
        let mut levels = Vec::with_capacity(4);
        for i in 0..4 {
            let t = outs[format!("f{i}").as_str()]
                .view()
                .downcast::<DynTensorValueType>()
                .map_err(|e| anyhow!("f{i}: {e}"))?;
            levels.push(t.data_ptr());
        }

        let t_head = Instant::now();
        let (ph, pw) = (h / PATCH, w / PATCH);
        let per_frame = ph * pw * FEATURES;
        let mut depth = Vec::with_capacity(g.frames * h * w);
        let mut conf = Vec::with_capacity(g.frames * h * w);
        for s0 in (0..g.frames).step_by(HEAD_CHUNK) {
            if cancel.is_cancelled() {
                bail!("cancelled");
            }
            let c = HEAD_CHUNK.min(g.frames - s0);
            let shape = [c as i64, ph as i64, pw as i64, FEATURES as i64];
            // SAFETY: each level is a device tensor [n, ph, pw, 1536] f16 owned
            // by `outs`, which outlives these views; s0 + c <= n.
            let views = levels
                .iter()
                .map(|&base| unsafe {
                    spl_ort::device_view::<f16>(device, base, s0 * per_frame, &shape)
                })
                .collect::<Result<Vec<_>>>()?;
            let mut it = views.into_iter();
            let (v0, v1, v2, v3) = (
                it.next().expect("4 levels"),
                it.next().expect("4 levels"),
                it.next().expect("4 levels"),
                it.next().expect("4 levels"),
            );
            let o = graphs
                .head
                .run(spl_ort::ort::inputs!["f0" => v0, "f1" => v1, "f2" => v2, "f3" => v3])
                .map_err(|e| anyhow!("DA3 head: {e}"))?;
            for (name, dst) in [("depth", &mut depth), ("depth_conf", &mut conf)] {
                let (_, v) = o[name]
                    .try_extract_tensor::<f16>()
                    .map_err(|e| anyhow!("{name}: {e}"))?;
                dst.extend(v.iter().map(|x| x.to_f32()));
            }
            progress("depth head", (s0 + c) as f32 / g.frames as f32);
        }
        drop(outs);
        self.times = EngineTimes {
            preprocess_ms: 0.0,
            trunk_ms,
            dense_ms: t_head.elapsed().as_secs_f32() * 1e3,
            forward_ms: started.elapsed().as_secs_f32() * 1e3,
        };
        Ok((pose_enc, depth, conf))
    }

    /// Cameras, depth, confidence and world points of prepared frames.
    pub fn run(
        &mut self,
        p: &Prepared,
        cancel: &CancellationToken,
        progress: &mut dyn FnMut(&str, f32),
    ) -> Result<(VggtCameras, DenseHost)> {
        let (pose_enc, depth, conf) = self.forward(p, cancel, progress)?;
        let cameras = cameras_from_c2w_pose_enc(&pose_enc, p.geom.width, p.geom.height);
        let world = unproject(&cameras, &depth);
        Ok((cameras, DenseHost { depth, conf, world }))
    }
}

impl Engine for Da3Engine {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            engine: "da3-base".into(),
            licence: self.manifest.licence.clone(),
            weights_sha256: Some(self.manifest.source_sha256.clone()),
        }
    }

    fn infer(
        &mut self,
        frames: &[Rgb8Image],
        cancel: &CancellationToken,
        progress: &mut dyn FnMut(&str, f32),
    ) -> Result<EngineOut> {
        let t = Instant::now();
        let prepared = self.preprocess(frames)?;
        let preprocess_ms = t.elapsed().as_secs_f32() * 1e3;
        let (cameras, dense) = self.run(&prepared, cancel, progress)?;
        let mut times = self.times.clone();
        times.preprocess_ms = preprocess_ms;
        Ok(EngineOut {
            prepared,
            cameras,
            dense,
            times,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        let b = |w, h| Bucket {
            width: w,
            height: h,
            backbone: String::new(),
            head: String::new(),
        };
        Manifest {
            model: "depth-anything/DA3-BASE".into(),
            licence: "apache-2.0".into(),
            source_sha256: String::new(),
            precision: "fp16".into(),
            buckets: vec![
                b(504, 336),
                b(504, 252),
                b(504, 378),
                b(336, 504),
                b(504, 504),
            ],
        }
    }

    #[test]
    fn bucket_follows_the_aspect_ratio() {
        let m = manifest();
        let pick = |w: f64, h: f64| {
            let b = m.bucket(h / w);
            (b.width, b.height)
        };
        assert_eq!(pick(6208.0, 4135.0), (504, 336)); // ETH3D DSLR
        assert_eq!(pick(720.0, 360.0), (504, 252)); // Sintel after the 2:1 crop
        assert_eq!(pick(4032.0, 3024.0), (504, 378)); // 4:3 phone photo
        assert_eq!(pick(3024.0, 4032.0), (336, 504)); // portrait: nearest is 2:3
        assert_eq!(pick(1000.0, 1000.0), (504, 504));
    }

    #[test]
    fn every_bucket_is_a_multiple_of_the_patch() {
        for b in manifest().buckets {
            assert!(b.width % PATCH == 0 && b.height % PATCH == 0);
        }
    }

    #[test]
    fn prepare_crops_and_resizes_to_the_bucket() {
        let m = manifest();
        let f = Rgb8Image {
            width: 846,
            height: 360,
            data: (0..846 * 360 * 3).map(|i| (i % 251) as u8).collect(),
        };
        let p = prepare(&[f.clone(), f], &m).unwrap();
        assert_eq!((p.geom.frames, p.geom.width, p.geom.height), (2, 504, 252));
        assert_eq!(p.pixels.len(), 2 * 504 * 252 * 3);
    }

    #[test]
    fn normalise_pads_short_inputs_and_uses_imagenet_stats() {
        let p = Prepared {
            geom: Geometry::new(1, 2, 2),
            pixels: vec![255; 12],
        };
        let (n, x) = normalise(&p, MIN_FRAMES);
        assert_eq!((n, x.len()), (3, 3 * 3 * 4));
        let want = (1.0 - MEAN[2]) / STD[2];
        assert!((x[2 * 4].to_f32() - want).abs() < 1e-2);
        assert_eq!(x[..12], x[24..]);
    }

    #[test]
    fn manifest_needs_a_permissive_licence_in_the_commercial_profile() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = serde_json::json!({
            "model": "depth-anything/DA3-LARGE", "licence": "cc-by-nc-4.0", "source_sha256": "",
            "precision": "fp16", "buckets": [{"width": 504, "height": 336, "backbone": "b", "head": "h"}]
        });
        std::fs::write(dir.path().join("manifest.json"), m.to_string()).unwrap();
        let err = Da3Engine::load(dir.path(), LicenceProfile::Commercial)
            .err()
            .unwrap();
        assert!(err.to_string().contains("does not allow"), "{err}");
        m["licence"] = "apache-2.0".into();
        std::fs::write(dir.path().join("manifest.json"), m.to_string()).unwrap();
        // Licence passes; on a machine without ONNX Runtime loading then fails
        // on the library, not on the licence.
        if let Err(e) = Da3Engine::load(dir.path(), LicenceProfile::Commercial) {
            assert!(!e.to_string().contains("does not allow"), "{e}");
        }
    }
}
