//! `image_to_3d.mesh`: a trained splat plus the cameras (and photos) of its
//! dataset → textured GLB. Owner: mesh-mode.

use anyhow::ensure;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use spl_core::{ArtifactKind, ArtifactRef, Op, Result, RunCx};
use ts_rs::TS;

use super::pipeline::{MeshConfig, MeshSummary, build};
use crate::export::ply::read_inria_ply;
use crate::export::spz::read_spz;
use crate::gpu::Gpu;
use crate::io::scene::{LoadOptions, load_colmap_scene};
use crate::ops::dataset_dir;

pub const MEDIA_GLB: &str = "model/gltf-binary";

#[derive(Debug, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct MeshInput {
    /// Artifact id of a trained splat (.ply or .spz).
    pub splat: String,
    /// The dataset the splat was trained on (folder under the dataset root):
    /// its cameras drive the fusion and its photos the texture. Give this or `zip`.
    #[serde(default)]
    #[ts(optional)]
    pub dataset: Option<String>,
    /// Artifact id of an uploaded COLMAP-posed dataset zip.
    #[serde(default)]
    #[ts(optional)]
    pub zip: Option<String>,
    /// Must match the training run (resolution, split): the train views are fused.
    #[serde(default)]
    pub load: LoadOptions,
    #[serde(default)]
    pub mesh: MeshConfig,
}

#[derive(Debug, Serialize, JsonSchema, TS)]
#[ts(export, export_to = "image-to-3d/")]
pub struct MeshOutput {
    /// Textured binary glTF.
    pub glb: ArtifactRef,
    pub summary: MeshSummary,
}

pub struct MeshOp;

impl Op for MeshOp {
    const ID: &'static str = "image_to_3d.mesh";
    const SUMMARY: &'static str = "Extract a textured mesh (GLB) from a trained splat: median depth, TSDF, marching cubes, decimation, UV, bake";

    type Input = MeshInput;
    type Output = MeshOutput;

    async fn run(&self, cx: &RunCx, input: MeshInput) -> Result<MeshOutput> {
        ensure!(
            input.mesh.resolution <= 8192 && input.mesh.atlas_size <= 16384,
            "resolution must be ≤ 8192 and atlas_size ≤ 16384"
        );
        let root = dataset_dir(cx, &input.dataset, &input.zip).await?;
        let bytes = cx.artifacts.get(&input.splat).await?;
        let (load, cfg) = (input.load.clone(), input.mesh.clone());
        cx.progress(0.0, "loading splat and cameras");
        let res = tokio::task::spawn_blocking(move || {
            let cloud = if bytes.starts_with(b"ply") {
                read_inria_ply(&bytes)?
            } else {
                read_spz(&bytes)?
            };
            let scene = load_colmap_scene(&root, &load)?;
            let views: Vec<_> = scene
                .train
                .iter()
                .map(|&i| scene.views[i].clone())
                .collect();
            let gpu = Gpu::open()?;
            build(&gpu, &cloud, &views, &cfg)
        })
        .await??;
        let s = &res.summary;
        cx.metric("mesh_time", s.total_s, "s");
        cx.metric("triangles", s.triangles as f64, "");
        cx.metric("views_fused", s.views_fused as f64, "");
        cx.log(format!(
            "{} views fused at voxel {:.5}: {} raw → {} triangles, {:.1} s",
            s.views_fused, s.voxel, s.raw_triangles, s.triangles, s.total_s
        ));
        let glb = cx
            .artifacts
            .put(res.glb, ArtifactKind::Mesh, MEDIA_GLB)
            .await?;
        Ok(MeshOutput {
            glb,
            summary: res.summary,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_input_parses() {
        let m: MeshInput =
            serde_json::from_str(r#"{"splat": "ab", "dataset": "dtu/scan24"}"#).unwrap();
        assert_eq!(m.mesh, MeshConfig::default());
    }
}
