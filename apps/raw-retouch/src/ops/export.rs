//! What the ops store: display-size JPEGs and the full-resolution files.

use std::sync::Arc;

use spl_core::{ArtifactKind, ArtifactRef, Context, Result, RunCx};
use tokio::sync::Semaphore;

use super::ExportOptions;
use super::cache::Photo;
use super::work::{Span, blocking, unless_cancelled};
use crate::encode;
use crate::finish::develop_full;
use crate::render::{RenderEngine, render_full};
use crate::types::{Gpu, Params, Rgb8Image};
use crate::xmp;

/// JPEG quality of the display-size previews.
const PREVIEW_QUALITY: u8 = 90;

/// Full-resolution exports at once in this process. One at 61 MP peaks
/// near 1.6 GB of GPU memory and holds about 1.6 GB on the host.
static FULL_RES: Semaphore = Semaphore::const_new(1);

/// A display-size render, upright, and its JPEG.
pub(super) struct Shown {
    pub upright: Rgb8Image,
    pub jpeg: Vec<u8>,
}

/// Renders `params` at display size, turns it upright and encodes the
/// preview JPEG. Every preview of a photo has the same size.
pub(super) fn display(
    engine: &mut RenderEngine,
    params: &Params,
    orientation: u8,
) -> Result<Shown> {
    let rendered = engine.display(params)?;
    let upright = encode::orient(&rendered, orientation);
    let jpeg = encode::jpeg(&upright, PREVIEW_QUALITY)?;
    Ok(Shown { upright, jpeg })
}

pub(super) async fn put_jpeg(cx: &RunCx, jpeg: Vec<u8>) -> Result<ArtifactRef> {
    cx.artifacts
        .put(jpeg, ArtifactKind::Image, "image/jpeg")
        .await
        .context("store a JPEG")
}

/// The export files of one edit.
#[derive(Debug, Default)]
pub(super) struct Files {
    pub jpeg: Option<ArtifactRef>,
    pub tiff: Option<ArtifactRef>,
    pub xmp: Option<ArtifactRef>,
    /// Upright full-resolution size, when anything was rendered at it.
    pub size: Option<(u32, u32)>,
}

/// Full-resolution JPEG and 16-bit TIFF (as `opts` asks) and the XMP
/// sidecar: demosaic with the lens corrections, render, sharpen and reduce
/// noise as the Detail settings say, turn upright, encode.
pub(super) async fn export(
    cx: &RunCx,
    gpu: &Gpu,
    photo: &Arc<Photo>,
    params: &Params,
    opts: &ExportOptions,
    span: Span,
) -> Result<Files> {
    let mut files = Files::default();
    if opts.jpeg || opts.tiff16 {
        let permit = match FULL_RES.try_acquire() {
            Ok(permit) => permit,
            Err(_) => {
                cx.progress(span.at(0.0), "Waiting for another full-resolution export");
                unless_cancelled(cx, async {
                    FULL_RES.acquire().await.context("export queue closed")
                })
                .await?
            }
        };
        let (gpu, photo, params, opts) = (gpu.clone(), photo.clone(), *params, opts.clone());
        let (jpeg, tiff, width, height) = blocking(cx, span, move |r| {
            // Held until the big buffers are gone.
            let _permit = permit;
            r.progress(0.0, "Exporting full resolution: demosaicing");
            let scene = photo.scene_for(&gpu, &params)?;
            let full = develop_full(&gpu, &photo.frame, photo.lens.as_ref(), &params)?;
            r.check()?;
            r.progress(0.35, "Exporting full resolution: rendering");
            let image = {
                let rendered = render_full(&gpu, &scene, &full, &params)?;
                drop(full);
                encode::orient16(&rendered, photo.scene.info.orientation)
            };
            r.check()?;
            let jpeg = if opts.jpeg {
                r.progress(0.6, "Exporting full resolution: encoding the JPEG");
                Some(encode::jpeg(&encode::to_rgb8(&image), opts.jpeg_quality)?)
            } else {
                None
            };
            let tiff = if opts.tiff16 {
                r.check()?;
                r.progress(0.8, "Exporting full resolution: encoding the TIFF");
                Some(encode::tiff16(&image)?)
            } else {
                None
            };
            Ok((jpeg, tiff, image.width as u32, image.height as u32))
        })
        .await?;
        files.size = Some((width, height));
        if let Some(jpeg) = jpeg {
            files.jpeg = Some(put_jpeg(cx, jpeg).await?);
        }
        if let Some(tiff) = tiff {
            files.tiff = Some(
                cx.artifacts
                    .put(tiff, ArtifactKind::Image, "image/tiff")
                    .await
                    .context("store the TIFF")?,
            );
        }
    }
    if opts.xmp {
        let sidecar = xmp::sidecar(params, &photo.scene.info);
        files.xmp = Some(
            cx.artifacts
                .put(
                    sidecar.into_bytes(),
                    ArtifactKind::Blob,
                    "application/rdf+xml",
                )
                .await
                .context("store the XMP sidecar")?,
        );
    }
    Ok(files)
}
