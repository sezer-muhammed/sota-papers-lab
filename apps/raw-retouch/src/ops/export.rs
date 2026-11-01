//! What the ops store: display-size JPEGs and the full-resolution files.

use std::sync::Arc;

use spl_core::{ArtifactKind, ArtifactRef, Context, Result, RunCx, bail};
use tokio::sync::{Semaphore, SemaphorePermit};

use super::ExportOptions;
use super::cache::Photo;
use super::region;
use super::work::{Span, blocking, unless_cancelled};
use crate::encode;
use crate::render::{RenderEngine, render_full};
use crate::types::{Gpu, Params, Rgb8Image};
use crate::xmp;

/// JPEG quality of the display-size previews.
const PREVIEW_QUALITY: u8 = 90;

/// Full-resolution renders at once in this process (exports and zoomed
/// regions). One at 61 MP peaks near 1.6 GB of GPU memory and holds about
/// 1.6 GB on the host.
static FULL_RES: Semaphore = Semaphore::const_new(1);

/// The right to render at full resolution, waiting (cancellably) while
/// another run does.
pub(super) async fn full_res_permit(cx: &RunCx, span: Span) -> Result<SemaphorePermit<'static>> {
    match FULL_RES.try_acquire() {
        Ok(permit) => Ok(permit),
        Err(_) => {
            cx.progress(span.at(0.0), "Waiting for another full-resolution render");
            unless_cancelled(cx, async {
                FULL_RES.acquire().await.context("export queue closed")
            })
            .await
        }
    }
}

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
    pub png: Option<ArtifactRef>,
    pub xmp: Option<ArtifactRef>,
    /// Upright full-resolution size, when anything was rendered at it.
    pub size: Option<(u32, u32)>,
}

impl ExportOptions {
    /// Whether any full-resolution image file is asked for.
    pub(super) fn full_res(&self) -> bool {
        self.jpeg || self.tiff16 || self.png
    }

    /// The options as given, or why they cannot be met.
    pub(super) fn check(&self) -> Result<()> {
        if self.png && !matches!(self.png_bits, 8 | 16) {
            bail!("png_bits must be 8 or 16, not {}", self.png_bits);
        }
        Ok(())
    }
}

/// Encoded full-resolution files, before they are stored.
struct Encoded {
    jpeg: Option<Vec<u8>>,
    tiff: Option<Vec<u8>>,
    png: Option<Vec<u8>>,
    /// The 8-bit image the JPEG (and an 8-bit PNG) hold, kept for zoomed views.
    rgb8: Option<Rgb8Image>,
    width: u32,
    height: u32,
}

/// Full-resolution JPEG, 16-bit TIFF and PNG (as `opts` asks) and the XMP
/// sidecar: demosaic with the lens corrections (or take that frame from
/// the zoom cache), render, sharpen and reduce noise as the Detail settings
/// say, turn upright, encode. The 8-bit image and the demosaiced frame stay
/// cached, so zooming into the result right after costs no render.
pub(super) async fn export(
    cx: &RunCx,
    gpu: &Gpu,
    photo: &Arc<Photo>,
    params: &Params,
    opts: &ExportOptions,
    span: Span,
) -> Result<Files> {
    opts.check()?;
    let mut files = Files::default();
    if opts.full_res() {
        let permit = full_res_permit(cx, span).await?;
        let full = region::frame(cx, gpu, photo, params, span.part(0.0, 0.3)).await?;
        let (gpu, owned, params, opts) = (gpu.clone(), photo.clone(), *params, opts.clone());
        let encoded = blocking(cx, span.part(0.3, 1.0), move |r| {
            // Held until the big buffers are gone.
            let _permit = permit;
            let photo = owned;
            r.progress(0.0, "Exporting full resolution: rendering");
            let scene = photo.scene_for(&gpu, &params)?;
            let image = {
                let rendered = render_full(&gpu, &scene, &full, &params)?;
                drop(full);
                encode::orient16(&rendered, photo.scene.info.orientation)
            };
            r.check()?;
            let png16 = opts.png && opts.png_bits == 16;
            let rgb8 = (opts.jpeg || (opts.png && !png16)).then(|| encode::to_rgb8(&image));
            let jpeg = match (&rgb8, opts.jpeg) {
                (Some(rgb8), true) => {
                    r.progress(0.3, "Exporting full resolution: encoding the JPEG");
                    Some(encode::jpeg(rgb8, opts.jpeg_quality)?)
                }
                _ => None,
            };
            let png = if opts.png {
                r.check()?;
                r.progress(0.5, "Exporting full resolution: encoding the PNG");
                Some(match &rgb8 {
                    Some(rgb8) if !png16 => encode::png8(rgb8)?,
                    _ => encode::png16(&image)?,
                })
            } else {
                None
            };
            let tiff = if opts.tiff16 {
                r.check()?;
                r.progress(0.75, "Exporting full resolution: encoding the TIFF");
                Some(encode::tiff16(&image)?)
            } else {
                None
            };
            Ok(Encoded {
                jpeg,
                tiff,
                png,
                rgb8,
                width: image.width as u32,
                height: image.height as u32,
            })
        })
        .await?;
        files.size = Some((encoded.width, encoded.height));
        if let Some(rgb8) = encoded.rgb8 {
            region::keep_render(photo, &params, rgb8);
        }
        if let Some(jpeg) = encoded.jpeg {
            files.jpeg = Some(put_jpeg(cx, jpeg).await?);
        }
        if let Some(png) = encoded.png {
            files.png = Some(
                cx.artifacts
                    .put(png, ArtifactKind::Image, "image/png")
                    .await
                    .context("store the PNG")?,
            );
        }
        if let Some(tiff) = encoded.tiff {
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
