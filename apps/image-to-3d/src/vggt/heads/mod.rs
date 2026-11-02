//! PLACEHOLDER until the heads branch (w/eng-heads) lands: the agreed
//! interface only. Owner: heads.

use spl_core::{Result, bail};
use spl_nn::Nn;
use spl_nn::cudarc::driver::{CudaSlice, CudaView, CudaViewMut};

use super::Geometry;

pub trait HeadSource {
    fn host_f32(&self, name: &str) -> Result<Vec<f32>>;
    fn shape(&self, name: &str) -> Result<Vec<usize>>;
}

pub struct DeviceCams {
    pub extrinsics: CudaSlice<f32>,
    pub intrinsics: CudaSlice<f32>,
}

impl DeviceCams {
    pub fn new(nn: &Nn, frames: usize) -> Result<Self> {
        let e = |e| spl_core::anyhow!("{e:?}");
        Ok(Self {
            extrinsics: nn.stream.alloc_zeros(frames * 12).map_err(e)?,
            intrinsics: nn.stream.alloc_zeros(frames * 9).map_err(e)?,
        })
    }
}

pub struct CameraHead;

impl CameraHead {
    pub fn new(_nn: &Nn, _w: &dyn HeadSource) -> Result<Self> {
        Ok(Self)
    }
    pub fn scratch_bytes(_g: &Geometry) -> usize {
        0
    }
    pub fn run(
        &self,
        _nn: &Nn,
        _cache23: CudaView<'_, u16>,
        _g: &Geometry,
        _cams: &mut DeviceCams,
    ) -> Result<Vec<[f32; 9]>> {
        bail!("heads not merged yet")
    }
}

pub struct DenseHead;

impl DenseHead {
    pub fn new(_nn: &Nn, _w: &dyn HeadSource) -> Result<Self> {
        Ok(Self)
    }
    pub fn scratch_bytes(_g: &Geometry) -> usize {
        0
    }
    #[allow(clippy::too_many_arguments)]
    pub fn run_chunk(
        &self,
        _nn: &Nn,
        _layers: [CudaView<'_, u16>; 4],
        _g: &Geometry,
        _depth: &mut CudaViewMut<'_, u16>,
        _conf: &mut CudaViewMut<'_, u16>,
        _cams: Option<(&CudaView<'_, f32>, &CudaView<'_, f32>)>,
        _world: Option<&mut CudaViewMut<'_, f32>>,
    ) -> Result<()> {
        bail!("heads not merged yet")
    }
}

pub fn encoding_to_camera(_pe: &[f32; 9], _h: usize, _w: usize) -> ([f32; 12], [f32; 9]) {
    ([0.0; 12], [0.0; 9])
}
