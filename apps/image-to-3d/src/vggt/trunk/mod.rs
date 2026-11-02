//! PLACEHOLDER until the trunk branch (w/eng-trunk) lands: the agreed
//! interface only. Owner: trunk.

use spl_core::{Result, bail};
use spl_nn::Nn;
use spl_nn::cudarc::driver::{CudaView, CudaViewMut};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterKind {
    Global,
    Register,
}

#[derive(Debug, Clone, Copy)]
pub struct TrunkGeom {
    pub frames: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

pub trait TrunkWeights {
    fn bf16(&self, name: &str) -> Result<CudaView<'_, u16>>;
    fn f32(&self, name: &str) -> Result<CudaView<'_, f32>>;
    fn host_f32(&self, name: &str) -> Result<Vec<f32>>;
    fn has(&self, name: &str) -> bool;
}

pub struct Trunk;

impl Trunk {
    pub fn new(_nn: &Nn, _w: &dyn TrunkWeights) -> Result<Self> {
        Ok(Self)
    }

    pub fn scratch_bytes(_g: &super::Geometry) -> usize {
        0
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        _nn: &Nn,
        _w: &dyn TrunkWeights,
        _g: TrunkGeom,
        _images: &CudaView<'_, f32>,
        _cache: [&mut CudaViewMut<'_, u16>; 4],
        _sched: &[InterKind; 24],
        _tap: &mut dyn FnMut(usize),
    ) -> Result<()> {
        bail!("trunk not merged yet")
    }
}
