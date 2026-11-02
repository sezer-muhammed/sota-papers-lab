//! Onesweep radix sort and tile binning. Owner: sort (E2).
//! Kernels: `cuda/radix.cu`, `cuda/binning.cu`. Spec: spec-raster 3.
//!
//! Buffer contract:
//! - `sort_depth`: (depth_key[0], depth_val[0]) → `depth_*[f.depth_sorted]`,
//!   stable, count from `status[ST_N_VIS]`.
//! - `bin_tiles`: scan_tiles → `inst_off`, `status[ST_N_INST]` (overflow
//!   check); emit → `tile_key[0]`, `inst_v[0]`; tile_hist; bin_scan →
//!   `tile_start`, `chain_off`, `tile_digit_offsets`, `status[ST_N_CHAINS]`;
//!   two u16 passes → `tile_key/inst_v[f.tile_sorted]`. Tile t occupies
//!   `[tile_start[t], tile_start[t + 1])` afterwards; no range-extraction pass.
//!
//! STUBS: the sorts and stage functions return "not implemented" until E2
//! lands them; `RadixSorter::new` allocates its scratch for real.

use spl_core::Result;
use spl_gpu::cudarc::driver::{CudaFunction, CudaSlice, CudaStream, CudaView};

use super::layout::{RADIX_BINS, SORT_IPT, SORT_THREADS};
use super::{FrameBuffers, SplatKernels};
use crate::gpu::{
    COMMON_CUH, CudaFile, CudaResultExt, Defines, Gpu, KernelSource, SPLAT_COMMON_CUH,
    not_implemented,
};

const OWNER: &str = "sort (E2)";

pub const RADIX_SOURCE: KernelSource = KernelSource {
    module: "radix",
    files: &[
        COMMON_CUH,
        SPLAT_COMMON_CUH,
        CudaFile {
            name: "radix.cu",
            text: include_str!("../../cuda/radix.cu"),
        },
    ],
    kernels: &["radix_hist32", "onesweep_u32", "onesweep_u16"],
};

pub const BINNING_SOURCE: KernelSource = KernelSource {
    module: "binning",
    files: &[
        COMMON_CUH,
        SPLAT_COMMON_CUH,
        CudaFile {
            name: "binning.cu",
            text: include_str!("../../cuda/binning.cu"),
        },
    ],
    kernels: &["scan_tiles", "emit", "tile_hist", "bin_scan"],
};

pub struct RadixKernels {
    pub radix_hist32: CudaFunction,
    pub onesweep_u32: CudaFunction,
    pub onesweep_u16: CudaFunction,
}

impl RadixKernels {
    pub fn load(gpu: &Gpu) -> Result<Self> {
        let r = gpu.kernels(&RADIX_SOURCE, &Defines::new())?;
        Ok(Self {
            radix_hist32: r.get("radix_hist32")?,
            onesweep_u32: r.get("onesweep_u32")?,
            onesweep_u16: r.get("onesweep_u16")?,
        })
    }
}

pub struct BinningKernels {
    pub scan_tiles: CudaFunction,
    pub emit: CudaFunction,
    pub tile_hist: CudaFunction,
    pub bin_scan: CudaFunction,
}

impl BinningKernels {
    pub fn load(gpu: &Gpu, d: &Defines) -> Result<Self> {
        let b = gpu.kernels(&BINNING_SOURCE, d)?;
        Ok(Self {
            scan_tiles: b.get("scan_tiles")?,
            emit: b.get("emit")?,
            tile_hist: b.get("tile_hist")?,
            bin_scan: b.get("bin_scan")?,
        })
    }
}

/// Items per onesweep partition.
pub const PARTITION: u32 = SORT_THREADS * SORT_IPT;

/// Generic stable LSD radix sort (spec-raster 3.2), 8-bit digits, count read
/// from the device. Also used for Morton re-sorting (N items).
pub struct RadixSorter {
    pub max_items: u32,
    pub kernels: RadixKernels,
    /// `[4 passes][partitions][RADIX_BINS]` look-back words (flag | count).
    pub lookback: CudaSlice<u32>,
    /// One partition counter per pass.
    pub part_ctr: CudaSlice<u32>,
    /// `[4][RADIX_BINS]` digit histograms, scanned in place.
    pub digit_hist: CudaSlice<u32>,
    pub done_ctr: CudaSlice<u32>,
}

impl RadixSorter {
    /// Loads the radix kernels and allocates scratch for up to `max_items`
    /// items (4 passes of 32-bit keys).
    pub fn new(gpu: &Gpu, max_items: u32) -> Result<Self> {
        let partitions = max_items.div_ceil(PARTITION).max(1) as usize;
        let s = &gpu.stream;
        Ok(Self {
            max_items,
            kernels: RadixKernels::load(gpu)?,
            lookback: s
                .alloc_zeros::<u32>(4 * partitions * RADIX_BINS as usize)
                .cuda("alloc sort look-back")?,
            part_ctr: s.alloc_zeros::<u32>(4).cuda("alloc partition counters")?,
            digit_hist: s
                .alloc_zeros::<u32>(4 * RADIX_BINS as usize)
                .cuda("alloc digit histograms")?,
            done_ctr: s.alloc_zeros::<u32>(1).cuda("alloc done counter")?,
        })
    }

    /// Sorts `n` (device) (key, value) pairs by the low `key_bits` bits.
    /// Ping-pongs between the two buffers; returns the index holding the result.
    pub fn sort_u32(
        &mut self,
        _s: &CudaStream,
        _keys: [&mut CudaSlice<u32>; 2],
        _vals: [&mut CudaSlice<u32>; 2],
        _n: &CudaView<'_, u32>,
        _key_bits: u32,
    ) -> Result<usize> {
        Err(not_implemented("RadixSorter::sort_u32", OWNER))
    }

    /// The tile sort: 16-bit keys whose digit offsets bin_scan already derived
    /// from the tile histogram (no histogram pass).
    pub fn sort_u16_with_hist(
        &mut self,
        _s: &CudaStream,
        _keys: [&mut CudaSlice<u16>; 2],
        _vals: [&mut CudaSlice<u32>; 2],
        _n: &CudaView<'_, u32>,
        _key_bits: u32,
        _digit_offsets: &CudaSlice<u32>,
    ) -> Result<usize> {
        Err(not_implemented("RadixSorter::sort_u16_with_hist", OWNER))
    }
}

/// 32-bit (or 24-bit, `DEPTH_BITS`) stable depth sort of the visible set.
pub fn sort_depth(
    _k: &SplatKernels,
    _s: &CudaStream,
    _sorter: &mut RadixSorter,
    _f: &mut FrameBuffers,
) -> Result<()> {
    Err(not_implemented("sort_depth", OWNER))
}

/// scan_tiles → emit → tile_hist → bin_scan → 2 × onesweep_u16.
pub fn bin_tiles(
    _k: &SplatKernels,
    _s: &CudaStream,
    _sorter: &mut RadixSorter,
    _f: &mut FrameBuffers,
) -> Result<()> {
    Err(not_implemented("bin_tiles", OWNER))
}
