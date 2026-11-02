//! Onesweep radix sort and tile binning. Owner: sort (E2).
//! Kernels: `cuda/radix.cu`, `cuda/binning.cu`. Spec: spec-raster 3.
//!
//! Buffer contract:
//! - `sort_depth`: (depth_key[0], depth_val[0]) → `depth_*[f.depth_sorted]`,
//!   stable, count from `status[ST_N_VIS]`.
//! - `bin_tiles`: scan_tiles → `inst_off`, `status[ST_N_INST]` (overflow
//!   check); emit → `tile_key[0]`, `inst_v[0]`; tile_hist; bin_scan →
//!   `tile_start`, `chain_off`, `tile_digit_offsets`, `status[ST_N_CHAINS]`,
//!   `status[ST_N_TILES_HIT]`; two u16 passes → `tile_key/inst_v[f.tile_sorted]`.
//!   Tile t occupies `[tile_start[t], tile_start[t + 1])` afterwards; no
//!   range-extraction pass.
//! - `count_tiles`: `n_tiles[v]` from `rec[v]` with the span rule emit uses
//!   (the reference count; preprocess_fwd computes the same with the same
//!   device function once it moves to `splat_common.cuh`).
//!
//! Everything only enqueues: no host sync, no allocation, counts read from the
//! device, no memsets (the sorter's scratch resets itself, see `radix.cu`), so
//! the whole sequence can be captured in a CUDA graph and replayed with any
//! count up to the capacities. Each kernel also has its own function so
//! benchmarks can time it alone; the stage functions just call them in order.

use anyhow::ensure;
use spl_core::Result;
use spl_gpu::cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, CudaView, LaunchArgs, LaunchConfig, PushKernelArg,
};

use super::layout::{RADIX_BINS, RADIX_BITS, SORT_IPT, SORT_THREADS, ST_N_INST, ST_N_VIS};
use super::{FrameBuffers, SplatKernels};
use crate::gpu::{
    COMMON_CUH, CudaFile, CudaResultExt, Defines, Gpu, KernelSource, SPLAT_COMMON_CUH,
};

#[cfg(test)]
mod bench;
#[cfg(test)]
mod tests;

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
    kernels: &["count_tiles", "scan_tiles", "emit", "tile_hist", "bin_scan"],
};

/// Depth keys are the float bits of z > 0 (monotone). spec-raster 2's
/// `DEPTH_BITS = 24` fast mode would write `bits >> 8` and sort 3 passes.
pub const DEPTH_KEY_BITS: u32 = 32;
/// Digit width of the tile sort's first pass; the second takes the rest
/// (13-bit tile ids: 8 + 5). `bin_scan` derives both passes' digit offsets
/// with the same split.
pub const TILE_LO_BITS: u32 = 8;

// Launch shapes, mirrored by `#define`s in binning.cu / radix.cu (test
// `launch_constants_match_cuda`).
pub(crate) const BIN_THREADS: u32 = 256;
pub(crate) const SCAN_THREADS: u32 = 256;
pub(crate) const SCAN_IPT: u32 = 16;
pub(crate) const SCAN_PART: u32 = SCAN_THREADS * SCAN_IPT;
pub(crate) const HIST_THREADS: u32 = 512;
pub(crate) const BIN_SCAN_THREADS: u32 = 1024;
const HIST_BLOCKS_PER_SM: u32 = 2;

/// `RadixSorter::state` words: the radix kernels' (`OS_*` of radix.cu), then
/// scan_tiles' (`SC_*` of binning.cu).
pub(crate) const OS_STATE: usize = 4;
pub(crate) const SC_STATE: usize = 3;

/// Bits needed for tile ids below `tiles` (at least 1).
pub fn tile_key_bits(tiles: u32) -> u32 {
    (u32::BITS - tiles.saturating_sub(1).leading_zeros()).max(1)
}

/// Compile-time shape of the onesweep kernels (one NVRTC module per value).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RadixConfig {
    /// Threads per pass block (multiple of 32, ≥ 256: threads 0..255 own a digit).
    pub threads: u32,
    /// Items per thread of a u32-key pass.
    pub ipt32: u32,
    /// Items per thread of a u16-key pass.
    pub ipt16: u32,
    /// `__launch_bounds__` minimum blocks per SM of the pass kernels.
    pub min_blocks: u32,
    /// Threads of the histogram kernel.
    pub hist_threads: u32,
    /// Rank with an 8-ballot multisplit instead of `__match_any_sync`.
    pub rank_ballot: bool,
}

impl Default for RadixConfig {
    fn default() -> Self {
        Self {
            threads: SORT_THREADS,
            ipt32: SORT_IPT,
            ipt16: SORT_IPT,
            min_blocks: 2,
            hist_threads: 512,
            rank_ballot: false,
        }
    }
}

impl RadixConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.threads.is_multiple_of(32) && (RADIX_BINS..=1024).contains(&self.threads),
            "radix threads must be a multiple of 32 in 256..=1024, got {}",
            self.threads
        );
        // Static shared memory: staging buffer + per-warp counters + digit arrays ≤ 48 KB.
        for ipt in [self.ipt32, self.ipt16] {
            ensure!(
                (1..=32).contains(&ipt),
                "items per thread {ipt} out of range"
            );
            let smem =
                4 * (self.threads * ipt + (self.threads / 32) * RADIX_BINS + 2 * RADIX_BINS + 16);
            ensure!(
                smem <= 48 * 1024,
                "radix pass needs {smem} B of shared memory (max 48 KB)"
            );
        }
        ensure!(
            self.hist_threads.is_multiple_of(32) && (128..=1024).contains(&self.hist_threads),
            "histogram threads {} out of range",
            self.hist_threads
        );
        Ok(())
    }

    pub fn defines(&self) -> Defines {
        Defines::new()
            .set("OS_THREADS", self.threads)
            .set("OS_IPT32", self.ipt32)
            .set("OS_IPT16", self.ipt16)
            .set("OS_MIN_BLOCKS", self.min_blocks)
            .set("OS_HIST_THREADS", self.hist_threads)
            .set("OS_RANK_BALLOT", u8::from(self.rank_ballot))
    }

    /// Items per partition of a u32-key pass.
    pub fn part32(&self) -> u32 {
        self.threads * self.ipt32
    }

    /// Items per partition of a u16-key pass.
    pub fn part16(&self) -> u32 {
        self.threads * self.ipt16
    }
}

pub struct RadixKernels {
    pub radix_hist32: CudaFunction,
    pub onesweep_u32: CudaFunction,
    pub onesweep_u16: CudaFunction,
}

impl RadixKernels {
    pub fn load(gpu: &Gpu, cfg: &RadixConfig) -> Result<Self> {
        cfg.validate()?;
        let r = gpu.kernels(&RADIX_SOURCE, &cfg.defines())?;
        Ok(Self {
            radix_hist32: r.get("radix_hist32")?,
            onesweep_u32: r.get("onesweep_u32")?,
            onesweep_u16: r.get("onesweep_u16")?,
        })
    }
}

pub struct BinningKernels {
    pub count_tiles: CudaFunction,
    pub scan_tiles: CudaFunction,
    pub emit: CudaFunction,
    pub tile_hist: CudaFunction,
    pub bin_scan: CudaFunction,
}

impl BinningKernels {
    pub fn load(gpu: &Gpu, d: &Defines) -> Result<Self> {
        let b = gpu.kernels(&BINNING_SOURCE, d)?;
        Ok(Self {
            count_tiles: b.get("count_tiles")?,
            scan_tiles: b.get("scan_tiles")?,
            emit: b.get("emit")?,
            tile_hist: b.get("tile_hist")?,
            bin_scan: b.get("bin_scan")?,
        })
    }
}

/// Items per onesweep partition with the default configuration.
pub const PARTITION: u32 = SORT_THREADS * SORT_IPT;

/// A null device pointer for optional kernel arguments.
static NULL_PTR: u64 = 0;

/// Pushes an optional status pointer (the overflow guard); null disables it.
fn push_guard<'a>(b: &mut LaunchArgs<'a>, guard: Option<&'a CudaView<'a, u32>>) {
    match guard {
        Some(g) => b.arg(g),
        None => b.arg(&NULL_PTR),
    };
}

fn grid(n: u32, per_block: u32) -> u32 {
    n.div_ceil(per_block).max(1)
}

fn cfg1d(blocks: u32, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// The `(shift, bits)` digits of a u16 sort with a `lo_bits` first pass.
pub fn u16_passes(key_bits: u32, lo_bits: u32) -> Vec<(u32, u32)> {
    if key_bits <= lo_bits {
        vec![(0, key_bits)]
    } else {
        vec![(0, lo_bits), (lo_bits, key_bits - lo_bits)]
    }
}

/// Generic stable LSD radix sort (spec-raster 3.2), 8-bit digits, onesweep
/// with decoupled look-back, count read from the device. Also used for
/// Morton re-sorting (N items). One stream at a time (`&mut self`).
pub struct RadixSorter {
    pub max_items: u32,
    pub config: RadixConfig,
    pub kernels: RadixKernels,
    /// `[partitions][RADIX_BINS]` epoch-tagged look-back words, shared by every
    /// pass (a word from an earlier pass reads as "not ready"; never cleared).
    pub lookback: CudaSlice<u64>,
    /// Self-resetting counters: `OS_STATE` radix words (epoch, partition
    /// ticket, check-out, histogram check-out), then `SC_STATE` scan words.
    pub state: CudaSlice<u32>,
    /// `[4][RADIX_BINS]` histogram accumulators (zeroed by the kernel that
    /// scans them).
    pub digit_hist: CudaSlice<u32>,
    /// `[4][RADIX_BINS]` exclusive digit offsets of the current u32 sort.
    pub digit_offsets: CudaSlice<u32>,
    /// scan_tiles look-back words, one per `SCAN_PART` items.
    pub scan_lookback: CudaSlice<u64>,
    sms: u32,
}

impl RadixSorter {
    /// Loads the radix kernels and allocates scratch for up to `max_items`
    /// items (any number of passes, u32 or u16 keys).
    pub fn new(gpu: &Gpu, max_items: u32) -> Result<Self> {
        Self::with_config(gpu, max_items, RadixConfig::default())
    }

    pub fn with_config(gpu: &Gpu, max_items: u32, config: RadixConfig) -> Result<Self> {
        let kernels = RadixKernels::load(gpu, &config)?;
        let part = config.part32().min(config.part16());
        let partitions = max_items.div_ceil(part).max(1) as usize;
        let s = &gpu.stream;
        Ok(Self {
            max_items,
            config,
            kernels,
            lookback: s
                .alloc_zeros::<u64>(partitions * RADIX_BINS as usize)
                .cuda("alloc sort look-back")?,
            state: s
                .alloc_zeros::<u32>(OS_STATE + SC_STATE)
                .cuda("alloc sort state")?,
            digit_hist: s
                .alloc_zeros::<u32>(4 * RADIX_BINS as usize)
                .cuda("alloc digit histograms")?,
            digit_offsets: s
                .alloc_zeros::<u32>(4 * RADIX_BINS as usize)
                .cuda("alloc digit offsets")?,
            scan_lookback: s
                .alloc_zeros::<u64>(max_items.div_ceil(SCAN_PART).max(1) as usize)
                .cuda("alloc scan look-back")?,
            sms: gpu.multiprocessors(),
        })
    }

    /// Bytes of device scratch.
    pub fn scratch_bytes(&self) -> u64 {
        (self.lookback.len() * 8
            + self.scan_lookback.len() * 8
            + (self.state.len() + self.digit_hist.len() + self.digit_offsets.len()) * 4)
            as u64
    }

    /// Items the launches cover: the sorter capacity, or less if the buffers
    /// are shorter. The device count must not exceed it.
    fn cover(&self, lens: &[usize]) -> u32 {
        lens.iter()
            .fold(self.max_items as usize, |m, &l| m.min(l))
            .min(u32::MAX as usize) as u32
    }

    /// Sorts `n` (device) (key, value) pairs by the low `key_bits` bits of the
    /// key (higher bits must be zero), stable. Ping-pongs between the two
    /// buffers, starting from index 0; returns the index holding the result
    /// (`key_bits.div_ceil(8) % 2`). `guard`: the frame's status words; the
    /// sort is skipped while `status[ST_OVERFLOW]` is set.
    pub fn sort_u32(
        &mut self,
        s: &CudaStream,
        keys: [&mut CudaSlice<u32>; 2],
        vals: [&mut CudaSlice<u32>; 2],
        n: &CudaView<'_, u32>,
        key_bits: u32,
        guard: Option<&CudaView<'_, u32>>,
    ) -> Result<usize> {
        let [k0, k1] = keys;
        let [v0, v1] = vals;
        self.hist_u32(s, k0, n, key_bits, guard)?;
        let passes = key_bits.div_ceil(RADIX_BITS);
        for p in 0..passes {
            if p % 2 == 0 {
                self.pass_u32(s, [&*k0, &*v0], [&mut *k1, &mut *v1], n, p, key_bits, guard)?;
            } else {
                self.pass_u32(s, [&*k1, &*v1], [&mut *k0, &mut *v0], n, p, key_bits, guard)?;
            }
        }
        Ok((passes % 2) as usize)
    }

    /// radix_hist32: every pass's digit histogram of a `key_bits`-bit sort,
    /// scanned into `digit_offsets`.
    pub fn hist_u32(
        &mut self,
        s: &CudaStream,
        keys: &CudaSlice<u32>,
        n: &CudaView<'_, u32>,
        key_bits: u32,
        guard: Option<&CudaView<'_, u32>>,
    ) -> Result<()> {
        ensure!(
            (1..=32).contains(&key_bits),
            "sort_u32: key_bits {key_bits} not in 1..=32"
        );
        let mut b = s.launch_builder(&self.kernels.radix_hist32);
        b.arg(keys)
            .arg(n)
            .arg(&key_bits)
            .arg(&mut self.digit_hist)
            .arg(&mut self.digit_offsets)
            .arg(&mut self.state);
        push_guard(&mut b, guard);
        let cfg = cfg1d(self.sms * HIST_BLOCKS_PER_SM, self.config.hist_threads);
        // SAFETY: radix_hist32(const u32* keys, const u32* n, u32 key_bits,
        // u32* hist_acc, u32* digit_offsets, u32* state, const u32* status):
        // keys hold ≥ n items (16 B aligned), both histogram arrays 4 ×
        // RADIX_BINS words, state ≥ OS_STATE words; status null or the frame's.
        unsafe { b.launch(cfg) }.map(drop).cuda("radix_hist32")
    }

    /// One onesweep_u32 pass (digit `pass` of a `key_bits`-bit sort) from
    /// `src` to `dst`, using the offsets of the last `hist_u32`.
    #[allow(clippy::too_many_arguments)]
    pub fn pass_u32(
        &mut self,
        s: &CudaStream,
        src: [&CudaSlice<u32>; 2],
        dst: [&mut CudaSlice<u32>; 2],
        n: &CudaView<'_, u32>,
        pass: u32,
        key_bits: u32,
        guard: Option<&CudaView<'_, u32>>,
    ) -> Result<()> {
        let shift = pass * RADIX_BITS;
        ensure!(shift < key_bits, "pass {pass} beyond {key_bits} bits");
        let bits = (key_bits - shift).min(RADIX_BITS);
        let [ki, vi] = src;
        let [ko, vo] = dst;
        let cover = self.cover(&[ki.len(), vi.len(), ko.len(), vo.len()]);
        let base = self
            .digit_offsets
            .slice((pass * RADIX_BINS) as usize..((pass + 1) * RADIX_BINS) as usize);
        let mut b = s.launch_builder(&self.kernels.onesweep_u32);
        b.arg(ki)
            .arg(vi)
            .arg(ko)
            .arg(vo)
            .arg(n)
            .arg(&shift)
            .arg(&bits)
            .arg(&base)
            .arg(&mut self.lookback)
            .arg(&mut self.state);
        push_guard(&mut b, guard);
        let cfg = cfg1d(grid(cover, self.config.part32()), self.config.threads);
        // SAFETY: onesweep_u32(keys_in, vals_in, keys_out, vals_out, const u32*
        // n, u32 shift, u32 bits, const u32* digit_base, u64* lookback, u32*
        // state, const u32* status): the four buffers hold ≥ cover ≥ n items,
        // digit_base RADIX_BINS words, lookback a RADIX_BINS row per block of
        // the grid (sized from max_items ≥ cover).
        unsafe { b.launch(cfg) }.map(drop).cuda("onesweep_u32")
    }

    /// The tile sort: 16-bit keys of at most `key_bits` (≤ 16) bits whose
    /// digit offsets `bin_scan` already derived from the tile histogram (no
    /// histogram pass): `digit_offsets[0][·]` for the low `lo_bits` bits,
    /// `[1][·]` for the rest. Returns the result's buffer index.
    #[allow(clippy::too_many_arguments)]
    pub fn sort_u16_with_hist(
        &mut self,
        s: &CudaStream,
        keys: [&mut CudaSlice<u16>; 2],
        vals: [&mut CudaSlice<u32>; 2],
        n: &CudaView<'_, u32>,
        key_bits: u32,
        lo_bits: u32,
        digit_offsets: &CudaView<'_, u32>,
        guard: Option<&CudaView<'_, u32>>,
    ) -> Result<usize> {
        ensure!(
            (1..=16).contains(&key_bits) && (1..=RADIX_BITS).contains(&lo_bits),
            "sort_u16_with_hist: key_bits {key_bits}, lo_bits {lo_bits}"
        );
        ensure!(
            key_bits <= lo_bits + RADIX_BITS,
            "sort_u16_with_hist: {key_bits} bits need more than two passes of {lo_bits} + 8"
        );
        ensure!(
            digit_offsets.len() >= 2 * RADIX_BINS as usize,
            "sort_u16_with_hist: digit offsets need 2 × {RADIX_BINS} words"
        );
        let [k0, k1] = keys;
        let [v0, v1] = vals;
        let passes = u16_passes(key_bits, lo_bits);
        for (p, &(shift, bits)) in passes.iter().enumerate() {
            let base = digit_offsets.slice(p * RADIX_BINS as usize..(p + 1) * RADIX_BINS as usize);
            if p % 2 == 0 {
                self.pass_u16(
                    s,
                    (&*k0, &*v0),
                    (&mut *k1, &mut *v1),
                    n,
                    shift,
                    bits,
                    &base,
                    guard,
                )?;
            } else {
                self.pass_u16(
                    s,
                    (&*k1, &*v1),
                    (&mut *k0, &mut *v0),
                    n,
                    shift,
                    bits,
                    &base,
                    guard,
                )?;
            }
        }
        Ok(passes.len() % 2)
    }

    /// One onesweep_u16 pass on digit `(key >> shift) & ((1 << bits) - 1)`.
    #[allow(clippy::too_many_arguments)]
    pub fn pass_u16(
        &mut self,
        s: &CudaStream,
        src: (&CudaSlice<u16>, &CudaSlice<u32>),
        dst: (&mut CudaSlice<u16>, &mut CudaSlice<u32>),
        n: &CudaView<'_, u32>,
        shift: u32,
        bits: u32,
        digit_base: &CudaView<'_, u32>,
        guard: Option<&CudaView<'_, u32>>,
    ) -> Result<()> {
        ensure!(
            (1..=RADIX_BITS).contains(&bits) && digit_base.len() >= RADIX_BINS as usize,
            "pass_u16: bits {bits}"
        );
        let (ki, vi) = src;
        let (ko, vo) = dst;
        let cover = self.cover(&[ki.len(), vi.len(), ko.len(), vo.len()]);
        let mut b = s.launch_builder(&self.kernels.onesweep_u16);
        b.arg(ki)
            .arg(vi)
            .arg(ko)
            .arg(vo)
            .arg(n)
            .arg(&shift)
            .arg(&bits)
            .arg(digit_base)
            .arg(&mut self.lookback)
            .arg(&mut self.state);
        push_guard(&mut b, guard);
        let cfg = cfg1d(grid(cover, self.config.part16()), self.config.threads);
        // SAFETY: onesweep_u16: as onesweep_u32 with u16 keys.
        unsafe { b.launch(cfg) }.map(drop).cuda("onesweep_u16")
    }
}

/// 32-bit (or 24-bit, `DEPTH_BITS`) stable depth sort of the visible set.
pub fn sort_depth(
    _k: &SplatKernels,
    s: &CudaStream,
    sorter: &mut RadixSorter,
    f: &mut FrameBuffers,
) -> Result<()> {
    let FrameBuffers {
        iter,
        depth_key: [k0, k1],
        depth_val: [v0, v1],
        ..
    } = f;
    let status = iter.status.slice(..);
    let n = iter.status.slice(ST_N_VIS..ST_N_VIS + 1);
    let sorted = sorter.sort_u32(s, [k0, k1], [v0, v1], &n, DEPTH_KEY_BITS, Some(&status))?;
    f.depth_sorted = sorted;
    Ok(())
}

/// `n_tiles[v]` for v < `status[ST_N_VIS]` from the finished records, with the
/// span rule `emit` uses (tests, and callers that need counts from records).
pub fn count_tiles(k: &SplatKernels, s: &CudaStream, f: &mut FrameBuffers) -> Result<()> {
    let mut b = s.launch_builder(&k.binning.count_tiles);
    b.arg(&f.iter.cam)
        .arg(&f.rec)
        .arg(&mut f.n_tiles)
        .arg(&f.iter.status);
    // SAFETY: count_tiles(const FrameCam* cam, const SplatRec* rec, u32*
    // n_tiles, const u32* status): rec and n_tiles hold V_cap ≥ status[ST_N_VIS].
    unsafe { b.launch(cfg1d(grid(f.cap.visible, BIN_THREADS), BIN_THREADS)) }
        .map(drop)
        .cuda("count_tiles")
}

/// scan_tiles → emit → tile_hist → bin_scan → 2 × onesweep_u16.
pub fn bin_tiles(
    k: &SplatKernels,
    s: &CudaStream,
    sorter: &mut RadixSorter,
    f: &mut FrameBuffers,
) -> Result<()> {
    scan_tiles(k, s, sorter, f)?;
    emit(k, s, f)?;
    tile_hist(k, s, sorter, f)?;
    bin_scan(k, s, f)?;
    sort_tiles(k, s, sorter, f)
}

/// inst_off = exclusive scan of n_tiles in depth order; status[ST_N_INST]
/// (true total) and OVF_INSTANCES.
pub fn scan_tiles(
    k: &SplatKernels,
    s: &CudaStream,
    sorter: &mut RadixSorter,
    f: &mut FrameBuffers,
) -> Result<()> {
    let i_cap = f.cap.instances;
    let mut sc_state = sorter.state.slice_mut(OS_STATE..OS_STATE + SC_STATE);
    let mut b = s.launch_builder(&k.binning.scan_tiles);
    b.arg(&f.depth_val[f.depth_sorted])
        .arg(&f.n_tiles)
        .arg(&i_cap)
        .arg(&mut f.inst_off)
        .arg(&mut sorter.scan_lookback)
        .arg(&mut sc_state)
        .arg(&mut f.iter.status);
    // SAFETY: scan_tiles(const u32* sorted_v, const u32* n_tiles, u32 i_cap,
    // u32* inst_off, u64* lookback, u32* state, u32* status): sorted_v and
    // n_tiles hold V_cap, inst_off V_cap + 1, lookback one word per SCAN_PART
    // of the grid (the sorter covers max(V_cap, I_cap)), state SC_STATE words.
    unsafe { b.launch(cfg1d(grid(f.cap.visible, SCAN_PART), SCAN_THREADS)) }
        .map(drop)
        .cuda("scan_tiles")
}

/// (tile, v) instances in depth order into tile_key[0] / inst_v[0].
pub fn emit(k: &SplatKernels, s: &CudaStream, f: &mut FrameBuffers) -> Result<()> {
    let [tk0, _] = &mut f.tile_key;
    let [iv0, _] = &mut f.inst_v;
    let mut b = s.launch_builder(&k.binning.emit);
    b.arg(&f.iter.cam)
        .arg(&f.depth_val[f.depth_sorted])
        .arg(&f.rec)
        .arg(&f.inst_off)
        .arg(tk0)
        .arg(iv0)
        .arg(&f.iter.status);
    // SAFETY: emit(const FrameCam*, const u32* sorted_v, const SplatRec* rec,
    // const u32* inst_off, u16* tile_key, u32* inst_v, const u32* status):
    // per-v buffers hold V_cap, instance buffers I_cap; emit runs only while
    // status[ST_N_INST] ≤ I_cap (no overflow) and writes [0, N_INST).
    unsafe { b.launch(cfg1d(grid(f.cap.visible, BIN_THREADS), BIN_THREADS)) }
        .map(drop)
        .cuda("emit")
}

/// Full tile histogram of the emitted keys into tile_hist.
pub fn tile_hist(
    k: &SplatKernels,
    s: &CudaStream,
    sorter: &RadixSorter,
    f: &mut FrameBuffers,
) -> Result<()> {
    let mut b = s.launch_builder(&k.binning.tile_hist);
    b.arg(&f.tile_key[0])
        .arg(&mut f.tile_hist)
        .arg(&f.iter.status);
    // SAFETY: tile_hist(const u16* tile_key, u32* hist, const u32* status):
    // keys hold I_cap ≥ N_INST entries (16 B aligned) below cam.n_tiles ≤
    // T_cap; hist T_cap + 1 words.
    unsafe { b.launch(cfg1d(sorter.sms * HIST_BLOCKS_PER_SM, HIST_THREADS)) }
        .map(drop)
        .cuda("tile_hist")
}

/// tile_start, chain_off, the tile sort's digit offsets, N_CHAINS, N_TILES_HIT.
pub fn bin_scan(k: &SplatKernels, s: &CudaStream, f: &mut FrameBuffers) -> Result<()> {
    let c_cap = f.cap.chains(&f.variant);
    let mut b = s.launch_builder(&k.binning.bin_scan);
    b.arg(&f.iter.cam)
        .arg(&mut f.tile_hist)
        .arg(&c_cap)
        .arg(&TILE_LO_BITS)
        .arg(&mut f.tile_start)
        .arg(&mut f.chain_off)
        .arg(&mut f.tile_digit_offsets)
        .arg(&mut f.iter.status);
    // SAFETY: bin_scan(const FrameCam*, u32* hist, u32 c_cap, u32 lo_bits, u32*
    // tile_start, u32* chain_off, u32* digit_offsets, u32* status): the tile
    // arrays hold T_cap + 1 ≥ cam.n_tiles + 1 words, digit_offsets 2 × 256.
    unsafe { b.launch(cfg1d(1, BIN_SCAN_THREADS)) }
        .map(drop)
        .cuda("bin_scan")
}

/// The stable two-pass tile sort of the emitted instances.
pub fn sort_tiles(
    _k: &SplatKernels,
    s: &CudaStream,
    sorter: &mut RadixSorter,
    f: &mut FrameBuffers,
) -> Result<()> {
    let key_bits = tile_key_bits(f.cap.tiles);
    let FrameBuffers {
        iter,
        tile_key: [tk0, tk1],
        inst_v: [iv0, iv1],
        tile_digit_offsets,
        ..
    } = f;
    let status = iter.status.slice(..);
    let n = iter.status.slice(ST_N_INST..ST_N_INST + 1);
    let offsets = tile_digit_offsets.slice(..);
    let sorted = sorter.sort_u16_with_hist(
        s,
        [tk0, tk1],
        [iv0, iv1],
        &n,
        key_bits,
        TILE_LO_BITS,
        &offsets,
        Some(&status),
    )?;
    f.tile_sorted = sorted;
    Ok(())
}
