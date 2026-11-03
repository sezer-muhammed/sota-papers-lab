//! VGGT-Ω weights: safetensors files (written by `scripts/convert_vggt.py`)
//! mapped into memory, the device region of every file copied into one
//! device arena with one bulk upload per file, typed views by tensor name.
//! Owner: runtime.
//!
//! The converter puts every `aggregator.*` tensor first (BF16 2-D weights and
//! their biases, F32 norms, gammas, tokens and periods), 256-byte aligned,
//! and records the length of that region as `device_bytes`. The heads'
//! tensors (`camera_head.*`, `dense_head.*`, F32) stay in the mapping: the
//! heads read them with [`Weights::host_f32`] and upload their own layouts.
//! An FP8 file (`<name>` F8_E4M3 + `<name>_scale` F32) loads into the same
//! arena.
//!
//! The checkpoints are under the FAIR non-commercial research licence:
//! research use only, never redistributed.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::ensure;
use memmap2::Mmap;
use spl_core::{Context, Result, anyhow, bail};
use spl_nn::Nn;
use spl_nn::cudarc::driver::{CudaSlice, CudaView};

/// Index prefix of tensors from an FP8 file (`format = spl-vggt-1-fp8`).
const FP8_PREFIX: &str = "fp8:";

/// Device offsets are multiples of this (the converter's alignment).
pub const ALIGN: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
    F8E4M3,
    U8,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "BF16" => Self::Bf16,
            "F16" => Self::F16,
            "F32" => Self::F32,
            "F8_E4M3" => Self::F8E4M3,
            "U8" => Self::U8,
            other => bail!("unsupported safetensors dtype {other}"),
        })
    }

    pub fn size(self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
            Self::F8E4M3 | Self::U8 => 1,
        }
    }
}

/// One tensor of a parsed header: byte range relative to the data section.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A parsed safetensors header.
#[derive(Debug, Clone)]
pub struct Header {
    /// Offset of the data section in the file.
    pub data_start: usize,
    pub tensors: HashMap<String, TensorInfo>,
    /// Length of the device region (`__metadata__.device_bytes`); 0 when the
    /// file has none (then nothing is uploaded).
    pub device_bytes: usize,
    pub metadata: HashMap<String, String>,
}

/// Parses the header of a safetensors file held in `bytes` (at least the
/// header; the data length is checked when `file_len` is given).
pub fn parse_header(bytes: &[u8], file_len: usize) -> Result<Header> {
    ensure!(bytes.len() >= 8, "safetensors: file too short");
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    ensure!(
        n <= 100 << 20 && 8 + n <= bytes.len(),
        "safetensors: bad header length {n}"
    );
    let json: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + n]).context("safetensors header JSON")?;
    let data_start = 8 + n;
    let mut tensors = HashMap::new();
    let mut metadata = HashMap::new();
    for (name, v) in json {
        if name == "__metadata__" {
            if let Some(m) = v.as_object() {
                for (k, v) in m {
                    metadata.insert(k.clone(), v.as_str().unwrap_or_default().to_string());
                }
            }
            continue;
        }
        let dtype = Dtype::parse(v["dtype"].as_str().unwrap_or(""))?;
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .ok_or_else(|| anyhow!("{name}: no shape"))?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize))
            .collect::<Option<_>>()
            .ok_or_else(|| anyhow!("{name}: bad shape"))?;
        let offs = v["data_offsets"]
            .as_array()
            .filter(|a| a.len() == 2)
            .ok_or_else(|| anyhow!("{name}: no data_offsets"))?;
        let (start, end) = (
            offs[0].as_u64().unwrap_or(u64::MAX) as usize,
            offs[1].as_u64().unwrap_or(0) as usize,
        );
        let t = TensorInfo {
            dtype,
            shape,
            start,
            end,
        };
        ensure!(
            start <= end && end - start == t.numel() * dtype.size(),
            "{name}: data_offsets {start}..{end} do not match {:?} {:?}",
            t.dtype,
            t.shape
        );
        ensure!(
            data_start + end <= file_len,
            "{name}: data past the end of the file"
        );
        tensors.insert(name, t);
    }
    let device_bytes = match metadata.get("device_bytes") {
        Some(s) => s.parse().map_err(|_| anyhow!("bad device_bytes {s:?}"))?,
        None => 0,
    };
    ensure!(
        data_start + device_bytes <= file_len,
        "device_bytes past the end of the file"
    );
    Ok(Header {
        data_start,
        tensors,
        device_bytes,
        metadata,
    })
}

struct Entry {
    file: usize,
    info: TensorInfo,
    /// Byte offset in the arena when the tensor is on the device.
    device: Option<usize>,
}

struct Mapped {
    path: PathBuf,
    map: Mmap,
    data_start: usize,
}

/// All weights: device arena plus the mapped files.
pub struct Weights {
    arena: CudaSlice<u8>,
    files: Vec<Mapped>,
    index: HashMap<String, Entry>,
    upload_s: f64,
}

impl Weights {
    /// Maps `paths` (the BF16 file first, then optional FP8 files) and
    /// uploads their device regions into one allocation. Names must be
    /// unique across files.
    pub fn load(nn: &Nn, paths: &[PathBuf]) -> Result<Self> {
        ensure!(!paths.is_empty(), "no weight files");
        let mut files = Vec::new();
        let mut index = HashMap::new();
        let mut regions = Vec::new();
        let mut total = 0usize;
        for (fi, path) in paths.iter().enumerate() {
            let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
            // SAFETY: read-only mapping of a file we do not modify; the
            // converter writes to a temporary name and renames.
            let map =
                unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
            let h = parse_header(&map, map.len())
                .with_context(|| format!("parse {}", path.display()))?;
            let base = total;
            // FP8 files reuse the BF16 names; keep them apart.
            let prefix = if h
                .metadata
                .get("format")
                .is_some_and(|f| f.ends_with("-fp8"))
            {
                FP8_PREFIX
            } else {
                ""
            };
            for (name, info) in h.tensors {
                if name.starts_with("__pad") {
                    continue;
                }
                let name = format!("{prefix}{name}");
                let device = (info.end <= h.device_bytes).then(|| base + info.start);
                if let Some(off) = device {
                    ensure!(off % ALIGN == 0, "{name}: device offset {off} not aligned");
                }
                let prev = index.insert(
                    name.clone(),
                    Entry {
                        file: fi,
                        info,
                        device,
                    },
                );
                ensure!(prev.is_none(), "tensor {name} appears twice");
            }
            regions.push((base, h.data_start, h.device_bytes));
            total = base + h.device_bytes.next_multiple_of(ALIGN);
            files.push(Mapped {
                path: path.clone(),
                map,
                data_start: h.data_start,
            });
        }
        let started = std::time::Instant::now();
        let mut arena = nn
            .stream
            .alloc_zeros::<u8>(total.max(ALIGN))
            .map_err(|e| anyhow!("allocate {total} bytes of weights: {e:?}"))?;
        upload_regions(nn, &files, &regions, &mut arena)?;
        nn.sync()?;
        Ok(Self {
            arena,
            files,
            index,
            upload_s: started.elapsed().as_secs_f64(),
        })
    }

    /// Bytes on the device.
    pub fn device_bytes(&self) -> usize {
        self.arena.len()
    }

    /// Seconds the bulk upload took.
    pub fn upload_seconds(&self) -> f64 {
        self.upload_s
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.entry(name).map(|e| &e.info)
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        self.info(name).map(|i| i.shape.as_slice())
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.index
            .get(name)
            .ok_or_else(|| anyhow!("weight {name} not found"))
    }

    fn view<T>(&self, name: &str, dtype: Dtype) -> Result<CudaView<'_, T>> {
        let e = self.entry(name)?;
        ensure!(
            e.info.dtype == dtype,
            "weight {name} is {:?}, wanted {dtype:?}",
            e.info.dtype
        );
        let off = e
            .device
            .ok_or_else(|| anyhow!("weight {name} is host-only"))?;
        let bytes = e.info.end - e.info.start;
        let v = self.arena.slice(off..off + bytes);
        // SAFETY: the range holds `numel` elements of `T` (checked against
        // the dtype above) and is 256-byte aligned.
        unsafe { v.transmute::<T>(e.info.numel()) }
            .ok_or_else(|| anyhow!("weight {name}: view out of range"))
    }

    /// A BF16 tensor on the device (bits as `u16`).
    pub fn bf16(&self, name: &str) -> Result<CudaView<'_, u16>> {
        self.view(name, Dtype::Bf16)
    }

    /// An F32 tensor on the device.
    pub fn f32(&self, name: &str) -> Result<CudaView<'_, f32>> {
        self.view(name, Dtype::F32)
    }

    /// The FP8 E4M3 copy of a weight (from the FP8 file) and its
    /// per-output-channel scales (`<name>_scale`).
    pub fn fp8(&self, name: &str) -> Result<(CudaView<'_, u8>, CudaView<'_, f32>)> {
        Ok((
            self.view(&format!("{FP8_PREFIX}{name}"), Dtype::F8E4M3)?,
            self.f32(&format!("{FP8_PREFIX}{name}_scale"))?,
        ))
    }

    /// The raw little-endian bytes in the mapping.
    pub fn host_bytes(&self, name: &str) -> Result<&[u8]> {
        let e = self.entry(name)?;
        let f = &self.files[e.file];
        Ok(&f.map[f.data_start + e.info.start..f.data_start + e.info.end])
    }

    /// Any float tensor as f32 on the host.
    pub fn host_f32(&self, name: &str) -> Result<Vec<f32>> {
        let e = self.entry(name)?;
        let b = self.host_bytes(name)?;
        Ok(match e.info.dtype {
            Dtype::F32 => b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            Dtype::Bf16 => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| spl_nn::host::bf16_to_f32(u16::from_le_bytes(*c)))
                .collect(),
            Dtype::F16 => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| spl_nn::host::f16_to_f32(u16::from_le_bytes(*c)))
                .collect(),
            Dtype::F8E4M3 => b.iter().map(|&v| spl_nn::host::fp8_to_f32(v)).collect(),
            Dtype::U8 => bail!("weight {name} is U8, not a float tensor"),
        })
    }
}

/// Bytes per pinned staging buffer (two of them alternate).
const STAGE_BYTES: usize = 64 << 20;

/// Copies each file's device region into the arena: the mapping is read
/// into two pinned buffers in turn (with every core, so page faults of a
/// cold file overlap), each sent with an asynchronous copy while the other
/// fills.
fn upload_regions(
    nn: &Nn,
    files: &[Mapped],
    regions: &[(usize, usize, usize)],
    arena: &mut CudaSlice<u8>,
) -> Result<()> {
    use rayon::prelude::*;
    let total: usize = regions.iter().map(|r| r.2).sum();
    if total == 0 {
        return Ok(());
    }
    let n = STAGE_BYTES.min(total);
    let mut stages = Vec::new();
    for _ in 0..2 {
        // Pageable when pinning is refused; each refill waits for its
        // previous copy either way.
        let buf = super::hostbuf::HostBuf::<u8>::new(nn, n, "weight staging");
        let done = nn
            .ctx
            .new_event(None)
            .map_err(|e| anyhow!("create event: {e:?}"))?;
        stages.push((buf, done, false));
    }
    let mut turn = 0usize;
    for (f, &(base, data_start, len)) in files.iter().zip(regions) {
        let src = &f.map[data_start..data_start + len];
        for off in (0..len).step_by(n) {
            let piece = &src[off..(off + n).min(len)];
            let (buf, done, used) = &mut stages[turn % 2];
            if *used {
                done.synchronize()
                    .map_err(|e| anyhow!("wait for staging: {e:?}"))?;
            }
            let host = buf.as_mut_slice()?;
            host[..piece.len()]
                .par_chunks_mut(1 << 20)
                .zip(piece.par_chunks(1 << 20))
                .for_each(|(d, s)| d.copy_from_slice(s));
            let mut dst = arena.slice_mut(base + off..base + off + piece.len());
            nn.stream
                .memcpy_htod(&host[..piece.len()], &mut dst)
                .map_err(|e| anyhow!("upload {}: {e:?}", f.path.display()))?;
            done.record(&nn.stream)
                .map_err(|e| anyhow!("record event: {e:?}"))?;
            *used = true;
            turn += 1;
        }
    }
    // Release the staging memory now, not with the engine.
    nn.sync()?;
    drop(stages);
    Ok(())
}

/// The FP8 companion of a weight file: `x.safetensors` → `x_fp8.safetensors`.
pub fn fp8_companion(path: &Path) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("weights");
    path.with_file_name(format!("{stem}_fp8.safetensors"))
}

impl super::trunk::TrunkWeights for Weights {
    fn bf16(&self, name: &str) -> Result<CudaView<'_, u16>> {
        Weights::bf16(self, name)
    }

    fn f32(&self, name: &str) -> Result<CudaView<'_, f32>> {
        Weights::f32(self, name)
    }

    fn host_f32(&self, name: &str) -> Result<Vec<f32>> {
        Weights::host_f32(self, name)
    }

    fn has(&self, name: &str) -> bool {
        Weights::has(self, name)
    }

    fn fp8(&self, name: &str) -> Result<(CudaView<'_, u8>, CudaView<'_, f32>)> {
        Weights::fp8(self, name)
    }
}

impl super::heads::HeadSource for Weights {
    fn host_f32(&self, name: &str) -> Result<Vec<f32>> {
        Weights::host_f32(self, name)
    }

    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Weights::shape(self, name).map(<[usize]>::to_vec)
    }
}
