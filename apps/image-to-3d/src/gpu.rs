//! Device handle, NVRTC compilation with a process-wide cache, and the CUDA
//! plumbing every module shares. Owner: trainer (E7).
//!
//! Kernel sources are header-free (spec-raster 1.6): a compilation unit is
//! `common.cuh + splat_common.cuh [+ adam.cuh] + <module>.cu`, concatenated with
//! `#line` markers so NVRTC errors name the right file, and compiled for the
//! device's own virtual architecture with `--use_fast_math` and `-D` defines
//! for the compile-time variant. PTX is cached per (module, defines, arch) for
//! the life of the process, modules per context.
//!
//! The arch-aware helper belongs in `spl-gpu` once a second app needs it (the
//! spl-nn work adds one); until it lands there, this file is the app's copy.

use std::collections::{BTreeMap, HashMap};
use std::fmt::{Display, Write as _};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use spl_core::{Result, anyhow, bail};
use spl_gpu::cudarc::driver::sys::CUdevice_attribute;
use spl_gpu::cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use spl_gpu::cudarc::nvrtc::{CompileOptions, Ptx};

/// One `.cu`/`.cuh` file, embedded at build time.
#[derive(Debug, Clone, Copy)]
pub struct CudaFile {
    pub name: &'static str,
    pub text: &'static str,
}

pub const COMMON_CUH: CudaFile = CudaFile {
    name: "common.cuh",
    text: include_str!("../cuda/common.cuh"),
};
pub const SPLAT_COMMON_CUH: CudaFile = CudaFile {
    name: "splat_common.cuh",
    text: include_str!("../cuda/splat_common.cuh"),
};

/// One NVRTC compilation unit and the kernels it must export.
#[derive(Debug, Clone, Copy)]
pub struct KernelSource {
    /// Unique per unit; part of the cache key.
    pub module: &'static str,
    /// Concatenated in order.
    pub files: &'static [CudaFile],
    /// Every `extern "C" __global__` the Rust side loads.
    pub kernels: &'static [&'static str],
}

impl KernelSource {
    /// The text NVRTC compiles.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for f in self.files {
            let _ = writeln!(out, "#line 1 \"{}\"", f.name);
            out.push_str(f.text);
            out.push('\n');
        }
        out
    }
}

/// `-D` defines of a compile-time variant. Ordered, so equal sets give equal keys.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Defines(BTreeMap<String, String>);

impl Defines {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(mut self, name: &str, value: impl Display) -> Self {
        self.0.insert(name.to_string(), value.to_string());
        self
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.0.get(name).map(String::as_str)
    }

    /// `-DNAME=VALUE` options in name order.
    pub fn options(&self) -> Vec<String> {
        self.0.iter().map(|(k, v)| format!("-D{k}={v}")).collect()
    }

    /// Stable text form, used in cache keys and logs.
    pub fn key(&self) -> String {
        self.options().join(" ")
    }
}

/// The NVRTC virtual architecture for a compute capability.
pub fn nvrtc_arch(cc: (i32, i32)) -> &'static str {
    match cc {
        (8, 0) => "compute_80",
        (8, 6) => "compute_86",
        (8, 7) => "compute_87",
        (8, 9) => "compute_89",
        (9, _) => "compute_90",
        (10, _) => "compute_100",
        (12, _) => "compute_120",
        (8, _) => "compute_80",
        // Newer families JIT PTX of an older target through the driver.
        _ => "compute_90",
    }
}

/// NVRTC options for every unit: the device arch, fast math (gsplat and
/// Faster-GS both build with it; cudarc's `use_fast_math` only sets `--fmad`),
/// C++17, the variant's defines, and `-lineinfo` when `SPL_NVRTC_LINEINFO` is
/// set (for ncu source views).
pub fn compile_options(arch: &'static str, defines: &Defines) -> CompileOptions {
    let mut options = vec!["--use_fast_math".to_string(), "-std=c++17".to_string()];
    if std::env::var("SPL_NVRTC_LINEINFO").is_ok_and(|v| !v.is_empty() && v != "0") {
        options.push("-lineinfo".into());
    }
    options.extend(defines.options());
    CompileOptions {
        arch: Some(arch),
        options,
        ..Default::default()
    }
}

/// One compilation, as the cache recorded it.
#[derive(Debug, Clone)]
pub struct CompileRecord {
    pub module: &'static str,
    pub defines: String,
    pub arch: &'static str,
    pub ms: f64,
}

type PtxKey = (&'static str, String, &'static str);
/// Modules are per context: (context address, module, defines, arch).
type ModuleKey = (usize, &'static str, String, &'static str);

#[derive(Default)]
struct Cache {
    ptx: HashMap<PtxKey, Ptx>,
    modules: HashMap<ModuleKey, Arc<CudaModule>>,
    log: Vec<CompileRecord>,
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);
static CONTEXT: OnceLock<Mutex<Option<Arc<CudaContext>>>> = OnceLock::new();

/// Compiles `src` with `defines` for `arch`, or returns the cached PTX.
pub fn compile_ptx(src: &KernelSource, defines: &Defines, arch: &'static str) -> Result<Ptx> {
    let key = (src.module, defines.key(), arch);
    if let Some(p) = with_cache(|c| c.ptx.get(&key).cloned())? {
        return Ok(p);
    }
    // SAFETY: only probes whether the library can be loaded (cudarc would
    // panic on first use otherwise, e.g. on a host with only the driver).
    if !unsafe { spl_gpu::cudarc::nvrtc::sys::is_culib_present() } {
        bail!(
            "the NVRTC library (libnvrtc) was not found: run in the CUDA runtime container or install the CUDA runtime"
        );
    }
    let started = Instant::now();
    let ptx =
        spl_gpu::cudarc::nvrtc::compile_ptx_with_opts(src.text(), compile_options(arch, defines))
            .map_err(|e| {
            anyhow!(
                "NVRTC compile of {} [{}] for {arch} failed: {e:?}",
                src.module,
                defines.key()
            )
        })?;
    let record = CompileRecord {
        module: src.module,
        defines: defines.key(),
        arch,
        ms: started.elapsed().as_secs_f64() * 1e3,
    };
    with_cache(|c| {
        c.ptx.insert(key, ptx.clone());
        c.log.push(record);
    })?;
    Ok(ptx)
}

/// Every NVRTC compilation this process has done, oldest first.
pub fn compile_log() -> Vec<CompileRecord> {
    with_cache(|c| c.log.clone()).unwrap_or_default()
}

fn with_cache<T>(f: impl FnOnce(&mut Cache) -> T) -> Result<T> {
    let mut guard = CACHE
        .lock()
        .map_err(|_| anyhow!("NVRTC cache lock poisoned"))?;
    Ok(f(guard.get_or_insert_with(Cache::default)))
}

/// Device 0's context (shared by every run in the process), a stream of this
/// handle's own, and the device facts kernels are specialized on.
#[derive(Clone)]
pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    cc: (i32, i32),
    sms: u32,
}

impl Gpu {
    /// Device 0 with a new stream. Fails below compute capability 8.0 (the
    /// kernels use `cp.async` and bf16 conversions).
    pub fn open() -> Result<Self> {
        let ctx = {
            let mut slot = CONTEXT
                .get_or_init(|| Mutex::new(None))
                .lock()
                .map_err(|_| anyhow!("GPU context lock poisoned"))?;
            match &*slot {
                Some(ctx) => ctx.clone(),
                None => {
                    let ctx = spl_gpu::context(0)?;
                    *slot = Some(ctx.clone());
                    ctx
                }
            }
        };
        let stream = ctx.new_stream().cuda("create CUDA stream")?;
        Self::new(stream)
    }

    /// Works on `stream`.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        let ctx = stream.context().clone();
        let cc = ctx.compute_capability().cuda("compute capability")?;
        if cc < (8, 0) {
            bail!(
                "image-to-3d needs compute capability 8.0 or newer; this GPU is {}.{}",
                cc.0,
                cc.1
            );
        }
        let sms = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .cuda("SM count")? as u32;
        Ok(Self {
            ctx,
            stream,
            cc,
            sms,
        })
    }

    pub fn compute_capability(&self) -> (i32, i32) {
        self.cc
    }

    pub fn multiprocessors(&self) -> u32 {
        self.sms
    }

    pub fn arch(&self) -> &'static str {
        nvrtc_arch(self.cc)
    }

    /// Compiles (or reuses) `src` with `defines` and loads it into this context.
    pub fn module(&self, src: &KernelSource, defines: &Defines) -> Result<Arc<CudaModule>> {
        let arch = self.arch();
        let key = (
            Arc::as_ptr(&self.ctx) as usize,
            src.module,
            defines.key(),
            arch,
        );
        if let Some(m) = with_cache(|c| c.modules.get(&key).cloned())? {
            return Ok(m);
        }
        let ptx = compile_ptx(src, defines, arch)?;
        let module = self
            .ctx
            .load_module(ptx)
            .cuda(&format!("load CUDA module {}", src.module))?;
        with_cache(|c| c.modules.insert(key, module.clone()))?;
        Ok(module)
    }

    /// Loads every kernel `src` declares, in declaration order.
    pub fn kernels(&self, src: &KernelSource, defines: &Defines) -> Result<Kernels> {
        let module = self.module(src, defines)?;
        let mut functions = Vec::with_capacity(src.kernels.len());
        for &name in src.kernels {
            let f = module
                .load_function(name)
                .cuda(&format!("load kernel {}::{name}", src.module))?;
            functions.push((name, f));
        }
        Ok(Kernels(functions))
    }

    /// Waits for everything enqueued on this handle's stream.
    pub fn sync(&self) -> Result<()> {
        self.stream.synchronize().cuda("CUDA stream synchronize")
    }

    /// (free, total) device memory in bytes.
    pub fn mem_info(&self) -> Result<(u64, u64)> {
        let (free, total) = self.ctx.mem_get_info().cuda("cuMemGetInfo")?;
        Ok((free as u64, total as u64))
    }
}

/// The loaded kernels of one unit, by name.
pub struct Kernels(Vec<(&'static str, CudaFunction)>);

impl Kernels {
    pub fn get(&self, name: &str) -> Result<CudaFunction> {
        self.0
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, f)| f.clone())
            .ok_or_else(|| anyhow!("kernel {name} was not loaded"))
    }

    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.0.iter().map(|(n, _)| *n)
    }
}

/// The error every unimplemented stage returns until its owner fills it in.
pub fn not_implemented(what: &str, owner: &str) -> spl_core::Error {
    anyhow!(
        "image-to-3d: {what} is not implemented yet (owner: {owner}; see apps/image-to-3d/DESIGN.md)"
    )
}

/// `Result<T, DriverError>` and friends to `spl_core::Result` with context.
pub trait CudaResultExt<T> {
    fn cuda(self, what: &str) -> Result<T>;
}

impl<T, E: std::fmt::Debug> CudaResultExt<T> for std::result::Result<T, E> {
    fn cuda(self, what: &str) -> Result<T> {
        self.map_err(|e| anyhow!("{what}: {e:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_covers_ampere_ada_hopper() {
        assert_eq!(nvrtc_arch((8, 9)), "compute_89");
        assert_eq!(nvrtc_arch((8, 6)), "compute_86");
        assert_eq!(nvrtc_arch((9, 0)), "compute_90");
        assert_eq!(nvrtc_arch((12, 0)), "compute_120");
    }

    #[test]
    fn defines_are_ordered_and_render_as_options() {
        let a = Defines::new().set("TILE_H", 16).set("CHAIN_K", 4);
        let b = Defines::new().set("CHAIN_K", 4).set("TILE_H", 16);
        assert_eq!(a, b);
        assert_eq!(a.key(), "-DCHAIN_K=4 -DTILE_H=16");
        let opts = compile_options("compute_89", &a);
        assert!(opts.options.contains(&"--use_fast_math".to_string()));
        assert!(opts.options.contains(&"-DTILE_H=16".to_string()));
        assert_eq!(opts.arch, Some("compute_89"));
    }

    #[test]
    fn source_text_marks_each_file() {
        const SRC: KernelSource = KernelSource {
            module: "t",
            files: &[COMMON_CUH, SPLAT_COMMON_CUH],
            kernels: &[],
        };
        let text = SRC.text();
        assert!(text.starts_with("#line 1 \"common.cuh\"\n"));
        assert!(text.contains("#line 1 \"splat_common.cuh\"\n"));
        assert!(
            !text.lines().any(|l| l.trim_start().starts_with("#include")),
            "kernel sources must be header-free"
        );
    }
}
