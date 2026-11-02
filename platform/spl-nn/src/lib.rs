//! Transformer building blocks on CUDA for the native VGGT-Ω engine
//! (`reports/Image to 3D.md`, Section 3, Stage A items 2-4 and 7), built on
//! `cudarc`: kernels are compiled at runtime by NVRTC, GEMMs go through
//! cuBLASLt, and both libraries load lazily, so the crate compiles on
//! machines without CUDA.
//!
//! | Module | Does |
//! |---|---|
//! | [`gemm`] | cuBLASLt `y = act(x · wᵀ · scales + bias)` in bf16 and FP8 E4M3 (per-output-channel weight scales), bias and GELU in the epilogue, bf16 out |
//! | [`norm`] | LayerNorm fused with FP8 quantisation (per token or static), optionally preceded by the LayerScale residual update; the row epilogue (scales, bias, GELU, re-quantisation) for GEMM outputs |
//! | [`qkv`] | QKV prologue: split, QK-norm, 2D RoPE (VGGT and DINOv3 layouts), head-major layout for attention |
//! | [`attention`] | FlashAttention-2-style bf16 attention (`mma.sync` m16n8k16, `ldmatrix`, `cp.async` double buffering, swizzled shared memory, exp2 online softmax), batched over frames or over one long global sequence; an fp16-P·V variant runs at twice the tensor rate on GeForce Ada |
//! | [`residual`] | LayerScale, bias and residual add in one pass |
//! | [`host`] | bf16/FP8 conversions and CPU reference implementations (test oracles) |
//!
//! Data types on the device: bf16 is `u16`, FP8 E4M3 is `u8`, scales are
//! `f32`. Activations are row-major `[tokens][channels]`; weights are
//! `[out][in]` like `torch.nn.Linear`. Every call enqueues work on the
//! [`Nn`] stream and returns without synchronising.

pub mod attention;
pub mod gemm;
pub mod host;
pub mod norm;
pub mod qkv;
pub mod residual;
/// SageAttention-style quantised global attention: INT8 QKᵀ, FP8 PV,
/// smoothing and KV merging.
pub mod sage;

#[cfg(test)]
mod testkit;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub use cudarc;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, DevicePtr, DevicePtrMut, SyncOnDrop,
};
use cudarc::nvrtc::{CompileOptions, Ptx};
use spl_core::{Result, anyhow, bail};

/// Helpers every kernel source starts with: bf16 and FP8 packing, warp
/// reductions, GELU. Prepend it to sources compiled with [`Nn::module`].
pub const COMMON_CUH: &str = include_str!("../cuda/common.cuh");

/// Pointwise activation applied by the custom kernels (values match
/// `ACT_*` in cuda/common.cuh).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    None = 0,
    /// `torch.nn.GELU()`: `0.5 x (1 + erf(x / √2))`.
    GeluErf = 1,
    /// `GELU(approximate="tanh")`.
    GeluTanh = 2,
}

/// A CUDA stream plus the device facts and compiled modules the kernels
/// need. One per engine; cheap to share by reference.
pub struct Nn {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    cc: (i32, i32),
    sms: u32,
    modules: Mutex<HashMap<&'static str, Arc<CudaModule>>>,
}

/// PTX per (module name, NVRTC architecture), shared by every [`Nn`] in the
/// process: PTX does not depend on the context, and NVRTC takes about a
/// second for the attention kernels.
static PTX: OnceLock<Mutex<HashMap<(&'static str, &'static str), Ptx>>> = OnceLock::new();

impl Nn {
    /// Device 0 with a stream of its own.
    pub fn open() -> Result<Self> {
        let ctx = spl_gpu::context(0)?;
        let stream = ctx
            .new_stream()
            .map_err(|e| anyhow!("create CUDA stream: {e:?}"))?;
        Self::new(stream)
    }

    /// Works on `stream`. Fails below compute capability 8.0: the kernels
    /// use `cp.async`, `ldmatrix` and bf16 `mma.sync`.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        let ctx = stream.context().clone();
        let cc = ctx
            .compute_capability()
            .map_err(|e| anyhow!("compute capability: {e:?}"))?;
        if cc < (8, 0) {
            bail!(
                "spl-nn needs compute capability 8.0 or newer (bf16 tensor cores); this GPU is {}.{}",
                cc.0,
                cc.1
            );
        }
        let sms = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .map_err(|e| anyhow!("SM count: {e:?}"))? as u32;
        Ok(Self {
            ctx,
            stream,
            cc,
            sms,
            modules: Mutex::new(HashMap::new()),
        })
    }

    pub fn compute_capability(&self) -> (i32, i32) {
        self.cc
    }

    pub fn multiprocessors(&self) -> u32 {
        self.sms
    }

    /// FP8 E4M3 conversions and tensor-core FP8 need sm_89 (Ada) or newer.
    pub fn has_fp8(&self) -> bool {
        self.cc >= (8, 9)
    }

    /// The NVRTC virtual architecture for this device, so the driver JITs
    /// PTX that uses everything the GPU has.
    pub fn arch(&self) -> &'static str {
        nvrtc_arch(self.cc)
    }

    /// Compiles `source` (prefixed with [`COMMON_CUH`] by the caller when it
    /// needs it) for this device on first use and keeps the module. `name`
    /// must be unique per source.
    pub fn module(&self, name: &'static str, source: &str) -> Result<Arc<CudaModule>> {
        let mut cache = self
            .modules
            .lock()
            .map_err(|_| anyhow!("module cache lock poisoned"))?;
        if let Some(m) = cache.get(name) {
            return Ok(m.clone());
        }
        let arch = self.arch();
        let ptx = {
            let mut ptx_cache = PTX
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .map_err(|_| anyhow!("PTX cache lock poisoned"))?;
            match ptx_cache.get(&(name, arch)) {
                Some(p) => p.clone(),
                None => {
                    let opts = CompileOptions {
                        arch: Some(arch),
                        options: vec!["--std=c++17".into()],
                        ..Default::default()
                    };
                    let p = cudarc::nvrtc::compile_ptx_with_opts(source, opts)
                        .map_err(|e| anyhow!("NVRTC compile of {name} failed: {e:?}"))?;
                    ptx_cache.insert((name, arch), p.clone());
                    p
                }
            }
        };
        let m = self
            .ctx
            .load_module(ptx)
            .map_err(|e| anyhow!("load CUDA module {name}: {e:?}"))?;
        cache.insert(name, m.clone());
        Ok(m)
    }

    /// Loads the named kernels of a module.
    pub(crate) fn functions<const N: usize>(
        &self,
        module: &'static str,
        source: &str,
        names: [&str; N],
    ) -> Result<[CudaFunction; N]> {
        let m = self.module(module, source)?;
        let mut out = Vec::with_capacity(N);
        for n in names {
            out.push(
                m.load_function(n)
                    .map_err(|e| anyhow!("load kernel {n}: {e:?}"))?,
            );
        }
        out.try_into()
            .map_err(|_| anyhow!("kernel count mismatch in {module}"))
    }

    /// Waits for everything enqueued so far.
    pub fn sync(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow!("CUDA stream synchronize: {e:?}"))
    }
}

fn nvrtc_arch(cc: (i32, i32)) -> &'static str {
    match cc {
        (8, 0) => "compute_80",
        (8, 6) => "compute_86",
        (8, 7) => "compute_87",
        (8, 9) => "compute_89",
        (9, _) => "compute_90",
        (10, _) => "compute_100",
        (12, _) => "compute_120",
        (8, _) => "compute_80",
        // Newer families run PTX for older targets through the driver JIT.
        _ => "compute_90",
    }
}

/// `CudaContextExt`-style error mapping for cudarc results.
pub(crate) trait Cuda<T> {
    fn cuda(self, what: &str) -> Result<T>;
}

impl<T, E: std::fmt::Debug> Cuda<T> for std::result::Result<T, E> {
    fn cuda(self, what: &str) -> Result<T> {
        self.map_err(|e| anyhow!("{what}: {e:?}"))
    }
}

/// Raw device pointers for a kernel argument struct, holding cudarc's
/// stream-ordering guards until the launch has been enqueued.
pub(crate) struct Ptrs<'a> {
    stream: &'a CudaStream,
    guards: Vec<SyncOnDrop<'a>>,
}

impl<'a> Ptrs<'a> {
    pub(crate) fn new(stream: &'a CudaStream) -> Self {
        Self {
            stream,
            guards: Vec::new(),
        }
    }

    /// Reads `buf`; checks it holds at least `len` elements.
    pub(crate) fn input<T>(
        &mut self,
        buf: &'a dyn DevicePtr<T>,
        len: usize,
        what: &str,
    ) -> Result<u64> {
        if buf.len() < len {
            bail!("{what}: buffer holds {} elements, needs {len}", buf.len());
        }
        let (p, g) = buf.device_ptr(self.stream);
        self.guards.push(g);
        Ok(p)
    }

    /// Like [`Self::input`]; `None` becomes a null pointer.
    pub(crate) fn opt<T>(
        &mut self,
        buf: Option<&'a dyn DevicePtr<T>>,
        len: usize,
        what: &str,
    ) -> Result<u64> {
        match buf {
            Some(b) => self.input(b, len, what),
            None => Ok(0),
        }
    }

    /// Writes `buf`; checks it holds at least `len` elements.
    pub(crate) fn output<T>(
        &mut self,
        buf: &'a mut dyn DevicePtrMut<T>,
        len: usize,
        what: &str,
    ) -> Result<u64> {
        if buf.len() < len {
            bail!("{what}: buffer holds {} elements, needs {len}", buf.len());
        }
        let (p, g) = buf.device_ptr_mut(self.stream);
        self.guards.push(g);
        Ok(p)
    }

    pub(crate) fn opt_out<T>(
        &mut self,
        buf: Option<&'a mut dyn DevicePtrMut<T>>,
        len: usize,
        what: &str,
    ) -> Result<u64> {
        match buf {
            Some(b) => self.output(b, len, what),
            None => Ok(0),
        }
    }
}

/// Converts a dimension to the `i32` kernels take.
pub(crate) fn dim(v: usize, what: &str) -> Result<i32> {
    i32::try_from(v).map_err(|_| anyhow!("{what} = {v} does not fit in i32"))
}

#[cfg(test)]
mod lib_tests {
    use super::*;

    #[test]
    fn arch_covers_ampere_ada_hopper() {
        assert_eq!(nvrtc_arch((8, 9)), "compute_89");
        assert_eq!(nvrtc_arch((8, 6)), "compute_86");
        assert_eq!(nvrtc_arch((9, 0)), "compute_90");
        assert_eq!(nvrtc_arch((8, 8)), "compute_80");
    }

    #[test]
    fn common_header_is_self_contained() {
        assert!(!COMMON_CUH.contains("#include"));
        assert!(COMMON_CUH.contains("cvt.rn.satfinite.e4m3x2.f32"));
    }
}
