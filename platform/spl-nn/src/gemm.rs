//! Linear layers on cuBLASLt (loaded at runtime through `cudarc`):
//! `y[m][n] = act(x[m][k] · w[n][k]ᵀ · scales + bias[n])`, bf16 out.
//!
//! - **bf16:** bf16 in, fp32 accumulate; bias and GELU in the epilogue.
//! - **FP8 E4M3:** FP8 activations and weights, fp32 accumulate, scales per
//!   [`WeightScale`] and [`Fp8Scales::act`]:
//!   - per-tensor weight and activation scales (cuBLASLt's A/B scale
//!     pointers): bias and GELU stay in the epilogue;
//!   - per-output-channel weight scales as cuBLASLt's per-row alpha vector
//!     ([`WeightScale::Channel`]), where cuBLASLt accepts it. cuBLASLt 12.8
//!     rejects FP8 with an alpha vector on Ada, so there
//!   - [`WeightScale::Deferred`] leaves channel scales, per-token activation
//!     scales and bias to the consumer kernel, which reads the output anyway
//!     ([`crate::qkv`], [`crate::residual`],
//!     [`crate::norm::Norm::residual_layer_norm`],
//!     [`crate::norm::Norm::row_epilogue`]): no extra pass.
//!
//! Measured on the RTX 4070 SUPER with cuBLASLt 12.8.4 (`bench_gemm`, VGGT
//! 1B layer shapes, ≥ 22k tokens): bf16 73-77 TFLOPS, FP8 141-153 TFLOPS
//! (about 97% of the FP8 peak at the 2.7 GHz the card boosts to). The FP8
//! GELU epilogue is slow on Ada (fc1 at 137,800 tokens: 13.0 ms with GELU,
//! 7.7 ms without), so the FP8 MLP defers bias and GELU to
//! [`crate::norm::Norm::row_epilogue`], which also re-quantises for fc2.
//! PyTorch's per-token × per-channel FP8 path (`torch._scaled_mm` rowwise)
//! runs at 39-55 TFLOPS on the same card.
//!
//! cuBLASLt is column-major, so the call computes `yᵀ = w · xᵀ`: A is the
//! weight (`k × n`, transposed), B the activations (`k × m`), D the output
//! (`n × m`, i.e. `[m][n]` row-major). That is the TN layout FP8 requires,
//! and the bias and alpha vector run along D's rows, the output channels.
//!
//! The first call of each shape builds the descriptors and picks an
//! algorithm (cuBLASLt's heuristic, or with [`Tuning::Measure`] the fastest
//! of its top candidates, timed on the call's own buffers); later calls
//! only update pointers.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ffi::c_void;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use cudarc::cublaslt::result as lt;
use cudarc::cublaslt::sys;
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use spl_core::{Result, anyhow, bail};

use crate::{Cuda, Nn, Ptrs};

/// `cublasOperation_t` values (cuBLAS's header, not cuBLASLt's).
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;

/// Workspace for split-K and stream-K algorithms.
const WORKSPACE_BYTES: usize = 32 << 20;

/// How the first call of a shape picks its algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tuning {
    /// cuBLASLt's top heuristic choice.
    Heuristic,
    /// Time up to this many heuristic candidates (one warm-up and three
    /// timed runs each) and keep the fastest.
    Measure(usize),
}

/// Activation fused into the epilogue. cuBLASLt's GELU is the tanh form
/// (`GELU(approximate="tanh")`); the `bench`/`torch` tests measure its
/// distance from the erf form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Epilogue {
    None,
    Gelu,
}

/// GEMM dimensions: `m` tokens, `n` output channels, `k` input channels,
/// and row strides of `x` and `y` in elements (`k` and `n` when packed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub ldx: usize,
    pub ldy: usize,
}

impl GemmShape {
    pub fn packed(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            ldx: k,
            ldy: n,
        }
    }

    pub fn flops(&self) -> f64 {
        2.0 * self.m as f64 * self.n as f64 * self.k as f64
    }
}

/// How an FP8 weight scale reaches the output.
pub enum WeightScale<'a> {
    /// One scale for the whole matrix (`[1]`), applied in the epilogue
    /// through cuBLASLt's A scale pointer; bias and GELU stay fused.
    Tensor(&'a dyn DevicePtr<f32>),
    /// One scale per output channel (`[n]`), applied in the epilogue as
    /// cuBLASLt's per-row alpha vector so bias and GELU stay fused.
    /// cuBLASLt 12.8 rejects FP8 with an alpha vector on Ada (measured:
    /// `CUBLAS_STATUS_NOT_SUPPORTED` on sm_89, see `gpu_fp8_scale_modes`);
    /// [`Gemm::supports_channel_alpha`] tells whether this device can.
    Channel(&'a dyn DevicePtr<f32>),
    /// Per-channel scales applied by the consumer: the GEMM writes
    /// `act_scale · xq · wqᵀ` and rejects a bias or activation, which must
    /// come after the channel scale. The consumers in this crate
    /// ([`crate::qkv`], [`crate::residual`], [`crate::norm`]) take the
    /// channel scales, the per-token scales and the bias, so on Ada this
    /// costs no extra pass.
    Deferred,
}

/// FP8 dequantisation scales.
pub struct Fp8Scales<'a> {
    pub weight: WeightScale<'a>,
    /// Per-tensor activation scale, `[1]`, applied in the epilogue
    /// (cuBLASLt's B scale pointer); `None` when the activations carry
    /// per-token scales that the consumer applies.
    pub act: Option<&'a dyn DevicePtr<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum WScale {
    None,
    Tensor,
    Channel,
    Deferred,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    Bf16,
    Fp8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PlanKey {
    kind: Kind,
    shape: GemmShape,
    bias: bool,
    epilogue: Epilogue,
    wscale: WScale,
    act_scale: bool,
    align: u32,
}

struct Plan {
    desc: sys::cublasLtMatmulDesc_t,
    layouts: [sys::cublasLtMatrixLayout_t; 3],
    algo: sys::cublasLtMatmulAlgo_t,
}

// SAFETY: cuBLASLt descriptors are plain heap objects with no thread
// affinity; access is serialised by the plan map's mutex.
unsafe impl Send for Plan {}

impl Drop for Plan {
    fn drop(&mut self) {
        // SAFETY: created by this plan and destroyed once.
        unsafe {
            for l in self.layouts.into_iter().filter(|l| !l.is_null()) {
                let _ = lt::destroy_matrix_layout(l);
            }
            let _ = lt::destroy_matmul_desc(self.desc);
        }
    }
}

struct Handle(sys::cublasLtHandle_t);

// SAFETY: a cuBLASLt handle may be used from any thread as long as calls on
// it are not concurrent; `Gemm` only uses it under the plan mutex.
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: created in `Gemm::new` and destroyed once.
        unsafe {
            let _ = lt::destroy_handle(self.0);
        }
    }
}

pub struct Gemm {
    handle: Handle,
    stream: Arc<CudaStream>,
    workspace: CudaSlice<u8>,
    /// Device `0.0f32`, the beta argument in alpha-vector mode.
    zero: CudaSlice<f32>,
    plans: Mutex<HashMap<PlanKey, Plan>>,
    tuning: Tuning,
}

fn blas<T>(r: std::result::Result<T, lt::CublasError>, what: &str) -> Result<T> {
    r.map_err(|e| anyhow!("cuBLASLt {what}: {:?}", e.0))
}

impl Gemm {
    pub fn new(nn: &Nn, tuning: Tuning) -> Result<Self> {
        nn.ctx.bind_to_thread().cuda("bind CUDA context")?;
        // SAFETY: probing for the library before the first call keeps
        // cudarc from panicking when cuBLASLt is missing.
        if !unsafe { sys::is_culib_present() } {
            bail!("cuBLASLt (libcublasLt.so.12) not found; install the CUDA 12 runtime libraries");
        }
        let handle = Handle(blas(lt::create_handle(), "create handle")?);
        let workspace = nn
            .stream
            .alloc_zeros::<u8>(WORKSPACE_BYTES)
            .cuda("cuBLASLt workspace")?;
        let zero = nn.stream.alloc_zeros::<f32>(1).cuda("cuBLASLt beta")?;
        Ok(Self {
            handle,
            stream: nn.stream.clone(),
            workspace,
            zero,
            plans: Mutex::new(HashMap::new()),
            tuning,
        })
    }

    /// Whether cuBLASLt accepts FP8 with a per-output-channel alpha vector
    /// (with bias and GELU) on this GPU, i.e. whether
    /// [`WeightScale::Channel`] works. Asks the heuristic; runs nothing.
    pub fn supports_channel_alpha(&self) -> bool {
        let s = GemmShape::packed(256, 256, 256);
        let key = PlanKey {
            kind: Kind::Fp8,
            shape: s,
            bias: true,
            epilogue: Epilogue::Gelu,
            wscale: WScale::Channel,
            act_scale: true,
            align: 256,
        };
        // Pointers are only recorded in the descriptor, never dereferenced.
        let p = Pointers {
            x: 256,
            w: 256,
            y: 256,
            bias: 256,
            wscale: (WScale::Channel, 256),
            act_scale: 256,
        };
        self.plan(&key, &p, false).is_ok()
    }

    /// cuBLASLt's version number, e.g. 120804 for 12.8.4.
    pub fn version() -> usize {
        // SAFETY: a pure query.
        unsafe { sys::cublasLtGetVersion() }
    }

    /// bf16 linear layer: `y = act(x · wᵀ + bias)`.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_bf16(
        &self,
        nn: &Nn,
        x: &dyn DevicePtr<u16>,
        w: &dyn DevicePtr<u16>,
        bias: Option<&dyn DevicePtr<u16>>,
        epilogue: Epilogue,
        y: &mut dyn DevicePtrMut<u16>,
        shape: GemmShape,
    ) -> Result<()> {
        check_shape(&shape, 8)?;
        self.check_stream(nn)?;
        let mut ptrs = Ptrs::new(&nn.stream);
        let p = Pointers {
            x: ptrs.input(x, x_len(&shape), "gemm x")?,
            w: ptrs.input(w, shape.n * shape.k, "gemm w")?,
            y: ptrs.output(y, y_len(&shape), "gemm y")?,
            bias: ptrs.opt(bias, shape.n, "gemm bias")?,
            wscale: (WScale::None, 0),
            act_scale: 0,
        };
        self.run(Kind::Bf16, shape, epilogue, &p)?;
        drop(ptrs);
        Ok(())
    }

    /// FP8 E4M3 linear layer:
    /// `y = act(scales.act · scales.weight · (xq · wqᵀ) + bias)`, with the
    /// weight scale per tensor or per output channel (see [`WeightScale`]).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_fp8(
        &self,
        nn: &Nn,
        x: &dyn DevicePtr<u8>,
        w: &dyn DevicePtr<u8>,
        scales: &Fp8Scales<'_>,
        bias: Option<&dyn DevicePtr<u16>>,
        epilogue: Epilogue,
        y: &mut dyn DevicePtrMut<u16>,
        shape: GemmShape,
    ) -> Result<()> {
        if !nn.has_fp8() {
            bail!("FP8 GEMMs need sm_89 or newer");
        }
        // FP8 needs k and the FP8 leading dimensions to be multiples of 16.
        check_shape(&shape, 16)?;
        self.check_stream(nn)?;
        let mut ptrs = Ptrs::new(&nn.stream);
        let p = Pointers {
            x: ptrs.input(x, x_len(&shape), "gemm x")?,
            w: ptrs.input(w, shape.n * shape.k, "gemm w")?,
            y: ptrs.output(y, y_len(&shape), "gemm y")?,
            bias: ptrs.opt(bias, shape.n, "gemm bias")?,
            wscale: match scales.weight {
                WeightScale::Tensor(s) => (WScale::Tensor, ptrs.input(s, 1, "gemm weight scale")?),
                WeightScale::Channel(s) => (
                    WScale::Channel,
                    ptrs.input(s, shape.n, "gemm weight scales")?,
                ),
                WeightScale::Deferred => (WScale::Deferred, 0),
            },
            act_scale: ptrs.opt(scales.act, 1, "gemm activation scale")?,
        };
        if p.wscale.0 == WScale::Deferred && (p.bias != 0 || epilogue != Epilogue::None) {
            bail!("gemm: deferred channel scales leave bias and activation to the consumer");
        }
        self.run(Kind::Fp8, shape, epilogue, &p)?;
        drop(ptrs);
        Ok(())
    }

    /// GEMMs run on the stream the handle was made for; buffers are ordered
    /// against `nn.stream`, so they must be the same.
    fn check_stream(&self, nn: &Nn) -> Result<()> {
        if !Arc::ptr_eq(&nn.stream, &self.stream) {
            bail!("gemm: called with a different Nn stream than Gemm::new");
        }
        Ok(())
    }

    fn run(&self, kind: Kind, shape: GemmShape, epilogue: Epilogue, p: &Pointers) -> Result<()> {
        if shape.m == 0 || shape.n == 0 {
            return Ok(());
        }
        self.stream
            .context()
            .bind_to_thread()
            .cuda("bind CUDA context")?;
        let key = PlanKey {
            kind,
            shape,
            bias: p.bias != 0,
            epilogue,
            wscale: p.wscale.0,
            act_scale: p.act_scale != 0,
            align: [p.x, p.w, p.y]
                .iter()
                .map(|&a| 1u32 << a.trailing_zeros().min(8))
                .min()
                .unwrap_or(256),
        };
        let mut plans = self
            .plans
            .lock()
            .map_err(|_| anyhow!("GEMM plan lock poisoned"))?;
        if let Entry::Vacant(slot) = plans.entry(key) {
            slot.insert(self.plan(&key, p, true)?);
        }
        let plan = &plans[&key];
        self.launch(plan, &key, p, &plan.algo)
    }

    /// Builds descriptors and chooses an algorithm for `key`; `measure`
    /// allows timing candidates on the buffers in `p`.
    fn plan(&self, key: &PlanKey, p: &Pointers, measure: bool) -> Result<Plan> {
        let s = &key.shape;
        let (ab_type, a_ld) = match key.kind {
            Kind::Bf16 => (sys::cudaDataType_t::CUDA_R_16BF, s.k),
            Kind::Fp8 => (sys::cudaDataType_t::CUDA_R_8F_E4M3, s.k),
        };
        let desc = blas(
            lt::create_matmul_desc(
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cudaDataType_t::CUDA_R_32F,
            ),
            "matmul desc",
        )?;
        let mut plan = Plan {
            desc,
            layouts: [std::ptr::null_mut(); 3],
            algo: sys::cublasLtMatmulAlgo_t { data: [0; 8] },
        };
        plan.layouts[0] = blas(
            lt::create_matrix_layout(ab_type, s.k as u64, s.n as u64, a_ld as i64),
            "layout A",
        )?;
        plan.layouts[1] = blas(
            lt::create_matrix_layout(ab_type, s.k as u64, s.m as u64, s.ldx as i64),
            "layout B",
        )?;
        plan.layouts[2] = blas(
            lt::create_matrix_layout(
                sys::cudaDataType_t::CUDA_R_16BF,
                s.n as u64,
                s.m as u64,
                s.ldy as i64,
            ),
            "layout D",
        )?;
        set_desc(desc, Attr::CUBLASLT_MATMUL_DESC_TRANSA, &CUBLAS_OP_T)?;
        set_desc(desc, Attr::CUBLASLT_MATMUL_DESC_TRANSB, &CUBLAS_OP_N)?;
        let epi = match (key.bias, key.epilogue) {
            (false, Epilogue::None) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
            (true, Epilogue::None) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
            (false, Epilogue::Gelu) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU,
            (true, Epilogue::Gelu) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
        };
        set_desc(desc, Attr::CUBLASLT_MATMUL_DESC_EPILOGUE, &(epi as u32))?;
        if key.bias {
            set_desc(
                desc,
                Attr::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                &(sys::cudaDataType_t::CUDA_R_16BF as i32),
            )?;
        }
        if key.wscale == WScale::Channel {
            set_desc(
                desc,
                Attr::CUBLASLT_MATMUL_DESC_POINTER_MODE,
                &(sys::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO
                    as i32),
            )?;
        }
        self.set_pointers(&plan, key, p)?;

        let pref = blas(lt::create_matmul_pref(), "matmul preference")?;
        let pref = Pref(pref);
        set_pref(
            pref.0,
            PrefAttr::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &(WORKSPACE_BYTES as u64),
        )?;
        for a in [
            PrefAttr::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
            PrefAttr::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            PrefAttr::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
            PrefAttr::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
        ] {
            set_pref(pref.0, a, &key.align)?;
        }
        let want = match self.tuning {
            Tuning::Measure(n) if measure => n.clamp(1, 32),
            _ => 1,
        };
        // SAFETY: the result structs are plain data that cuBLASLt fills.
        let mut results: Vec<sys::cublasLtMatmulHeuristicResult_t> =
            vec![unsafe { std::mem::zeroed() }; want];
        let mut found = 0;
        // SAFETY: descriptors are live; `results` holds `want` entries.
        unsafe {
            sys::cublasLtMatmulAlgoGetHeuristic(
                self.handle.0,
                desc,
                plan.layouts[0],
                plan.layouts[1],
                plan.layouts[2],
                plan.layouts[2],
                pref.0,
                want as i32,
                results.as_mut_ptr(),
                &mut found,
            )
        }
        .result()
        .map_err(|e| {
            let hint = if key.wscale == WScale::Channel {
                " (per-channel alpha vectors with FP8 are unsupported on this GPU; use WeightScale::Deferred)"
            } else {
                ""
            };
            anyhow!("cuBLASLt heuristic for {key:?}: {:?}{hint}", e.0)
        })?;
        let candidates: Vec<_> = results[..found.max(0) as usize]
            .iter()
            .filter(|r| r.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS)
            .filter(|r| r.workspaceSize <= WORKSPACE_BYTES)
            .map(|r| r.algo)
            .collect();
        if candidates.is_empty() {
            bail!("cuBLASLt has no algorithm for {key:?}");
        }
        plan.algo = if candidates.len() == 1 {
            candidates[0]
        } else {
            self.fastest(&plan, key, p, &candidates)?
        };
        Ok(plan)
    }

    fn fastest(
        &self,
        plan: &Plan,
        key: &PlanKey,
        p: &Pointers,
        candidates: &[sys::cublasLtMatmulAlgo_t],
    ) -> Result<sys::cublasLtMatmulAlgo_t> {
        let ctx = self.stream.context();
        let start = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .cuda("event")?;
        let end = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .cuda("event")?;
        let mut best = (f32::INFINITY, candidates[0]);
        for algo in candidates {
            // Some candidates fail at launch for this shape; skip them.
            if self.launch(plan, key, p, algo).is_err() {
                continue;
            }
            start.record(&self.stream).cuda("record")?;
            for _ in 0..3 {
                self.launch(plan, key, p, algo)?;
            }
            end.record(&self.stream).cuda("record")?;
            let ms = start.elapsed_ms(&end).cuda("elapsed")?;
            if ms < best.0 {
                best = (ms, *algo);
            }
        }
        Ok(best.1)
    }

    fn set_pointers(&self, plan: &Plan, key: &PlanKey, p: &Pointers) -> Result<()> {
        if key.bias {
            set_desc(plan.desc, Attr::CUBLASLT_MATMUL_DESC_BIAS_POINTER, &p.bias)?;
        }
        if key.act_scale {
            // B holds the activations.
            set_desc(
                plan.desc,
                Attr::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &p.act_scale,
            )?;
        }
        if key.wscale == WScale::Tensor {
            set_desc(
                plan.desc,
                Attr::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                &p.wscale.1,
            )?;
        }
        Ok(())
    }

    fn launch(
        &self,
        plan: &Plan,
        key: &PlanKey,
        p: &Pointers,
        algo: &sys::cublasLtMatmulAlgo_t,
    ) -> Result<()> {
        self.set_pointers(plan, key, p)?;
        let one = 1.0f32;
        let zero = 0.0f32;
        let (zero_dev, _g) = self.zero.device_ptr(&self.stream);
        let (ws, _gw) = self.workspace.device_ptr(&self.stream);
        let (alpha, beta): (*const c_void, *const c_void) = match key.wscale {
            WScale::Channel => (p.wscale.1 as *const c_void, zero_dev as *const c_void),
            _ => ((&one as *const f32).cast(), (&zero as *const f32).cast()),
        };
        // SAFETY: descriptors are live and match the buffers, which were
        // checked for size; the workspace holds WORKSPACE_BYTES.
        unsafe {
            lt::matmul(
                self.handle.0,
                plan.desc,
                alpha,
                beta,
                p.w as *const c_void,
                plan.layouts[0],
                p.x as *const c_void,
                plan.layouts[1],
                p.y as *const c_void,
                plan.layouts[2],
                p.y as *mut c_void,
                plan.layouts[2],
                algo,
                ws as *mut c_void,
                WORKSPACE_BYTES,
                self.stream.cu_stream() as sys::cudaStream_t,
            )
        }
        .map_err(|e| anyhow!("cuBLASLt matmul {key:?}: {:?}", e.0))
    }
}

struct Pointers {
    x: u64,
    w: u64,
    y: u64,
    bias: u64,
    /// Weight scale mode and its pointer (0 when none).
    wscale: (WScale, u64),
    act_scale: u64,
}

struct Pref(sys::cublasLtMatmulPreference_t);

impl Drop for Pref {
    fn drop(&mut self) {
        // SAFETY: created in `Gemm::plan` and destroyed once.
        unsafe {
            let _ = lt::destroy_matmul_pref(self.0);
        }
    }
}

type Attr = sys::cublasLtMatmulDescAttributes_t;
type PrefAttr = sys::cublasLtMatmulPreferenceAttributes_t;

fn set_desc<T>(desc: sys::cublasLtMatmulDesc_t, attr: Attr, v: &T) -> Result<()> {
    // SAFETY: `v` is a live value of the attribute's documented type.
    blas(
        unsafe {
            lt::set_matmul_desc_attribute(desc, attr, (v as *const T).cast(), size_of::<T>())
        },
        &format!("set {attr:?}"),
    )
}

fn set_pref<T>(pref: sys::cublasLtMatmulPreference_t, attr: PrefAttr, v: &T) -> Result<()> {
    // SAFETY: `v` is a live value of the attribute's documented type.
    blas(
        unsafe {
            lt::set_matmul_pref_attribute(pref, attr, (v as *const T).cast(), size_of::<T>())
        },
        &format!("set {attr:?}"),
    )
}

fn x_len(s: &GemmShape) -> usize {
    if s.m == 0 { 0 } else { (s.m - 1) * s.ldx + s.k }
}

fn y_len(s: &GemmShape) -> usize {
    if s.m == 0 { 0 } else { (s.m - 1) * s.ldy + s.n }
}

fn check_shape(s: &GemmShape, mult: usize) -> Result<()> {
    if s.k == 0
        || !s.k.is_multiple_of(mult)
        || !s.ldx.is_multiple_of(mult)
        || !s.n.is_multiple_of(8)
        || !s.ldy.is_multiple_of(8)
    {
        bail!("gemm: k and ldx must be multiples of {mult} and n, ldy of 8: {s:?}");
    }
    if s.ldx < s.k || s.ldy < s.n {
        bail!("gemm: leading dimensions smaller than the rows: {s:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_are_validated() {
        assert!(check_shape(&GemmShape::packed(689, 3072, 1024), 16).is_ok());
        assert!(check_shape(&GemmShape::packed(1, 8, 8), 8).is_ok());
        assert!(
            check_shape(&GemmShape::packed(4, 8, 24), 16).is_err(),
            "FP8 k % 16"
        );
        assert!(
            check_shape(&GemmShape::packed(4, 12, 16), 16).is_err(),
            "n % 8"
        );
        let s = GemmShape {
            ldy: 4,
            ..GemmShape::packed(4, 8, 16)
        };
        assert!(check_shape(&s, 16).is_err());
        assert_eq!(x_len(&GemmShape::packed(3, 8, 16)), 48);
        assert_eq!(GemmShape::packed(2, 3, 4).flops(), 48.0);
    }
}
