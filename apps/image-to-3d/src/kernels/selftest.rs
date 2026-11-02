//! GPU self-tests of the shared headers and the skeleton: every unit compiles
//! header-free under NVRTC for this device and exports its kernels (every
//! variant), struct layouts agree between Rust and CUDA, the PTX helpers and
//! Philox behave, the store and buffers round-trip and match the VRAM model.
//! Owner: preprocess (E1). GPU tests are `#[ignore]`; run them on the GPU host
//! with `apps/image-to-3d/scripts/gpu-test.sh <checkout> selftest`.

use std::mem::{offset_of, size_of};

use spl_gpu::cudarc::driver::{LaunchConfig, PushKernelArg};

use super::adam::AdamIterPlan;
use super::*;
use crate::frame::{CamBase, FrameCam, FrameParams, Intr, IterParams};
use crate::gpu::{CudaFile, Defines, compile_log};
use crate::model::{Gaussians, Grads, Group};
use crate::testkit::synthetic::{SyntheticOptions, synthetic_scene};
use crate::testkit::timing::{bench, gpu_snapshot};

const SELFTEST: KernelSource = KernelSource {
    module: "selftest",
    files: &[
        crate::gpu::COMMON_CUH,
        crate::gpu::SPLAT_COMMON_CUH,
        CudaFile {
            name: "selftest.cu",
            text: include_str!("../../cuda/selftest.cu"),
        },
    ],
    kernels: &["layout_probe", "philox_probe", "half_probe", "ptx_probe"],
};

/// What `layout_probe` writes, in order, as Rust sees it.
fn layout_fields() -> Vec<(&'static str, usize)> {
    vec![
        ("sizeof CamBase", size_of::<CamBase>()),
        ("CamBase.intr_id", offset_of!(CamBase, intr_id)),
        ("CamBase.width", offset_of!(CamBase, width)),
        ("sizeof Intr", size_of::<Intr>()),
        ("sizeof FrameParams", size_of::<FrameParams>()),
        ("FrameParams.bg", offset_of!(FrameParams, bg)),
        ("FrameParams.eps2d", offset_of!(FrameParams, eps2d)),
        ("FrameParams.flags", offset_of!(FrameParams, flags)),
        (
            "FrameParams.adam_lr_scale",
            offset_of!(FrameParams, adam_lr_scale),
        ),
        ("FrameParams.render_w", offset_of!(FrameParams, render_w)),
        ("sizeof FrameCam", size_of::<FrameCam>()),
        ("FrameCam.fx", offset_of!(FrameCam, fx)),
        ("FrameCam.campos", offset_of!(FrameCam, campos)),
        ("FrameCam.near_z", offset_of!(FrameCam, near_z)),
        ("FrameCam.lim_x0", offset_of!(FrameCam, lim_x0)),
        ("FrameCam.width", offset_of!(FrameCam, width)),
        ("FrameCam.image_index", offset_of!(FrameCam, image_index)),
        ("FrameCam.intr_id", offset_of!(FrameCam, intr_id)),
        ("sizeof AdamIterPlan", size_of::<AdamIterPlan>()),
        (
            "AdamIterPlan.bc2_sqrt_rcp",
            offset_of!(AdamIterPlan, bc2_sqrt_rcp),
        ),
        ("AdamIterPlan.due_mask", offset_of!(AdamIterPlan, due_mask)),
        (
            "AdamIterPlan.elem_prefix",
            offset_of!(AdamIterPlan, elem_prefix),
        ),
        ("sizeof IterParams", size_of::<IterParams>()),
        (
            "IterParams.target_offset",
            offset_of!(IterParams, target_offset),
        ),
        ("IterParams.target_w", offset_of!(IterParams, target_w)),
        ("IterParams.res_scale", offset_of!(IterParams, res_scale)),
        ("IterParams.means_lr", offset_of!(IterParams, means_lr)),
        (
            "IterParams.noise_scale",
            offset_of!(IterParams, noise_scale),
        ),
        ("IterParams.seed", offset_of!(IterParams, seed)),
        ("IterParams.adam", offset_of!(IterParams, adam)),
        ("sizeof SplatRec", size_of::<SplatRec>()),
        ("SplatRec.ca", offset_of!(SplatRec, ca)),
        ("SplatRec.o", offset_of!(SplatRec, o)),
        ("SplatRec.r", offset_of!(SplatRec, r)),
        ("SplatRec.pthr", offset_of!(SplatRec, pthr)),
        ("SplatRec.z", offset_of!(SplatRec, z)),
        ("SplatRec.flags", offset_of!(SplatRec, flags)),
    ]
}

#[test]
fn layout_probe_lists_one_entry_per_rust_field() {
    let src = include_str!("../../cuda/selftest.cu");
    let body = &src[src.find("layout_probe").unwrap()..src.find("philox_probe").unwrap()];
    let entries = body.matches("OFF(").count() + body.matches("(uint32_t)sizeof(").count();
    assert_eq!(
        entries,
        layout_fields().len(),
        "selftest.cu layout_probe vs layout_fields()"
    );
}

fn gpu() -> Gpu {
    Gpu::open().expect("open GPU")
}

/// The variants worth compiling: the default, each knob moved once.
fn variants() -> Vec<RasterVariant> {
    let d = RasterVariant::default();
    vec![
        d,
        RasterVariant { tile_h: 12, ..d },
        RasterVariant { chain_k: 1, ..d },
        RasterVariant { chain_k: 8, ..d },
        RasterVariant {
            ckpt_f16: true,
            ..d
        },
        RasterVariant { rec32: true, ..d },
        RasterVariant { extra_ch: 1, ..d },
        RasterVariant { extra_ch: 4, ..d },
        RasterVariant {
            blend: BlendSemantics::FasterGs,
            ..d
        },
        RasterVariant {
            deterministic: true,
            ..d
        },
    ]
}

#[test]
#[ignore]
fn gpu_every_unit_compiles_and_exports_its_kernels() {
    let gpu = gpu();
    println!(
        "device arch {} with {} SMs",
        gpu.arch(),
        gpu.multiprocessors()
    );
    let mut n = 0;
    for src in SOURCES.iter().chain([&SELFTEST]) {
        let k = gpu
            .kernels(src, &Defines::new())
            .unwrap_or_else(|e| panic!("{}: {e:#}", src.module));
        n += k.names().count();
    }
    let bf16 = adam::AdamKernels::load(&gpu, true).expect("adam bf16 moments variant");
    drop(bf16);
    for v in variants() {
        SplatKernels::load(&gpu, v).unwrap_or_else(|e| panic!("{v:?}: {e:#}"));
    }
    for r in compile_log() {
        println!("nvrtc {:<11} {:>7.1} ms  {}", r.module, r.ms, r.defines);
    }
    let total: f64 = compile_log().iter().map(|r| r.ms).sum();
    println!(
        "{n} kernels in {} units, {} compilations, {total:.0} ms",
        SOURCES.len() + 1,
        compile_log().len()
    );
    // The cache answers the second request without compiling.
    let before = compile_log().len();
    SplatKernels::load(&gpu, RasterVariant::default()).unwrap();
    assert_eq!(compile_log().len(), before);
}

#[test]
#[ignore]
fn gpu_layout_probe_matches_rust() {
    let gpu = gpu();
    let f = gpu
        .kernels(&SELFTEST, &Defines::new())
        .unwrap()
        .get("layout_probe")
        .unwrap();
    let fields = layout_fields();
    let mut out = gpu.stream.alloc_zeros::<u32>(fields.len()).unwrap();
    let mut l = gpu.stream.launch_builder(&f);
    l.arg(&mut out);
    // SAFETY: layout_probe(uint32_t* out) writes fields.len() words.
    unsafe { l.launch(LaunchConfig::for_num_elems(1)) }.unwrap();
    let got = gpu.stream.clone_dtoh(&out).unwrap();
    for ((name, rust), cuda) in fields.iter().zip(&got) {
        assert_eq!(*rust, *cuda as usize, "{name}");
    }
    println!("{} layout fields agree", fields.len());
}

#[test]
#[ignore]
fn gpu_philox_matches_cpu() {
    let gpu = gpu();
    let f = gpu
        .kernels(&SELFTEST, &Defines::new())
        .unwrap()
        .get("philox_probe")
        .unwrap();
    let n = 4096usize;
    let ctr: Vec<[u32; 4]> = (0..n as u32)
        .map(|i| {
            [
                i,
                i.wrapping_mul(2_654_435_761),
                0xdead_beef ^ i,
                u32::MAX - i,
            ]
        })
        .collect();
    let key: Vec<[u32; 2]> = (0..n as u32).map(|i| [i ^ 0x1234_5678, !i]).collect();
    let flat_c: Vec<u32> = ctr.iter().flatten().copied().collect();
    let flat_k: Vec<u32> = key.iter().flatten().copied().collect();
    let s = &gpu.stream;
    let (dc, dk) = (
        s.clone_htod(&flat_c).unwrap(),
        s.clone_htod(&flat_k).unwrap(),
    );
    let mut out = s.alloc_zeros::<u32>(4 * n).unwrap();
    let nn = n as u32;
    let mut l = s.launch_builder(&f);
    l.arg(&dc).arg(&dk).arg(&nn).arg(&mut out);
    // SAFETY: philox_probe(const uint4*, const uint2*, uint32_t n, uint4* out), n elements each.
    unsafe { l.launch(LaunchConfig::for_num_elems(nn)) }.unwrap();
    let got = s.clone_dtoh(&out).unwrap();
    for i in 0..n {
        let want = crate::rng::philox4x32_10(ctr[i], key[i]);
        assert_eq!(&got[4 * i..4 * i + 4], &want, "counter {i}");
    }
}

/// bf16 round-to-nearest-even of a finite f32, as bits.
fn bf16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

#[test]
#[ignore]
fn gpu_half_and_bfloat16_conversions() {
    let gpu = gpu();
    let f = gpu
        .kernels(&SELFTEST, &Defines::new())
        .unwrap()
        .get("half_probe")
        .unwrap();
    let p = crate::rng::Philox::new(9);
    let n = 1 << 16;
    let x: Vec<f32> = (0..n as u32)
        .map(|i| {
            (crate::rng::u01_open(p.draw4(i, 0, 0)[0]) - 0.5) * 2f32.powi((i % 24) as i32 - 12)
        })
        .collect();
    // Same x repeated with random words: SR must be unbiased.
    let rand: Vec<u32> = (0..n as u32).map(|i| p.draw4(i, 1, 0)[0]).collect();
    let s = &gpu.stream;
    let (dx, dr) = (s.clone_htod(&x).unwrap(), s.clone_htod(&rand).unwrap());
    let mut f16 = s.alloc_zeros::<u16>(n).unwrap();
    let mut rn = s.alloc_zeros::<u16>(n).unwrap();
    let mut sr = s.alloc_zeros::<u16>(n).unwrap();
    let mut back = s.alloc_zeros::<f32>(n).unwrap();
    let nn = n as u32;
    let mut l = s.launch_builder(&f);
    l.arg(&dx)
        .arg(&dr)
        .arg(&nn)
        .arg(&mut f16)
        .arg(&mut rn)
        .arg(&mut sr)
        .arg(&mut back);
    // SAFETY: half_probe(x, rand, n, f16, bf16_rn, bf16_sr, f16_back), n elements each.
    unsafe { l.launch(LaunchConfig::for_num_elems(nn)) }.unwrap();
    let (rn, sr, back) = (
        s.clone_dtoh(&rn).unwrap(),
        s.clone_dtoh(&sr).unwrap(),
        s.clone_dtoh(&back).unwrap(),
    );
    for i in 0..n {
        assert_eq!(rn[i], bf16_rne(x[i]), "bf16 RNE of {}", x[i]);
        let want = ((x[i].to_bits() + (rand[i] & 0xffff)) >> 16) as u16;
        assert_eq!(sr[i], want, "bf16 SR of {} with {:#x}", x[i], rand[i]);
        if x[i].abs() > 1e-4 {
            assert!(
                (back[i] - x[i]).abs() <= x[i].abs() * 2f32.powi(-11),
                "f16 {}",
                x[i]
            );
        }
    }
    // Unbiasedness: one value between two bf16 neighbours, 65536 draws.
    let v = 1.0f32 + 3.0 * 2f32.powi(-10);
    let up_fraction = rand
        .iter()
        .filter(|&&r| ((v.to_bits() + (r & 0xffff)) >> 16) as u16 != bf16_rne(1.0))
        .count() as f64
        / n as f64;
    let exact = (v.to_bits() & 0xffff) as f64 / 65536.0;
    assert!(
        (up_fraction - exact).abs() < 0.01,
        "{up_fraction} vs {exact}"
    );
}

#[test]
#[ignore]
fn gpu_ptx_helpers() {
    let gpu = gpu();
    let f = gpu
        .kernels(&SELFTEST, &Defines::new())
        .unwrap()
        .get("ptx_probe")
        .unwrap();
    let s = &gpu.stream;
    let input: Vec<f32> = (0..128).map(|i| i as f32 * 0.5).collect();
    let din = s.clone_htod(&input).unwrap();
    let mut dout = s.alloc_zeros::<f32>(128).unwrap();
    let mut lanes = s.alloc_zeros::<u32>(32).unwrap();
    let mut flag = s.alloc_zeros::<u32>(2).unwrap();
    let mut l = s.launch_builder(&f);
    l.arg(&din).arg(&mut dout).arg(&mut lanes).arg(&mut flag);
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: ptx_probe(const float4* in, float4* out, uint32_t* lanes, uint32_t* flag), one warp.
    unsafe { l.launch(cfg) }.unwrap();
    assert_eq!(s.clone_dtoh(&dout).unwrap(), input, "cp.async round trip");
    let lanes = s.clone_dtoh(&lanes).unwrap();
    for (t, v) in lanes.iter().enumerate() {
        assert_eq!(
            (v & 0xff, v >> 8),
            (t as u32, t as u32),
            "lane id and lanemask_lt"
        );
    }
    assert_eq!(
        s.clone_dtoh(&flag).unwrap(),
        vec![7, 14],
        "release/acquire/volatile"
    );
}

#[test]
#[ignore]
fn gpu_store_round_trips() {
    let gpu = gpu();
    let scene = synthetic_scene(&SyntheticOptions {
        n: 3000,
        ..Default::default()
    });
    let mut g = Gaussians::alloc(&gpu, 3000, true).unwrap();
    assert_eq!(g.cap, 3072);
    g.upload(&gpu, &scene.gaussians).unwrap();
    let mut back = g.download(&gpu).unwrap();
    back.sh_degree = scene.gaussians.sh_degree;
    back.antialiased = scene.gaussians.antialiased;
    assert_eq!(back, scene.gaussians);
    // Planar layout on the device: coefficient j of Gaussian p at j·cap + p.
    let planar = gpu.stream.clone_dtoh(&g.sh_rest).unwrap();
    assert_eq!(planar[7 * 3072 + 11], scene.gaussians.sh_rest[11 * 45 + 7]);
    let mut grads = Grads::alloc(&gpu, 3000).unwrap();
    for grp in Group::ALL {
        grads.zero(&gpu, grp).unwrap();
        assert_eq!(grads.get(grp).len(), 3072 * grp.floats());
    }
}

#[test]
#[ignore]
fn gpu_raster_buffers_match_the_vram_model() {
    let gpu = gpu();
    let v = RasterVariant::default();
    let cap = Capacity::new(1 << 20, 1 << 22, 1297, 840, &v);
    gpu.sync().unwrap();
    let (free0, total) = gpu.mem_info().unwrap();
    let f = FrameBuffers::alloc(&gpu, cap, &v).unwrap();
    let out = RenderOut::alloc(&gpu, cap, &v, false).unwrap();
    let pg = PixelGrads::alloc(&gpu, cap, &v, true).unwrap();
    let g2d = Grad2d::alloc(&gpu, cap, &v, true).unwrap();
    gpu.sync().unwrap();
    let (free1, _) = gpu.mem_info().unwrap();
    let used = free0 - free1;
    let model = FrameBuffers::vram_bytes(cap, &v);
    println!(
        "N 1M, I 4M, 1297x840: model {:.1} MB ({:?}), measured {:.1} MB of {:.1} GB",
        model.total() as f64 / 1e6,
        model,
        used as f64 / 1e6,
        total as f64 / 1e9
    );
    let ratio = used as f64 / model.total() as f64;
    assert!((0.95..1.15).contains(&ratio), "measured/model = {ratio:.3}");
    drop((f, out, pg, g2d));
}

#[test]
#[ignore]
fn gpu_stage_stubs_name_their_owners() {
    let gpu = gpu();
    let v = RasterVariant::default();
    let cap = Capacity::new(4096, 65_536, 64, 48, &v);
    let mut r = raster::GpuRaster::new(&gpu, cap, v, false).unwrap();
    let mut g = Gaussians::alloc(&gpu, 4096, false).unwrap();
    let cams = crate::frame::CameraTable::upload(
        &gpu,
        &synthetic_scene(&SyntheticOptions::default()).views,
    )
    .unwrap();
    let p = IterParams::default();
    use raster::SplatRaster;
    let err = r
        .forward(
            &gpu,
            &mut g,
            &cams,
            raster::FrameSource::Eager(&p),
            RenderMode::Infer,
        )
        .unwrap_err();
    assert!(format!("{err}").contains("owner: raster (E3)"), "{err}");
    let err = crate::frame::FrameRing::new(&gpu, 8).err().unwrap();
    assert!(format!("{err}").contains("owner: trainer (E7)"), "{err}");
}

/// Device-to-device copy bandwidth through the CUDA-event timer (spec-raster
/// 0.1 measured 428 GB/s on the 4070 SUPER).
#[test]
#[ignore]
fn gpu_timer_measures_copy_bandwidth() {
    let gpu = gpu();
    let before = gpu_snapshot();
    let n = 128usize << 20; // 512 MiB of f32
    let src = gpu.stream.alloc_zeros::<f32>(n).unwrap();
    let mut dst = gpu.stream.alloc_zeros::<f32>(n).unwrap();
    let stats = bench(&gpu, 3, 20, || {
        gpu.stream
            .memcpy_dtod(&src, &mut dst)
            .map_err(|e| spl_core::anyhow!("{e:?}"))
    })
    .unwrap();
    let gbs = 2.0 * (n * 4) as f64 / (stats.median_ms * 1e-3) / 1e9;
    println!(
        "D2D 512 MiB: median {:.3} ms ({gbs:.0} GB/s read+write), {stats:?}",
        stats.median_ms
    );
    println!("nvidia-smi before {before:?} after {:?}", gpu_snapshot());
    assert!(gbs > 100.0, "{gbs} GB/s");
}
