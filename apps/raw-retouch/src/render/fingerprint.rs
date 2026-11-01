//! Output fingerprints: blake3 of the display, search and full-resolution
//! renders and of the statistics, for a neutral and an edited parameter
//! set on the test samples. Run it before and after a renderer change that
//! must not alter existing output (a new setting at its default, say) and
//! compare the printed lines:
//! `gpu-test <checkout> render::fingerprint --ignored --nocapture`.

use super::testkit;
use super::*;

const SAMPLES: [&str; 2] = [
    "ILCE-7M3___DSC0009.ARW",
    "ILCE-7RM3__Sony_-_ILCE-7RM3_-_12bit_uncompressed__3_2_.ARW",
];

#[test]
fn hooks_are_spliced_in_ahead_of_render_core() {
    let at = |needle: &str| SOURCE.find(needle).unwrap_or_else(|| panic!("{needle}"));
    let core = at("__device__ Lab render_core(");
    for hook in [
        "void tone_curve_apply(",
        "void color_mixer_apply(",
        "void presence_apply(",
    ] {
        assert!(at(hook) > at(HOOKS) && at(hook) < core, "{hook}");
    }
}

/// Writes the renderer's PTX to the output directory, to diff two builds.
#[test]
#[ignore]
fn gpu_ptx_dump() {
    let ptx = spl_gpu::cudarc::nvrtc::compile_ptx(SOURCE.as_str()).unwrap();
    let path = testkit::out_dir().join("render.ptx");
    std::fs::write(&path, ptx.to_src()).unwrap();
    println!("PTX {}", path.display());
}

fn hash(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex()[..16].to_owned()
}

fn hash16(data: &[u16]) -> String {
    hash(
        &data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

#[test]
#[ignore]
fn gpu_fingerprint() {
    let gpu = Gpu::open().expect("gpu");
    for file in SAMPLES {
        let s = testkit::load(file);
        let mut e = RenderEngine::with_color(&gpu, &s.scene, s.color.clone()).expect("engine");
        let n = Params::neutral(s.scene.info.as_shot_temperature, s.scene.info.as_shot_tint);
        let edit = Params {
            temperature: n.temperature + 600.0,
            exposure: 0.4,
            contrast: 30.0,
            highlights: -70.0,
            shadows: 60.0,
            whites: 15.0,
            blacks: -20.0,
            vibrance: 40.0,
            saturation: -10.0,
            ..n
        };
        for (label, p) in [("neutral", n), ("edit", edit)] {
            let display = e.display(&p).unwrap();
            let search = e.search(&p).unwrap();
            let full = full::render(&gpu, &s.scene, &s.half, &p, &*s.color).unwrap();
            println!(
                "FP {file} {label}: display {} search {} full {}x{} {}",
                hash(&display.data),
                hash(&search.data),
                full.width,
                full.height,
                hash16(&full.data)
            );
        }
        let stats = e.stats(&[n, edit]).unwrap();
        println!(
            "FP {file} stats {}",
            hash(serde_json::to_string(&stats).unwrap().as_bytes())
        );
    }
}
