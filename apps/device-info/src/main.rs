//! Standalone: `device-info` prints the GPU report as JSON. No server, no database.

fn main() {
    let report = spl_gpu::probe();
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("report serializes")
    );
    if report.devices.is_empty() {
        std::process::exit(1);
    }
}
