# Benchmarks

Suites (`<suite>.toml`) and dataset manifests (Kodak, CLIC, DIV2K, …).
A suite is pipeline × dataset × parameter sweep; results land in the
database next to each paper's claimed numbers from its `paper.toml`.

Datasets are never committed. Manifests list URLs and blake3 hashes; the
host downloads them into `<data_dir>/datasets/`.
