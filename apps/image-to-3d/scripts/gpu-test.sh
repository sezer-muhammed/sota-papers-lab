#!/usr/bin/env bash
# Usage (on the GPU host): apps/image-to-3d/scripts/gpu-test.sh <checkout> [test-filter] [test args...]
#
# Builds spl-app-image-to-3d's tests natively (release) and runs every test
# binary in the CUDA runtime container (NVRTC available), single-threaded,
# ignored (GPU and dataset) tests included, output shown. Adapted from
# ~/spl-work/bin/gpu-test (raw-retouch) with these mounts:
#   <checkout>               at the same path
#   ~/spl-work/out           at /out        (tests write under /out/image-to-3d/)
#   ~/spl-work/datasets      at /datasets   (read-only; SPL_DATASETS=/datasets)
#
# SPL_GPU_LOCK=1 takes `flock -x ~/spl-work/gpu.lock` around the run (timing
# runs whose numbers get reported). SPL_NVRTC_LINEINFO=1 compiles kernels with
# -lineinfo (ncu source views).
set -euo pipefail
dir=$(cd "$1" && pwd)
shift
export PATH=$HOME/.cargo/bin:$PATH
cd "$dir"
cargo test --release -p spl-app-image-to-3d --no-run 2>&1 | grep -E "^(error|warning)|-->|Finished" || true
bins=$(cargo test --release -p spl-app-image-to-3d --no-run --message-format=json 2>/dev/null | python3 -c '
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d.get("executable") and d.get("profile", {}).get("test"):
        print(d["executable"])
')
[ -n "$bins" ] || { echo "no test binaries (build failed?)"; exit 1; }

image=${SPL_GPU_IMAGE:-sota-papers-lab-spl}
datasets=${SPL_DATASETS_HOST:-$HOME/spl-work/datasets}
mkdir -p "$HOME/spl-work/out/image-to-3d"
mounts=(-v "$dir:$dir" -v "$HOME/spl-work/out:/out")
[ -d "$datasets" ] && mounts+=(-v "$datasets:/datasets:ro")
lock=()
[ "${SPL_GPU_LOCK:-0}" = 1 ] && lock=(flock -x "$HOME/spl-work/gpu.lock")
for b in $bins; do
  "${lock[@]}" docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e RUST_BACKTRACE=1 -e SPL_DATASETS=/datasets \
    -e SPL_NVRTC_LINEINFO="${SPL_NVRTC_LINEINFO:-}" -e SPL_SORT_DUMP="${SPL_SORT_DUMP:-}" \
    "${mounts[@]}" -w "$dir" --entrypoint "" "$image" \
    "$b" --nocapture --test-threads=1 --include-ignored "$@"
done
