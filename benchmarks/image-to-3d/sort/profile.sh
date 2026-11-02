#!/usr/bin/env bash
# Usage (on the GPU host): benchmarks/image-to-3d/sort/profile.sh <checkout> <out-name> [kernel-regex] [test-filter]
#
# Nsight Compute on one image-to-3d test binary run: builds the tests natively (release),
# then runs ncu from the CUDA devel container (spl-cuda-devel) as root with SYS_ADMIN (the
# driver restricts performance counters to admins), kernels compiled with -lineinfo.
# Writes ~/spl-work/out/k-sort/<out-name>.ncu-rep and <out-name>.txt (the details page of
# the SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, LaunchStats, WarpStateStats and
# SchedulerStats sections). Takes the GPU lock: run it when the card is otherwise idle.
#
# Default test: sort::bench::gpu_profile_pipeline_once (late-training garden workload).
set -euo pipefail
dir=$(cd "$1" && pwd)
name=$2
kernels=${3:-'regex:radix_hist32|onesweep_u32|onesweep_u16|scan_tiles|emit|tile_hist|bin_scan|count_tiles'}
filter=${4:-kernels::sort::bench::gpu_profile_pipeline_once}
export PATH=$HOME/.cargo/bin:$PATH
cd "$dir"
bin=$(cargo test --release -p spl-app-image-to-3d --no-run --message-format=json 2>/dev/null | python3 -c '
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d.get("executable") and d.get("profile", {}).get("test") and "spl_app_image_to_3d" in d["executable"]:
        print(d["executable"])
' | head -1)
[ -n "$bin" ] || { echo "no test binary (build failed?)"; exit 1; }
out=$HOME/spl-work/out/k-sort
mkdir -p "$out"
flock -x "$HOME/spl-work/gpu.lock" docker run --rm --gpus all --cap-add=SYS_ADMIN \
  -e HOME=/tmp -e SPL_DATASETS=/datasets -e SPL_NVRTC_LINEINFO=1 \
  -e SPL_SORT_PROFILE_SCALE="${SPL_SORT_PROFILE_SCALE:-10}" \
  -v "$dir:$dir" -v "$HOME/spl-work/out:/out" -v "$HOME/spl-work/datasets:/datasets:ro" \
  -w "$dir" spl-cuda-devel \
  ncu --target-processes all -k "$kernels" \
    --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
    --section LaunchStats --section WarpStateStats --section SchedulerStats \
    --section SourceCounters --import-source yes \
    -f -o "/out/k-sort/$name" \
    "$bin" "$filter" --exact --include-ignored --nocapture --test-threads=1
docker run --rm -v "$HOME/spl-work/out:/out" spl-cuda-devel chown "$(id -u):$(id -g)" "/out/k-sort/$name.ncu-rep"
docker run --rm --user "$(id -u):$(id -g)" -v "$HOME/spl-work/out:/out" spl-cuda-devel \
  ncu --import "/out/k-sort/$name.ncu-rep" --page details > "$out/$name.txt"
echo "wrote $out/$name.ncu-rep and $out/$name.txt"
