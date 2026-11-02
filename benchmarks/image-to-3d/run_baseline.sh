#!/usr/bin/env bash
# Usage: run_baseline.sh <tool> <scene> [--steps N] [--tag TAG] [--keep] [-- <extra tool args>]
#
# Runs one baseline tool on one scene of datasets.toml under the exclusive GPU
# lock (flock -x ~/spl-work/gpu.lock), then appends one JSON line to
# ~/spl-work/bench/results.jsonl with: wall time, the tool's own train time,
# peak VRAM and utilization from nvidia-smi polling (200 ms), PSNR / SSIM /
# LPIPS from gs-metrics on the held-out views, Gaussian count, versions.
# A failed run is recorded too (status "failed").
#
# Tools are adapters in tools/<tool>.sh (see README.md, "Adding a tool").
# --steps overrides the tool's iteration count (smoke tests); --keep keeps
# checkpoints and PLY files (deleted by default to save disk).
# Run directory: ~/spl-work/bench/runs/<UTC time>_<tool>_<scene>[_<tag>]/
set -euo pipefail

HERE=$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)
WORK=${SPL_WORK:-$HOME/spl-work}
BENCH=$WORK/bench
LOCK=$WORK/gpu.lock
export IMAGE=${SPL_GS_REF_IMAGE:-spl-gs-ref}

usage() { sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 2; }
[[ $# -ge 2 ]] || usage
tool=$1 scene=$2
shift 2
if [[ ! -x $HERE/tools/$tool.sh ]]; then
  echo "unknown tool '$tool'; tools: $(find "$HERE/tools" -name '*.sh' -printf '%f ' | sed 's/\.sh//g')" >&2
  exit 2
fi

# Take the GPU lock first; everything below runs while holding it.
if [[ -z ${SPL_BENCH_LOCKED:-} ]]; then
  mkdir -p "$BENCH"
  SPL_BENCH_LOCK_T0=$(date +%s.%N)
  export SPL_BENCH_LOCK_T0
  echo "waiting for $LOCK ..." >&2
  exec env SPL_BENCH_LOCKED=1 flock -x "$LOCK" "${BASH_SOURCE[0]}" "$tool" "$scene" "$@"
fi
lock_wait=$(python3 -c "import sys; print(round($(date +%s.%N) - float(sys.argv[1]), 2))" "$SPL_BENCH_LOCK_T0")

steps="" tag="" keep=0 extra=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --steps) steps=$2; shift 2 ;;
    --tag) tag=$2; shift 2 ;;
    --keep) keep=1; shift ;;
    --) shift; extra=("$@"); break ;;
    *) echo "unknown option $1" >&2; usage ;;
  esac
done

# Scene -> paths and protocol, from the manifest.
eval "$(python3 -I - "$HERE/datasets.toml" "$scene" <<'EOF'
import os, shlex, sys, tomllib
m = tomllib.load(open(sys.argv[1], "rb"))
sid = sys.argv[2]
if sid not in m["scenes"]:
    sys.exit(f"unknown scene {sid!r}; scenes: {' '.join(m['scenes'])}")
s = m["scenes"][sid]
root = os.path.expanduser(m["root"])
for k, v in {"SCENE_DIR": os.path.join(root, s["dataset"], sid), "IMAGES": s["images"],
             "DATASET": s["dataset"], "TEST_EVERY": m["protocol"]["test_every"]}.items():
    print(f"export {k}={shlex.quote(str(v))}")
EOF
)"
# shellcheck disable=SC2153 # IMAGES comes from the eval above
[[ -d $SCENE_DIR/$IMAGES ]] || { echo "missing $SCENE_DIR/$IMAGES: run install.sh --datasets" >&2; exit 1; }

stamp=$(date -u +%Y%m%dT%H%M%SZ)
export RUN_DIR=$BENCH/runs/${stamp}_${tool}_${scene}${tag:+_$tag}
export STEPS=$steps
mkdir -p "$RUN_DIR"
echo "run dir: $RUN_DIR" >&2

# GPU state before the run (other processes make contention visible).
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,clocks.max.sm,power.limit \
  --format=csv,noheader,nounits > "$RUN_DIR/gpu_before.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$RUN_DIR/gpu_apps_before.csv" || true

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu \
  --format=csv,noheader,nounits -lms 200 > "$RUN_DIR/smi.csv" &
smi_pid=$!
trap 'kill $smi_pid 2>/dev/null || true' EXIT

t0=$(date +%s.%N)
status=ok
"$HERE/tools/$tool.sh" "${extra[@]}" > "$RUN_DIR/adapter.log" 2>&1 || status=failed
t1=$(date +%s.%N)
kill $smi_pid 2>/dev/null || true
wait $smi_pid 2>/dev/null || true
[[ $status == ok ]] || { echo "adapter failed, tail of $RUN_DIR/adapter.log:" >&2; tail -n 20 "$RUN_DIR/adapter.log" >&2; }

m0=$(date +%s.%N)
if [[ $status == ok ]]; then
  "$HERE/bin/gs-metrics" "$RUN_DIR/renders" "$SCENE_DIR/$IMAGES" --test-every "$TEST_EVERY" \
    --out "$RUN_DIR/metrics.json" --quiet 2> "$RUN_DIR/metrics.log" || status=metrics_failed
  tail -n 1 "$RUN_DIR/metrics.log" >&2
fi
m1=$(date +%s.%N)

if [[ $keep == 0 ]]; then
  find "$RUN_DIR" \( -name '*.pt' -o -name '*.ply' -o -name '*.spz' -o -name 'events.out.tfevents*' \) -delete 2>/dev/null || true
fi

image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || echo unknown)
harness=$(cd "$HERE" && find . -type f \( -name '*.sh' -o -name '*.py' -o -name '*.toml' -o -name Dockerfile \) -print0 \
  | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16)

python3 -I - "$RUN_DIR" "$BENCH/results.jsonl" <<EOF
import csv, json, os, socket, sys, time
run, out = sys.argv[1], sys.argv[2]
def load(name):
    p = os.path.join(run, name)
    return json.load(open(p)) if os.path.exists(p) else {}
tool, metrics = load("tool.json"), load("metrics.json")
before = next(csv.reader(open(os.path.join(run, "gpu_before.csv"))), [])
before = [c.strip() for c in before]
rows = []
for r in csv.reader(open(os.path.join(run, "smi.csv"))):
    try:
        rows.append([float(x) for x in r[1:]])
    except ValueError:
        pass
base = float(before[2]) if len(before) > 2 else (rows[0][0] if rows else 0.0)
mem = [r[0] for r in rows]; util = [r[1] for r in rows]; power = [r[3] for r in rows]; sm = [r[4] for r in rows]
mean = lambda v: round(sum(v) / len(v), 2) if v else None
apps = [l.strip() for l in open(os.path.join(run, "gpu_apps_before.csv")) if l.strip()]
rec = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "tool": "$tool", "scene": "$scene", "dataset": "$DATASET", "images": "$IMAGES",
    "test_every": $TEST_EVERY, "steps": int("$steps") if "$steps" else None, "tag": "$tag" or None,
    "status": "$status",
    "wall_s": round($t1 - $t0, 2),
    "train_s": tool.get("train_s"),
    "metrics_s": round($m1 - $m0, 2),
    "lock_wait_s": $lock_wait,
    "psnr": metrics.get("psnr"), "ssim": metrics.get("ssim"), "lpips": metrics.get("lpips"),
    "ssim_tm": metrics.get("ssim_tm"), "lpips_alex": metrics.get("lpips_alex"),
    "n_test": metrics.get("n"), "eval_width": metrics.get("width"), "eval_height": metrics.get("height"),
    "num_gaussians": tool.get("num_gaussians"),
    "peak_vram_mib": round(max(mem) - base, 1) if mem else None,
    "peak_vram_total_mib": max(mem) if mem else None,
    "vram_before_mib": base,
    "util_mean": mean(util), "util_max": max(util) if util else None,
    "power_mean_w": mean(power), "sm_clock_mean_mhz": mean(sm), "smi_samples": len(rows),
    "gpu_apps_before": apps,
    "gpu": before[0] if before else None, "driver": before[1] if len(before) > 1 else None,
    "host": socket.gethostname(),
    "tool_version": tool.get("tool_version"), "tool_metrics": tool.get("tool_metrics"),
    "tool_extra": {k: v for k, v in tool.items() if k not in ("train_s", "num_gaussians", "tool_version", "tool_metrics")},
    "image": "$IMAGE", "image_id": "$image_id", "harness": "$harness",
    "run_dir": run,
}
line = json.dumps(rec)
with open(out, "a") as f:
    f.write(line + "\n")
json.dump(rec, open(os.path.join(run, "result.json"), "w"), indent=2)
print(line)
EOF
[[ $status == ok ]]
