#!/usr/bin/env bash
# Validate GLB files with the Khronos glTF validator.
# Usage: validate_glbs.sh <directory>
set -euo pipefail

DIR="${1:?usage: validate_glbs.sh <directory>}"

if ! command -v npx &>/dev/null; then
  echo "npx not found. Install Node.js or use: npm i -g gltf-validator"
  exit 1
fi

TOTAL=0
PASS=0
FAIL=0

for glb in "$DIR"/*.glb; do
  [ -f "$glb" ] || continue
  TOTAL=$((TOTAL + 1))
  name=$(basename "$glb")

  output=$(npx --yes gltf-validator "$glb" 2>&1)
  errors=$(echo "$output" | grep -c '"severity":0' || true)

  if [ "$errors" -eq 0 ]; then
    echo "[PASS] $name"
    PASS=$((PASS + 1))
  else
    echo "[FAIL] $name ($errors error(s))"
    echo "$output" | grep '"message"' | head -3
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "=== $TOTAL GLBs: $PASS passed, $FAIL failed ==="
