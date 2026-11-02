#!/usr/bin/env bash
# Validate GLB files with the Khronos glTF validator (npm gltf-validator,
# a library without a CLI, so it is driven from a small node script).
# Usage: validate_glbs.sh <directory>
set -euo pipefail

DIR="${1:?usage: validate_glbs.sh <directory>}"
TOOL="${TMPDIR:-/tmp}/spl-gltf-validator"
command -v node >/dev/null || { echo "node not found"; exit 1; }
[ -d "$TOOL/node_modules/gltf-validator" ] || {
  mkdir -p "$TOOL" && (cd "$TOOL" && npm init -y >/dev/null && npm i --silent gltf-validator >/dev/null)
}

NODE_PATH="$TOOL/node_modules" node - "$DIR"/*.glb <<'JS'
const v = require('gltf-validator'), fs = require('fs');
(async () => {
  let fail = 0;
  for (const f of process.argv.slice(2)) {
    const r = await v.validateBytes(new Uint8Array(fs.readFileSync(f)), { maxIssues: 20 });
    const i = r.issues;
    console.log(`[${i.numErrors ? 'FAIL' : 'PASS'}] ${f} errors=${i.numErrors} warnings=${i.numWarnings} tris=${r.info.totalTriangleCount}`);
    for (const m of i.messages.filter(m => m.severity === 0).slice(0, 3)) console.log('   ', m.code, m.message);
    if (i.numErrors) fail++;
  }
  process.exit(fail ? 1 : 0);
})();
JS
