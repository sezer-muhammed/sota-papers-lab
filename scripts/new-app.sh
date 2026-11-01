#!/usr/bin/env bash
# Usage: scripts/new-app.sh <name> "<one-line summary>"
# Creates apps/<name> from templates/app and wires it into the catalog.
set -euo pipefail

name="${1:?usage: scripts/new-app.sh <name> \"<summary>\"}"
summary="${2:?give a one-line summary}"
[[ "$name" =~ ^[a-z][a-z0-9-]*$ ]] || { echo "name must be kebab-case: $name" >&2; exit 1; }
[[ "$summary" == *[\"\\]* ]] && { echo "summary must not contain quotes or backslashes" >&2; exit 1; }
snake="${name//-/_}"
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/apps/$name"
[[ -e "$dest" ]] && { echo "apps/$name already exists" >&2; exit 1; }

cp -R "$root/templates/app" "$dest"
NAME="$name" SNAKE="$snake" SUMMARY="$summary" find "$dest" -type f -exec \
  perl -pi -e 's/__NAME__/$ENV{NAME}/g; s/__SNAKE__/$ENV{SNAKE}/g; s/__SUMMARY__/$ENV{SUMMARY}/g' {} +

# Workspace dependency, catalog feature and registration line.
NAME="$name" perl -0pi -e 's/(# apps\n)/$1spl-app-$ENV{NAME} = { path = "apps\/$ENV{NAME}" }\n/' "$root/Cargo.toml"
NAME="$name" perl -0pi -e 's/default = \[([^\]]*)\]\n/default = [$1, "$ENV{NAME}"]\n$ENV{NAME} = ["dep:spl-app-$ENV{NAME}"]\n/; s/\z/spl-app-$ENV{NAME} = { workspace = true, optional = true }\n/' "$root/catalog/Cargo.toml"
NAME="$name" SNAKE="$snake" perl -0pi -e 's/(    registry\n\})/    #[cfg(feature = "$ENV{NAME}")]\n    spl_app_$ENV{SNAKE}::register(&mut registry);\n$1/' "$root/catalog/src/lib.rs"

echo "Created apps/$name. Next: cargo check -p spl-app-$name"
