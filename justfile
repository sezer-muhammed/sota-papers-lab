# `brew install just` (macOS) or `cargo install just`. Every recipe is a thin
# wrapper; the commands work without just too.

default:
    @just --list

# Format, lint, test Rust; typecheck web. Runs anywhere (no GPU needed).
check:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace
    pnpm typecheck

# Regenerate web/bindings from the Rust types.
bindings:
    cargo test --workspace --quiet

# Serve the API and the built shell (run on the GPU host).
serve: web
    SPL_WEB_DIR=web/shell/dist cargo run --release -p spl-cli -- serve

# Shell with hot reload, proxied to a GPU host: just dev http://gpu-box:7070
dev api="http://127.0.0.1:7070":
    SPL_API={{api}} pnpm dev

web:
    pnpm install
    pnpm build

ops:
    cargo run -q -p spl-cli -- ops

# just run device_info.list '{}'
run op input="{}":
    cargo run -q --release -p spl-cli -- run {{op}} --input '{{input}}'

# just new-app imgcodec "Learned image compression from <paper>"
new-app name summary:
    scripts/new-app.sh {{name}} "{{summary}}"

docker:
    docker build -f deploy/docker/Dockerfile --build-arg SPL_GIT_SHA=$(git rev-parse --short HEAD) -t sota-papers-lab .
