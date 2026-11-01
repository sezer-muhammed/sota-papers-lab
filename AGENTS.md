# SOTA Papers Lab: agent guide

Read `docs/ARCHITECTURE.md` and `docs/APP_CONTRACT.md` before structural changes.

## Rules

- **Dependency direction:** `apps/*` depend only on `platform/*`. Apps never import each other; they compose through op IDs and artifacts. `platform/*` never depends on an app.
- **New app:** `scripts/new-app.sh <kebab-name> "<summary>"`. It wires the workspace, `catalog/Cargo.toml` and `catalog/src/lib.rs`. Nothing else in the platform changes.
- **Op IDs** are `<app_snake>.<verb>` and permanent once used in a recorded run.
- **GPU work** happens inside `tokio::task::spawn_blocking`. Kernels live in `apps/<app>/cuda/*.cu`.
- **Database:** only `platform/spl-db` writes SQL. Follow `docs/DATABASE.md`; every migration exists for sqlite, postgres and oracle with the same file name (a test enforces it).
- **Generated code:** `web/bindings/*.ts` come from ts-rs. Regenerate with `cargo test`, never edit by hand, and commit the result (CI checks).
- **UI:** use `@spl/ui` tokens and components; follow `docs/DESIGN.md`. No raw hex colors in components.

## Verify

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
pnpm typecheck && pnpm build
```

CUDA is not present on macOS. Code compiles and non-GPU tests run there; GPU ops return a "driver not found" failed run. Run GPU paths on the Linux GPU host.
