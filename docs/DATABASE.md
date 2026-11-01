# Database

SQLite today. PostgreSQL and **Oracle Autonomous Database** are planned. The
code is shaped so each move is a new backend, not a rewrite.

## How it's isolated

- Only `platform/spl-db` contains SQL. Everything else uses the `Store` trait.
- Each backend implements `Store` in its own module with its own dialect (placeholders, upsert, pagination, JSON type).
- One logical schema, three physical versions: `migrations/{sqlite,postgres,oracle}/NNNN_name.sql`. A unit test fails if the three folders ever differ in file names.
- Migrations run through our own small runner and a `spl_schema_migrations` table (same shape on every backend) instead of a tool that only knows some databases.
- `spl_db::connect(url)` picks the backend by URL scheme: `sqlite:`, `postgres:`, `oracle:`.

## Schema rules (portable to SQLite, PostgreSQL, Oracle)

| Rule | Why |
|---|---|
| Table names prefixed `spl_`, unquoted, lowercase | Oracle upper-cases unquoted names; a prefix avoids clashes in a shared ADB schema |
| No reserved words as column names: `size`, `level`, `comment`, `date`, `user`, `uid`, `mode`, `number`, `session`, `file`, `start`, `rows`, `access`… | Oracle reserves far more words than SQLite/Postgres. Use `size_bytes`, `metric_key`, `metric_value` |
| IDs are app-generated strings (UUIDv7 for rows, blake3 hex for artifacts) | No auto-increment or sequence differences; IDs sort by time |
| Timestamps are `*_at_ms` integers (Unix epoch ms, UTC) | No timezone or `DATE`/`TIMESTAMP` semantics differences |
| JSON stored in `*_json` columns: TEXT (SQLite), JSONB (Postgres), CLOB `IS JSON` (Oracle 19c; native JSON on 23ai) | Each backend keeps a real JSON type where it has one |
| Never rely on empty string vs NULL; optional text is `NULL` | Oracle stores `''` as NULL |
| No booleans; use small text enums (`status = 'succeeded'`) | Oracle before 23ai has no BOOLEAN column type |
| `INTEGER` → Postgres `BIGINT` → Oracle `NUMBER(19)`; `REAL` → `DOUBLE PRECISION` → `BINARY_DOUBLE` | Same range everywhere |
| Identifiers ≤ 30 characters | Safe even on older Oracle compatibility settings |
| No `LIMIT`/`ON CONFLICT`/`RETURNING` in shared code | Each backend writes them in its own dialect (`FETCH FIRST`, `MERGE`) |

## Backend plan

| Backend | Driver | Notes |
|---|---|---|
| SQLite | `sqlx` (done) | WAL mode, one file per host |
| PostgreSQL | `sqlx` postgres feature | Queries use `$1`; JSONB binds with `sqlx::types::Json`. Contract tests run against a Postgres service in CI |
| Oracle ADB | an Oracle driver crate (`sqlx` has no Oracle support). Candidates: `oracle` (ODPI-C, needs Instant Client) or a pure-Rust thin driver such as `oracle-rs` (Tokio, TLS, wallets) | ADB needs TLS (wallet or one-way TLS). Statements in `migrations/oracle` are separated by `/` lines for the runner |

## Adding a migration

1. Add `NNNN_name.sql` to **all three** folders.
2. Register it in `src/migrations.rs` for all three constants.
3. Follow the rules above; `cargo test -p spl-db` checks parity and runs the SQLite round-trip test.
