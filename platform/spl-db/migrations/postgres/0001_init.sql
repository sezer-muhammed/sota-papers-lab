-- Logical schema v1 for PostgreSQL. Mirrors ../sqlite/0001_init.sql.

CREATE TABLE spl_runs (
    id             TEXT PRIMARY KEY,
    op_id          TEXT NOT NULL,
    status         TEXT NOT NULL,
    input_json     JSONB NOT NULL,
    output_json    JSONB,
    error_message  TEXT,
    host_name      TEXT,
    git_sha        TEXT,
    env_json       JSONB,
    created_at_ms  BIGINT NOT NULL,
    started_at_ms  BIGINT,
    finished_at_ms BIGINT
);
CREATE INDEX spl_runs_op_created ON spl_runs (op_id, created_at_ms);

CREATE TABLE spl_artifacts (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    media_type    TEXT NOT NULL,
    size_bytes    BIGINT NOT NULL,
    uri           TEXT NOT NULL,
    meta_json     JSONB,
    created_at_ms BIGINT NOT NULL
);

CREATE TABLE spl_run_artifacts (
    run_id      TEXT NOT NULL REFERENCES spl_runs (id),
    artifact_id TEXT NOT NULL REFERENCES spl_artifacts (id),
    direction   TEXT NOT NULL,
    slot_name   TEXT NOT NULL,
    PRIMARY KEY (run_id, direction, slot_name)
);

CREATE TABLE spl_metrics (
    run_id       TEXT NOT NULL REFERENCES spl_runs (id),
    seq          BIGINT NOT NULL,
    metric_key   TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    unit         TEXT,
    PRIMARY KEY (run_id, seq)
);
