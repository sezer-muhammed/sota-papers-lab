-- Logical schema v1 for Oracle Database / Autonomous Database (19c+).
-- Mirrors ../sqlite/0001_init.sql. Statements are separated by lines holding
-- only "/" so the runner can execute them one at a time.
-- On 23ai, the CLOB ... IS JSON columns can become the native JSON type.

CREATE TABLE spl_runs (
    id             VARCHAR2(36) PRIMARY KEY,
    op_id          VARCHAR2(200) NOT NULL,
    status         VARCHAR2(16) NOT NULL,
    input_json     CLOB NOT NULL CHECK (input_json IS JSON),
    output_json    CLOB CHECK (output_json IS JSON),
    error_message  CLOB,
    host_name      VARCHAR2(255),
    git_sha        VARCHAR2(64),
    env_json       CLOB CHECK (env_json IS JSON),
    created_at_ms  NUMBER(19) NOT NULL,
    started_at_ms  NUMBER(19),
    finished_at_ms NUMBER(19)
)
/
CREATE INDEX spl_runs_op_created ON spl_runs (op_id, created_at_ms)
/
CREATE TABLE spl_artifacts (
    id            VARCHAR2(64) PRIMARY KEY,
    kind          VARCHAR2(32) NOT NULL,
    media_type    VARCHAR2(255) NOT NULL,
    size_bytes    NUMBER(19) NOT NULL,
    uri           VARCHAR2(2000) NOT NULL,
    meta_json     CLOB CHECK (meta_json IS JSON),
    created_at_ms NUMBER(19) NOT NULL
)
/
CREATE TABLE spl_run_artifacts (
    run_id      VARCHAR2(36) NOT NULL REFERENCES spl_runs (id),
    artifact_id VARCHAR2(64) NOT NULL REFERENCES spl_artifacts (id),
    direction   VARCHAR2(8) NOT NULL,
    slot_name   VARCHAR2(200) NOT NULL,
    PRIMARY KEY (run_id, direction, slot_name)
)
/
CREATE TABLE spl_metrics (
    run_id       VARCHAR2(36) NOT NULL REFERENCES spl_runs (id),
    seq          NUMBER(19) NOT NULL,
    metric_key   VARCHAR2(200) NOT NULL,
    metric_value BINARY_DOUBLE NOT NULL,
    unit         VARCHAR2(64),
    PRIMARY KEY (run_id, seq)
)
/
