-- Logical schema v2 for PostgreSQL. Mirrors ../sqlite/0002_announcements.sql.

CREATE TABLE spl_run_announcements (
    run_id          TEXT NOT NULL REFERENCES spl_runs (id),
    slot_name       TEXT NOT NULL,
    artifact_id     TEXT NOT NULL REFERENCES spl_artifacts (id),
    revision        BIGINT NOT NULL,
    info_json       JSONB,
    announced_at_ms BIGINT NOT NULL,
    PRIMARY KEY (run_id, slot_name)
);
CREATE INDEX spl_run_ann_artifact ON spl_run_announcements (artifact_id);

CREATE INDEX spl_run_art_artifact ON spl_run_artifacts (artifact_id);
