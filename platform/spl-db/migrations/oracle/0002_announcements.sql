-- Logical schema v2 for Oracle Database / Autonomous Database (19c+).
-- Mirrors ../sqlite/0002_announcements.sql. Statements are separated by
-- lines holding only "/".

CREATE TABLE spl_run_announcements (
    run_id          VARCHAR2(36) NOT NULL REFERENCES spl_runs (id),
    slot_name       VARCHAR2(200) NOT NULL,
    artifact_id     VARCHAR2(64) NOT NULL REFERENCES spl_artifacts (id),
    revision        NUMBER(19) NOT NULL,
    info_json       CLOB CHECK (info_json IS JSON),
    announced_at_ms NUMBER(19) NOT NULL,
    PRIMARY KEY (run_id, slot_name)
)
/
CREATE INDEX spl_run_ann_artifact ON spl_run_announcements (artifact_id)
/
CREATE INDEX spl_run_art_artifact ON spl_run_artifacts (artifact_id)
/
