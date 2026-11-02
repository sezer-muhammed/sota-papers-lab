-- Logical schema v2. Keep in step with ../postgres and ../oracle.
--
-- What a run announced while it ran (`cx.announce`): its current artifact
-- per slot ("points", "cameras", "splat"), so clients show intermediate
-- results before the run ends. One row per slot; a newer announcement
-- replaces it and bumps the revision.

CREATE TABLE spl_run_announcements (
    run_id          TEXT NOT NULL REFERENCES spl_runs (id),
    slot_name       TEXT NOT NULL,
    artifact_id     TEXT NOT NULL REFERENCES spl_artifacts (id),
    revision        INTEGER NOT NULL,  -- 1, 2, ... per slot
    info_json       TEXT,
    announced_at_ms INTEGER NOT NULL,
    PRIMARY KEY (run_id, slot_name)
);
CREATE INDEX spl_run_ann_artifact ON spl_run_announcements (artifact_id);

-- Who references an artifact: asked before a superseded snapshot is deleted.
CREATE INDEX spl_run_art_artifact ON spl_run_artifacts (artifact_id);
