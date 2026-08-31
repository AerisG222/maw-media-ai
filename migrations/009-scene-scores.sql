-- =============================================================================
-- 009 — scene scores (docs/place-covers.md §3)
--
-- One Places365 pass over the library produces two things: how outdoor a photo
-- looks, used to rank candidates for a location's cover image, and the scene
-- categories themselves, which maw-media folds into search at the lowest
-- tsvector weight.
--
-- Idempotent; safe to re-run.  The same DDL is in schema.sql for fresh
-- databases -- this file brings an already-populated one forward.
--
--   psql "$FACE_SCANNER_DSN" -f migrations/009-scene-scores.sql
-- =============================================================================

-- Probability mass over the 204 outdoor categories of the 365, so the number
-- comes from the model's own indoor/outdoor taxonomy rather than a threshold
-- someone picked.  NULL means "not scored yet", which is also the work queue.
ALTER TABLE media ADD COLUMN IF NOT EXISTS outdoor_score REAL;
ALTER TABLE media ADD COLUMN IF NOT EXISTS scene_scored_at TIMESTAMPTZ;

-- ---------------------------------------------------------------------------
-- Top-K scene categories per media
--
-- A child table rather than JSONB on media: both consumers want it relational.
-- The search vector aggregates labels per category (a join), and the
-- probability floor that keeps softmax noise out of the index is then a WHERE
-- rather than a json path expression.
--
-- Storing K labels rather than just the winner is deliberate -- see the "store
-- more than is currently used" note in the design.  The floor and the cover
-- heuristics are exactly the decisions that will move after the first review,
-- and keeping the output means moving them costs a query rather than a re-scan.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS media_scene_label (
    media_id    UUID NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    rank        SMALLINT NOT NULL,      -- 1 = most probable
    code        TEXT NOT NULL,          -- places365 category, e.g. 'botanical_garden'
    probability REAL NOT NULL,

    PRIMARY KEY (media_id, rank)
);

-- "which media look like a museum" -- the admin cover picker filters on this
-- when the outdoor ranking has nothing good to offer for a place.
CREATE INDEX IF NOT EXISTS media_scene_label_code_idx
    ON media_scene_label(code);

-- The work queue: rows still needing a pass.  Partial, so it stays small as the
-- library fills in and is empty once everything is scored.
CREATE INDEX IF NOT EXISTS media_unscored_idx
    ON media(id) WHERE scene_scored_at IS NULL;

-- Cover ranking reads highest-first over a subset of media, so the index is
-- worth having even though the table is small enough to seq-scan today.
CREATE INDEX IF NOT EXISTS media_outdoor_score_idx
    ON media(outdoor_score DESC) WHERE outdoor_score IS NOT NULL;
