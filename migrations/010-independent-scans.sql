-- =============================================================================
-- 010 — let the scanners run in either order
--
-- scan-faces.py and scan-scenes.py both walk the media directory now, so either
-- can be the first to meet a new file.  Whichever gets there creates the media
-- row; the other must still do its own work on it.
--
-- That did not hold before.  scan-faces.py decided what to skip with "does this
-- path already have a media row?", which conflates *registered* with
-- *face-scanned*.  A row created by any other writer would therefore be skipped
-- by the face scan permanently, and no face in that photo would ever be found.
--
-- The fix is to say what is actually meant.  `faces_scanned_at` records that the
-- face scan ran, mirroring `scene_scored_at` for scenes, and each scanner now
-- claims work by its own column.  `scanned_at` keeps its existing meaning of
-- "last touched" and is left alone.
--
-- Idempotent; safe to re-run.  The same DDL is in schema.sql for fresh
-- databases.
--
--   psql "$FACE_SCANNER_DSN" -f migrations/010-independent-scans.sql
-- =============================================================================

ALTER TABLE media ADD COLUMN IF NOT EXISTS faces_scanned_at TIMESTAMPTZ;

-- Backfill: every row that exists today was created by the face scan, which is
-- precisely the population the old predicate treated as scanned.  Without this
-- the next run would re-detect faces across the entire library.
--
-- COALESCE because scanned_at is nullable in principle; a row that somehow has
-- neither still needs a non-null marker or it would be re-scanned forever.
UPDATE media
SET faces_scanned_at = COALESCE(scanned_at, now())
WHERE faces_scanned_at IS NULL;

-- The work queue for the face scan, matching media_unscored_idx for scenes.
CREATE INDEX IF NOT EXISTS media_faces_unscanned_idx
    ON media(id) WHERE faces_scanned_at IS NULL;
