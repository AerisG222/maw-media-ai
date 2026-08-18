-- =============================================================================
-- 008 — face image publish tracking (docs/face-sync.md §6, "Face images")
--
-- maw-media stores the preferred face crop as a file keyed by face id.  A given
-- face's crop never changes once published, so this records which ones have
-- already been uploaded.
--
-- Deliberately separate from published_revision.  person.revision bumps on any
-- meaningful change -- face_count shifts on essentially every cluster run -- so
-- driving uploads from it would re-send byte-identical images for every named
-- person, every run.  Tracking at the face makes the upload happen exactly once.
--
-- Idempotent; safe to re-run.  The same DDL is in schema.sql for fresh
-- databases -- this file exists to bring an already-populated one forward.
--
--   psql "$FACE_SCANNER_DSN" -f migrations/008-face-image-published.sql
-- =============================================================================

ALTER TABLE face_detection ADD COLUMN IF NOT EXISTS image_published_at TIMESTAMPTZ;

-- the publisher looks for preferred faces that still need an upload; the set is
-- small (one per named person) but the scan is over every face without it
CREATE INDEX IF NOT EXISTS face_detection_image_unpublished_idx
    ON face_detection(id)
    WHERE image_published_at IS NULL;
