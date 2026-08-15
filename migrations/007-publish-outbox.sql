-- =============================================================================
-- 007 — publish outbox (docs/face-sync.md §4.1–4.3)
--
-- Adds the state that lets publish-faces.py answer "what has changed since the
-- last successful publish?" without comparing timestamps across two machines'
-- clocks (see §5, "Why not timestamps").
--
-- Idempotent; safe to re-run.  The same DDL is in schema.sql for fresh
-- databases -- this file exists to bring an already-populated one forward.
--
--   psql "$FACE_SCANNER_DSN" -f migrations/007-publish-outbox.sql
--
-- Note: adding `revision` rewrites both tables (a volatile DEFAULT cannot use
-- the fast path), so expect this to take a few seconds against ~255k faces.
-- Every existing row lands with published_revision NULL, i.e. "never
-- published", which is exactly right for the first run.
-- =============================================================================

CREATE SEQUENCE IF NOT EXISTS revision_seq;

-- ---------------------------------------------------------------------------
-- Outbox columns
--
-- "needs publishing" is  published_revision IS NULL OR published_revision < revision.
-- The publisher stamps published_revision only after a 2xx, so a crash anywhere
-- in between simply republishes -- at-least-once, which the API's revision guard
-- makes safe.
-- ---------------------------------------------------------------------------
ALTER TABLE person ADD COLUMN IF NOT EXISTS
    revision BIGINT NOT NULL DEFAULT nextval('revision_seq');
ALTER TABLE person ADD COLUMN IF NOT EXISTS published_revision BIGINT;

ALTER TABLE face_detection ADD COLUMN IF NOT EXISTS
    revision BIGINT NOT NULL DEFAULT nextval('revision_seq');
ALTER TABLE face_detection ADD COLUMN IF NOT EXISTS published_revision BIGINT;

-- §4.3 — diagnostics only.  Deliberately NOT a cached maw-media media id: that
-- would be derived data going stale, the mistake face_count already taught us.
ALTER TABLE media ADD COLUMN IF NOT EXISTS last_published_at TIMESTAMPTZ;
ALTER TABLE media ADD COLUMN IF NOT EXISTS publish_error TEXT;

CREATE INDEX IF NOT EXISTS person_unpublished_idx
    ON person(revision)
    WHERE published_revision IS NULL OR published_revision < revision;

CREATE INDEX IF NOT EXISTS face_detection_unpublished_idx
    ON face_detection(revision)
    WHERE published_revision IS NULL OR published_revision < revision;

-- Retraction scans: rows that ARE published and may have left publish scope.
CREATE INDEX IF NOT EXISTS person_published_idx
    ON person(id) WHERE published_revision IS NOT NULL;

CREATE INDEX IF NOT EXISTS face_detection_published_idx
    ON face_detection(person_id) WHERE published_revision IS NOT NULL;

-- ---------------------------------------------------------------------------
-- §4.2 Tombstones
--
-- `cluster` deletes and merges persons, and absence from a batch cannot express
-- that -- a batch is a delta, so a missing row means "unchanged", not "gone".
--
-- Only rows that were actually published are recorded.  A `cluster` run deletes
-- hundreds of unnamed clusters that maw-media has never heard of; tombstoning
-- those would generate delete calls whose every row comes back `not_found`.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS deleted_entity (
    entity_type        TEXT NOT NULL,          -- 'person' | 'face'
    entity_id          UUID NOT NULL,
    revision           BIGINT NOT NULL DEFAULT nextval('revision_seq'),
    published_revision BIGINT,
    deleted_at         TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (entity_type, entity_id)
);

CREATE INDEX IF NOT EXISTS deleted_entity_unpublished_idx
    ON deleted_entity(entity_type)
    WHERE published_revision IS NULL;

-- ---------------------------------------------------------------------------
-- Revision triggers
--
-- A revision bumps only when a column that is actually PUBLISHED changes.
-- Notably not `embedding`: a re-scan that produces identical assignments must
-- not enqueue a quarter of a million no-op updates.
-- ---------------------------------------------------------------------------

-- person.slug is derived from name by the publisher, so name covers it.
CREATE OR REPLACE FUNCTION person_bump_revision() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.name IS DISTINCT FROM OLD.name
       OR NEW.status_code IS DISTINCT FROM OLD.status_code
       OR NEW.preferred_face_id IS DISTINCT FROM OLD.preferred_face_id
    THEN
        NEW.revision := nextval('revision_seq');
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS person_bump_revision_trg ON person;
CREATE TRIGGER person_bump_revision_trg
    BEFORE UPDATE ON person
    FOR EACH ROW EXECUTE FUNCTION person_bump_revision();

CREATE OR REPLACE FUNCTION face_bump_revision() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.person_id IS DISTINCT FROM OLD.person_id
       OR NEW.media_id IS DISTINCT FROM OLD.media_id
       OR NEW.bounding_box IS DISTINCT FROM OLD.bounding_box
       OR NEW.detection_score IS DISTINCT FROM OLD.detection_score
    THEN
        NEW.revision := nextval('revision_seq');
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS face_bump_revision_trg ON face_detection;
CREATE TRIGGER face_bump_revision_trg
    BEFORE UPDATE ON face_detection
    FOR EACH ROW EXECUTE FUNCTION face_bump_revision();

-- ---------------------------------------------------------------------------
-- face_count propagation
--
-- PersonSync carries faceCount, and an upsert replaces the whole row -- but
-- face_count lives in face_detection, so moving a face changes two person rows'
-- published state without either row being touched.  Without this the count on
-- the website silently rots until the person changes for some other reason.
--
-- Statement level with transition tables: `cluster` reassigns ~60k faces in one
-- statement, and a row-level trigger would issue 60k separate person updates.
-- Restricted to already-published persons -- an unpublished one is sent in full
-- the first time it is seen, so bumping it achieves nothing.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION person_revision_from_face_insert() RETURNS TRIGGER AS $$
BEGIN
    UPDATE person p SET revision = nextval('revision_seq')
    WHERE p.published_revision IS NOT NULL
      AND p.id IN (SELECT n.person_id FROM new_faces n WHERE n.person_id IS NOT NULL);

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION person_revision_from_face_update() RETURNS TRIGGER AS $$
BEGIN
    -- Only where person_id actually moved; a bounding_box-only update must not
    -- re-publish every named person.
    UPDATE person p SET revision = nextval('revision_seq')
    WHERE p.published_revision IS NOT NULL
      AND p.id IN (
            SELECT n.person_id
            FROM new_faces n JOIN old_faces o ON o.id = n.id
            WHERE n.person_id IS DISTINCT FROM o.person_id AND n.person_id IS NOT NULL
            UNION
            SELECT o.person_id
            FROM new_faces n JOIN old_faces o ON o.id = n.id
            WHERE n.person_id IS DISTINCT FROM o.person_id AND o.person_id IS NOT NULL
      );

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION person_revision_from_face_delete() RETURNS TRIGGER AS $$
BEGIN
    UPDATE person p SET revision = nextval('revision_seq')
    WHERE p.published_revision IS NOT NULL
      AND p.id IN (SELECT o.person_id FROM old_faces o WHERE o.person_id IS NOT NULL);

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS person_revision_face_ins_trg ON face_detection;
CREATE TRIGGER person_revision_face_ins_trg
    AFTER INSERT ON face_detection
    REFERENCING NEW TABLE AS new_faces
    FOR EACH STATEMENT EXECUTE FUNCTION person_revision_from_face_insert();

DROP TRIGGER IF EXISTS person_revision_face_upd_trg ON face_detection;
CREATE TRIGGER person_revision_face_upd_trg
    AFTER UPDATE ON face_detection
    REFERENCING NEW TABLE AS new_faces OLD TABLE AS old_faces
    FOR EACH STATEMENT EXECUTE FUNCTION person_revision_from_face_update();

DROP TRIGGER IF EXISTS person_revision_face_del_trg ON face_detection;
CREATE TRIGGER person_revision_face_del_trg
    AFTER DELETE ON face_detection
    REFERENCING OLD TABLE AS old_faces
    FOR EACH STATEMENT EXECUTE FUNCTION person_revision_from_face_delete();

-- ---------------------------------------------------------------------------
-- Tombstone triggers
--
-- ON CONFLICT resets published_revision so that an id deleted, somehow revived,
-- and deleted again is re-sent rather than swallowed by the old tombstone.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION record_person_deletion() RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO deleted_entity (entity_type, entity_id)
    SELECT 'person', o.id FROM old_persons o WHERE o.published_revision IS NOT NULL
    ON CONFLICT (entity_type, entity_id) DO UPDATE
        SET revision = nextval('revision_seq'),
            published_revision = NULL,
            deleted_at = now();

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION record_face_deletion() RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO deleted_entity (entity_type, entity_id)
    SELECT 'face', o.id FROM old_faces o WHERE o.published_revision IS NOT NULL
    ON CONFLICT (entity_type, entity_id) DO UPDATE
        SET revision = nextval('revision_seq'),
            published_revision = NULL,
            deleted_at = now();

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS person_tombstone_trg ON person;
CREATE TRIGGER person_tombstone_trg
    AFTER DELETE ON person
    REFERENCING OLD TABLE AS old_persons
    FOR EACH STATEMENT EXECUTE FUNCTION record_person_deletion();

DROP TRIGGER IF EXISTS face_tombstone_trg ON face_detection;
CREATE TRIGGER face_tombstone_trg
    AFTER DELETE ON face_detection
    REFERENCING OLD TABLE AS old_faces
    FOR EACH STATEMENT EXECUTE FUNCTION record_face_deletion();
