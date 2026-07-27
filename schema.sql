-- =============================================================================
-- Face Scanner Schema (UUID, no default)
-- Run this once against your Postgres 18 database before the first scan.
-- Requires the pgvector extension: https://github.com/pgvector/pgvector
--
-- Table names are singular nouns (photo, person, face_detection).
-- Upgrading an existing database created with the old plural names?
-- Run migrations/001-singular-table-names.sql instead of this file.
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- photo
-- One row per image file that has been scanned (or attempted).
-- You may already have a photo table — if so, add the scanner columns to it
-- and remove the REFERENCES below that point here.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS photo (
    id              UUID PRIMARY KEY,
    file_path       TEXT NOT NULL UNIQUE,
    file_name       TEXT NOT NULL,
    scanned_at      TIMESTAMPTZ DEFAULT now(),
    scan_error      TEXT
);

-- ---------------------------------------------------------------------------
-- person
-- One row per distinct person identified by clustering.
-- name is NULL until a human labels the cluster via your admin UI.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS person (
    id                          UUID PRIMARY KEY,
    name                        VARCHAR(255),            -- set by human operator
    representative_embedding    vector(512),             -- centroid of all face embeddings
    created_at                  TIMESTAMPTZ DEFAULT now(),
    updated_at                  TIMESTAMPTZ DEFAULT now(),
    preferred_face_id           UUID REFERENCES face_detection(id) ON DELETE SET NULL,
    status_code                 TEXT REFERENCES person_status(code)
);

-- ---------------------------------------------------------------------------
-- person_status
-- Triage states for clusters you do not intend to name.  A lookup table rather
-- than a CHECK so new states are an INSERT, the UI can build its selector from
-- the data, and a .NET consumer can discover the valid values.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS person_status (
    code        TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    description TEXT,
    sort_order  INT NOT NULL DEFAULT 0
);

INSERT INTO person_status (code, label, description, sort_order) VALUES
    ('unknown',      'Unknown',
     'A real person you have decided not to name. Kept grouped so it can be '
     'named later if you recognise them.', 1),
    ('not_a_person', 'Not a person',
     'Bad detection: not a face, or not a person worth tracking.', 2)
ON CONFLICT (code) DO NOTHING;

-- ---------------------------------------------------------------------------
-- face_detection
-- One row per detected face in a photo.
-- bounding_box stores normalised coordinates (0–1) so they survive resizes,
-- plus raw pixel values for convenience.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS face_detection (
    id                   UUID PRIMARY KEY,
    photo_id             UUID NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    person_id            UUID REFERENCES person(id) ON DELETE SET NULL,
    bounding_box         JSONB NOT NULL,
    embedding            vector(512),
    detection_score      FLOAT NOT NULL,
    face_width_px        INT NOT NULL,
    face_height_px       INT NOT NULL,
    created_at           TIMESTAMPTZ DEFAULT now(),
    suggestion_score     FLOAT,
    suggested_person_id  UUID REFERENCES person(id) ON DELETE SET NULL
);

ALTER TABLE person DROP CONSTRAINT IF EXISTS person_name_status_excl;
ALTER TABLE person ADD CONSTRAINT person_name_status_excl
    CHECK (name IS NULL OR status_code IS NULL);


-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

-- ANN (approximate nearest neighbour) index for fast similarity search.
-- ivfflat is good up to ~1M rows. For larger collections switch to hnsw:
--   CREATE INDEX … USING hnsw (embedding vector_cosine_ops);
-- lists = 100 is appropriate for up to ~1M rows.
-- Rebuild after large bulk inserts for best performance:
--   REINDEX INDEX face_detection_embedding_idx;
CREATE INDEX IF NOT EXISTS face_detection_embedding_idx
    ON face_detection USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS face_detection_photo_id_idx
    ON face_detection(photo_id);

CREATE INDEX IF NOT EXISTS face_detection_person_id_idx
    ON face_detection(person_id);

CREATE INDEX IF NOT EXISTS person_name_idx
    ON person(name)
    WHERE name IS NOT NULL;

CREATE INDEX IF NOT EXISTS face_detection_suggested_person_idx
    ON face_detection(suggested_person_id)
    WHERE suggested_person_id IS NOT NULL;

-- Candidate lookups (suggest, cluster) filter on "not yet assigned".
CREATE INDEX IF NOT EXISTS face_detection_unassigned_idx
    ON face_detection(person_id)
    WHERE person_id IS NULL;

-- Triage queues filter on this constantly.
CREATE INDEX IF NOT EXISTS person_status_code_idx
    ON person(status_code);

-- Untriaged clusters are the hot lookup for cluster/suggest/merge-clusters.
CREATE INDEX IF NOT EXISTS person_untriaged_idx
    ON person(id) WHERE name IS NULL AND status_code IS NULL;

-- ---------------------------------------------------------------------------
-- Views
-- ---------------------------------------------------------------------------

-- person, with face_count computed live.  face_count is deliberately NOT stored
-- on the person table: a denormalised counter drifted in practice (direct SQL
-- bypassed the application paths that maintained it), and Postgres generated
-- columns cannot aggregate over another table.  Write to `person`; read from
-- `person_v` whenever you need the count.
-- person_v predates status_code; refresh it so the UI can read triage state
-- without a second query.  (New columns may be appended to a view in place.)
CREATE OR REPLACE VIEW person_v AS
SELECT p.id,
       p.name,
       p.representative_embedding,
       p.created_at,
       p.updated_at,
       p.preferred_face_id,
       coalesce(c.n, 0)::int AS face_count,
       p.status_code
FROM person p
LEFT JOIN (
    SELECT person_id, count(*) AS n
    FROM face_detection
    WHERE person_id IS NOT NULL
    GROUP BY person_id
) c ON c.person_id = p.id;

-- All photos that contain at least one recognised (named) person.
CREATE OR REPLACE VIEW photos_with_named_persons AS
SELECT
    p.id            AS photo_id,
    p.file_path,
    p.file_name,
    per.id          AS person_id,
    per.name        AS person_name,
    fd.detection_score,
    fd.bounding_box
FROM photo p
JOIN face_detection fd ON fd.photo_id = p.id
JOIN person per        ON per.id = fd.person_id
WHERE per.name IS NOT NULL;

-- Aggregate: photos with a JSON array of the named people in them.
CREATE OR REPLACE VIEW photo_person_summary AS
SELECT
    p.id        AS photo_id,
    p.file_path,
    p.file_name,
    jsonb_agg(DISTINCT per.name ORDER BY per.name) AS people
FROM photo p
JOIN face_detection fd ON fd.photo_id = p.id
JOIN person per        ON per.id = fd.person_id
WHERE per.name IS NOT NULL
GROUP BY p.id, p.file_path, p.file_name;
