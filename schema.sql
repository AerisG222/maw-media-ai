-- =============================================================================
-- Face Scanner Schema
-- Run this once against your Postgres 18 database before the first scan.
-- Requires the pgvector extension: https://github.com/pgvector/pgvector
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- photos
-- One row per image file that has been scanned (or attempted).
-- You may already have a photos table — if so, add the scanner columns to it
-- and remove the REFERENCES below that point here.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS photos (
    id              BIGSERIAL PRIMARY KEY,
    file_path       TEXT NOT NULL UNIQUE,
    file_name       TEXT NOT NULL,
    scanned_at      TIMESTAMPTZ DEFAULT now(),
    scan_error      TEXT                     -- NULL = scanned OK
);

-- ---------------------------------------------------------------------------
-- persons
-- One row per distinct person identified by clustering.
-- name is NULL until a human labels the cluster via your admin UI.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS persons (
    id                          BIGSERIAL PRIMARY KEY,
    name                        VARCHAR(255),            -- set by human operator
    cluster_label               INT,                     -- HDBSCAN cluster id
    representative_embedding    vector(512),             -- centroid of all face embeddings
    face_count                  INT DEFAULT 0,
    created_at                  TIMESTAMPTZ DEFAULT now(),
    updated_at                  TIMESTAMPTZ DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- face_detections
-- One row per detected face in a photo.
-- bounding_box stores normalised coordinates (0–1) so they survive resizes,
-- plus raw pixel values for convenience.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS face_detections (
    id              BIGSERIAL PRIMARY KEY,
    photo_id        BIGINT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    person_id       BIGINT REFERENCES persons(id) ON DELETE SET NULL,
    bounding_box    JSONB NOT NULL,
    -- Example bounding_box:
    -- { "x": 0.12, "y": 0.05, "width": 0.18, "height": 0.24,
    --   "x_px": 154, "y_px": 64, "width_px": 230, "height_px": 307 }
    embedding       vector(512),
    detection_score FLOAT NOT NULL,
    face_width_px   INT NOT NULL,
    face_height_px  INT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

-- ANN (approximate nearest neighbour) index for fast similarity search.
-- ivfflat is good up to ~1M rows. For larger collections switch to hnsw:
--   CREATE INDEX … USING hnsw (embedding vector_cosine_ops);
-- lists = 100 is appropriate for up to ~1M rows.
-- Rebuild after large bulk inserts for best performance:
--   REINDEX INDEX face_detections_embedding_idx;
CREATE INDEX IF NOT EXISTS face_detections_embedding_idx
    ON face_detections USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS face_detections_photo_id_idx
    ON face_detections(photo_id);

CREATE INDEX IF NOT EXISTS face_detections_person_id_idx
    ON face_detections(person_id);

CREATE INDEX IF NOT EXISTS persons_name_idx
    ON persons(name)
    WHERE name IS NOT NULL;


-- ---------------------------------------------------------------------------
-- Useful views for your .NET application
-- ---------------------------------------------------------------------------

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
FROM photos p
JOIN face_detections fd ON fd.photo_id = p.id
JOIN persons per        ON per.id = fd.person_id
WHERE per.name IS NOT NULL;

-- Aggregate: photos with a JSON array of the named people in them.
CREATE OR REPLACE VIEW photo_person_summary AS
SELECT
    p.id        AS photo_id,
    p.file_path,
    p.file_name,
    jsonb_agg(DISTINCT per.name ORDER BY per.name) AS people
FROM photos p
JOIN face_detections fd ON fd.photo_id = p.id
JOIN persons per        ON per.id = fd.person_id
WHERE per.name IS NOT NULL
GROUP BY p.id, p.file_path, p.file_name;
