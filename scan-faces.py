#!/usr/bin/env python3
"""
Face Detection & Recognition Scanner
=====================================
Scans a photo directory tree, detects faces using InsightFace (buffalo_l),
generates 512-dim embeddings, clusters them with HDBSCAN, and writes
everything to Postgres (with pgvector) for querying from your .NET app.

Usage:
    # First-time full scan:
    python scan_faces.py scan --photo-dir /path/to/photos

    # Cluster detected faces into persons:
    python scan_faces.py cluster

    # Process only new/unscanned photos (incremental):
    python scan_faces.py scan --photo-dir /path/to/photos --incremental

    # Show stats about current database state:
    python scan_faces.py stats
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import hdbscan
import numpy as np
import psycopg
from insightface.app import FaceAnalysis
from psycopg.rows import dict_row
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration — override via environment variables or edit defaults below
# ---------------------------------------------------------------------------

DB_DSN = os.getenv(
    "FACE_SCANNER_DSN",
    "postgresql://face_scanner:face_scanner_secret@localhost:5433/face_scanner",
)

# InsightFace model name. buffalo_l is the best quality; buffalo_s is faster.
INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")

# Minimum face detection confidence (0–1). Faces below this are discarded.
DET_SCORE_THRESHOLD = float(os.getenv("DET_SCORE_THRESHOLD", "0.70"))

# Minimum face size in pixels (width or height). Tiny faces degrade accuracy.
MIN_FACE_SIZE_PX = int(os.getenv("MIN_FACE_SIZE_PX", "40"))

# Supported image extensions (lowercase).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp", ".avif"}

# HDBSCAN clustering parameters.
HDBSCAN_MIN_CLUSTER_SIZE = int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "5"))
HDBSCAN_MIN_SAMPLES = int(os.getenv("HDBSCAN_MIN_SAMPLES", "3"))
HDBSCAN_CLUSTER_THRESHOLD = float(os.getenv("HDBSCAN_CLUSTER_THRESHOLD", "0.4"))

# Cosine distance threshold for incremental recognition (0 = identical, 2 = opposite).
# Faces with nearest-neighbor distance below this are assigned to the existing person.
RECOGNITION_DISTANCE_THRESHOLD = float(
    os.getenv("RECOGNITION_DISTANCE_THRESHOLD", "0.40")
)

# How many images to process between database commits.
BATCH_COMMIT_SIZE = int(os.getenv("BATCH_COMMIT_SIZE", "50"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS photos (
    id              BIGSERIAL PRIMARY KEY,
    file_path       TEXT NOT NULL UNIQUE,
    file_name       TEXT NOT NULL,
    scanned_at      TIMESTAMPTZ DEFAULT now(),
    scan_error      TEXT
);

CREATE TABLE IF NOT EXISTS persons (
    id                       BIGSERIAL PRIMARY KEY,
    name                     VARCHAR(255),
    cluster_label            INT,
    representative_embedding vector(512),
    face_count               INT DEFAULT 0,
    created_at               TIMESTAMPTZ DEFAULT now(),
    updated_at               TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS face_detections (
    id              BIGSERIAL PRIMARY KEY,
    photo_id        BIGINT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    person_id       BIGINT REFERENCES persons(id) ON DELETE SET NULL,
    bounding_box    JSONB NOT NULL,
    embedding       vector(512),
    detection_score FLOAT NOT NULL,
    face_width_px   INT NOT NULL,
    face_height_px  INT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- ANN index for fast similarity search. Rebuild after bulk inserts if needed:
--   DROP INDEX IF EXISTS face_detections_embedding_idx;
--   CREATE INDEX face_detections_embedding_idx ON face_detections
--       USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS face_detections_embedding_idx
    ON face_detections USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS face_detections_photo_id_idx
    ON face_detections(photo_id);

CREATE INDEX IF NOT EXISTS face_detections_person_id_idx
    ON face_detections(person_id);
"""


def get_connection(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn, row_factory=dict_row)


def ensure_schema(conn: psycopg.Connection) -> None:
    log.info("Ensuring database schema is up to date…")
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()
    log.info("Schema OK.")


def get_already_scanned_paths(conn: psycopg.Connection) -> set[str]:
    """Return file paths that already have a photos row (success or error)."""
    with conn.cursor() as cur:
        cur.execute("SELECT file_path FROM photos")
        return {row["file_path"] for row in cur.fetchall()}


def upsert_photo(
    conn: psycopg.Connection, file_path: str, error: str | None = None
) -> int:
    """Insert or update a photos row; return the photo id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO photos (file_path, file_name, scan_error)
            VALUES (%s, %s, %s)
            ON CONFLICT (file_path) DO UPDATE
                SET scanned_at = now(),
                    scan_error = EXCLUDED.scan_error
            RETURNING id
            """,
            (file_path, Path(file_path).name, error),
        )
        return cur.fetchone()["id"]


def insert_face(
    conn: psycopg.Connection,
    photo_id: int,
    bbox: dict,
    embedding: np.ndarray,
    det_score: float,
    face_w: int,
    face_h: int,
) -> int:
    """Insert a face detection row; return the face id."""
    import json

    vec_str = "[" + ",".join(f"{v:.8f}" for v in embedding.tolist()) + "]"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO face_detections
                (photo_id, bounding_box, embedding, detection_score, face_width_px, face_height_px)
            VALUES (%s, %s, %s::vector, %s, %s, %s)
            RETURNING id
            """,
            (photo_id, json.dumps(bbox), vec_str, det_score, face_w, face_h),
        )
        return cur.fetchone()["id"]


# ---------------------------------------------------------------------------
# InsightFace model loader
# ---------------------------------------------------------------------------

_face_app: FaceAnalysis | None = None


def get_face_app() -> FaceAnalysis:
    global _face_app
    if _face_app is None:
        log.info(
            f"Loading InsightFace model '{INSIGHTFACE_MODEL}' (first call may download ~300MB)…"
        )
        _face_app = FaceAnalysis(
            name=INSIGHTFACE_MODEL,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        # det_size: larger = more accurate but slower. 640 is a good balance.
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
        log.info("InsightFace model loaded.")
    return _face_app


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------


def load_image(file_path: str) -> np.ndarray | None:
    """Load an image as BGR numpy array. Returns None on failure."""
    try:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        # Cap very large images to avoid OOM — InsightFace handles resizing internally,
        # but extremely large images still consume memory during loading.
        h, w = img.shape[:2]
        max_dim = 4096
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        return img
    except Exception as e:
        log.warning(f"Failed to load image {file_path}: {e}")
        return None


def detect_faces(img: np.ndarray) -> list[dict]:
    """
    Run InsightFace detection on a BGR image.
    Returns a list of face dicts with keys: bbox, embedding, det_score, face_w, face_h.
    """
    app = get_face_app()
    try:
        faces = app.get(img)
    except Exception as e:
        log.warning(f"InsightFace inference error: {e}")
        return []

    results = []
    for face in faces:
        if face.det_score < DET_SCORE_THRESHOLD:
            continue
        if face.embedding is None:
            continue

        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        face_w = max(0, x2 - x1)
        face_h = max(0, y2 - y1)

        if face_w < MIN_FACE_SIZE_PX or face_h < MIN_FACE_SIZE_PX:
            continue

        img_h, img_w = img.shape[:2]
        bbox = {
            "x": round(x1 / img_w, 6),
            "y": round(y1 / img_h, 6),
            "width": round(face_w / img_w, 6),
            "height": round(face_h / img_h, 6),
            "x_px": x1,
            "y_px": y1,
            "width_px": face_w,
            "height_px": face_h,
        }

        # L2-normalise the embedding so cosine distance = euclidean distance.
        embedding = face.embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        results.append(
            {
                "bbox": bbox,
                "embedding": embedding,
                "det_score": float(face.det_score),
                "face_w": face_w,
                "face_h": face_h,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------


def iter_images(photo_dir: str) -> list[Path]:
    root = Path(photo_dir)
    if not root.exists():
        log.error(f"Photo directory does not exist: {photo_dir}")
        sys.exit(1)
    paths = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and "full" in p.parts  # only images inside a directory named 'full'
    ]
    paths.sort()
    return paths


# ---------------------------------------------------------------------------
# SCAN command
# ---------------------------------------------------------------------------


def cmd_scan(photo_dir: str, incremental: bool) -> None:
    log.info(f"Scanning photos in: {photo_dir}")
    log.info(f"Incremental mode: {incremental}")

    conn = get_connection(DB_DSN)
    ensure_schema(conn)

    all_images = iter_images(photo_dir)
    log.info(f"Found {len(all_images):,} image(s) in directory tree.")

    if incremental:
        already_scanned = get_already_scanned_paths(conn)
        images_to_scan = [p for p in all_images if str(p) not in already_scanned]
        log.info(
            f"{len(already_scanned):,} already scanned; {len(images_to_scan):,} new image(s) to process."
        )
    else:
        images_to_scan = all_images
        log.info("Full scan: processing all images.")

    if not images_to_scan:
        log.info("Nothing to scan.")
        conn.close()
        return

    total_faces = 0
    errors = 0
    batch_count = 0

    with tqdm(images_to_scan, unit="img", desc="Scanning") as pbar:
        for img_path in pbar:
            path_str = str(img_path)
            pbar.set_postfix(faces=total_faces, errors=errors)

            img = load_image(path_str)
            if img is None:
                photo_id = upsert_photo(conn, path_str, error="Failed to load image")
                errors += 1
                batch_count += 1
            else:
                faces = detect_faces(img)
                photo_id = upsert_photo(conn, path_str)

                for face in faces:
                    insert_face(
                        conn,
                        photo_id,
                        face["bbox"],
                        face["embedding"],
                        face["det_score"],
                        face["face_w"],
                        face["face_h"],
                    )
                    total_faces += 1

                batch_count += 1

            if batch_count >= BATCH_COMMIT_SIZE:
                conn.commit()
                batch_count = 0

    conn.commit()
    conn.close()

    log.info(
        f"Scan complete. {len(images_to_scan):,} images processed, "
        f"{total_faces:,} faces detected, {errors:,} errors."
    )
    log.info("Run 'cluster' command next to group faces into persons.")


# ---------------------------------------------------------------------------
# CLUSTER command
# ---------------------------------------------------------------------------


def cmd_cluster() -> None:
    """
    Load all embeddings from face_detections, run HDBSCAN, write persons rows,
    and update face_detections.person_id.
    """
    log.info("Loading embeddings from database for clustering…")
    conn = get_connection(DB_DSN)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, embedding::text
            FROM face_detections
            WHERE embedding IS NOT NULL
            ORDER BY id
            """
        )
        rows = cur.fetchall()

    if not rows:
        log.warning("No embeddings found. Run 'scan' first.")
        conn.close()
        return

    log.info(f"Loaded {len(rows):,} face embeddings.")

    face_ids = []
    embeddings = []
    for row in rows:
        face_ids.append(row["id"])
        vec = np.fromstring(row["embedding"].strip("[]"), sep=",", dtype=np.float32)
        embeddings.append(vec)

    X = np.array(embeddings, dtype=np.float32)

    log.info(
        f"Running HDBSCAN (min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, "
        f"min_samples={HDBSCAN_MIN_SAMPLES})…"
    )
    t0 = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",  # cosine ≈ euclidean on L2-normalised vectors
        cluster_selection_epsilon=HDBSCAN_CLUSTER_THRESHOLD,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(X)
    elapsed = time.time() - t0

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int(np.sum(labels == -1))
    log.info(
        f"Clustering done in {elapsed:.1f}s. "
        f"{n_clusters} clusters found, {n_noise} noise faces (unclustered)."
    )

    # Build cluster → face index mapping
    cluster_to_indices: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        cluster_to_indices.setdefault(label, []).append(i)

    log.info("Writing persons and updating face_detections…")

    # Clear existing auto-generated persons (keep manually named ones).
    with conn.cursor() as cur:
        cur.execute("DELETE FROM persons WHERE name IS NULL")
    conn.commit()

    person_count = 0
    face_update_count = 0

    for cluster_label, indices in tqdm(
        cluster_to_indices.items(), desc="Writing clusters"
    ):
        cluster_embeddings = X[indices]
        centroid = cluster_embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        centroid_str = "[" + ",".join(f"{v:.8f}" for v in centroid.tolist()) + "]"

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO persons (cluster_label, representative_embedding, face_count)
                VALUES (%s, %s::vector, %s)
                RETURNING id
                """,
                (int(cluster_label), centroid_str, len(indices)),
            )
            person_id = cur.fetchone()["id"]

        # Update each face in this cluster
        cluster_face_ids = [face_ids[i] for i in indices]
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE face_detections SET person_id = %s WHERE id = ANY(%s)",
                (person_id, cluster_face_ids),
            )
        face_update_count += len(cluster_face_ids)
        person_count += 1

    conn.commit()
    conn.close()

    log.info(
        f"Clustering complete. {person_count} persons created, "
        f"{face_update_count} faces assigned, {n_noise} faces left unassigned (noise)."
    )
    log.info("Next step: label persons via your admin UI, or inspect with 'stats'.")


# ---------------------------------------------------------------------------
# STATS command
# ---------------------------------------------------------------------------


def cmd_stats() -> None:
    conn = get_connection(DB_DSN)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM photos")
        n_photos = cur.fetchone()["n"]

        cur.execute("SELECT COUNT(*) AS n FROM photos WHERE scan_error IS NOT NULL")
        n_errors = cur.fetchone()["n"]

        cur.execute("SELECT COUNT(*) AS n FROM face_detections")
        n_faces = cur.fetchone()["n"]

        cur.execute("SELECT COUNT(*) AS n FROM face_detections WHERE person_id IS NULL")
        n_unassigned = cur.fetchone()["n"]

        cur.execute("SELECT COUNT(*) AS n FROM persons")
        n_persons = cur.fetchone()["n"]

        cur.execute("SELECT COUNT(*) AS n FROM persons WHERE name IS NOT NULL")
        n_named = cur.fetchone()["n"]

        cur.execute(
            """
            SELECT per.name, per.face_count
            FROM persons per
            WHERE per.name IS NOT NULL
            ORDER BY per.face_count DESC
            LIMIT 20
            """
        )
        named_persons = cur.fetchall()

    conn.close()

    print("\n=== Face Scanner Stats ===")
    print(f"  Photos scanned:      {n_photos:>8,}")
    print(f"  Photos with errors:  {n_errors:>8,}")
    print(f"  Faces detected:      {n_faces:>8,}")
    print(f"  Faces assigned:      {n_faces - n_unassigned:>8,}")
    print(f"  Faces unassigned:    {n_unassigned:>8,}")
    print(f"  Persons (clusters):  {n_persons:>8,}")
    print(f"  Named persons:       {n_named:>8,}")

    if named_persons:
        print("\n  Top named persons:")
        for row in named_persons:
            print(f"    {row['name']:<30} {row['face_count']:>6} faces")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Face detection & recognition scanner for local photo libraries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = sub.add_parser("scan", help="Detect faces in photos and store embeddings.")
    p_scan.add_argument(
        "--photo-dir",
        required=True,
        help="Root directory of your photo library.",
    )
    p_scan.add_argument(
        "--incremental",
        action="store_true",
        help="Skip photos that have already been scanned.",
    )

    # cluster
    sub.add_parser(
        "cluster", help="Cluster stored embeddings into persons using HDBSCAN."
    )

    # stats
    sub.add_parser("stats", help="Print database statistics.")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args.photo_dir, args.incremental)
    elif args.command == "cluster":
        cmd_cluster()
    elif args.command == "stats":
        cmd_stats()


if __name__ == "__main__":
    main()
