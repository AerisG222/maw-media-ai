#!/usr/bin/env python3
"""
Face Detection & Recognition Scanner
=====================================
Scans a photo directory tree, detects faces using InsightFace (buffalo_l),
generates 512-dim embeddings, clusters them with HDBSCAN, and writes
everything to Postgres (with pgvector) for querying from your .NET app.

Usage:
    # Scan new/unscanned photos (already-scanned photos are skipped):
    python scan_faces.py scan --photo-dir /path/to/photos

    # Cluster detected faces into persons:
    python scan_faces.py cluster

    # Suggest person assignments for unvalidated faces (writes suggested_person_id):
    python scan_faces.py suggest

    # Show stats about current database state:
    python scan_faces.py stats
"""

import argparse
import itertools
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import hdbscan
import numpy as np
import psycopg
from insightface.app import FaceAnalysis
from psycopg.rows import dict_row
from tqdm import tqdm

from face_cache import face_crop_path

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

# Cosine distance threshold for the suggest command.
# Only faces whose nearest named-person centroid is within this distance receive a
# suggestion.  Lower = more conservative (fewer but higher-confidence suggestions).
SUGGEST_DISTANCE_THRESHOLD = float(os.getenv("SUGGEST_DISTANCE_THRESHOLD", "0.35"))

# Cosine distance threshold for merge-clusters.
# Unnamed clusters whose centroid is within this distance of a named person's
# centroid are merged into that person.  Tighter than suggest because a whole
# cluster is moved at once.
MERGE_DISTANCE_THRESHOLD = float(os.getenv("MERGE_DISTANCE_THRESHOLD", "0.25"))

# Laplacian variance threshold below which a face crop is considered blurry.
# Higher = sharper.  Typical face crops: <50 very blurry, 50-150 moderate, >150 sharp.
BLUR_SCORE_THRESHOLD = float(os.getenv("BLUR_SCORE_THRESHOLD", "80.0"))

# Longest-edge cap (px) for the JPEG face crops cached on disk for the UI.
# Large enough for crisp thumbnails, small enough to keep the cache compact.
FACE_CROP_MAX_DIM = int(os.getenv("FACE_CROP_MAX_DIM", "400"))

# How many images to process between database commits.
BATCH_COMMIT_SIZE = int(os.getenv("BATCH_COMMIT_SIZE", "50"))

# Background threads used to decode/resize images ahead of detection.  Detection
# itself runs serially on the single model; this just overlaps the (slow, e.g.
# AVIF) image I/O with inference.  Override with --workers or SCAN_LOADER_THREADS.
SCAN_LOADER_THREADS = int(os.getenv("SCAN_LOADER_THREADS", "4"))

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


def get_connection(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn, row_factory=dict_row)


def check_schema(conn: psycopg.Connection) -> bool:
    """Check if the required tables exist. Returns True if schema is present."""
    required_tables = {"photos", "persons", "face_detections"}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = {row["table_name"] for row in cur.fetchall()}
    return required_tables.issubset(tables)


def get_already_scanned_paths(conn: psycopg.Connection) -> set[str]:
    """Return file paths that already have a photos row (success or error)."""
    with conn.cursor() as cur:
        cur.execute("SELECT file_path FROM photos")
        return {row["file_path"] for row in cur.fetchall()}


import uuid


def upsert_photo(
    conn: psycopg.Connection, file_path: str, error: str | None = None
) -> str:
    """Insert or update a photos row; return the photo id (UUID string)."""
    with conn.cursor() as cur:
        # Try to fetch existing photo id first
        cur.execute("SELECT id FROM photos WHERE file_path = %s", (file_path,))
        row = cur.fetchone()
        if row:
            # Update scan_error if needed
            cur.execute(
                """
                UPDATE photos SET scanned_at = now(), scan_error = %s WHERE id = %s
                """,
                (error, row["id"]),
            )
            return row["id"]
        # Insert new photo with generated UUIDv7
        photo_id = str(uuid.uuid7())
        cur.execute(
            """
            INSERT INTO photos (id, file_path, file_name, scan_error)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (photo_id, file_path, Path(file_path).name, error),
        )
        return cur.fetchone()["id"]


def insert_face(
    conn: psycopg.Connection,
    photo_id: str,
    face_id: str,
    bbox: dict,
    embedding: np.ndarray,
    det_score: float,
    face_w: int,
    face_h: int,
    blur_score: float | None = None,
    person_id: str | None = None,
) -> str:
    """Insert a face detection row; return the face id (UUID string).

    The caller supplies ``face_id`` so it can name the cached crop file before
    the row exists.
    """
    vec_str = "[" + ",".join(f"{v:.8f}" for v in embedding.tolist()) + "]"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO face_detections
                (id, photo_id, bounding_box, embedding, detection_score, face_width_px, face_height_px, blur_score, person_id)
            VALUES (%s, %s, %s, %s::vector, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                face_id,
                photo_id,
                json.dumps(bbox),
                vec_str,
                det_score,
                face_w,
                face_h,
                blur_score,
                person_id,
            ),
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


def _crop_region(img_bgr: np.ndarray, bbox: dict) -> np.ndarray | None:
    """Return the face crop (BGR ndarray) for a bbox, or None if degenerate.

    Uses the normalized (0–1) coordinates scaled to the actual image size, so
    the crop is correct regardless of whether the image was downscaled since
    detection.  The stored ``*_px`` values assume the resolution detection ran
    at and would mis-crop a full-resolution re-read of a large photo.
    """
    img_h, img_w = img_bgr.shape[:2]
    x = bbox.get("x")
    y = bbox.get("y")
    w = bbox.get("width")
    h = bbox.get("height")

    if None not in (x, y, w, h):
        x1 = int(round(x * img_w))
        y1 = int(round(y * img_h))
        x2 = int(round((x + w) * img_w))
        y2 = int(round((y + h) * img_h))
    else:
        # Fall back to pixel coords for older rows lacking normalized values.
        x1 = int(bbox.get("x_px", 0))
        y1 = int(bbox.get("y_px", 0))
        x2 = x1 + int(bbox.get("width_px", 0))
        y2 = y1 + int(bbox.get("height_px", 0))

    x1, y1 = max(0, min(x1, img_w)), max(0, min(y1, img_h))
    x2, y2 = max(0, min(x2, img_w)), max(0, min(y2, img_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2]


def blur_score_for_crop(img_bgr: np.ndarray, bbox: dict) -> float:
    """Return the Laplacian variance of a face crop. Higher = sharper.

    Computed on the full-resolution crop so scores stay comparable regardless
    of the (possibly downscaled) JPEG saved for the UI.
    """
    crop = _crop_region(img_bgr, bbox)
    if crop is None:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def write_face_crop(img_bgr: np.ndarray, bbox: dict, crop_path: Path) -> bool:
    """Write a JPEG of the face crop to *crop_path* (creating parent dirs).

    The crop is downscaled so its longest edge is at most FACE_CROP_MAX_DIM and
    written atomically.  Returns True on success.
    """
    crop = _crop_region(img_bgr, bbox)
    if crop is None:
        return False

    ch, cw = crop.shape[:2]
    longest = max(ch, cw)
    if longest > FACE_CROP_MAX_DIM:
        scale = FACE_CROP_MAX_DIM / longest
        crop = cv2.resize(
            crop,
            (max(1, round(cw * scale)), max(1, round(ch * scale))),
            interpolation=cv2.INTER_AREA,
        )

    crop_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = crop_path.with_name(f"{crop_path.stem}.tmp-{uuid.uuid4().hex}.jpg")
    ok = cv2.imwrite(str(tmp), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if ok:
        os.replace(str(tmp), str(crop_path))
    else:
        tmp.unlink(missing_ok=True)
    return bool(ok)


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


def _iter_prefetched_images(paths: list[str | Path], n_workers: int):
    """Yield ``(path_str, img)`` in order, decoding images in worker threads.

    A bounded look-ahead keeps at most ~2×workers images resident so image I/O
    overlaps with detection on the main thread without ballooning memory.
    ``img`` is None when a file could not be loaded (same as load_image).
    """
    n_workers = max(1, n_workers)
    window = n_workers * 2
    with ThreadPoolExecutor(
        max_workers=n_workers, thread_name_prefix="img-load"
    ) as pool:
        it = iter(paths)
        pending: deque = deque()
        for p in itertools.islice(it, window):
            pending.append((str(p), pool.submit(load_image, str(p))))

        while pending:
            path_str, fut = pending.popleft()
            img = fut.result()
            nxt = next(it, None)
            if nxt is not None:
                pending.append((str(nxt), pool.submit(load_image, str(nxt))))
            yield path_str, img


def cmd_scan(photo_dir: str, workers: int = SCAN_LOADER_THREADS) -> None:
    log.info(f"Scanning photos in: {photo_dir}")

    conn = get_connection(DB_DSN)
    if not check_schema(conn):
        log.error(
            "Database schema is missing. Please run setup-db.sh to initialize the schema."
        )
        sys.exit(1)

    all_images = iter_images(photo_dir)
    log.info(f"Found {len(all_images):,} image(s) in directory tree.")

    # Skip images that already have a photos row.
    already_scanned = get_already_scanned_paths(conn)
    images_to_scan = [p for p in all_images if str(p) not in already_scanned]
    log.info(
        f"{len(already_scanned):,} already scanned; {len(images_to_scan):,} new image(s) to process."
    )

    if not images_to_scan:
        log.info("Nothing to scan.")
        conn.close()
        return

    # Load the model before the loop so its startup cost isn't attributed to the
    # first image and the decode threads have detection to overlap with.
    get_face_app()
    log.info(f"Decoding images with {max(1, workers)} loader thread(s).")

    total_faces = 0
    errors = 0
    batch_count = 0

    with tqdm(total=len(images_to_scan), unit="img", desc="Scanning") as pbar:
        for path_str, img in _iter_prefetched_images(images_to_scan, workers):
            pbar.update(1)
            pbar.set_postfix(faces=total_faces, errors=errors)

            if img is None:
                photo_id = upsert_photo(conn, path_str, error="Failed to load image")
                errors += 1
                batch_count += 1
            else:
                faces = detect_faces(img)
                photo_id = upsert_photo(conn, path_str)

                for face in faces:
                    face_id = str(uuid.uuid7())
                    bbox = face["bbox"]
                    # Extract the crop for the UI and score its sharpness while
                    # the source image is already in memory.  Only compute blur
                    # when the crop isn't already cached on disk.
                    crop_path = face_crop_path(path_str, face_id)
                    if crop_path.exists():
                        blur = None
                    else:
                        blur = blur_score_for_crop(img, bbox)
                        write_face_crop(img, bbox, crop_path)
                    insert_face(
                        conn,
                        photo_id,
                        face_id,
                        bbox,
                        face["embedding"],
                        face["det_score"],
                        face["face_w"],
                        face["face_h"],
                        blur_score=blur,
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
            person_id = str(uuid.uuid7())
            cur.execute(
                """
                INSERT INTO persons (id, cluster_label, representative_embedding, face_count)
                VALUES (%s, %s, %s::vector, %s)
                RETURNING id
                """,
                (person_id, int(cluster_label), centroid_str, len(indices)),
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
# SUGGEST command
# ---------------------------------------------------------------------------


def cmd_suggest(threshold: float) -> None:
    """
    For every unassigned or unnamed-cluster face with an embedding, find its
    nearest named-person centroid via pgvector and write a suggestion when the
    cosine distance is below *threshold*.

    Candidates are faces where:
      - person_id IS NULL (completely unassigned), OR
      - person_id points to a person with name IS NULL (assigned to an unnamed cluster)

    Already-suggested faces are re-evaluated so that re-running after labelling
    more clusters can improve or replace earlier suggestions.  Validated faces
    (is_validated = TRUE) in NAMED clusters are never touched.
    """
    log.info(f"Running suggest with distance threshold {threshold:.3f}…")
    conn = get_connection(DB_DSN)

    # Count candidates: unassigned faces + faces in unnamed clusters
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM face_detections fd
            LEFT JOIN persons p ON p.id = fd.person_id
            WHERE (fd.person_id IS NULL OR p.name IS NULL)
              AND fd.is_validated = FALSE
              AND fd.embedding IS NOT NULL
            """
        )
        n_candidates = cur.fetchone()["count"]

    if n_candidates == 0:
        log.info("No candidate faces found — nothing to do.")
        conn.close()
        return

    log.info(f"{n_candidates:,} candidate face(s) to evaluate.")

    # Reset any previous unvalidated suggestions so stale ones don't linger.
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE face_detections fd
            SET suggested_person_id = NULL,
                suggestion_score    = NULL
            FROM (
                SELECT fd2.id
                FROM face_detections fd2
                LEFT JOIN persons p ON p.id = fd2.person_id
                WHERE (fd2.person_id IS NULL OR p.name IS NULL)
                  AND fd2.is_validated = FALSE
            ) candidates
            WHERE fd.id = candidates.id
            """
        )
    conn.commit()
    log.info("Cleared previous unvalidated suggestions.")

    # For each candidate, find the single closest named-person centroid and
    # write a suggestion if within the threshold.
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE face_detections fd
            SET suggested_person_id = best.person_id,
                suggestion_score    = best.distance
            FROM (
                SELECT
                    fd.id AS face_id,
                    nearest.id       AS person_id,
                    (fd.embedding <=> nearest.representative_embedding) AS distance
                FROM face_detections fd
                LEFT JOIN persons src_p ON src_p.id = fd.person_id
                CROSS JOIN LATERAL (
                    SELECT p.id, p.representative_embedding
                    FROM persons p
                    WHERE p.name IS NOT NULL
                      AND p.representative_embedding IS NOT NULL
                    ORDER BY p.representative_embedding <=> fd.embedding
                    LIMIT 1
                ) nearest
                WHERE (fd.person_id IS NULL OR src_p.name IS NULL)
                  AND fd.is_validated = FALSE
                  AND fd.embedding   IS NOT NULL
                  AND (fd.embedding <=> nearest.representative_embedding) < %(threshold)s
            ) best
            WHERE fd.id = best.face_id
            """,
            {"threshold": threshold},
        )
        n_suggested = cur.rowcount
    conn.commit()

    n_skipped = n_candidates - n_suggested
    log.info(
        f"Suggest complete. {n_suggested:,} suggestion(s) written, "
        f"{n_skipped:,} face(s) had no match within threshold {threshold:.3f}."
    )
    log.info("Open the Streamlit UI and use 'Review suggestions' to confirm or reject.")
    conn.close()


# ---------------------------------------------------------------------------
# MERGE-CLUSTERS command
# ---------------------------------------------------------------------------


def cmd_merge_clusters(threshold: float, dry_run: bool) -> None:
    """
    For each unnamed cluster, find the nearest named-person centroid via
    pgvector.  If the cosine distance is below *threshold*, merge the cluster
    into that person (move all its faces, recompute face_count, delete the
    empty cluster).

    Always run with --dry-run first to review the plan before committing.
    """
    prefix = "[DRY RUN] " if dry_run else ""
    log.info(f"{prefix}Running merge-clusters (threshold={threshold:.3f})…")
    conn = get_connection(DB_DSN)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                p_unnamed.id            AS unnamed_id,
                p_unnamed.cluster_label AS cluster_label,
                p_unnamed.face_count    AS face_count,
                nearest.id              AS named_id,
                nearest.name            AS name,
                (p_unnamed.representative_embedding <=> nearest.representative_embedding)
                                        AS distance
            FROM persons p_unnamed
            CROSS JOIN LATERAL (
                SELECT p.id, p.name, p.representative_embedding
                FROM persons p
                WHERE p.name IS NOT NULL
                  AND p.representative_embedding IS NOT NULL
                ORDER BY p.representative_embedding <=> p_unnamed.representative_embedding
                LIMIT 1
            ) nearest
            WHERE p_unnamed.name IS NULL
              AND p_unnamed.representative_embedding IS NOT NULL
              AND (p_unnamed.representative_embedding <=> nearest.representative_embedding)
                  < %(threshold)s
            ORDER BY distance ASC
            """,
            {"threshold": threshold},
        )
        candidates = cur.fetchall()

    if not candidates:
        log.info("No unnamed clusters found within threshold — nothing to merge.")
        conn.close()
        return

    total_faces = sum(row["face_count"] or 0 for row in candidates)
    log.info(
        f"{prefix}Found {len(candidates)} unnamed cluster(s) to merge "
        f"({total_faces:,} faces total):"
    )
    for row in candidates:
        label = (
            f"cluster_{row['cluster_label']}"
            if row["cluster_label"] is not None
            else str(row["unnamed_id"])[:8]
        )
        log.info(
            f"  {label:<20} ({row['face_count'] or 0:>5} faces)"
            f"  →  \"{row['name']}\"  distance={row['distance']:.4f}"
        )

    if dry_run:
        log.info("No changes made. Re-run without --dry-run to apply.")
        conn.close()
        return

    for row in tqdm(candidates, desc="Merging clusters"):
        unnamed_id = row["unnamed_id"]
        named_id = row["named_id"]
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE face_detections SET person_id = %s WHERE person_id = %s",
                (named_id, unnamed_id),
            )
            cur.execute(
                """
                UPDATE persons
                SET face_count = (SELECT COUNT(*) FROM face_detections WHERE person_id = %s),
                    updated_at = now()
                WHERE id = %s
                """,
                (named_id, named_id),
            )
            cur.execute("DELETE FROM persons WHERE id = %s", (unnamed_id,))
        conn.commit()

    log.info(
        f"Merged {len(candidates)} unnamed cluster(s) into named persons "
        f"({total_faces:,} faces reassigned)."
    )
    log.info("Tip: run 'suggest' again to pick up any remaining unassigned faces.")
    conn.close()


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


# ---------------------------------------------------------------------------
# DETECT-BLUR command
# ---------------------------------------------------------------------------


def cmd_detect_blur(overwrite: bool = False, workers: int = SCAN_LOADER_THREADS) -> None:
    """
    Backfill the UI face-crop cache and blur scores for already-scanned faces.

    For every face this ensures both:
      * a cached JPEG crop under image-cache/faces (written when missing), and
      * a blur_score (Laplacian variance of the crop; higher = sharper).

    Faces are grouped by source photo so each image file is loaded only once,
    and a photo is only opened when at least one of its faces still needs work.
    Images are decoded in background threads to overlap I/O with scoring.
    Re-run with --overwrite to re-write every crop and re-score every face.
    """
    log.info("Running detect-blur%s…", " (overwrite)" if overwrite else "")
    conn = get_connection(DB_DSN)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT fd.id, ph.file_path, fd.bounding_box, fd.blur_score
            FROM face_detections fd
            JOIN photos ph ON ph.id = fd.photo_id
            ORDER BY ph.file_path
            """
        )
        rows = cur.fetchall()

    if not rows:
        log.info("No faces found — nothing to do.")
        conn.close()
        return

    # Group by file path to load each image only once.
    by_photo: dict[str, list] = defaultdict(list)
    for row in rows:
        by_photo[row["file_path"]].append(row)

    # Decide what each face needs before touching any (slow) source file, so we
    # only decode photos that actually have work to do.
    photos_with_work: list[tuple[str, list]] = []
    for file_path, faces in by_photo.items():
        work = []
        for face in faces:
            crop_path = face_crop_path(file_path, face["id"])
            need_blur = overwrite or face["blur_score"] is None
            need_crop = overwrite or not crop_path.exists()
            if need_blur or need_crop:
                work.append((face, crop_path, need_blur, need_crop))
        if work:
            photos_with_work.append((file_path, work))

    if not photos_with_work:
        log.info("All faces already have crops and blur scores — nothing to do.")
        conn.close()
        return

    log.info(
        "%d photo(s) need work; decoding with %d loader thread(s).",
        len(photos_with_work),
        max(1, workers),
    )
    work_by_path = dict(photos_with_work)
    paths = [fp for fp, _ in photos_with_work]

    n_scored = 0
    n_crops = 0
    n_errors = 0
    processed = 0

    with conn.cursor() as cur:
        for file_path, img in tqdm(
            _iter_prefetched_images(paths, workers),
            total=len(paths),
            unit="photo",
            desc="Backfilling",
        ):
            work = work_by_path[file_path]
            if img is None:
                log.warning(
                    "Could not read %s — setting blur_score=0 for %d face(s).",
                    file_path,
                    len(work),
                )
                ids = [f["id"] for f, _, need_blur, _ in work if need_blur]
                if ids:
                    cur.execute(
                        "UPDATE face_detections SET blur_score = 0 WHERE id = ANY(%s::uuid[])",
                        (ids,),
                    )
                n_errors += len(work)
                continue

            for face, crop_path, need_blur, need_crop in work:
                bbox = face["bounding_box"]
                if isinstance(bbox, str):
                    bbox = json.loads(bbox)
                if need_blur:
                    score = blur_score_for_crop(img, bbox)
                    cur.execute(
                        "UPDATE face_detections SET blur_score = %s WHERE id = %s",
                        (score, face["id"]),
                    )
                    n_scored += 1
                if need_crop and write_face_crop(img, bbox, crop_path):
                    n_crops += 1

            processed += 1
            if processed % BATCH_COMMIT_SIZE == 0:
                conn.commit()

    conn.commit()
    log.info(
        "detect-blur complete. %d scored, %d crop(s) written, %d unreadable (blur set to 0).",
        n_scored,
        n_crops,
        n_errors,
    )
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Face detection & recognition scanner for local photo libraries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = sub.add_parser(
        "scan",
        help="Detect faces in new photos and store embeddings (skips already-scanned).",
    )
    p_scan.add_argument(
        "--photo-dir",
        required=True,
        help="Root directory of your photo library.",
    )
    p_scan.add_argument(
        "--workers",
        type=int,
        default=SCAN_LOADER_THREADS,
        help=f"Background image-decode threads (default: {SCAN_LOADER_THREADS}). "
             "Detection stays serial; this overlaps image I/O with inference.",
    )

    # cluster
    sub.add_parser(
        "cluster", help="Cluster stored embeddings into persons using HDBSCAN."
    )

    # suggest
    p_suggest = sub.add_parser(
        "suggest",
        help="Suggest person assignments for unassigned faces using nearest-centroid matching.",
    )
    p_suggest.add_argument(
        "--threshold",
        type=float,
        default=SUGGEST_DISTANCE_THRESHOLD,
        help=f"Cosine distance threshold for suggestions (default: {SUGGEST_DISTANCE_THRESHOLD}). "
             "Lower = more conservative.",
    )

    # merge-clusters
    p_merge = sub.add_parser(
        "merge-clusters",
        help="Merge unnamed clusters into the nearest named person when centroids are close.",
    )
    p_merge.add_argument(
        "--threshold",
        type=float,
        default=MERGE_DISTANCE_THRESHOLD,
        help=f"Cosine distance threshold (default: {MERGE_DISTANCE_THRESHOLD}). "
             "Lower = more conservative.",
    )
    p_merge.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which clusters would be merged without making any changes.",
    )

    # detect-blur
    p_blur = sub.add_parser(
        "detect-blur",
        help="Backfill cached face crops and blur scores for already-scanned faces.",
    )
    p_blur.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-write every crop and re-score every face, even if already present.",
    )
    p_blur.add_argument(
        "--workers",
        type=int,
        default=SCAN_LOADER_THREADS,
        help=f"Background image-decode threads (default: {SCAN_LOADER_THREADS}).",
    )

    # stats
    sub.add_parser("stats", help="Print database statistics.")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args.photo_dir, args.workers)
    elif args.command == "cluster":
        cmd_cluster()
    elif args.command == "suggest":
        cmd_suggest(args.threshold)
    elif args.command == "merge-clusters":
        cmd_merge_clusters(args.threshold, args.dry_run)
    elif args.command == "detect-blur":
        cmd_detect_blur(args.overwrite, args.workers)
    elif args.command == "stats":
        cmd_stats()


if __name__ == "__main__":
    main()
