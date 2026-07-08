import base64
import html
import io
import json
import math
import os
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import numpy as np
import psycopg2
import streamlit as st
from PIL import Image

import face_cache

# --- Configuration & Constants ---
DSN = os.environ.get("FACE_SCANNER_DSN")
if not DSN:
    st.error("FACE_SCANNER_DSN environment variable not set.")
    st.stop()

# Thumbnails live at the cache root; pre-extracted face crops live under it in
# faces/.  Shared with the scanner via face_cache so both agree on the location.
CACHE_DIR = face_cache.CACHE_ROOT
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_MAX_ENTRIES = 1200
# Eviction fires when file count exceeds this, trimming back to CACHE_MAX_ENTRIES.
# The buffer avoids eviction on every single new write.
_CACHE_EVICT_THRESHOLD = CACHE_MAX_ENTRIES + 200

GRID_COLS = 6
CELL_WIDTH = 160
CELL_HEIGHT = 200
CAPTION_HEIGHT = 72
IMAGE_HEIGHT = CELL_HEIGHT - CAPTION_HEIGHT

FACES_PAGE_SIZE = 24
PERSONS_PAGE_SIZE = 24

# Which view is showing. Navigation stays within one websocket session (set
# these and st.rerun()), so filter/page state in st.session_state persists.
VIEW_KEY = "view"  # "persons" | "faces" | "unknown" | "review"
VIEW_PERSON_KEY = "view_person_id"


# --- Database Connection ---
def get_connection():
    return psycopg2.connect(DSN)


def execute_query(query: str, params: tuple = ()):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()


def execute_single(query: str, params: tuple = ()):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone()


def execute_update(query: str, params: tuple = ()):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            conn.commit()


def remove_faces_from_person(person_id: str, face_ids: list[str]):
    """Unassign the given faces from a person and sync the face count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE face_detections SET person_id = NULL WHERE id = ANY(%s::uuid[])",
                (face_ids,),
            )
            cur.execute(
                "UPDATE persons SET face_count = (SELECT COUNT(*) FROM face_detections WHERE person_id = %s) WHERE id = %s",
                (person_id, person_id),
            )
            conn.commit()


def assign_faces_to_person(person_id: str, face_ids: list[str]):
    """Assign faces to a person and sync the face count and centroid."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE face_detections SET person_id = %s WHERE id = ANY(%s::uuid[])",
                (person_id, face_ids),
            )
            cur.execute(
                "UPDATE persons SET face_count = (SELECT COUNT(*) FROM face_detections WHERE person_id = %s) WHERE id = %s",
                (person_id, person_id),
            )
            cur.execute(
                """
                UPDATE persons
                SET representative_embedding = (
                    SELECT avg(embedding)::vector
                    FROM face_detections
                    WHERE person_id = %s AND embedding IS NOT NULL
                ),
                updated_at = now()
                WHERE id = %s
                """,
                (person_id, person_id),
            )
            conn.commit()


def clear_cluster(person_id: str):
    """Unassign all faces from a cluster and delete it."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE face_detections SET person_id = NULL, is_validated = FALSE WHERE person_id = %s",
                (person_id,),
            )
            cur.execute("DELETE FROM persons WHERE id = %s", (person_id,))
            conn.commit()


def merge_persons_into(target_id: str, source_ids: list[str]):
    """Move all faces from source persons into target, recompute count, delete sources."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE face_detections SET person_id = %s WHERE person_id = ANY(%s::uuid[])",
                (target_id, source_ids),
            )
            cur.execute(
                "UPDATE persons SET face_count = (SELECT COUNT(*) FROM face_detections WHERE person_id = %s) WHERE id = %s",
                (target_id, target_id),
            )
            cur.execute(
                """
                UPDATE persons
                SET representative_embedding = (
                    SELECT avg(embedding)::vector
                    FROM face_detections
                    WHERE person_id = %s AND embedding IS NOT NULL
                ),
                updated_at = now()
                WHERE id = %s
                """,
                (target_id, target_id),
            )
            # If the target is unnamed, inherit the name from the first named source.
            cur.execute("SELECT name FROM persons WHERE id = %s", (target_id,))
            target_row = cur.fetchone()
            if target_row and target_row[0] is None:
                cur.execute(
                    "SELECT name FROM persons WHERE id = ANY(%s::uuid[]) AND name IS NOT NULL ORDER BY name LIMIT 1",
                    (source_ids,),
                )
                name_row = cur.fetchone()
                if name_row:
                    cur.execute(
                        "UPDATE persons SET name = %s WHERE id = %s",
                        (name_row[0], target_id),
                    )
            cur.execute(
                "DELETE FROM persons WHERE id = ANY(%s::uuid[])",
                (source_ids,),
            )
            conn.commit()


# --- Data Fetching ---
def fetch_persons_count(
    search: str | None = None,
    unnamed_only: bool = False,
) -> int:
    like = f"%{search}%" if search else None
    result = execute_single(
        "SELECT COUNT(1) FROM persons WHERE (%s IS NULL OR name ILIKE %s OR id::text ILIKE %s) AND (NOT %s OR name IS NULL)",
        (search, like, like, unnamed_only),
    )
    return result[0] if result else 0


def fetch_persons_page(
    search: str | None = None,
    limit: int = PERSONS_PAGE_SIZE,
    offset: int = 0,
    unnamed_only: bool = False,
) -> list:
    """Return a page of persons with one sample face for preview.

    Each row: (id, name, cluster_label, face_count, sample_path, sample_score, sample_bbox, sample_face_id)
    """
    like = f"%{search}%" if search else None
    return execute_query(
        """
        SELECT p.id, p.name, p.cluster_label, p.face_count,
               ph.file_path AS sample_path, fd.detection_score AS sample_score,
               fd.bounding_box AS sample_bbox, fd.id AS sample_face_id
        FROM persons p
        LEFT JOIN LATERAL (
            SELECT fd.id, fd.photo_id, fd.detection_score, fd.bounding_box
            FROM face_detections fd
            WHERE fd.person_id = p.id
            ORDER BY fd.detection_score DESC NULLS LAST, fd.id
            LIMIT 1
        ) fd ON true
        LEFT JOIN photos ph ON ph.id = fd.photo_id
        WHERE (%s IS NULL OR p.name ILIKE %s OR p.id::text ILIKE %s)
          AND (NOT %s OR p.name IS NULL)
        ORDER BY p.face_count DESC NULLS LAST, p.id
        LIMIT %s OFFSET %s
        """,
        (search, like, like, unnamed_only, limit, offset),
    )


def fetch_all_persons_embeddings(
    search: str | None = None,
    unnamed_only: bool = False,
) -> list[tuple]:
    """Fetch all matching persons with centroids for similarity-order computation.

    Returns list of (id, face_count, embedding_floats | None), sorted by face_count desc.
    """
    like = f"%{search}%" if search else None
    rows = execute_query(
        """
        SELECT p.id, p.face_count, p.representative_embedding::text
        FROM persons p
        WHERE (%s IS NULL OR p.name ILIKE %s OR p.id::text ILIKE %s)
          AND (NOT %s OR p.name IS NULL)
        ORDER BY p.face_count DESC NULLS LAST, p.id
        """,
        (search, like, like, unnamed_only),
    )
    result = []
    for pid, face_count, emb_text in rows:
        emb = json.loads(emb_text) if emb_text else None
        result.append((str(pid), face_count, emb))
    return result


def _compute_similarity_order(persons_data: list[tuple]) -> list[str]:
    """Greedy nearest-neighbor traversal ordering persons by centroid cosine similarity.

    Starts from the largest cluster and chains each step to the most similar
    unvisited cluster.  Persons without embeddings are appended at the end.

    persons_data: list of (id, face_count, embedding | None)
    Returns: ordered list of person ID strings.
    """
    with_emb = [(p[0], p[2]) for p in persons_data if p[2] is not None]
    without_emb = [p[0] for p in persons_data if p[2] is None]

    if not with_emb:
        return without_emb

    ids = [p[0] for p in with_emb]
    mat = np.array([p[1] for p in with_emb], dtype=np.float32)

    # L2-normalize so dot product equals cosine similarity
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat /= norms

    sim = mat @ mat.T  # (n, n) cosine similarity matrix

    n = len(ids)
    visited = np.zeros(n, dtype=bool)
    order: list[int] = []
    current = 0  # largest cluster is index 0 (sorted by face_count desc)
    visited[current] = True
    order.append(current)

    for _ in range(n - 1):
        row = sim[current].copy()
        row[visited] = -np.inf
        nearest = int(np.argmax(row))
        visited[nearest] = True
        order.append(nearest)
        current = nearest

    return [ids[i] for i in order] + without_emb


def fetch_persons_by_ids(page_ids: list[str]) -> list:
    """Fetch a page of persons in the given ID order (used for similarity sort).

    Returns same tuple format as fetch_persons_page.
    """
    if not page_ids:
        return []
    return execute_query(
        """
        SELECT p.id, p.name, p.cluster_label, p.face_count,
               ph.file_path AS sample_path, fd.detection_score AS sample_score,
               fd.bounding_box AS sample_bbox, fd.id AS sample_face_id
        FROM persons p
        LEFT JOIN LATERAL (
            SELECT fd.id, fd.photo_id, fd.detection_score, fd.bounding_box
            FROM face_detections fd
            WHERE fd.person_id = p.id
            ORDER BY fd.detection_score DESC NULLS LAST, fd.id
            LIMIT 1
        ) fd ON true
        LEFT JOIN photos ph ON ph.id = fd.photo_id
        WHERE p.id = ANY(%s::uuid[])
        ORDER BY array_position(%s::uuid[], p.id)
        """,
        (page_ids, page_ids),
    )


def fetch_person(person_id: str) -> tuple | None:
    """Return person data: (id, name, cluster_label, face_count)."""
    return execute_single(
        "SELECT id, name, cluster_label, face_count FROM persons WHERE id = %s",
        (person_id,),
    )


def fetch_face_count_for_person(person_id: str) -> int:
    """Return the number of faces for a person."""
    result = execute_single(
        "SELECT COUNT(*) FROM face_detections WHERE person_id = %s",
        (person_id,),
    )
    return result[0] if result else 0


def fetch_faces_for_person(
    person_id: str,
    limit: int = FACES_PAGE_SIZE,
    offset: int = 0,
) -> list:
    """Return a page of faces for a person, ordered by detection score.

    Each row: (id, file_path, bounding_box, detection_score)
    """
    return execute_query(
        """
        SELECT fd.id, p.file_path, fd.bounding_box, fd.detection_score
        FROM face_detections fd
        JOIN photos p ON fd.photo_id = p.id
        WHERE fd.person_id = %s
        ORDER BY fd.detection_score DESC NULLS LAST, fd.id
        LIMIT %s OFFSET %s
        """,
        (person_id, limit, offset),
    )


def fetch_face_count_for_unknown() -> int:
    """Return the number of faces that are not associated with any person."""
    result = execute_single(
        "SELECT COUNT(*) FROM face_detections WHERE person_id IS NULL",
    )
    return result[0] if result else 0


def fetch_faces_for_unknown(
    limit: int = FACES_PAGE_SIZE,
    offset: int = 0,
) -> list:
    """Return a page of unassigned faces ordered by detection score.

    Each row: (id, file_path, bounding_box, detection_score)
    """
    return execute_query(
        """
        SELECT fd.id, p.file_path, fd.bounding_box, fd.detection_score
        FROM face_detections fd
        JOIN photos p ON fd.photo_id = p.id
        WHERE fd.person_id IS NULL
        ORDER BY fd.detection_score DESC NULLS LAST, fd.id
        LIMIT %s OFFSET %s
        """,
        (limit, offset),
    )


def fetch_faces_for_unknown_by_similarity(
    person_id: str,
    limit: int = FACES_PAGE_SIZE,
    offset: int = 0,
) -> list:
    """Return unassigned faces sorted by cosine similarity to a person's centroid.

    Each row: (id, file_path, bounding_box, detection_score)
    """
    return execute_query(
        """
        SELECT fd.id, p.file_path, fd.bounding_box, fd.detection_score
        FROM face_detections fd
        JOIN photos p ON fd.photo_id = p.id
        WHERE fd.person_id IS NULL
          AND fd.embedding IS NOT NULL
        ORDER BY fd.embedding <=> (
            SELECT representative_embedding FROM persons WHERE id = %s
        ) ASC
        LIMIT %s OFFSET %s
        """,
        (person_id, limit, offset),
    )


def fetch_all_persons_for_merge(exclude_id: str) -> list:
    """Return all named persons except exclude_id, sorted by name, for the merge picker.

    Each row: (id, name, face_count, sample_path, sample_bbox, sample_face_id)
    """
    return execute_query(
        """
        SELECT p.id, p.name, p.face_count,
               ph.file_path  AS sample_path,
               fd.bounding_box AS sample_bbox,
               fd.id AS sample_face_id
        FROM persons p
        LEFT JOIN LATERAL (
            SELECT fd.id, fd.photo_id, fd.bounding_box
            FROM face_detections fd
            WHERE fd.person_id = p.id
            ORDER BY fd.detection_score DESC NULLS LAST
            LIMIT 1
        ) fd ON true
        LEFT JOIN photos ph ON ph.id = fd.photo_id
        WHERE p.id != %s
          AND p.name IS NOT NULL
        ORDER BY p.name
        """,
        (exclude_id,),
    )


def fetch_named_persons_for_assign() -> list:
    """Return all named persons for use in the assign-unknown dropdown.

    Each row: (id, name, face_count)
    """
    return execute_query(
        """
        SELECT id, name, face_count
        FROM persons
        WHERE name IS NOT NULL
        ORDER BY name
        """,
        (),
    )


def cleanup_persons() -> tuple[int, int]:
    """Recompute face_count for all persons, then delete any with zero faces.

    Returns (n_updated, n_deleted).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE persons
                SET face_count = (
                    SELECT COUNT(*) FROM face_detections WHERE person_id = persons.id
                )
                """
            )
            n_updated = cur.rowcount
            cur.execute("DELETE FROM persons WHERE face_count = 0")
            n_deleted = cur.rowcount
            conn.commit()
    return n_updated, n_deleted


def fetch_suggested_persons() -> list:
    """Return persons that have pending suggestions, with their suggestion count.

    Each row: (id, name, suggestion_count)
    """
    return execute_query(
        """
        SELECT per.id, per.name, COUNT(*) AS suggestion_count
        FROM face_detections fd
        JOIN persons per ON per.id = fd.suggested_person_id
        WHERE fd.suggested_person_id IS NOT NULL
          AND fd.is_validated = FALSE
        GROUP BY per.id, per.name
        ORDER BY per.name
        """,
        (),
    )


def fetch_suggestion_count(person_id: str | None = None) -> int:
    if person_id:
        result = execute_single(
            """
            SELECT COUNT(*)
            FROM face_detections
            WHERE suggested_person_id = %s
              AND is_validated = FALSE
            """,
            (person_id,),
        )
    else:
        result = execute_single(
            """
            SELECT COUNT(*)
            FROM face_detections
            WHERE suggested_person_id IS NOT NULL
              AND is_validated = FALSE
            """
        )
    return result[0] if result else 0


def fetch_suggestions_page(
    limit: int = FACES_PAGE_SIZE,
    offset: int = 0,
    person_id: str | None = None,
) -> list:
    """Return a page of pending suggestions ordered by score (best first).

    Each row: (face_id, file_path, bounding_box, detection_score,
               suggested_person_id, suggested_name, suggestion_score)
    """
    if person_id:
        where_extra = "AND fd.suggested_person_id = %s"
        params = (person_id, limit, offset)
    else:
        where_extra = ""
        params = (limit, offset)
    return execute_query(
        f"""
        SELECT fd.id,
               ph.file_path,
               fd.bounding_box,
               fd.detection_score,
               fd.suggested_person_id,
               per.name  AS suggested_name,
               fd.suggestion_score
        FROM face_detections fd
        JOIN photos  ph  ON ph.id  = fd.photo_id
        JOIN persons per ON per.id = fd.suggested_person_id
        WHERE fd.suggested_person_id IS NOT NULL
          AND fd.is_validated = FALSE
          {where_extra}
        ORDER BY fd.suggestion_score ASC
        LIMIT %s OFFSET %s
        """,
        params,
    )


def confirm_suggestions(face_ids: list[str]) -> None:
    """Accept suggestions: move suggested_person_id → person_id and mark validated."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE face_detections
                SET person_id           = suggested_person_id,
                    suggested_person_id = NULL,
                    suggestion_score    = NULL,
                    is_validated        = TRUE
                WHERE id = ANY(%s::uuid[])
                  AND suggested_person_id IS NOT NULL
                """,
                (face_ids,),
            )
            # Recompute face_count and centroid for every affected person
            cur.execute(
                """
                UPDATE persons p
                SET face_count = (
                    SELECT COUNT(*) FROM face_detections WHERE person_id = p.id
                ),
                representative_embedding = (
                    SELECT avg(fd.embedding)::vector
                    FROM face_detections fd
                    WHERE fd.person_id = p.id AND fd.embedding IS NOT NULL
                ),
                updated_at = now()
                WHERE p.id IN (
                    SELECT DISTINCT person_id FROM face_detections
                    WHERE id = ANY(%s::uuid[]) AND person_id IS NOT NULL
                )
                """,
                (face_ids,),
            )
            conn.commit()


def reject_suggestions(face_ids: list[str]) -> None:
    """Reject suggestions: clear them and mark validated so they won't reappear."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE face_detections
                SET suggested_person_id = NULL,
                    suggestion_score    = NULL,
                    is_validated        = TRUE
                WHERE id = ANY(%s::uuid[])
                """,
                (face_ids,),
            )
            conn.commit()


# --- Image Processing ---
def load_and_crop_face(
    image_path: str, bounding_box: dict | None
) -> Image.Image | None:
    """Load an image and crop to bounding box if provided."""
    try:
        img = Image.open(image_path)
        if not bounding_box:
            return img

        x = bounding_box.get("x")
        y = bounding_box.get("y")
        w = bounding_box.get("width")
        h = bounding_box.get("height")

        if None not in (x, y, w, h):
            img_w, img_h = img.size
            left = int(x * img_w)
            top = int(y * img_h)
            right = int((x + w) * img_w)
            bottom = int((y + h) * img_h)
            return img.crop((left, top, right, bottom))

        return img
    except Exception:
        return None


def _get_file_metadata(file_path: str) -> tuple[float, bool]:
    """Return (mtime, exists) for a file."""
    if not os.path.exists(file_path):
        return 0.0, False
    try:
        return os.path.getmtime(file_path), True
    except Exception:
        return 0.0, False


def _get_cache_key(file_path: str, bbox: dict | None, mtime: float) -> Path:
    """Generate a cache key (SHA256) for an image crop."""
    bbox_json = json.dumps(bbox, sort_keys=True) if bbox else ""
    key_str = f"{file_path}|{bbox_json}|{int(mtime)}"
    hash_hex = sha256(key_str.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{hash_hex}.jpg"


def _atomic_write(path: Path, data: bytes):
    """Write data to a file atomically using a temporary file."""
    tmp = path.with_suffix(f".tmp-{uuid4().hex}")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(str(tmp), str(path))


def _evict_disk_cache_if_needed():
    """Delete the oldest cached thumbnails when the count exceeds _CACHE_EVICT_THRESHOLD."""
    try:
        entries = [
            e for e in os.scandir(CACHE_DIR) if e.name.endswith((".jpg", ".png"))
        ]
        if len(entries) <= _CACHE_EVICT_THRESHOLD:
            return
        entries.sort(key=lambda e: e.stat().st_mtime)
        for entry in entries[: len(entries) - CACHE_MAX_ENTRIES]:
            try:
                os.unlink(entry.path)
            except OSError:
                pass
    except Exception:
        pass


def get_cached_thumbnail_path(file_path: str, bbox: dict | None = None) -> Path | None:
    """Return path to cached thumbnail PNG, creating it if missing."""
    mtime, exists = _get_file_metadata(file_path)
    if not exists:
        return None

    cache_path = _get_cache_key(file_path, bbox, mtime)
    if cache_path.exists():
        return cache_path

    # Create the cached JPEG thumbnail at display resolution
    try:
        img = load_and_crop_face(file_path, bbox)
        if img is None:
            return None

        img.thumbnail((CELL_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        _atomic_write(cache_path, buf.getvalue())
        return cache_path
    except Exception:
        return None


def _get_streamlit_cache_decorator(
    ttl_seconds: int = 3600, max_entries: int = CACHE_MAX_ENTRIES
):
    """Return the best available Streamlit cache decorator."""
    if hasattr(st, "cache_data"):
        return lambda func: st.cache_data(ttl=ttl_seconds, max_entries=max_entries)(
            func
        )
    if hasattr(st, "experimental_memo"):
        return lambda func: st.experimental_memo(func)
    if hasattr(st, "cache"):
        return lambda func: st.cache(func)
    return lambda func: func


@_get_streamlit_cache_decorator(ttl_seconds=3600, max_entries=CACHE_MAX_ENTRIES)
def get_image_data_url_cached(
    file_path: str, bbox: dict | None = None, crop_path: str | None = None
) -> str | None:
    """Return a data URL for a face image.

    When *crop_path* points to a pre-extracted face crop that exists on disk,
    the crop is embedded as-is — no decoding or resizing — since the scanner
    already sized it for display.  Only when no crop exists yet do we fall back
    to cropping/thumbnailing the full-resolution source photo.
    """
    try:
        if crop_path and os.path.exists(crop_path):
            with open(crop_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"

        # No pre-extracted crop — crop from the source on the fly (legacy path).
        cache_path = get_cached_thumbnail_path(file_path, bbox)
        if not cache_path:
            return None
        with open(cache_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _crop_str(file_path: str | None, face_id) -> str | None:
    """String path to a face's cached crop, or None when not derivable."""
    if not (file_path and face_id):
        return None
    return str(face_cache.face_crop_path(file_path, str(face_id)))


def face_thumb_url(file_path: str, face_id, bbox: dict | None = None) -> str | None:
    """Data URL for a face, preferring its pre-extracted crop when present."""
    return get_image_data_url_cached(
        file_path, bbox, crop_path=_crop_str(file_path, face_id)
    )


def _prefetch_next_page(items: list[tuple]) -> None:
    """Warm the disk + memory cache for the next page's images.

    Each item is (file_path, bbox) or (file_path, bbox, crop_path).
    """
    for item in items:
        file_path = item[0]
        bbox = item[1]
        crop_path = item[2] if len(item) > 2 else None
        if file_path:
            get_image_data_url_cached(file_path, bbox, crop_path=crop_path)


# --- HTML Rendering ---
def render_person_card_image(
    data_url: str | None, width: int = 160, height: int = 160, filename: str = "Image"
):
    """Render a face preview image with a placeholder for missing images."""
    if data_url:
        style = f"width:{width}px;height:{height}px;object-fit:contain;display:block;margin:auto;user-select:none;"
        html_fragment = (
            f'<img src="{data_url}" alt="{html.escape(filename)}" '
            f'style="{style}" draggable="false" />'
        )
        st.markdown(html_fragment, unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="width:{width}px;height:{height}px;background:#EEE;'
            f'display:flex;align-items:center;justify-content:center;">'
            f'<span style="color:#999;font-size:12px;">No preview</span></div>',
            unsafe_allow_html=True,
        )


def render_person_card_meta(name: str | None, face_count: int, top_score: float | None):
    """Render the title and metadata for a person card."""
    st.markdown(
        f'<div style="margin-bottom:0;"><strong>{html.escape(name or "Unnamed")}</strong></div>',
        unsafe_allow_html=True,
    )

    if top_score is not None:
        try:
            score_text = f"{float(top_score):.2f}"
        except (ValueError, TypeError):
            score_text = str(top_score)
        st.markdown(
            f'<span style="font-size:0.8rem;">Faces: {face_count} | Top: {score_text}</span>',
            unsafe_allow_html=True,
        )


def render_clickable_person_card(
    person_id: str | int,
    data_url: str | None,
    name: str | None,
    face_count: int,
    top_score: float | None,
    width: int = 160,
    height: int = 160,
    filename: str = "Image",
) -> None:
    """Render a cluster/person card's image and metadata (no action button)."""
    if data_url:
        style = f"width:{width}px;height:{height}px;object-fit:contain;display:block;margin:auto;user-select:none;border-radius:8px;"
        img_html = (
            f'<img src="{data_url}" alt="{html.escape(filename)}" width="{width}" height="{height}" '
            f'style="{style}" draggable="false" />'
        )
    else:
        img_html = (
            f'<div style="width:{width}px;height:{height}px;background:var(--background-color);'
            f'display:flex;align-items:center;justify-content:center;border-radius:4px;margin:auto;">'
            f'<span style="color:#999;font-size:12px;">No preview</span></div>'
        )

    if top_score is not None:
        try:
            score_text = f"{float(top_score):.2f}"
        except (ValueError, TypeError):
            score_text = str(top_score)
        meta = f"Faces: {face_count} | Top: {score_text}"
    else:
        meta = f"Faces: {face_count}"

    st.markdown(
        f'<div style="text-align:center;">{img_html}'
        f'<div style="font-size:0.8rem;opacity:0.8;margin-top:6px;">{html.escape(meta)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_face_grid_cell_html(
    data_url: str | None,
    width: int,
    height: int,
    filename: str,
    score: float | None,
    file_path: str | None = None,
) -> str:
    if data_url:
        img_html = (
            f'<img src="{data_url}" alt="{html.escape(filename)}" width="{width}" height="{height}" '
            f'style="width:{width}px;height:{height}px;object-fit:contain;'
            f'display:block;margin:auto;pointer-events:none;border-radius:8px" loading="lazy" />'
        )
    else:
        img_html = (
            f"<div style='width:{width}px;height:{height}px;background:#EEE;"
            f"display:flex;align-items:center;justify-content:center;'>"
            f"<span style='color:#999;font-size:12px;'>Could not load</span></div>"
        )

    parts = []
    if score is not None:
        try:
            parts.append(f"Det: {float(score):.2f}")
        except (ValueError, TypeError):
            parts.append(f"Det: {html.escape(str(score))}")
    info_html = " · ".join(parts)

    score_div = (
        f"<div style='height:20px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;"
        f"-webkit-box-orient:vertical;text-overflow:ellipsis;white-space:normal;"
        f"font-size:11px;color:#666;text-align:center;'>{info_html}</div>"
    )

    title_attr = f" title='{html.escape(file_path)}'" if file_path else ""

    return (
        f"<div{title_attr} style='width:{CELL_WIDTH}px;display:flex;flex-direction:column;align-items:center;cursor:default;'>"
        f"<div style='width:{CELL_WIDTH}px;height:{IMAGE_HEIGHT}px;display:flex;"
        f"align-items:center;justify-content:center;'>{img_html}</div>"
        f"{score_div}</div>"
    )


# --- UI Helper Functions ---
def render_pagination_controls_persons() -> tuple[bool, bool, bool, bool, bool, bool]:
    cols = st.columns(6)
    with cols[0]:
        first = st.button("<< First", key="person_page_first")
    with cols[1]:
        prev10 = st.button("◀ -10", key="person_page_prev10")
    with cols[2]:
        prev = st.button("◀ Prev", key="person_page_prev")
    with cols[3]:
        next = st.button("Next ▶", key="person_page_next")
    with cols[4]:
        next10 = st.button("+10 ▶", key="person_page_next10")
    with cols[5]:
        last = st.button("Last >>", key="person_page_last")
    return first, prev10, prev, next, next10, last


def render_pagination_controls_faces(
    page: int, total_pages: int
) -> tuple[bool, bool, bool, bool, bool, bool]:
    cols = st.columns(6)
    with cols[0]:
        first = st.button("<< First", key="face_page_first")
    with cols[1]:
        prev10 = st.button("◀ -10", key="face_page_prev10")
    with cols[2]:
        prev = st.button("◀ Prev", key="face_page_prev")
    with cols[3]:
        next = st.button("Next ▶", key="face_page_next")
    with cols[4]:
        next10 = st.button("+10 ▶", key="face_page_next10")
    with cols[5]:
        last = st.button("Last >>", key="face_page_last")
    return first, prev10, prev, next, next10, last


def render_pagination_controls_unknown(
    page: int, total_pages: int
) -> tuple[bool, bool, bool, bool, bool, bool]:
    cols = st.columns(6)
    with cols[0]:
        first = st.button("<< First", key="unknown_page_first")
    with cols[1]:
        prev10 = st.button("◀ -10", key="unknown_page_prev10")
    with cols[2]:
        prev = st.button("◀ Prev", key="unknown_page_prev")
    with cols[3]:
        next = st.button("Next ▶", key="unknown_page_next")
    with cols[4]:
        next10 = st.button("+10 ▶", key="unknown_page_next10")
    with cols[5]:
        last = st.button("Last >>", key="unknown_page_last")
    return first, prev10, prev, next, next10, last


def update_view_page(action: str, page: int, total_pages: int) -> int:
    """Update the current page based on action."""
    if action == "start":
        return 1
    elif action == "prev10":
        return max(1, page - 10)
    elif action == "prev":
        return max(1, page - 1)
    elif action == "next":
        return min(total_pages, page + 1)
    elif action == "next10":
        return min(total_pages, page + 10)
    elif action == "end":
        return total_pages
    return page


def _face_selection_key(person_id: str, face_id) -> str:
    return f"remove_face_{person_id}_{face_id}"


def _clear_face_selections(person_id: str):
    prefix = f"remove_face_{person_id}_"
    for k in [k for k in st.session_state if k.startswith(prefix)]:
        del st.session_state[k]


def _get_selected_face_ids(person_id: str) -> list[str]:
    prefix = f"remove_face_{person_id}_"
    return [
        k[len(prefix) :]
        for k, v in st.session_state.items()
        if k.startswith(prefix) and v
    ]


def main():
    st.set_page_config(page_title="Face Clusters Viewer", layout="wide")
    st.title("Face Cluster Explorer")

    # Streamlit drops widget-keyed state when the widget isn't rendered in a run
    # (e.g. the persons-list filters while viewing a cluster). Re-assigning each
    # value to itself keeps it alive so the filters survive navigating in and out.
    for _k in (
        "choose_search",
        "choose_unnamed_only",
        "choose_sim_sort",
    ):
        if _k in st.session_state:
            st.session_state[_k] = st.session_state[_k]

    view = st.session_state.get(VIEW_KEY, "persons")
    person_id = st.session_state.get(VIEW_PERSON_KEY)

    if view == "faces" and person_id:
        render_faces_step(person_id)
    elif view == "unknown":
        render_unknown_step()
    elif view == "review":
        render_review_step()
    else:
        render_persons_step()

    _evict_disk_cache_if_needed()


def render_persons_step():
    control_col1, control_col3, control_col_sim, control_col4 = st.columns(
        [3, 1, 1, 3]
    )

    with control_col1:
        search = st.text_input(
            "Search by name or id",
            key="choose_search",
        )

    with control_col3:
        st.checkbox("Unnamed", key="choose_unnamed_only")

    with control_col_sim:
        st.checkbox("Sort by similarity", key="choose_sim_sort")

    with control_col4:
        first, prev10, prev, next, next10, last = render_pagination_controls_persons()

    n_suggestions = fetch_suggestion_count()
    suggestion_badge = f" ({n_suggestions:,})" if n_suggestions else ""
    link_col1, link_col2, link_col3 = st.columns([1, 2, 1])
    with link_col1:
        if st.button("Unknown Faces", key="goto_unknown"):
            navigate_to_unknown()
    with link_col2:
        if st.button(
            f"Review Suggestions{suggestion_badge}", key="goto_review"
        ):
            navigate_to_review()
    with link_col3:
        if st.button("Clean up", key="cleanup_empty_persons"):
            try:
                n_updated, n_deleted = cleanup_persons()
                if n_deleted:
                    st.success(
                        f"Resynced {n_updated} cluster(s), deleted {n_deleted} empty."
                    )
                else:
                    st.info(f"Resynced {n_updated} cluster(s), none were empty.")
                st.rerun()
            except Exception as e:
                st.error(f"Cleanup failed: {e}")

    unnamed_only = st.session_state.get("choose_unnamed_only", False)
    sim_sort = st.session_state.get("choose_sim_sort", False)

    # Cache key uniquely identifies the current filter combination.
    # When it changes, the similarity order must be recomputed and the page reset.
    sim_cache_key = f"{search or ''}:{unnamed_only}"

    if sim_sort:
        prev_key = st.session_state.get("sim_order_key")
        if prev_key != sim_cache_key or "sim_order" not in st.session_state:
            with st.spinner("Computing similarity order…"):
                persons_data = fetch_all_persons_embeddings(
                    search if search else None, unnamed_only
                )
                st.session_state["sim_order"] = _compute_similarity_order(persons_data)
                # Reset to page 1 only when the user actually changed the filter,
                # not on a fresh restore (prev_key is None) where the URL page stands.
                if prev_key is not None and prev_key != sim_cache_key:
                    st.session_state["choose_page"] = 1
                st.session_state["sim_order_key"] = sim_cache_key

        ordered_ids = st.session_state["sim_order"]
        total = len(ordered_ids)
    else:
        # Clear stale cache when sim sort is turned off
        st.session_state.pop("sim_order", None)
        st.session_state.pop("sim_order_key", None)
        total = fetch_persons_count(
            search if search else None,
            unnamed_only=unnamed_only,
        )

    page_count = max(1, math.ceil(total / PERSONS_PAGE_SIZE))

    # Clamp current page
    st.session_state["choose_page"] = max(
        1, min(st.session_state.get("choose_page", 1), page_count)
    )

    # Handle navigation
    cur = st.session_state["choose_page"]
    if first and cur > 1:
        st.session_state["choose_page"] = 1
        st.rerun()
    if prev10 and cur > 1:
        st.session_state["choose_page"] = max(1, cur - 10)
        st.rerun()
    if prev and cur > 1:
        st.session_state["choose_page"] = cur - 1
        st.rerun()
    if next and cur < page_count:
        st.session_state["choose_page"] = cur + 1
        st.rerun()
    if next10 and cur < page_count:
        st.session_state["choose_page"] = min(page_count, cur + 10)
        st.rerun()
    if last and cur < page_count:
        st.session_state["choose_page"] = page_count
        st.rerun()

    st.write(
        f"{total} clusters — page {st.session_state['choose_page']} of {page_count}"
    )

    # Fetch page
    offset = (st.session_state["choose_page"] - 1) * PERSONS_PAGE_SIZE
    if sim_sort:
        page_ids = ordered_ids[offset : offset + PERSONS_PAGE_SIZE]
        persons = fetch_persons_by_ids(page_ids)
    else:
        persons = fetch_persons_page(
            search if search else None,
            limit=PERSONS_PAGE_SIZE,
            offset=offset,
            unnamed_only=unnamed_only,
        )

    for row_start in range(0, len(persons), GRID_COLS):
        row = persons[row_start : row_start + GRID_COLS]
        cols = st.columns(GRID_COLS)

        for col_idx, person_row in enumerate(row):
            (
                person_id,
                name,
                _,
                face_count,
                sample_path,
                sample_score,
                sample_bbox,
                sample_face_id,
            ) = person_row

            with cols[col_idx]:
                # Get cached image
                data_url = None
                if sample_path:
                    data_url = face_thumb_url(sample_path, sample_face_id, sample_bbox)

                render_clickable_person_card(
                    person_id=person_id,
                    data_url=data_url,
                    name=name,
                    face_count=face_count,
                    top_score=sample_score,
                    width=IMAGE_HEIGHT,
                    height=IMAGE_HEIGHT,
                    filename=os.path.basename(sample_path)
                    if sample_path
                    else "Unknown",
                )

                # Named clusters get a single full-width name button. Unnamed
                # clusters share the row with a trash button (two-step confirm so
                # a misclick can't delete a cluster).
                if name:
                    if st.button(
                        name,
                        key=f"open_person_{person_id}",
                        use_container_width=True,
                    ):
                        navigate_to_faces(str(person_id))
                else:
                    confirm_key = f"confirm_clear_list_{person_id}"
                    st.session_state.setdefault(confirm_key, False)
                    if st.session_state[confirm_key]:
                        # Confirm state replaces the row with ✓ / ✗.
                        yes_col, no_col = st.columns(2)
                        with yes_col:
                            if st.button(
                                "✓",
                                key=f"clear_list_yes_{person_id}",
                                type="primary",
                                use_container_width=True,
                                help="Confirm delete",
                            ):
                                try:
                                    clear_cluster(str(person_id))
                                    st.session_state.pop(confirm_key, None)
                                    # A cluster was removed — invalidate the
                                    # cached similarity order so it recomputes.
                                    st.session_state.pop("sim_order", None)
                                    st.session_state.pop("sim_order_key", None)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to clear: {e}")
                        with no_col:
                            if st.button(
                                "✗",
                                key=f"clear_list_no_{person_id}",
                                use_container_width=True,
                                help="Cancel",
                            ):
                                st.session_state[confirm_key] = False
                                st.rerun()
                    else:
                        name_col, trash_col = st.columns([4, 1])
                        with name_col:
                            if st.button(
                                "Unnamed",
                                key=f"open_person_{person_id}",
                                use_container_width=True,
                            ):
                                navigate_to_faces(str(person_id))
                        with trash_col:
                            if st.button(
                                "🗑️",
                                key=f"clear_list_{person_id}",
                                use_container_width=True,
                                help="Clear this cluster",
                            ):
                                st.session_state[confirm_key] = True
                                st.rerun()

    current_page = st.session_state["choose_page"]
    if current_page < page_count:
        if sim_sort:
            next_ids = ordered_ids[
                current_page * PERSONS_PAGE_SIZE : (current_page + 1)
                * PERSONS_PAGE_SIZE
            ]
            next_persons = fetch_persons_by_ids(next_ids)
        else:
            next_persons = fetch_persons_page(
                search if search else None,
                limit=PERSONS_PAGE_SIZE,
                offset=current_page * PERSONS_PAGE_SIZE,
                unnamed_only=unnamed_only,
            )
        _prefetch_next_page(
            [
                (row[4], row[6], _crop_str(row[4], row[7]))
                for row in next_persons
                if row[4]
            ]
        )


def navigate_to_persons():
    st.session_state[VIEW_KEY] = "persons"
    st.session_state[VIEW_PERSON_KEY] = None
    st.rerun()


def navigate_to_faces(person_id: str):
    st.session_state[VIEW_KEY] = "faces"
    st.session_state[VIEW_PERSON_KEY] = person_id
    st.rerun()


def navigate_to_unknown():
    st.session_state[VIEW_KEY] = "unknown"
    st.rerun()


@st.dialog("Original photo", width="large")
def _show_original_photo(file_path: str):
    """Modal showing the full original image so a face can be seen in context."""
    st.caption(file_path)
    if os.path.exists(file_path):
        st.image(file_path, use_container_width=True)
    else:
        st.error("Original file not found.")


def render_faces_step(person_id: str):
    if not person_id:
        st.error("No person selected; returning to list.")
        navigate_to_persons()

    person_row = fetch_person(person_id)
    if not person_row:
        st.error("Selected person not found in database.")
        navigate_to_persons()

    _, current_name, _, face_count = (
        person_row if person_row else (None, "Unknown", None, 0)
    )

    select_mode_key = f"select_mode_{person_id}"
    if select_mode_key not in st.session_state:
        st.session_state[select_mode_key] = False
    in_select_mode = st.session_state[select_mode_key]

    confirm_clear_key = f"confirm_clear_{person_id}"
    st.session_state.setdefault(confirm_clear_key, False)

    header_col, select_col, clear_col, back_col = st.columns([6, 2, 2, 1])
    with header_col:
        st.markdown(f"## {html.escape(current_name or 'Unnamed')}")
        st.markdown(f"ID: `{html.escape(str(person_id))}` — Faces: {face_count}")

    with select_col:
        toggle_label = "Cancel selection" if in_select_mode else "Select to remove"
        if st.button(toggle_label, key=f"toggle_select_{person_id}"):
            new_mode = not in_select_mode
            st.session_state[select_mode_key] = new_mode
            if not new_mode:
                _clear_face_selections(person_id)
            st.rerun()

    with clear_col:
        if not st.session_state[confirm_clear_key]:
            if st.button("Clear cluster", key=f"clear_cluster_btn_{person_id}"):
                st.session_state[confirm_clear_key] = True
                st.rerun()
        else:
            st.warning("Remove all faces and delete this cluster?")
            yes_col, no_col = st.columns(2)
            with yes_col:
                if st.button(
                    "Yes, delete", key=f"clear_cluster_yes_{person_id}", type="primary"
                ):
                    try:
                        clear_cluster(person_id)
                        navigate_to_persons()
                    except Exception as e:
                        st.error(f"Failed to clear cluster: {e}")
            with no_col:
                if st.button("Cancel", key=f"clear_cluster_no_{person_id}"):
                    st.session_state[confirm_clear_key] = False
                    st.rerun()

    with back_col:
        if st.button("Back to list", key="back_to_list"):
            navigate_to_persons()

    edit_name_col, save_button_col = st.columns([4, 1])
    with edit_name_col:
        edit_key = f"name_edit_{person_id}"
        new_name = st.text_input(
            "Person/Cluster name",
            value=current_name or "",
            key=edit_key,
        )
    with save_button_col:
        if st.button("Save name", key=f"save_name_{person_id}"):
            try:
                execute_update(
                    "UPDATE persons SET name = %s WHERE id = %s",
                    (new_name if new_name else None, person_id),
                )
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save name: {e}")

    with st.expander("Merge this cluster into another person"):
        other_persons = fetch_all_persons_for_merge(person_id)
        if not other_persons:
            st.info("No other named clusters found.")
        else:
            merge_target = st.selectbox(
                "Select the person to merge this cluster into (this cluster will be deleted):",
                options=other_persons,
                format_func=lambda x: (
                    f"{x[1] or 'Unnamed'} — {x[2]} faces  [{str(x[0])[:8]}…]"
                ),
                index=None,
                key=f"merge_select_{person_id}",
            )
            if merge_target:
                preview_col, btn_col = st.columns([1, 4])
                with preview_col:
                    sample_path, sample_bbox = merge_target[3], merge_target[4]
                    if sample_path:
                        data_url = face_thumb_url(
                            sample_path, merge_target[5], sample_bbox
                        )
                        if data_url:
                            st.markdown(
                                f'<img src="{data_url}" style="width:80px;height:80px;'
                                f'object-fit:contain;border-radius:6px;" />',
                                unsafe_allow_html=True,
                            )
                with btn_col:
                    if st.button(
                        f"Merge this cluster into {merge_target[1]}",
                        key=f"merge_btn_{person_id}",
                        type="primary",
                    ):
                        try:
                            merge_persons_into(str(merge_target[0]), [person_id])
                            st.success(f"Merged into {merge_target[1]}.")
                            navigate_to_persons()
                        except Exception as e:
                            st.error(f"Merge failed: {e}")

    # Faces pagination
    view_page_key = f"view_page_{person_id}"
    if view_page_key not in st.session_state:
        st.session_state[view_page_key] = 1

    total_faces = fetch_face_count_for_person(person_id)
    total_pages = max(1, math.ceil(total_faces / FACES_PAGE_SIZE))
    st.session_state[view_page_key] = max(
        1, min(st.session_state[view_page_key], total_pages)
    )

    # Navigation
    start, prev10, prev, next_btn, next10, end = render_pagination_controls_faces(
        st.session_state[view_page_key], total_pages
    )

    cur = st.session_state[view_page_key]
    if start:
        st.session_state[view_page_key] = 1
    elif prev10:
        st.session_state[view_page_key] = max(1, cur - 10)
    elif prev:
        st.session_state[view_page_key] = max(1, cur - 1)
    elif next_btn:
        st.session_state[view_page_key] = min(total_pages, cur + 1)
    elif next10:
        st.session_state[view_page_key] = min(total_pages, cur + 10)
    elif end:
        st.session_state[view_page_key] = total_pages

    # Fetch faces
    page = st.session_state[view_page_key]
    offset = (page - 1) * FACES_PAGE_SIZE
    faces = fetch_faces_for_person(
        person_id, limit=FACES_PAGE_SIZE, offset=offset
    )

    # Summary + remove action row
    start_idx = offset + 1 if total_faces > 0 else 0
    end_idx = offset + len(faces)
    summary_col, remove_col = st.columns([6, 4])
    with summary_col:
        st.markdown(
            f"**Showing {start_idx}-{end_idx} of {total_faces} (page {page}/{total_pages})**"
        )
    if in_select_mode:
        selected_ids = _get_selected_face_ids(person_id)
        with remove_col:
            if selected_ids:
                if st.button(
                    f"Remove {len(selected_ids)} selected face(s)",
                    key=f"do_remove_{person_id}",
                    type="primary",
                ):
                    try:
                        remove_faces_from_person(person_id, selected_ids)
                        st.session_state[select_mode_key] = False
                        _clear_face_selections(person_id)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to remove faces: {e}")

    # Render face grid — one column per face so each can carry a "view original"
    # button (and, in select mode, a selection checkbox).
    for row_start in range(0, len(faces), GRID_COLS):
        row_faces = faces[row_start : row_start + GRID_COLS]
        cols = st.columns(GRID_COLS)
        for col_idx, (
            face_id,
            file_path,
            bounding_box,
            score,
        ) in enumerate(row_faces):
            with cols[col_idx]:
                data_url = face_thumb_url(file_path, face_id, bounding_box)
                st.markdown(
                    render_face_grid_cell_html(
                        data_url,
                        width=CELL_WIDTH,
                        height=IMAGE_HEIGHT,
                        filename=os.path.basename(file_path),
                        score=score,
                        file_path=file_path,
                    ),
                    unsafe_allow_html=True,
                )
                if st.button(
                    "🔍",
                    key=f"view_orig_{person_id}_{face_id}",
                    help="Open the original photo to see this face in context",
                    use_container_width=True,
                ):
                    _show_original_photo(file_path)
                if in_select_mode:
                    st.checkbox(
                        "Select",
                        key=_face_selection_key(person_id, face_id),
                        label_visibility="collapsed",
                    )

    if page < total_pages:
        next_faces = fetch_faces_for_person(
            person_id,
            limit=FACES_PAGE_SIZE,
            offset=page * FACES_PAGE_SIZE,
        )
        _prefetch_next_page([(f[1], f[2], _crop_str(f[1], f[0])) for f in next_faces])


def render_unknown_step():
    # Use an explicit set to track selections rather than individual widget keys.
    # Pre-populating many checkbox session-state keys at once (old approach) was
    # disrupting sort_by_similarity / target_person state on the following rerun.
    SEL_KEY = "unknown_selected_set"
    st.session_state.setdefault(SEL_KEY, set())

    assign_mode_key = "assign_mode_unknown"
    st.session_state.setdefault(assign_mode_key, False)
    in_assign_mode = st.session_state[assign_mode_key]

    header_col, select_col, back_col = st.columns([6, 2, 1])
    with header_col:
        st.markdown("## Uncategorized Faces")
    with select_col:
        toggle_label = "Cancel selection" if in_assign_mode else "Select multiple"
        if st.button(toggle_label, key="toggle_assign_unknown"):
            new_mode = not in_assign_mode
            st.session_state[assign_mode_key] = new_mode
            if not new_mode:
                st.session_state[SEL_KEY] = set()
            st.rerun()
    with back_col:
        if st.button("Back to list", key="back_to_list"):
            navigate_to_persons()

    # Target person + sort controls — always visible so browsing by similarity
    # doesn't require entering bulk-select mode first.
    # Manual state management (no key= on selectbox) avoids the widget-state
    # reset that occurs when st.rerun() is called from inside a column context.
    TARGET_KEY = "unknown_target_pid"
    st.session_state.setdefault(TARGET_KEY, "")

    named_persons = fetch_named_persons_for_assign()
    target_person = None
    sort_by_similarity = False
    if named_persons:
        person_map = {str(p[0]): p for p in named_persons}
        person_options = [""] + list(person_map.keys())

        stored_target = st.session_state[TARGET_KEY]
        if stored_target not in person_options:
            stored_target = ""
            st.session_state[TARGET_KEY] = ""

        ctrl_col1, ctrl_col2 = st.columns([4, 2])
        with ctrl_col1:
            selected_target = st.selectbox(
                "Target person:",
                options=person_options,
                format_func=lambda pid: (
                    "— none —"
                    if pid == ""
                    else f"{person_map[pid][1]}  ({person_map[pid][2]} faces)"
                ),
                index=person_options.index(stored_target),
            )  # No key= — prevents Streamlit from overwriting state on rerun
        if selected_target != stored_target:
            st.session_state[TARGET_KEY] = selected_target
            st.session_state[SEL_KEY] = set()
        target_person = person_map.get(st.session_state[TARGET_KEY])

        with ctrl_col2:
            st.session_state.setdefault("sort_by_similarity", True)
            sort_by_similarity = st.checkbox(
                "Sort by similarity",
                key="sort_by_similarity",
                disabled=target_person is None,
            )
    else:
        st.warning("No named persons found. Label some clusters first.")

    # Pagination
    view_page_key = "view_page_unknown"
    st.session_state.setdefault(view_page_key, 1)

    total_faces = fetch_face_count_for_unknown()
    total_pages = max(1, math.ceil(total_faces / FACES_PAGE_SIZE))
    st.session_state[view_page_key] = max(
        1, min(st.session_state[view_page_key], total_pages)
    )

    start, prev10, prev, next_btn, next10, end = render_pagination_controls_unknown(
        st.session_state[view_page_key], total_pages
    )

    cur = st.session_state[view_page_key]
    if start:
        st.session_state[view_page_key] = 1
    elif prev10:
        st.session_state[view_page_key] = max(1, cur - 10)
    elif prev:
        st.session_state[view_page_key] = max(1, cur - 1)
    elif next_btn:
        st.session_state[view_page_key] = min(total_pages, cur + 1)
    elif next10:
        st.session_state[view_page_key] = min(total_pages, cur + 10)
    elif end:
        st.session_state[view_page_key] = total_pages

    page = st.session_state[view_page_key]
    offset = (page - 1) * FACES_PAGE_SIZE

    if sort_by_similarity and target_person:
        faces = fetch_faces_for_unknown_by_similarity(
            str(target_person[0]),
            limit=FACES_PAGE_SIZE,
            offset=offset,
        )
    else:
        faces = fetch_faces_for_unknown(
            limit=FACES_PAGE_SIZE, offset=offset
        )

    start_idx = offset + 1 if total_faces > 0 else 0
    end_idx = offset + len(faces)

    summary_col, action_col = st.columns([4, 4])
    with summary_col:
        st.markdown(
            f"**Showing {start_idx}–{end_idx} of {total_faces} (page {page}/{total_pages})**"
        )

    if target_person:
        page_face_ids = [str(face[0]) for face in faces]
        with action_col:
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Select all on page", key="select_all_unknown"):
                    st.session_state[SEL_KEY].update(page_face_ids)
            with btn_col2:
                # Read AFTER the select-all button may have updated the set
                # in this same render pass — a list copy taken earlier would
                # be stale and leave the assign button permanently disabled.
                selected_ids = list(st.session_state[SEL_KEY])
                assign_label = (
                    f"Assign {len(selected_ids)} face(s)"
                    if selected_ids
                    else "Assign faces"
                )
                if st.button(
                    assign_label,
                    key="do_assign_unknown",
                    type="primary",
                    disabled=not selected_ids,
                ):
                    try:
                        assign_faces_to_person(str(target_person[0]), selected_ids)
                        st.session_state[SEL_KEY] = set()
                        st.session_state[assign_mode_key] = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Assignment failed: {e}")

    # Use the Streamlit-columns layout whenever we need per-face widgets
    # (quick-assign buttons or bulk-select checkboxes).  Fall back to a single
    # static HTML block when just browsing — it's faster to render.
    needs_columns = target_person is not None or in_assign_mode
    if needs_columns:
        selected_set = st.session_state[SEL_KEY]
        for row_start in range(0, len(faces), GRID_COLS):
            row_faces = faces[row_start : row_start + GRID_COLS]
            cols = st.columns(GRID_COLS)
            for col_idx, (
                face_id,
                file_path,
                bounding_box,
                score,
            ) in enumerate(row_faces):
                face_id_str = str(face_id)
                with cols[col_idx]:
                    data_url = face_thumb_url(file_path, face_id, bounding_box)
                    st.markdown(
                        render_face_grid_cell_html(
                            data_url,
                            width=CELL_WIDTH,
                            height=IMAGE_HEIGHT,
                            filename=os.path.basename(file_path),
                            score=score,
                            file_path=file_path,
                        ),
                        unsafe_allow_html=True,
                    )

                    if target_person is not None:
                        if st.button(
                            "Assign",
                            key=f"quick_assign_{face_id_str}",
                            use_container_width=True,
                        ):
                            try:
                                assign_faces_to_person(
                                    str(target_person[0]), [face_id_str]
                                )
                                st.session_state[SEL_KEY].discard(face_id_str)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Assignment failed: {e}")

                    cb_key = f"unknown_face_cb_{face_id_str}"

                    def _toggle(fid=face_id_str):
                        s = st.session_state[SEL_KEY]
                        if fid in s:
                            s.discard(fid)
                        else:
                            s.add(fid)

                    st.session_state[cb_key] = face_id_str in selected_set
                    st.checkbox(
                        "Select",
                        key=cb_key,
                        on_change=_toggle,
                        label_visibility="collapsed",
                    )
    else:
        cells_html = []
        for face_id, file_path, bounding_box, score in faces:
            data_url = face_thumb_url(file_path, face_id, bounding_box)
            cells_html.append(
                render_face_grid_cell_html(
                    data_url,
                    width=CELL_WIDTH,
                    height=IMAGE_HEIGHT,
                    filename=os.path.basename(file_path),
                    score=score,
                    file_path=file_path,
                )
            )
        flex_style = "display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;justify-content:flex-start;"
        st.markdown(
            f"<div style='{flex_style}'>{''.join(cells_html)}</div>",
            unsafe_allow_html=True,
        )

    if page < total_pages:
        next_offset = page * FACES_PAGE_SIZE
        if sort_by_similarity and target_person:
            next_faces = fetch_faces_for_unknown_by_similarity(
                str(target_person[0]),
                limit=FACES_PAGE_SIZE,
                offset=next_offset,
            )
        else:
            next_faces = fetch_faces_for_unknown(
                limit=FACES_PAGE_SIZE,
                offset=next_offset,
            )
        _prefetch_next_page([(f[1], f[2], _crop_str(f[1], f[0])) for f in next_faces])


def navigate_to_review():
    st.session_state[VIEW_KEY] = "review"
    st.session_state[VIEW_PERSON_KEY] = None
    st.rerun()


def render_review_face_cell(
    data_url, file_path, det_score, suggested_name, suggestion_score
):
    """Render a suggestion card: face crop + suggested person + confidence."""
    if data_url:
        img_html = (
            f'<img src="{data_url}" alt="face" width="{CELL_WIDTH}" height="{IMAGE_HEIGHT}" '
            f'style="width:{CELL_WIDTH}px;height:{IMAGE_HEIGHT}px;'
            f'object-fit:contain;display:block;margin:auto;border-radius:8px" loading="lazy" />'
        )
    else:
        img_html = (
            f"<div style='width:{CELL_WIDTH}px;height:{IMAGE_HEIGHT}px;background:#EEE;"
            f"display:flex;align-items:center;justify-content:center;'>"
            f"<span style='color:#999;font-size:12px;'>No image</span></div>"
        )

    try:
        score_pct = f"{(1 - float(suggestion_score)) * 100:.0f}%"
    except (TypeError, ValueError):
        score_pct = "?"

    name_html = f"<div style='font-size:0.85rem;font-weight:bold;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{html.escape(suggested_name or '?')}</div>"
    meta_html = (
        f"<div style='font-size:0.75rem;opacity:0.75;'>Confidence: {score_pct}</div>"
    )

    return (
        f"<div style='width:{CELL_WIDTH}px;display:flex;flex-direction:column;align-items:center;'>"
        f"<div style='width:{CELL_WIDTH}px;height:{IMAGE_HEIGHT}px;display:flex;"
        f"align-items:center;justify-content:center;'>{img_html}</div>"
        f"<div style='width:100%;text-align:center;padding-top:4px;'>{name_html}{meta_html}</div>"
        f"</div>"
    )


def render_review_step():
    SEL_KEY = "review_selected_set"
    st.session_state.setdefault(SEL_KEY, set())

    header_col, back_col = st.columns([8, 1])
    with header_col:
        st.markdown("## Review Suggestions")
    with back_col:
        if st.button("Back to list", key="review_back"):
            navigate_to_persons()

    # Person filter — state is managed manually so that confirm/reject reruns
    # never accidentally clear the filter.  We store the chosen person_id under
    # FILTER_KEY and pass index= explicitly; the widget has no key= binding so
    # Streamlit cannot overwrite our state on its own.
    FILTER_KEY = "review_filter_pid"
    st.session_state.setdefault(FILTER_KEY, "")

    suggested_persons = fetch_suggested_persons()
    person_map = {str(p[0]): (p[1], p[2]) for p in suggested_persons}
    filter_options = [""] + list(person_map.keys())
    filter_labels = {
        "": f"All suggestions ({fetch_suggestion_count():,})",
        **{pid: f"{name}  ({count:,})" for pid, (name, count) in person_map.items()},
    }

    stored_filter = st.session_state[FILTER_KEY]
    # If the stored person has no more pending suggestions, drop back to "All".
    if stored_filter not in filter_options:
        stored_filter = ""
        st.session_state[FILTER_KEY] = ""

    selected_filter = st.selectbox(
        "Filter by person",
        options=filter_options,
        format_func=lambda x: filter_labels.get(x, x),
        index=filter_options.index(stored_filter),
        label_visibility="collapsed",
    )

    if selected_filter != stored_filter:
        st.session_state[FILTER_KEY] = selected_filter
        st.session_state["review_page"] = 1
        st.session_state[SEL_KEY] = set()

    active_person_id = selected_filter or None

    total = fetch_suggestion_count(active_person_id)
    if total == 0:
        st.info(
            "No pending suggestions. Run `python scan-faces.py suggest` to generate some."
        )
        return

    # Pagination
    view_page_key = "review_page"
    st.session_state.setdefault(view_page_key, 1)
    total_pages = max(1, math.ceil(total / FACES_PAGE_SIZE))
    st.session_state[view_page_key] = max(
        1, min(st.session_state[view_page_key], total_pages)
    )

    pg_cols = st.columns(6)
    with pg_cols[0]:
        if st.button("<< First", key="review_first"):
            st.session_state[view_page_key] = 1
    with pg_cols[1]:
        if st.button("◀ -10", key="review_prev10"):
            st.session_state[view_page_key] = max(
                1, st.session_state[view_page_key] - 10
            )
    with pg_cols[2]:
        if st.button("◀ Prev", key="review_prev"):
            st.session_state[view_page_key] = max(
                1, st.session_state[view_page_key] - 1
            )
    with pg_cols[3]:
        if st.button("Next ▶", key="review_next"):
            st.session_state[view_page_key] = min(
                total_pages, st.session_state[view_page_key] + 1
            )
    with pg_cols[4]:
        if st.button("+10 ▶", key="review_next10"):
            st.session_state[view_page_key] = min(
                total_pages, st.session_state[view_page_key] + 10
            )
    with pg_cols[5]:
        if st.button("Last >>", key="review_last"):
            st.session_state[view_page_key] = total_pages

    page = st.session_state[view_page_key]
    offset = (page - 1) * FACES_PAGE_SIZE
    faces = fetch_suggestions_page(
        limit=FACES_PAGE_SIZE, offset=offset, person_id=active_person_id
    )

    start_idx = offset + 1 if total > 0 else 0
    end_idx = offset + len(faces)

    selected_ids = list(st.session_state[SEL_KEY])
    page_face_ids = [str(r[0]) for r in faces]

    summary_col, action_col = st.columns([4, 6])
    with summary_col:
        st.markdown(
            f"**Showing {start_idx}–{end_idx} of {total:,} (page {page}/{total_pages})**"
        )
    with action_col:
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            if st.button("Select all on page", key="review_select_all"):
                st.session_state[SEL_KEY].update(page_face_ids)
                st.rerun()
        with a2:
            if st.button(
                f"Confirm {len(selected_ids)}" if selected_ids else "Confirm",
                key="review_confirm",
                type="primary",
                disabled=not selected_ids,
            ):
                try:
                    confirm_suggestions(selected_ids)
                    st.session_state[SEL_KEY] = set()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        with a3:
            if st.button(
                f"Reject {len(selected_ids)}" if selected_ids else "Reject",
                key="review_reject",
                disabled=not selected_ids,
            ):
                try:
                    reject_suggestions(selected_ids)
                    st.session_state[SEL_KEY] = set()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        with a4:
            if st.button("Confirm all on page", key="review_confirm_page"):
                try:
                    confirm_suggestions(page_face_ids)
                    st.session_state[SEL_KEY] = set()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

    selected_set = st.session_state[SEL_KEY]
    for row_start in range(0, len(faces), GRID_COLS):
        row_faces = faces[row_start : row_start + GRID_COLS]
        cols = st.columns(GRID_COLS)
        for col_idx, (
            face_id,
            file_path,
            bbox,
            det_score,
            _,
            suggested_name,
            suggestion_score,
        ) in enumerate(row_faces):
            face_id_str = str(face_id)
            cb_key = f"review_face_cb_{face_id_str}"
            with cols[col_idx]:
                data_url = face_thumb_url(file_path, face_id, bbox)
                st.markdown(
                    render_review_face_cell(
                        data_url, file_path, det_score, suggested_name, suggestion_score
                    ),
                    unsafe_allow_html=True,
                )

                def _toggle(fid=face_id_str):
                    s = st.session_state[SEL_KEY]
                    if fid in s:
                        s.discard(fid)
                    else:
                        s.add(fid)

                st.session_state[cb_key] = face_id_str in selected_set
                st.checkbox(
                    "Select",
                    key=cb_key,
                    on_change=_toggle,
                    label_visibility="collapsed",
                )

    if page < total_pages:
        next_suggestions = fetch_suggestions_page(
            limit=FACES_PAGE_SIZE,
            offset=page * FACES_PAGE_SIZE,
            person_id=active_person_id,
        )
        _prefetch_next_page(
            [(r[1], r[2], _crop_str(r[1], r[0])) for r in next_suggestions]
        )


if __name__ == "__main__":
    main()
