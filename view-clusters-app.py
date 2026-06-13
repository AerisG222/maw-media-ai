import base64
import html
import io
import json
import math
import os
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import psycopg2
import streamlit as st
from PIL import Image

# --- Configuration & Constants ---
DSN = os.environ.get("FACE_SCANNER_DSN")
if not DSN:
    st.error("FACE_SCANNER_DSN environment variable not set.")
    st.stop()

CACHE_DIR = Path(__file__).resolve().parent / "image-cache"
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

FACES_PAGE_SIZE = 48
PERSONS_PAGE_SIZE = 48

QUERY_PARAM_PERSON = "person"
QUERY_PARAM_UNKNOWN = "unknown"
QUERY_PARAM_REVIEW = "review"


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
    """Assign faces to a person and sync the face count."""
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
def fetch_persons_count(search: str | None = None, unnamed_only: bool = False) -> int:
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

    Each row: (id, name, cluster_label, face_count, sample_path, sample_score, sample_bbox)
    """
    like = f"%{search}%" if search else None
    return execute_query(
        """
        SELECT p.id, p.name, p.cluster_label, p.face_count,
               ph.file_path AS sample_path, fd.detection_score AS sample_score, fd.bounding_box AS sample_bbox
        FROM persons p
        LEFT JOIN LATERAL (
            SELECT fd.photo_id, fd.detection_score, fd.bounding_box
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
    person_id: str, limit: int = FACES_PAGE_SIZE, offset: int = 0
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


def fetch_faces_for_unknown(limit: int = FACES_PAGE_SIZE, offset: int = 0) -> list:
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
    person_id: str, limit: int = FACES_PAGE_SIZE, offset: int = 0
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

    Each row: (id, name, face_count)
    """
    return execute_query(
        """
        SELECT id, name, face_count
        FROM persons
        WHERE id != %s
          AND name IS NOT NULL
        ORDER BY name
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


def delete_empty_persons() -> int:
    """Delete persons with no face_detections. Returns the number deleted."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM persons
                WHERE NOT EXISTS (
                    SELECT 1 FROM face_detections WHERE person_id = persons.id
                )
                """
            )
            deleted = cur.rowcount
            conn.commit()
    return deleted


def fetch_suggestion_count() -> int:
    result = execute_single(
        """
        SELECT COUNT(*)
        FROM face_detections
        WHERE suggested_person_id IS NOT NULL
          AND is_validated = FALSE
        """
    )
    return result[0] if result else 0


def fetch_suggestions_page(limit: int = FACES_PAGE_SIZE, offset: int = 0) -> list:
    """Return a page of pending suggestions ordered by score (best first).

    Each row: (face_id, file_path, bounding_box, detection_score,
               suggested_person_id, suggested_name, suggestion_score)
    """
    return execute_query(
        """
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
        ORDER BY fd.suggestion_score ASC
        LIMIT %s OFFSET %s
        """,
        (limit, offset),
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
            # Recompute face_count for every affected person
            cur.execute(
                """
                UPDATE persons p
                SET face_count = (
                    SELECT COUNT(*) FROM face_detections WHERE person_id = p.id
                )
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
        entries = [e for e in os.scandir(CACHE_DIR) if e.name.endswith((".jpg", ".png"))]
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


def _get_streamlit_cache_decorator(ttl_seconds: int = 3600, max_entries: int = CACHE_MAX_ENTRIES):
    """Return the best available Streamlit cache decorator."""
    if hasattr(st, "cache_data"):
        return lambda func: st.cache_data(ttl=ttl_seconds, max_entries=max_entries)(func)
    if hasattr(st, "experimental_memo"):
        return lambda func: st.experimental_memo(func)
    if hasattr(st, "cache"):
        return lambda func: st.cache(func)
    return lambda func: func


@_get_streamlit_cache_decorator(ttl_seconds=3600, max_entries=CACHE_MAX_ENTRIES)
def get_image_data_url_cached(file_path: str, bbox: dict | None = None) -> str | None:
    """Return a data URL for a cached thumbnail image."""
    try:
        cache_path = get_cached_thumbnail_path(file_path, bbox)
        if not cache_path:
            return None
        with open(cache_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _prefetch_next_page(file_path_bbox_pairs: list[tuple[str, dict | None]]) -> None:
    """Warm the disk + memory cache for the next page's images."""
    for file_path, bbox in file_path_bbox_pairs:
        if file_path:
            get_image_data_url_cached(file_path, bbox)


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
):
    """Render a single clickable card for a cluster/person."""
    # Build Image HTML
    if data_url:
        style = f"width:{width}px;height:{height}px;object-fit:contain;display:block;margin:auto;user-select:none;border-radius:8px;"
        img_html = (
            f'<img src="{data_url}" alt="{html.escape(filename)}" '
            f'style="{style}" draggable="false" />'
        )
    else:
        img_html = (
            f'<div style="width:{width}px;height:{height}px;background:var(--background-color);'
            f'display:flex;align-items:center;justify-content:center;border-radius:4px;margin:auto;">'
            f'<span style="color:#999;font-size:12px;">No preview</span></div>'
        )

    # Build Metadata HTML
    name_html = f'<div style="margin-top:8px;margin-bottom:2px;font-size:0.95rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"><strong>{html.escape(name or "Unnamed")}</strong></div>'

    if top_score is not None:
        try:
            score_text = f"{float(top_score):.2f}"
        except (ValueError, TypeError):
            score_text = str(top_score)
        meta_html = f'<span style="font-size:0.8rem;opacity:0.8;">Faces: {face_count} | Top: {score_text}</span>'
    else:
        meta_html = (
            f'<span style="font-size:0.8rem;opacity:0.8;">Faces: {face_count}</span>'
        )

    card_html = f"""
    <a href="?{QUERY_PARAM_PERSON}={html.escape(str(person_id))}" target="_self" style="text-decoration:none;color:inherit;display:block;">
        <div style="
            border: 1px solid var(--secondary-background-color);
            border-radius: 8px;
            padding: 12px;
            background-color: var(--secondary-background-color);
            transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
            cursor: pointer;
            text-align: center;
            height: {height + 75}px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        " onmouseover="this.style.transform='scale(1.02)';this.style.borderColor='var(--primary-color)';"
           onmouseout="this.style.transform='scale(1)';this.style.borderColor='var(--secondary-background-color)';">
            <div>
                {img_html}
            </div>
            <div>
                {name_html}
                {meta_html}
            </div>
        </div>
    </a>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_face_grid_cell_html(
    data_url: str | None, width: int, height: int, filename: str, score: float | None
) -> str:
    if data_url:
        img_html = (
            f'<img src="{data_url}" alt="{html.escape(filename)}" '
            f'style="max-width:{width}px;max-height:{height}px;object-fit:contain;'
            f'display:block;margin:auto;pointer-events:none;border-radius:8px" loading="lazy" />'
        )
    else:
        img_html = (
            f"<div style='width:{width}px;height:{height}px;background:#EEE;"
            f"display:flex;align-items:center;justify-content:center;'>"
            f"<span style='color:#999;font-size:12px;'>Could not load</span></div>"
        )

    score_text = ""
    if score is not None:
        try:
            score_text = f"Score: {float(score):.2f}"
        except (ValueError, TypeError):
            score_text = f"Score: {html.escape(str(score))}"

    score_div = (
        f"<div style='height:20px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;"
        f"-webkit-box-orient:vertical;text-overflow:ellipsis;white-space:normal;"
        f"font-size:11px;color:#666;text-align:center;'>{html.escape(score_text)}</div>"
    )

    return (
        f"<div style='width:{CELL_WIDTH}px;display:flex;flex-direction:column;align-items:center;'>"
        f"<div style='width:{CELL_WIDTH}px;height:{IMAGE_HEIGHT}px;display:flex;"
        f"align-items:center;justify-content:center;'>{img_html}</div>"
        f"{score_div}</div>"
    )


# --- UI Helper Functions ---
def render_pagination_controls_persons() -> tuple[bool, bool, bool, bool]:
    cols = st.columns([1, 1, 1, 1])
    with cols[0]:
        first = st.button("<< First", key="person_page_first")
    with cols[1]:
        prev = st.button("◀ Prev", key="person_page_prev")
    with cols[2]:
        next = st.button("Next ▶", key="person_page_next")
    with cols[3]:
        last = st.button("Last >>", key="person_page_last")

    return first, prev, next, last


def render_pagination_controls_faces(
    page: int, total_pages: int
) -> tuple[bool, bool, bool, bool]:
    cols = st.columns([1, 1, 1, 6])
    with cols[0]:
        first = st.button("<< First", key="face_page_first")
    with cols[1]:
        prev = st.button("◀ Prev", key="face_page_prev")
    with cols[2]:
        next = st.button("Next ▶", key="face_page_next")
    with cols[3]:
        last = st.button("Last >>", key="face_page_last")
    return first, prev, next, last


def render_pagination_controls_unknown(
    page: int, total_pages: int
) -> tuple[bool, bool, bool, bool]:
    cols = st.columns([1, 1, 1, 6])
    with cols[0]:
        first = st.button("<< First", key="unknown_page_first")
    with cols[1]:
        prev = st.button("◀ Prev", key="unknown_page_prev")
    with cols[2]:
        next = st.button("Next ▶", key="unknown_page_next")
    with cols[3]:
        last = st.button("Last >>", key="unknown_page_last")
    return first, prev, next, last


def update_view_page(action: str, page: int, total_pages: int) -> int:
    """Update the current page based on action."""
    if action == "start":
        return 1
    elif action == "prev":
        return max(1, page - 1)
    elif action == "next":
        return min(total_pages, page + 1)
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

    person_id = st.query_params.get(QUERY_PARAM_PERSON)
    show_unknown = st.query_params.get(QUERY_PARAM_UNKNOWN, "false") == "true"
    show_review = st.query_params.get(QUERY_PARAM_REVIEW, "false") == "true"

    if person_id:
        render_faces_step(person_id)
    elif show_unknown:
        render_unknown_step()
    elif show_review:
        render_review_step()
    else:
        render_persons_step()

    _evict_disk_cache_if_needed()


def render_persons_step():
    control_col1, control_col3, control_col4 = st.columns([3, 1, 2])

    with control_col1:
        search = st.text_input(
            "Search by name or id",
            value=st.session_state.get("choose_search", ""),
            key="choose_search",
        )

    with control_col3:
        st.checkbox("Unnamed only", key="choose_unnamed_only")

    with control_col4:
        first, prev, next, last = render_pagination_controls_persons()

    n_suggestions = fetch_suggestion_count()
    suggestion_badge = f" ({n_suggestions:,})" if n_suggestions else ""
    link_col1, link_col2, link_col3 = st.columns([1, 2, 1])
    with link_col1:
        st.markdown(
            f'<a href="/?{QUERY_PARAM_UNKNOWN}=true">Unknown Faces</a>',
            unsafe_allow_html=True,
        )
    with link_col2:
        st.markdown(
            f'<a href="/?{QUERY_PARAM_REVIEW}=true">Review Suggestions{html.escape(suggestion_badge)}</a>',
            unsafe_allow_html=True,
        )
    with link_col3:
        if st.button("Clean up empty clusters", key="cleanup_empty_persons"):
            try:
                n_deleted = delete_empty_persons()
                if n_deleted:
                    st.success(f"Deleted {n_deleted} empty cluster(s).")
                    st.rerun()
                else:
                    st.info("No empty clusters found.")
            except Exception as e:
                st.error(f"Cleanup failed: {e}")

    unnamed_only = st.session_state.get("choose_unnamed_only", False)

    # Pagination
    total = fetch_persons_count(search if search else None, unnamed_only=unnamed_only)
    page_count = max(1, math.ceil(total / PERSONS_PAGE_SIZE))

    # Clamp current page
    st.session_state["choose_page"] = max(
        1, min(st.session_state.get("choose_page", 1), page_count)
    )

    # Handle navigation
    if first and st.session_state["choose_page"] > 1:
        st.session_state["choose_page"] = 1
        st.rerun()
    if prev and st.session_state["choose_page"] > 1:
        st.session_state["choose_page"] -= 1
        st.rerun()
    if next and st.session_state["choose_page"] < page_count:
        st.session_state["choose_page"] += 1
        st.rerun()
    if last and st.session_state["choose_page"] < page_count:
        st.session_state["choose_page"] = page_count
        st.rerun()

    st.write(
        f"{total} clusters — page {st.session_state['choose_page']} of {page_count}"
    )

    # Fetch page
    offset = (st.session_state["choose_page"] - 1) * PERSONS_PAGE_SIZE
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
            person_id, name, _, face_count, sample_path, sample_score, sample_bbox = (
                person_row
            )

            with cols[col_idx]:
                # Get cached image
                data_url = None
                if sample_path:
                    data_url = get_image_data_url_cached(sample_path, sample_bbox)

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

    current_page = st.session_state["choose_page"]
    if current_page < page_count:
        next_persons = fetch_persons_page(
            search if search else None,
            limit=PERSONS_PAGE_SIZE,
            offset=current_page * PERSONS_PAGE_SIZE,
            unnamed_only=unnamed_only,
        )
        _prefetch_next_page([(row[4], row[6]) for row in next_persons if row[4]])


def navigate_to_persons():
    dict = st.query_params.to_dict()
    dict.pop(QUERY_PARAM_PERSON, None)
    dict.pop(QUERY_PARAM_UNKNOWN, None)
    dict.pop(QUERY_PARAM_REVIEW, None)
    st.query_params.from_dict(dict)
    st.rerun()


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

    header_col, select_col, back_col = st.columns([6, 2, 1])
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
    start, prev, next_btn, end = render_pagination_controls_faces(
        st.session_state[view_page_key], total_pages
    )

    if start:
        st.session_state[view_page_key] = 1
    elif prev:
        st.session_state[view_page_key] = max(1, st.session_state[view_page_key] - 1)
    elif next_btn:
        st.session_state[view_page_key] = min(
            total_pages, st.session_state[view_page_key] + 1
        )
    elif end:
        st.session_state[view_page_key] = total_pages

    # Fetch faces
    page = st.session_state[view_page_key]
    offset = (page - 1) * FACES_PAGE_SIZE
    faces = fetch_faces_for_person(person_id, limit=FACES_PAGE_SIZE, offset=offset)

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

    # Render face grid
    if in_select_mode:
        for row_start in range(0, len(faces), GRID_COLS):
            row_faces = faces[row_start : row_start + GRID_COLS]
            cols = st.columns(GRID_COLS)
            for col_idx, (face_id, file_path, bounding_box, score) in enumerate(
                row_faces
            ):
                with cols[col_idx]:
                    data_url = get_image_data_url_cached(file_path, bounding_box)
                    st.markdown(
                        render_face_grid_cell_html(
                            data_url,
                            width=CELL_WIDTH,
                            height=IMAGE_HEIGHT,
                            filename=os.path.basename(file_path),
                            score=score,
                        ),
                        unsafe_allow_html=True,
                    )
                    st.checkbox(
                        "Select",
                        key=_face_selection_key(person_id, face_id),
                        label_visibility="collapsed",
                    )
    else:
        cells_html = []
        for face_id, file_path, bounding_box, score in faces:
            data_url = get_image_data_url_cached(file_path, bounding_box)
            cell_html = render_face_grid_cell_html(
                data_url,
                width=CELL_WIDTH,
                height=IMAGE_HEIGHT,
                filename=os.path.basename(file_path),
                score=score,
            )
            cells_html.append(cell_html)

        flex_style = "display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;justify-content:flex-start;"
        container_html = f"<div style='{flex_style}'>{''.join(cells_html)}</div>"
        st.markdown(container_html, unsafe_allow_html=True)

    if page < total_pages:
        next_faces = fetch_faces_for_person(
            person_id, limit=FACES_PAGE_SIZE, offset=page * FACES_PAGE_SIZE
        )
        _prefetch_next_page([(f[1], f[2]) for f in next_faces])


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
        toggle_label = "Cancel selection" if in_assign_mode else "Select to assign"
        if st.button(toggle_label, key="toggle_assign_unknown"):
            new_mode = not in_assign_mode
            st.session_state[assign_mode_key] = new_mode
            if not new_mode:
                st.session_state[SEL_KEY] = set()
            st.rerun()
    with back_col:
        if st.button("Back to list", key="back_to_list"):
            navigate_to_persons()

    # Target person + sort controls
    sort_by_similarity = False
    target_person = None
    if in_assign_mode:
        named_persons = fetch_named_persons_for_assign()
        if named_persons:
            person_map = {str(p[0]): p for p in named_persons}
            ctrl_col1, ctrl_col2 = st.columns([4, 4])
            with ctrl_col1:
                target_person_id = st.selectbox(
                    "Assign selected faces to:",
                    options=list(person_map.keys()),
                    format_func=lambda pid: (
                        f"{person_map[pid][1]}  ({person_map[pid][2]} faces)"
                    ),
                    key="assign_target_person",
                )
                target_person = person_map.get(target_person_id)
            with ctrl_col2:
                # setdefault so value= never overrides existing state
                st.session_state.setdefault("sort_by_similarity", True)
                sort_by_similarity = st.checkbox(
                    "Sort by similarity to selected person",
                    key="sort_by_similarity",
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

    start, prev, next_btn, end = render_pagination_controls_unknown(
        st.session_state[view_page_key], total_pages
    )

    if start:
        st.session_state[view_page_key] = 1
    elif prev:
        st.session_state[view_page_key] = max(1, st.session_state[view_page_key] - 1)
    elif next_btn:
        st.session_state[view_page_key] = min(
            total_pages, st.session_state[view_page_key] + 1
        )
    elif end:
        st.session_state[view_page_key] = total_pages

    page = st.session_state[view_page_key]
    offset = (page - 1) * FACES_PAGE_SIZE

    if in_assign_mode and sort_by_similarity and target_person:
        faces = fetch_faces_for_unknown_by_similarity(
            str(target_person[0]), limit=FACES_PAGE_SIZE, offset=offset
        )
    else:
        faces = fetch_faces_for_unknown(limit=FACES_PAGE_SIZE, offset=offset)

    start_idx = offset + 1 if total_faces > 0 else 0
    end_idx = offset + len(faces)

    summary_col, action_col = st.columns([4, 4])
    with summary_col:
        st.markdown(
            f"**Showing {start_idx}–{end_idx} of {total_faces} (page {page}/{total_pages})**"
        )

    if in_assign_mode and target_person:
        page_face_ids = [str(face[0]) for face in faces]
        selected_ids = list(st.session_state[SEL_KEY])
        with action_col:
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Select all on page", key="select_all_unknown"):
                    st.session_state[SEL_KEY].update(page_face_ids)
                    st.rerun()
            with btn_col2:
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

    if in_assign_mode:
        selected_set = st.session_state[SEL_KEY]
        for row_start in range(0, len(faces), GRID_COLS):
            row_faces = faces[row_start : row_start + GRID_COLS]
            cols = st.columns(GRID_COLS)
            for col_idx, (face_id, file_path, bounding_box, score) in enumerate(
                row_faces
            ):
                face_id_str = str(face_id)
                cb_key = f"unknown_face_cb_{face_id_str}"
                with cols[col_idx]:
                    data_url = get_image_data_url_cached(file_path, bounding_box)
                    st.markdown(
                        render_face_grid_cell_html(
                            data_url,
                            width=CELL_WIDTH,
                            height=IMAGE_HEIGHT,
                            filename=os.path.basename(file_path),
                            score=score,
                        ),
                        unsafe_allow_html=True,
                    )

                    def _toggle(fid=face_id_str):
                        s = st.session_state[SEL_KEY]
                        if fid in s:
                            s.discard(fid)
                        else:
                            s.add(fid)

                    # Sync key from the authoritative set before rendering so
                    # Select All (which only updates the set) is reflected here.
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
            data_url = get_image_data_url_cached(file_path, bounding_box)
            cells_html.append(
                render_face_grid_cell_html(
                    data_url,
                    width=CELL_WIDTH,
                    height=IMAGE_HEIGHT,
                    filename=os.path.basename(file_path),
                    score=score,
                )
            )
        flex_style = "display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;justify-content:flex-start;"
        st.markdown(
            f"<div style='{flex_style}'>{''.join(cells_html)}</div>",
            unsafe_allow_html=True,
        )

    if page < total_pages:
        next_offset = page * FACES_PAGE_SIZE
        if in_assign_mode and sort_by_similarity and target_person:
            next_faces = fetch_faces_for_unknown_by_similarity(
                str(target_person[0]), limit=FACES_PAGE_SIZE, offset=next_offset
            )
        else:
            next_faces = fetch_faces_for_unknown(limit=FACES_PAGE_SIZE, offset=next_offset)
        _prefetch_next_page([(f[1], f[2]) for f in next_faces])


def navigate_to_review():
    d = st.query_params.to_dict()
    d.pop(QUERY_PARAM_PERSON, None)
    d.pop(QUERY_PARAM_UNKNOWN, None)
    d[QUERY_PARAM_REVIEW] = "true"
    st.query_params.from_dict(d)
    st.rerun()


def render_review_face_cell(
    data_url, file_path, det_score, suggested_name, suggestion_score
):
    """Render a suggestion card: face crop + suggested person + confidence."""
    if data_url:
        img_html = (
            f'<img src="{data_url}" alt="face" '
            f'style="max-width:{CELL_WIDTH}px;max-height:{IMAGE_HEIGHT}px;'
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

    total = fetch_suggestion_count()
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

    pg_cols = st.columns([1, 1, 1, 6])
    with pg_cols[0]:
        if st.button("<< First", key="review_first"):
            st.session_state[view_page_key] = 1
    with pg_cols[1]:
        if st.button("◀ Prev", key="review_prev"):
            st.session_state[view_page_key] = max(
                1, st.session_state[view_page_key] - 1
            )
    with pg_cols[2]:
        if st.button("Next ▶", key="review_next"):
            st.session_state[view_page_key] = min(
                total_pages, st.session_state[view_page_key] + 1
            )
    with pg_cols[3]:
        if st.button("Last >>", key="review_last"):
            st.session_state[view_page_key] = total_pages

    page = st.session_state[view_page_key]
    offset = (page - 1) * FACES_PAGE_SIZE
    faces = fetch_suggestions_page(limit=FACES_PAGE_SIZE, offset=offset)

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
                data_url = get_image_data_url_cached(file_path, bbox)
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
            limit=FACES_PAGE_SIZE, offset=page * FACES_PAGE_SIZE
        )
        _prefetch_next_page([(r[1], r[2]) for r in next_suggestions])


if __name__ == "__main__":
    main()
