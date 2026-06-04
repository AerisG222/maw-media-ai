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

GRID_COLS = 6
CELL_WIDTH = 160
CELL_HEIGHT = 200
CAPTION_HEIGHT = 72
IMAGE_HEIGHT = CELL_HEIGHT - CAPTION_HEIGHT

FACES_PAGE_SIZE = 24
PERSONS_DEFAULT_PAGE_SIZE = 24


# --- Database Connection ---
def get_connection():
    """Get a PostgreSQL connection."""
    return psycopg2.connect(DSN)


def _execute_query(query: str, params: tuple = ()):
    """Execute a SELECT query and return all results."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()


def _execute_single(query: str, params: tuple = ()):
    """Execute a SELECT query and return a single row."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone()


def _execute_update(query: str, params: tuple = ()):
    """Execute an UPDATE/INSERT query and commit."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            conn.commit()


# --- Data Fetching ---
def fetch_persons_count(search: str | None = None) -> int:
    """Return the number of persons matching optional search."""
    like = f"%{search}%" if search else None
    result = _execute_single(
        "SELECT COUNT(*) FROM persons WHERE (%s IS NULL OR name ILIKE %s OR id::text ILIKE %s)",
        (search, like, like),
    )
    return result[0] if result else 0


def fetch_persons_page(
    search: str | None = None, limit: int = 24, offset: int = 0
) -> list:
    """Return a page of persons with one sample face for preview.

    Each row: (id, name, cluster_label, face_count, sample_path, sample_score, sample_bbox)
    """
    like = f"%{search}%" if search else None
    return _execute_query(
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
        ORDER BY COALESCE(p.name, ''), p.id
        LIMIT %s OFFSET %s
        """,
        (search, like, like, limit, offset),
    )


def fetch_person(person_id: str) -> tuple | None:
    """Return person data: (id, name, cluster_label, face_count)."""
    return _execute_single(
        "SELECT id, name, cluster_label, face_count FROM persons WHERE id = %s",
        (person_id,),
    )


def fetch_face_count_for_person(person_id: str) -> int:
    """Return the number of faces for a person."""
    result = _execute_single(
        "SELECT COUNT(*) FROM face_detections WHERE person_id = %s",
        (person_id,),
    )
    return result[0] if result else 0


def fetch_faces_for_person(person_id: str, limit: int = 24, offset: int = 0) -> list:
    """Return a page of faces for a person, ordered by detection score.

    Each row: (id, file_path, bounding_box, detection_score)
    """
    return _execute_query(
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
    return CACHE_DIR / f"{hash_hex}.png"


def _atomic_write(path: Path, data: bytes):
    """Write data to a file atomically using a temporary file."""
    tmp = path.with_suffix(f".tmp-{uuid4().hex}")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(str(tmp), str(path))


def get_cached_thumbnail_path(file_path: str, bbox: dict | None = None) -> Path | None:
    """Return path to cached thumbnail PNG, creating it if missing."""
    mtime, exists = _get_file_metadata(file_path)
    if not exists:
        return None

    cache_path = _get_cache_key(file_path, bbox, mtime)
    if cache_path.exists():
        return cache_path

    # Create the cached PNG
    try:
        img = load_and_crop_face(file_path, bbox)
        if img is None:
            return None

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG", optimize=True)
        _atomic_write(cache_path, buf.getvalue())
        return cache_path
    except Exception:
        return None


def _get_streamlit_cache_decorator(ttl_seconds: int = 3600):
    """Return the best available Streamlit cache decorator."""
    if hasattr(st, "cache_data"):
        return lambda func: st.cache_data(ttl=ttl_seconds)(func)
    if hasattr(st, "experimental_memo"):
        return lambda func: st.experimental_memo(func)
    if hasattr(st, "cache"):
        return lambda func: st.cache(func)
    return lambda func: func


@_get_streamlit_cache_decorator(ttl_seconds=3600)
def get_image_data_url_cached(file_path: str, bbox: dict | None = None) -> str | None:
    """Return a data URL for a cached thumbnail image."""
    try:
        cache_path = get_cached_thumbnail_path(file_path, bbox)
        if not cache_path:
            return None
        with open(cache_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
            return f"data:image/png;base64,{b64}"
    except Exception:
        return None


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


def render_face_grid_cell_html(
    data_url: str | None, width: int, height: int, filename: str, score: float | None
) -> str:
    """Return HTML for a single face cell in the grid."""
    if data_url:
        img_html = (
            f'<img src="{data_url}" alt="{html.escape(filename)}" '
            f'style="max-width:{width}px;max-height:{height}px;object-fit:contain;'
            f'display:block;margin:auto;pointer-events:none;" loading="lazy" />'
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

    filename_esc = html.escape(filename)
    filename_div = (
        f"<div style='height:20px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;"
        f"-webkit-box-orient:vertical;text-overflow:ellipsis;white-space:normal;"
        f"font-size:12px;color:#222;text-align:center;margin-top:6px;'>{filename_esc}</div>"
    )

    score_div = (
        f"<div style='height:20px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;"
        f"-webkit-box-orient:vertical;text-overflow:ellipsis;white-space:normal;"
        f"font-size:11px;color:#666;text-align:center;'>{html.escape(score_text)}</div>"
    )

    return (
        f"<div style='width:{CELL_WIDTH}px;display:flex;flex-direction:column;align-items:center;'>"
        f"<div style='width:{CELL_WIDTH}px;height:{IMAGE_HEIGHT}px;display:flex;"
        f"align-items:center;justify-content:center;'>{img_html}</div>"
        f"{filename_div}{score_div}</div>"
    )


# --- Query Parameter Navigation ---
def _get_query_param_handlers():
    """Get get/set query param functions, with fallbacks across Streamlit versions."""
    get_fn = getattr(st, "experimental_get_query_params", None) or getattr(
        st, "get_query_params", None
    )
    set_fn = getattr(st, "experimental_set_query_params", None) or getattr(
        st, "set_query_params", None
    )
    return get_fn, set_fn


def _try_set_query_param(key: str, value: str | None) -> bool:
    """Attempt to set a query parameter. Return True if successful."""
    _, set_fn = _get_query_param_handlers()
    if not callable(set_fn):
        return False

    try:
        if value is None:
            set_fn()  # Clear all params
        else:
            set_fn(**{key: value})
        return True
    except TypeError:
        try:
            set_fn({key: value} if value else {})
            return True
        except Exception:
            return False
    except Exception:
        return False


def _get_query_param(key: str) -> str | None:
    """Attempt to get a query parameter. Return None if not found."""
    get_fn, _ = _get_query_param_handlers()
    if not callable(get_fn):
        return None

    try:
        params = get_fn() or {}
        if not isinstance(params, dict):
            return None
        val = params.get(key)
        if isinstance(val, (list, tuple)):
            return val[0] if val else None
        return val
    except Exception:
        return None


def _do_rerun():
    """Attempt to rerun the Streamlit app."""
    rerun_fn = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if callable(rerun_fn):
        try:
            rerun_fn()
        except Exception:
            pass


# --- Session State Initialization ---
def _init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "ui_step": "choose",
        "selected_person_id": None,
        "choose_search": "",
        "choose_page": 1,
        "choose_page_size": PERSONS_DEFAULT_PAGE_SIZE,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --- UI Helper Functions ---
def render_pagination_controls_choose() -> tuple[bool, bool]:
    """Render pagination controls for the choose step. Return (prev_clicked, next_clicked)."""
    cols = st.columns([1, 1])
    with cols[0]:
        prev_clicked = st.button("◀ Prev", key="prev_choose_page")
    with cols[1]:
        next_clicked = st.button("Next ▶", key="next_choose_page")
    return prev_clicked, next_clicked


def render_bottom_navigation_choose(page: int, page_count: int):
    """Render bottom navigation for the choose step."""
    cols = st.columns([1, 1, 6])
    with cols[0]:
        if st.button("<< First", key="first_choose"):
            st.session_state["choose_page"] = 1
    with cols[1]:
        if st.button(">> Last", key="last_choose"):
            st.session_state["choose_page"] = page_count


def render_pagination_controls_view(
    page: int, total_pages: int
) -> tuple[bool, bool, bool, bool]:
    """Render pagination controls for the view step. Return (start, prev, next, end)."""
    cols = st.columns([1, 1, 1, 6])
    with cols[0]:
        start = st.button("⏮ Start", key="start_view")
    with cols[1]:
        prev = st.button("◀ Prev", key="prev_view")
    with cols[2]:
        next = st.button("Next ▶", key="next_view")
    with cols[3]:
        end = st.button("End ⏭", key="end_view")
    return start, prev, next, end


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


# --- Main App ---
def main():
    """Main application logic."""
    st.set_page_config(page_title="Face Clusters Viewer", layout="wide")
    st.title("Face Clusters Viewer")

    _init_session_state()

    # Check for query param to open a cluster
    open_id = _get_query_param("open")
    if open_id:
        st.session_state["selected_person_id"] = open_id
        st.session_state["ui_step"] = "view"
        _try_set_query_param("open", None)  # Clear the param

    # Step header
    if st.session_state["ui_step"] == "choose":
        st.header("1 — Choose a cluster to inspect")
        _render_choose_step()
    else:
        st.header("2 — Inspect cluster")
        _render_view_step()


def _render_choose_step():
    """Render the choose cluster step."""
    # Controls
    control_col1, control_col2, control_col3 = st.columns([4, 2, 2])

    with control_col1:
        search = st.text_input(
            "Search by name or id",
            value=st.session_state.get("choose_search", ""),
            key="choose_search",
        )

    with control_col2:
        st.selectbox(
            "Per page",
            options=[12, 24, 48, 96],
            index=[12, 24, 48, 96].index(st.session_state.get("choose_page_size", 24)),
            key="choose_page_size",
        )

    with control_col3:
        prev_clicked, next_clicked = render_pagination_controls_choose()

    # Pagination
    total = fetch_persons_count(search if search else None)
    page_count = max(1, math.ceil(total / st.session_state["choose_page_size"]))

    # Clamp current page
    st.session_state["choose_page"] = max(
        1, min(st.session_state.get("choose_page", 1), page_count)
    )

    # Handle navigation
    if prev_clicked and st.session_state["choose_page"] > 1:
        st.session_state["choose_page"] -= 1
    if next_clicked and st.session_state["choose_page"] < page_count:
        st.session_state["choose_page"] += 1

    st.write(
        f"{total} clusters — page {st.session_state['choose_page']} of {page_count}"
    )

    # Fetch page
    offset = (st.session_state["choose_page"] - 1) * st.session_state[
        "choose_page_size"
    ]
    persons = fetch_persons_page(
        search if search else None,
        limit=st.session_state["choose_page_size"],
        offset=offset,
    )

    # Render grid
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

                render_person_card_image(
                    data_url,
                    width=IMAGE_HEIGHT,
                    height=IMAGE_HEIGHT,
                    filename=os.path.basename(sample_path)
                    if sample_path
                    else "Unknown",
                )

                render_person_card_meta(name, face_count, sample_score)

                # Open button
                if st.button("Open", key=f"open_person_{person_id}"):
                    # Try query param navigation
                    if _try_set_query_param("open", str(person_id)):
                        st.stop()

                    # Fallback: server-side navigation
                    st.session_state["selected_person_id"] = person_id
                    st.session_state["ui_step"] = "view"
                    st.session_state[f"view_page_{person_id}"] = 1
                    _do_rerun()

    # Bottom navigation
    render_bottom_navigation_choose(st.session_state["choose_page"], page_count)


def _render_view_step():
    """Render the view cluster step."""
    person_id = st.session_state.get("selected_person_id")
    if not person_id:
        st.error("No person selected; returning to list.")
        st.session_state["ui_step"] = "choose"
        st.stop()

    person_row = fetch_person(person_id)
    if not person_row:
        st.error("Selected person not found in database.")
        st.session_state["ui_step"] = "choose"
        st.stop()

    _, current_name, _, face_count = (
        person_row if person_row else (None, "Unknown", None, 0)
    )

    # Header
    header_col, back_col = st.columns([8, 1])
    with header_col:
        st.markdown(f"## {html.escape(current_name or 'Unnamed')}")
        st.markdown(f"ID: `{html.escape(str(person_id))}` — Faces: {face_count}")

    with back_col:
        if st.button("Back to list", key="back_to_list"):
            if _try_set_query_param("open", None):
                st.stop()
            st.session_state["ui_step"] = "choose"
            _do_rerun()

    # Rename
    edit_key = f"name_edit_{person_id}"
    new_name = st.text_input(
        "Cluster name (optional)",
        value=current_name or "",
        key=edit_key,
    )

    if st.button("Save name", key=f"save_name_{person_id}"):
        try:
            _execute_update(
                "UPDATE persons SET name = %s WHERE id = %s",
                (new_name if new_name else None, person_id),
            )
            st.success("Name saved")
        except Exception as e:
            st.error(f"Failed to save name: {e}")

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
    start, prev, next_btn, end = render_pagination_controls_view(
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

    # Summary
    start_idx = offset + 1 if total_faces > 0 else 0
    end_idx = offset + len(faces)
    st.markdown(
        f"**Showing {start_idx}-{end_idx} of {total_faces} (page {page}/{total_pages})**"
    )

    # Render face grid
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


if __name__ == "__main__":
    main()
