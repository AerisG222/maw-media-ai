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

QUERY_PARAM_PERSON = "person"
QUERY_PARAM_UNKNOWN = "unknown"


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


# --- Data Fetching ---
def fetch_persons_count(search: str | None = None, unnamed_only: bool = False) -> int:
    like = f"%{search}%" if search else None
    result = execute_single(
        "SELECT COUNT(1) FROM persons WHERE (%s IS NULL OR name ILIKE %s OR id::text ILIKE %s) AND (NOT %s OR name IS NULL)",
        (search, like, like, unnamed_only),
    )
    return result[0] if result else 0


def fetch_persons_page(
    search: str | None = None, limit: int = 24, offset: int = 0, unnamed_only: bool = False
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


def fetch_faces_for_person(person_id: str, limit: int = 24, offset: int = 0) -> list:
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


def fetch_faces_for_unknown(limit: int = 24, offset: int = 0) -> list:
    """Return a page of faces for a person, ordered by detection score.

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
        k[len(prefix):]
        for k, v in st.session_state.items()
        if k.startswith(prefix) and v
    ]


def main():
    st.set_page_config(page_title="Face Clusters Viewer", layout="wide")
    st.title("Face Cluster Explorer")

    person_id = st.query_params.get(QUERY_PARAM_PERSON)
    show_unknown = st.query_params.get(QUERY_PARAM_UNKNOWN, "false") == "true"

    if person_id:
        render_faces_step(person_id)
    elif show_unknown:
        render_unknown_step()
    else:
        render_persons_step()


def render_persons_step():
    control_col1, control_col2, control_col3, control_col4 = st.columns([3, 1, 1, 2])

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
        st.checkbox("Unnamed only", key="choose_unnamed_only")

    with control_col4:
        first, prev, next, last = render_pagination_controls_persons()

    st.markdown(
        f'<a href="/?{QUERY_PARAM_UNKNOWN}=true">Unknown Faces</a>',
        unsafe_allow_html=True,
    )

    unnamed_only = st.session_state.get("choose_unnamed_only", False)

    # Pagination
    total = fetch_persons_count(search if search else None, unnamed_only=unnamed_only)
    page_count = max(1, math.ceil(total / st.session_state["choose_page_size"]))

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
    if last and st.session_state["choose_page"] > 1:
        st.session_state["choose_page"] = page_count - 1
        st.rerun()

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


def navigate_to_persons():
    dict = st.query_params.to_dict()
    dict.pop(QUERY_PARAM_PERSON, None)
    dict.pop(QUERY_PARAM_UNKNOWN, None)
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


def render_unknown_step():
    header_col, back_col = st.columns([8, 1])
    with header_col:
        st.markdown("UNCATEGORIZED FACES")

    with back_col:
        if st.button("Back to list", key="back_to_list"):
            navigate_to_persons()

    # Faces pagination
    view_page_key = "view_page_unknown"
    if view_page_key not in st.session_state:
        st.session_state[view_page_key] = 1

    total_faces = fetch_face_count_for_unknown()
    total_pages = max(1, math.ceil(total_faces / FACES_PAGE_SIZE))
    st.session_state[view_page_key] = max(
        1, min(st.session_state[view_page_key], total_pages)
    )

    # Navigation
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

    # Fetch faces
    page = st.session_state[view_page_key]
    offset = (page - 1) * FACES_PAGE_SIZE
    faces = fetch_faces_for_unknown(limit=FACES_PAGE_SIZE, offset=offset)

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
