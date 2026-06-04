import base64
import html
import io
import json
import math
import mimetypes
import os
import time
import urllib.parse
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import psycopg2
import streamlit as st
from PIL import Image

# --- Database Connection ---
DSN = os.environ.get("FACE_SCANNER_DSN")
if not DSN:
    st.error("FACE_SCANNER_DSN environment variable not set.")
    st.stop()


def get_connection():
    return psycopg2.connect(DSN)


# --- Data Fetching Helpers ---
def fetch_persons_count(search=None):
    """Return the number of persons matching optional search (name or id)."""
    like = f"%{search}%" if search else None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM persons WHERE (%s IS NULL OR name ILIKE %s OR id::text ILIKE %s)",
                (search, like, like),
            )
            row = cur.fetchone()
            return row[0] if row else 0


def fetch_persons_page(search=None, limit=24, offset=0):
    """Return a page of persons with one sample face (photo_path, score) for preview.

    Each row: (id, name, cluster_label, face_count, sample_path, sample_score)
    """
    like = f"%{search}%" if search else None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
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
            return cur.fetchall()


def fetch_person(person_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, cluster_label, face_count FROM persons WHERE id = %s",
                (person_id,),
            )
            return cur.fetchone()


def fetch_face_count_for_person(person_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM face_detections WHERE person_id = %s",
                (person_id,),
            )
            row = cur.fetchone()
            return row[0] if row else 0


def fetch_faces_for_person(person_id, limit=24, offset=0):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
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
            return cur.fetchall()


# --- Image Utilities ---
def load_and_crop_face(image_path, bounding_box):
    try:
        img = Image.open(image_path)
        if bounding_box:
            # bounding_box is a dict with normalized coordinates
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
                face_img = img.crop((left, top, right, bottom))
                return face_img
        return img
    except Exception:
        return None


def _pil_image_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _file_to_data_url(file_path: str):
    try:
        mime, _ = mimetypes.guess_type(file_path)
        if not mime:
            mime = "image/png"
        with open(file_path, "rb") as f:
            data = f.read()
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    except Exception:
        return None


def render_image_html(data_url: str, width: int, height: int, alt: str = ""):
    alt_esc = html.escape(alt or "")
    style = (
        f"width:{width}px;height:{height}px;object-fit:contain;display:block;margin:auto;"
        "pointer-events:none;user-select:none;"
    )
    html_fragment = (
        f'<img src="{data_url}" alt="{alt_esc}" style="{style}" draggable="false" />'
    )
    st.markdown(html_fragment, unsafe_allow_html=True)


def render_clickable_image(
    data_url: str, width: int, height: int, person_id: str, alt: str = ""
):
    def render_clickable_image(
        data_url: str, width: int, height: int, person_id: str, alt: str = ""
    ):
        """Render an image as a Markdown image link to ?open=<person_id>.

        Using Markdown image-link (with angle-bracketed data URL) avoids HTML escaping
        and generally renders cleanly in Streamlit. Clicking the image navigates to
        the same page with the open parameter.
        """
        alt_esc = html.escape(alt or "")
        open_param = urllib.parse.quote_plus(str(person_id))
        href = f"?open={open_param}"
        # Use angle-bracketed data URL to avoid parentheses parsing issues in Markdown
        md = f"[![{alt_esc}](<{data_url}>)]({href})"
        st.markdown(md, unsafe_allow_html=False)


# --- Image encoding cache ---
def _get_cache_decorator(ttl_seconds: int = 3600):
    """Return a Streamlit-compatible cache decorator.

    Prefers st.cache_data (Streamlit >=1.18), falls back to experimental_memo or st.cache.
    """
    if hasattr(st, "cache_data"):

        def deco(func):
            return st.cache_data(ttl=ttl_seconds)(func)

        return deco
    if hasattr(st, "experimental_memo"):

        def deco(func):
            return st.experimental_memo(func)

        return deco
    if hasattr(st, "cache"):

        def deco(func):
            return st.cache(func)

        return deco

    # no-op
    def deco(func):
        return func

    return deco


# Disk cache directory next to this script
CACHE_DIR = Path(__file__).resolve().parent / "image-cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key_file(file_path: str, bbox_json: str, mtime: float) -> Path:
    key = f"{file_path}|{bbox_json}|{int(mtime)}"
    h = sha256(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.png"


def _atomic_write(path: Path, data: bytes):
    tmp = path.with_suffix(f".tmp-{uuid4().hex}")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(str(tmp), str(path))


def get_cached_thumbnail_path(
    file_path: str, bbox_json: str, mtime: float
) -> Path | None:
    """Return a Path to a cached PNG thumbnail. Create it if missing.

    Cached file names include the file mtime so updates to the image create new cache entries.
    """
    path = _cache_key_file(file_path, bbox_json, mtime)
    if path.exists():
        return path

    # Need to create the cached PNG
    try:
        bbox = None
        if bbox_json:
            try:
                bbox = json.loads(bbox_json)
            except Exception:
                bbox = None

        if bbox:
            img = load_and_crop_face(file_path, bbox)
        else:
            img = Image.open(file_path)

        if img is None:
            return None

        buf = io.BytesIO()
        # convert to RGB to make PNG consistent
        img.convert("RGB").save(buf, format="PNG", optimize=True)
        data = buf.getvalue()

        # write atomically
        _atomic_write(path, data)
        return path
    except Exception:
        # creation failed
        return None


@_get_cache_decorator(ttl_seconds=3600)
def get_image_data_url_cached(file_path: str, bbox_json: str, mtime: float):
    """Return a data URL for the cached PNG, creating the cached PNG if necessary."""
    try:
        cached_path = get_cached_thumbnail_path(file_path, bbox_json, mtime)
        if not cached_path:
            return None
        with open(cached_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None


# --- UI state initialization ---
if "ui_step" not in st.session_state:
    st.session_state["ui_step"] = "choose"

if "selected_person_id" not in st.session_state:
    st.session_state["selected_person_id"] = None

if "choose_search" not in st.session_state:
    st.session_state["choose_search"] = ""

if "choose_page" not in st.session_state:
    st.session_state["choose_page"] = 1

if "choose_page_size" not in st.session_state:
    st.session_state["choose_page_size"] = 24


# allow opening a cluster via query param ?open=<id>
# support multiple Streamlit APIs across versions
_qp_get = getattr(st, "experimental_get_query_params", None) or getattr(
    st, "get_query_params", None
)
_qp_set = getattr(st, "experimental_set_query_params", None) or getattr(
    st, "set_query_params", None
)
if callable(_qp_get):
    try:
        _qp = _qp_get() or {}
        # some Streamlit versions return dict[str, list[str]]; some return dict[str,str]
        _open_val = _qp.get("open") if isinstance(_qp, dict) else None
        if _open_val:
            _v = _open_val[0] if isinstance(_open_val, (list, tuple)) else _open_val
            if _v:
                st.session_state["selected_person_id"] = _v
                st.session_state["ui_step"] = "view"
                # attempt to clear the query param afterwards (best-effort)
                try:
                    if callable(_qp_set):
                        if isinstance(_qp, dict):
                            _qp.pop("open", None)
                            # set_query_params expects keyword args or a mapping depending on version
                            try:
                                _qp_set(**_qp)
                            except Exception:
                                # fallback: if it expects a single mapping
                                try:
                                    _qp_set(_qp)
                                except Exception:
                                    pass
                except Exception:
                    pass
    except Exception:
        pass


# --- Streamlit UI ---
st.set_page_config(page_title="Face Clusters Viewer", layout="wide")
st.title("Face Clusters Viewer")


# Top-level navigation: ensure a small breadcrumb so user knows which step they're on
if st.session_state["ui_step"] == "choose":
    st.header("1 — Choose a cluster to inspect")
else:
    st.header("2 — Inspect cluster")


# -----------------------------
# STEP A — Choose a cluster
# -----------------------------
if st.session_state["ui_step"] == "choose":
    # Controls
    control_col1, control_col2, control_col3 = st.columns([4, 2, 2])
    with control_col1:
        search = st.text_input(
            "Search by name or id",
            value=st.session_state.get("choose_search", ""),
            key="choose_search",
        )
    with control_col2:
        page_size = st.selectbox(
            "Per page",
            options=[12, 24, 48, 96],
            index=[12, 24, 48, 96].index(st.session_state.get("choose_page_size", 24)),
            key="choose_page_size",
        )
    with control_col3:
        # simple page controls
        prev_btn = st.button("Prev page")
        next_btn = st.button("Next page")

    # Pagination calculation
    total = fetch_persons_count(search if search else None)
    page_count = max(1, math.ceil(total / st.session_state["choose_page_size"]))
    # clamp
    st.session_state["choose_page"] = max(
        1, min(st.session_state.get("choose_page", 1), page_count)
    )

    if prev_btn and st.session_state["choose_page"] > 1:
        st.session_state["choose_page"] -= 1
    if next_btn and st.session_state["choose_page"] < page_count:
        st.session_state["choose_page"] += 1

    st.write(
        f"{total} clusters — page {st.session_state['choose_page']} of {page_count}"
    )

    # Fetch page
    offset = (st.session_state["choose_page"] - 1) * st.session_state[
        "choose_page_size"
    ]
    persons_page = fetch_persons_page(
        search if search else None,
        limit=st.session_state["choose_page_size"],
        offset=offset,
    )

    # Render results as a compact grid using Streamlit columns so buttons are inside each cell
    COLS = 6
    CELL_W = 160
    CELL_H = 200
    CAPTION_H = 72
    IMG_H = CELL_H - CAPTION_H

    for row_start in range(0, len(persons_page), COLS):
        row = persons_page[row_start : row_start + COLS]
        cols = st.columns(COLS)
        for col_idx, pr in enumerate(row):
            (
                person_id,
                name,
                cluster_label,
                face_count,
                sample_path,
                sample_score,
                sample_bbox,
            ) = pr
            with cols[col_idx]:
                # Prepare data URL using cache
                data_url = None
                if sample_path:
                    try:
                        mtime = (
                            os.path.getmtime(sample_path)
                            if os.path.exists(sample_path)
                            else 0
                        )
                    except Exception:
                        mtime = 0
                    try:
                        bbox_key = (
                            json.dumps(sample_bbox, sort_keys=True)
                            if sample_bbox
                            else ""
                        )
                    except Exception:
                        bbox_key = str(sample_bbox) if sample_bbox else ""
                    data_url = get_image_data_url_cached(sample_path, bbox_key, mtime)

                if data_url:
                    render_image_html(
                        data_url,
                        width=IMG_H,
                        height=IMG_H,
                        alt=os.path.basename(sample_path),
                    )
                else:
                    st.markdown(
                        f"<div style='width:{CELL_W}px;height:{IMG_H}px;background:#EEE;display:flex;align-items:center;justify-content:center;'><span style='color:#999;font-size:12px;'>No preview</span></div>",
                        unsafe_allow_html=True,
                    )

                # title and meta
                st.markdown(f"**{name if name else 'Unnamed'}**")
                st.markdown(f"Faces: {face_count}")
                if sample_score is not None:
                    try:
                        st.markdown(f"Top score: {float(sample_score):.2f}")
                    except Exception:
                        st.markdown(f"Top score: {sample_score}")

                # action button inside the same cell
                if st.button("Open", key=f"open_btn_{person_id}"):
                    # Preferred: set query param so browser navigates and server reads it on load
                    set_qp_fn = getattr(
                        st, "experimental_set_query_params", None
                    ) or getattr(st, "set_query_params", None)
                    did_navigate = False
                    if callable(set_qp_fn):
                        try:
                            # try keyword form
                            set_qp_fn(open=person_id)
                            did_navigate = True
                        except TypeError:
                            try:
                                set_qp_fn({"open": person_id})
                                did_navigate = True
                            except Exception:
                                did_navigate = False
                        except Exception:
                            did_navigate = False

                    if did_navigate:
                        # Rely on the client to navigate; end this run.
                        st.stop()

                    # Fallback: set session state and rerun server-side
                    st.session_state["selected_person_id"] = person_id
                    st.session_state["ui_step"] = "view"
                    st.session_state[f"view_page_{person_id}"] = 1
                    try:
                        rerun_fn = getattr(st, "experimental_rerun", None)
                        if callable(rerun_fn):
                            rerun_fn()
                    except Exception:
                        pass

    # Pagination controls at bottom
    nav_c1, nav_c2, nav_c3 = st.columns([1, 1, 6])
    with nav_c1:
        if st.button("<< First"):
            st.session_state["choose_page"] = 1
    with nav_c2:
        if st.button(">> Last"):
            st.session_state["choose_page"] = page_count


# -----------------------------
# STEP B — View / edit a selected cluster
# -----------------------------
elif st.session_state["ui_step"] == "view":
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

    # person_row: (id, name, cluster_label, face_count)
    _, current_name, cluster_label, face_count = person_row

    header_col, back_col = st.columns([8, 1])
    with header_col:
        st.markdown(f"## {current_name if current_name else 'Unnamed'}")
        st.markdown(f"ID: `{person_id}` — Faces: {face_count}")
    with back_col:
        if st.button("Back to list"):
            # Preferred: clear open param and navigate via query params if supported
            set_qp_fn = getattr(st, "experimental_set_query_params", None) or getattr(
                st, "set_query_params", None
            )
            did_navigate = False
            if callable(set_qp_fn):
                try:
                    set_qp_fn()
                    did_navigate = True
                except TypeError:
                    try:
                        set_qp_fn({})
                        did_navigate = True
                    except Exception:
                        did_navigate = False
                except Exception:
                    did_navigate = False

            if did_navigate:
                st.stop()

            st.session_state["ui_step"] = "choose"
            try:
                rerun_fn = getattr(st, "experimental_rerun", None)
                if callable(rerun_fn):
                    rerun_fn()
            except Exception:
                pass

    # Rename input and save
    edit_key = f"name_edit_{person_id}"
    new_name = st.text_input(
        "Cluster name (optional)", value=current_name or "", key=edit_key
    )
    if st.button("Save name", key=f"save_name_{person_id}"):
        try:
            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE persons SET name = %s WHERE id = %s",
                    (new_name if new_name else None, person_id),
                )
                conn.commit()
            conn.close()
            st.success("Name saved")
            # update local variable so header reflects immediately
            current_name = new_name
        except Exception as e:
            st.error(f"Failed to save name: {e}")

    # Faces pagination settings
    PAGE_SIZE = 24
    view_page_key = f"view_page_{person_id}"
    if view_page_key not in st.session_state:
        st.session_state[view_page_key] = 1
    # clamp page
    total_faces = fetch_face_count_for_person(person_id)
    total_pages = max(1, math.ceil(total_faces / PAGE_SIZE))
    st.session_state[view_page_key] = max(
        1, min(st.session_state[view_page_key], total_pages)
    )

    # Navigation for faces
    nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 6])
    if nav1.button("⏮ Start"):
        st.session_state[view_page_key] = 1
    if nav2.button("◀ Prev"):
        st.session_state[view_page_key] = max(1, st.session_state[view_page_key] - 1)
    if nav3.button("Next ▶"):
        st.session_state[view_page_key] = min(
            total_pages, st.session_state[view_page_key] + 1
        )
    if nav4.button("End ⏭"):
        st.session_state[view_page_key] = total_pages

    page = st.session_state[view_page_key]
    offset = (page - 1) * PAGE_SIZE
    faces = fetch_faces_for_person(person_id, limit=PAGE_SIZE, offset=offset)

    start_idx = offset + 1 if total_faces > 0 else 0
    end_idx = offset + len(faces)
    st.markdown(
        f"**Showing {start_idx}-{end_idx} of {total_faces} (page {page}/{total_pages})**"
    )

    # Render faces as flexbox grid (browser will scale images to fit cells)
    COLS = 6
    CELL_W = 160
    CELL_H = 200
    CAPTION_H = 40
    IMG_H = CELL_H - CAPTION_H

    cells = []
    for idx, row in enumerate(faces):
        try:
            face_id, file_path, bounding_box, score = row
        except ValueError:
            face_id, file_path, bounding_box = row
            score = None

        # Use cached encoder (file path + bbox + mtime) to avoid re-encoding on every rerun
        mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
        bbox_key = json.dumps(bounding_box, sort_keys=True) if bounding_box else ""
        data_url = get_image_data_url_cached(file_path, bbox_key, mtime)

        if not data_url:
            img_html = (
                f"<div style='width:{CELL_W}px;height:{IMG_H}px;background:#EEE;display:flex;align-items:center;justify-content:center;'>"
                f"<span style='color:#999;font-size:12px;'>Could not load</span></div>"
            )
        else:
            alt_attr = html.escape(os.path.basename(file_path))
            img_html = (
                f'<img src="{data_url}" alt="{alt_attr}" loading="lazy" '
                f'style="max-width:{CELL_W}px;max-height:{IMG_H}px;object-fit:contain;display:block;margin:auto;pointer-events:none;"/>'
            )

        score_text = ""
        if score is not None:
            try:
                score_text = f"Score: {float(score):.2f}"
            except Exception:
                score_text = f"Score: {html.escape(str(score))}"

        cell_html = (
            f"<div style='width:{CELL_W}px;height:{CELL_H}px;display:flex;flex-direction:column;align-items:center;'>"
            f"<div style='width:{CELL_W}px;height:{IMG_H}px;display:flex;align-items:center;justify-content:center;'>"
            f"{img_html}"
            f"</div>"
            f"<div style='width:{CELL_W}px;height:{CAPTION_H // 2}px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;text-overflow:ellipsis;white-space:normal;font-size:12px;color:#222;text-align:center;margin-top:6px;'>{html.escape(os.path.basename(file_path))}</div>"
            f"<div style='width:{CELL_W}px;height:{CAPTION_H - (CAPTION_H // 2)}px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;text-overflow:ellipsis;white-space:normal;font-size:11px;color:#666;text-align:center;'>{html.escape(score_text)}</div>"
            f"</div>"
        )
        cells.append(cell_html)

    flex_style = "display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;justify-content:flex-start;"
    container_html = "<div style='" + flex_style + "'>" + "".join(cells) + "</div>"
    st.markdown(container_html, unsafe_allow_html=True)
