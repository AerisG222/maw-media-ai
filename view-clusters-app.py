import base64
import html
import io
import mimetypes
import os
import time

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


# --- Data Fetching ---
def fetch_persons():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, cluster_label, face_count
                FROM persons
                ORDER BY COALESCE(name, ''), id
            """)
            return cur.fetchall()


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


# --- Streamlit UI ---
st.set_page_config(page_title="Face Clusters Viewer", layout="wide")
st.title("Face Clusters Viewer")

persons = fetch_persons()
if not persons:
    st.warning("No persons found in database.")
    st.stop()

person_options = [
    f"{p[1] if p[1] else 'Unnamed'} (ID: {p[0]}) [Faces: {p[3]}]" for p in persons
]
person_id_map = {person_options[i]: persons[i][0] for i in range(len(persons))}

# Persist selected person id in session state so selection survives reruns and name edits
if "selected_person_id" not in st.session_state:
    st.session_state["selected_person_id"] = persons[0][0]

# Compute index of the currently selected person for the selectbox
selected_index = 0
for i, p in enumerate(persons):
    if p[0] == st.session_state["selected_person_id"]:
        selected_index = i
        break

selected = st.sidebar.selectbox(
    "Select a person/cluster:",
    person_options,
    index=selected_index,
    key="person_select",
)
selected_person_id = person_id_map[selected]
st.session_state["selected_person_id"] = selected_person_id

# Text input for naming the selected person/cluster
person_name_map = {p[0]: p[1] for p in persons}
current_name = person_name_map.get(selected_person_id) or ""
name_key = f"name_{selected_person_id}"
new_name = st.sidebar.text_input(
    "Cluster name (optional)", value=current_name, key=name_key
)
if st.sidebar.button("Save name"):
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE persons SET name = %s WHERE id = %s",
                (new_name if new_name else None, selected_person_id),
            )
            conn.commit()
        conn.close()
        st.sidebar.success("Name saved")
        # Trigger a rerun in a way that's compatible across Streamlit versions.
        rerun_fn = getattr(st, "experimental_rerun", None)
        if callable(rerun_fn):
            rerun_fn()
        else:
            # Fallback: try to change query params which causes a rerun in many versions
            set_qp = getattr(st, "experimental_set_query_params", None)
            get_qp = getattr(st, "experimental_get_query_params", None)
            if callable(set_qp) and callable(get_qp):
                try:
                    params = get_qp() or {}
                    params["_refresh"] = str(time.time())
                    set_qp(**params)
                except Exception:
                    st.sidebar.info(
                        "Name saved — please refresh the page to see changes."
                    )
            else:
                st.sidebar.info("Name saved — please refresh the page to see changes.")
    except Exception as e:
        st.sidebar.error(f"Failed to save name: {e}")

# Pagination settings
PAGE_SIZE = 25
total_faces = fetch_face_count_for_person(selected_person_id)
total_pages = max(1, (total_faces + PAGE_SIZE - 1) // PAGE_SIZE)

# Per-person page stored in session state (so page is preserved when switching persons)
page_key = f"page_{selected_person_id}"
if page_key not in st.session_state:
    st.session_state[page_key] = 1

# Clamp stored page into current valid range
st.session_state[page_key] = max(1, min(st.session_state[page_key], total_pages))

# Sidebar page input (synchronized with session_state)
st.sidebar.markdown("### Navigation")
page_input = st.sidebar.number_input(
    "Page",
    min_value=1,
    max_value=max(1, total_pages),
    value=st.session_state[page_key],
    step=1,
)
# Update stored page to the widget's value (widget has its own internal key, so this is safe)
st.session_state[page_key] = int(page_input)

# Header in the main area
st.header(f"{selected}")

curr_page = st.session_state[page_key]

# Use the (possibly updated) page value from session state
page = st.session_state[page_key]

offset = (page - 1) * PAGE_SIZE
faces = fetch_faces_for_person(selected_person_id, limit=PAGE_SIZE, offset=offset)

start_idx = offset + 1 if total_faces > 0 else 0
end_idx = offset + len(faces)


# Use callbacks on the buttons to update session state so we don't attempt to modify
# a widget-owned key after widget creation. The callbacks are small helper lambdas.
def _set_page(new_page):
    st.session_state[page_key] = new_page


nav_col1, nav_col2, nav_col3, nav_col4 = st.sidebar.columns([1, 1, 1, 1])

start_btn = nav_col1.button(
    "⏮ Start", disabled=(curr_page == 1), on_click=_set_page, args=(1,)
)
prev_btn = nav_col2.button(
    "◀ Prev",
    disabled=(curr_page == 1),
    on_click=_set_page,
    args=(max(1, curr_page - 1),),
)
next_btn = nav_col3.button(
    "Next ▶",
    disabled=(curr_page == total_pages),
    on_click=_set_page,
    args=(min(total_pages, curr_page + 1),),
)
end_btn = nav_col4.button(
    "End ⏭",
    disabled=(curr_page == total_pages),
    on_click=_set_page,
    args=(total_pages,),
)

# Move navigation/status summary to the left sidebar
st.sidebar.markdown(f"Page **{curr_page}** of **{total_pages}**")
st.sidebar.markdown(f"Faces {start_idx}-{end_idx} of {total_faces}")

if not faces:
    st.write("No faces to show for this page.")
else:
    # Build an HTML grid and let the browser scale the images to fit the cells.
    COLS = 6
    CELL_W = 160
    CELL_H = 200
    CAPTION_H = 40
    IMG_H = CELL_H - CAPTION_H

    def _image_data_url(file_path, pil_img=None):
        """Return a data URL for the image. If pil_img is provided, encode it as PNG
        (no resizing). Otherwise return the original file bytes with guessed mime type.
        """
        try:
            if pil_img is None:
                mime, _ = mimetypes.guess_type(file_path)
                if not mime:
                    mime = "image/png"
                with open(file_path, "rb") as f:
                    data = f.read()
                return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
            else:
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        except Exception:
            return None

    cells = []
    for idx, row in enumerate(faces):
        try:
            face_id, file_path, bounding_box, score = row
        except ValueError:
            face_id, file_path, bounding_box = row
            score = None

        pil_img = load_and_crop_face(file_path, bounding_box)
        data_url = _image_data_url(file_path, pil_img)
        if not data_url:
            # fallback placeholder
            placeholder = (
                f"<div style='width:{CELL_W}px;height:{IMG_H}px;background:#EEE;display:flex;align-items:center;justify-content:center;'>"
                f"<span style='color:#999;font-size:12px;'>Could not load</span></div>"
            )
            img_html = placeholder
        else:
            alt_attr = html.escape(os.path.basename(file_path))
            img_html = (
                f'<img src="{data_url}" alt="{alt_attr}" loading="lazy" '
                f'style="max-width:{CELL_W}px;max-height:{IMG_H}px;object-fit:contain;display:block;margin:auto;"/>'
            )

        fn = html.escape(os.path.basename(file_path))
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
            f"<div style='width:{CELL_W}px;height:{CAPTION_H - (CAPTION_H // 2)}px;font-size:11px;color:#666;text-align:center;'>{html.escape(score_text)}</div>"
            f"</div>"
        )
        cells.append(cell_html)

    # Use a flexbox container so cells automatically wrap to the available width.
    flex_style = "display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;justify-content:flex-start;"
    container_html = "<div style='" + flex_style + "'>" + "".join(cells) + "</div>"
    st.markdown(container_html, unsafe_allow_html=True)
