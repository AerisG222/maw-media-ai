import os

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

selected = st.sidebar.selectbox("Select a person/cluster:", person_options)
selected_person_id = person_id_map[selected]

# Pagination settings
PAGE_SIZE = 24
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
    cols = st.columns(6)
    for idx, row in enumerate(faces):
        # row: (face_id, file_path, bounding_box, detection_score)
        try:
            face_id, file_path, bounding_box, score = row
        except ValueError:
            # backward compatibility if query returns fewer columns
            face_id, file_path, bounding_box = row
            score = None

        img = load_and_crop_face(file_path, bounding_box)
        col = cols[idx % 6]
        with col:
            if img:
                # Show the image, then display filename and score below it
                st.image(img, width=160)
                st.caption(os.path.basename(file_path))
                if score is not None:
                    try:
                        st.caption(f"Score: {float(score):.2f}")
                    except Exception:
                        st.caption(f"Score: {score}")
            else:
                st.write(f"Could not load: {file_path}")
